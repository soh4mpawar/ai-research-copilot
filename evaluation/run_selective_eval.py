"""
Controlled Offline Experimental Runner for Selective Scientific Question Answering.

Features:
1. Complete 2x2 Factorial Logistic Design:
   - M1_Relevance_Raw: Relevance features, raw passages.
   - M2_Relevance_Dedup: Relevance features, deduplicated passages.
   - M3_Full_Raw: Relevance + paper/section structural features, raw passages.
   - M4_Full_Dedup: Relevance + paper/section structural features, deduplicated passages.
2. Baselines:
   - B0_AlwaysAccept: Full-coverage risk baseline (unconditional acceptance).
   - B1_ValidationTunedThreshold: Top-score threshold tuned on validation data.
   - B2_FrozenHistoricalGate: Historical 4-condition gate evaluated as discrete binary operating point.
3. Label Integrity & Provenance:
   - Derives result status from annotation provenance:
     'PENDING_SUFFICIENCY_ANNOTATION', 'PROVENANCE_FIXTURE_SYNTHETIC',
     'PROVENANCE_AI_REVIEWED_DEVELOPMENT_PILOT', 'PROVENANCE_AI_REVIEWED_HELDOUT',
     'PROVENANCE_HUMAN_VALIDATED'.
   - Never calls provisional or AI-reviewed labels 'verified'.
   - Classifiers are not trained on pending labels.
   - Zero paid API calls.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Sequence, Tuple, Optional, Union
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.selective.context import (
    FinalContextRecord,
    Passage
)
from backend.selective.dataset import (
    QuestionFamily,
    build_corpus_grounded_12_pilot,
    load_pilot_dataset_from_json,
    partition_families_by_paper_components,
    partition_families_by_manifest
)
from backend.selective.models import (
    AlwaysAcceptBaseline,
    ValidationTunedMaxThreshold,
    FrozenHistoricalGateBaseline,
    LogisticGatingModel,
    LengthDiagnosticBaseline
)
from backend.selective.metrics import (
    compute_evidence_selection_risk,
    compute_type_errors,
    compute_risk_coverage_curve,
    evaluate_at_matched_coverage,
    cluster_bootstrap_evaluation
)


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy data types and tuples to native JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _to_json_serializable(obj.tolist())
    return obj


def flatten_family_contexts(
    families: Sequence[QuestionFamily],
    family_to_component_id: Optional[Dict[str, str]] = None,
    include_variants: Optional[Sequence[str]] = None
) -> Tuple[List[FinalContextRecord], List[Optional[int]], List[str], List[str], List[str]]:
    """
    Flatten family records for evaluation while preserving:
    1. final context records
    2. sufficiency labels
    3. variant record IDs (family_id:variant_name)
    4. true family IDs (family_id)
    5. stable connected-component cluster IDs
    """
    records: List[FinalContextRecord] = []
    labels: List[Optional[int]] = []
    record_ids: List[str] = []
    true_family_ids: List[str] = []
    cluster_ids: List[str] = []

    for fam in families:
        if family_to_component_id and fam.family_id in family_to_component_id:
            cid = family_to_component_id[fam.family_id]
        elif fam.target_paper_ids:
            cid = f"paper_{fam.target_paper_ids[0]}"
        else:
            cid = f"family_{fam.family_id}"

        variants_to_use = include_variants if include_variants is not None else sorted(fam.variant_contexts.keys())
        for vname in variants_to_use:
            if vname in fam.variant_contexts:
                records.append(fam.variant_contexts[vname])
                lbl = fam.variant_sufficiency_labels.get(vname)
                labels.append(lbl)
                record_ids.append(f"{fam.family_id}:{vname}")
                true_family_ids.append(fam.family_id)
                cluster_ids.append(cid)

    return records, labels, record_ids, true_family_ids, cluster_ids


def run_selective_evaluation_pipeline(
    families: Union[Sequence[QuestionFamily], str],
    output_path: Optional[str] = None,
    annotation_provenance: Optional[str] = None,
    n_bootstrap: int = 500,
    split_manifest: Optional[Union[Dict[str, Any], str]] = None
) -> Dict[str, Any]:
    """
    Run the controlled experimental harness.
    Evaluates strictly offline with zero paid API calls.
    Can load saved QuestionFamily objects or a JSON path without rebuilding or rescoring.
    If split_manifest is provided, respects external pre-assigned splits rather than recomputing.
    """
    if isinstance(families, str):
        families = load_pilot_dataset_from_json(families)

    if output_path is None:
        output_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "data", "metadata", "selective_pilot_eval_results.json"
        ))

    t_start = time.time()

    # 1. Group-separated connected-component splits or externally supplied manifest
    if split_manifest is not None:
        train_fams, val_fams, test_fams, split_meta = partition_families_by_manifest(
            families, split_manifest
        )
    else:
        train_fams, val_fams, test_fams, split_meta = partition_families_by_paper_components(
            families, train_ratio=0.50, val_ratio=0.25, random_seed=42
        )

    fam_to_cid = split_meta.get("family_to_component_id", {})

    train_recs, train_lbls, train_rids, train_fids, train_cids = flatten_family_contexts(train_fams, fam_to_cid)
    val_recs, val_lbls, val_rids, val_fids, val_cids = flatten_family_contexts(val_fams, fam_to_cid)
    test_recs, test_lbls, test_rids, test_fids, test_cids = flatten_family_contexts(test_fams, fam_to_cid)

    # Check if labels are pending
    labels_pending = any(lbl is None for lbl in train_lbls + val_lbls + test_lbls)

    # Provenance guard: detect heuristic proxy annotations
    is_heuristic_proxy = any(
        (
            isinstance(getattr(fam, "annotation_provenance", None), str)
            and getattr(fam, "annotation_provenance", None) == "DEVELOPMENT_EXCERPT_PRESENCE_PROXY_LABELS"
        )
        or (
            isinstance(getattr(fam, "annotation_provenance", None), dict)
            and (
                getattr(fam, "annotation_provenance", {}).get("provenance_type") == "DEVELOPMENT_EXCERPT_PRESENCE_PROXY_LABELS"
                or "heuristic" in str(getattr(fam, "annotation_provenance", {}).get("method", "")).lower()
            )
        )
        for fam in families
    )
    if is_heuristic_proxy and annotation_provenance in (
        "PROVENANCE_AI_REVIEWED_DEVELOPMENT_PILOT",
        "PROVENANCE_AI_REVIEWED_HELDOUT",
        "PROVENANCE_HUMAN_VALIDATED"
    ):
        raise ValueError(
            f"Provenance guard rejection: Dataset contains heuristic proxy labels (DEVELOPMENT_EXCERPT_PRESENCE_PROXY_LABELS) "
            f"and cannot be evaluated under semantic review status '{annotation_provenance}'."
        )

    # Determine status & provenance
    if labels_pending:
        status = "PENDING_SUFFICIENCY_ANNOTATION"
        provenance_note = (
            "Genuine corpus development pilot across 12 families. "
            "Passages extracted from real ChromaDB chunks; scored via BAAI/bge-reranker-base. "
            "Sufficiency labels are pending independent human review. "
            "Classifiers are NOT trained on pending labels."
        )
    elif annotation_provenance:
        status = annotation_provenance
        provenance_note = f"Evaluated under specified annotation provenance: {annotation_provenance}."
    else:
        # Default provenance for non-pending genuine pilot
        status = "PROVENANCE_AI_REVIEWED_DEVELOPMENT_PILOT"
        provenance_note = (
            "Provisional AI-reviewed development pilot labels (reviewed by AI assistant with evidence-based rationales). "
            "NOT independent human-validated ground truth. Strictly internal development benchmark; "
            "not claimed as publication-ready or human-verified."
        )

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "provenance_note": provenance_note,
        "split_metadata": split_meta,
        "sample_counts": {
            "total_families": len(families),
            "train_contexts": len(train_recs),
            "val_contexts": len(val_recs),
            "test_contexts": len(test_recs)
        },
        "model_evaluations": {}
    }

    if labels_pending:
        # Output audit table of scores and contexts without fabricating metrics
        audit_records = []
        b2 = FrozenHistoricalGateBaseline()
        for fam in families:
            s_ev = fam.supporting_evidence
            rec_clean = fam.variant_contexts["clean"]
            audit_records.append({
                "family_id": fam.family_id,
                "target_paper": fam.target_paper_ids[0] if fam.target_paper_ids else "unknown",
                "support_chunk_id": s_ev.chunk_id,
                "support_section": s_ev.section,
                "support_cross_encoder_score": s_ev.cross_encoder_score,
                "verified_excerpt": s_ev.verified_excerpt,
                "b2_frozen_gate_decision_clean": b2.decide(rec_clean),
                "variant_status": fam.variant_sufficiency_status
            })
        results["dataset_audit_table"] = audit_records
        results["total_elapsed_sec"] = round(time.time() - t_start, 3)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(_to_json_serializable(results), fp, indent=2)

        print(f"Pilot evaluation audit written to {output_path} (Status: {status})")
        return results

    # Full evaluation with non-null labels
    valid_train_lbls = [int(x) for x in train_lbls]
    valid_val_lbls = [int(x) for x in val_lbls]
    valid_test_lbls = [int(x) for x in test_lbls]

    train_pos = sum(valid_train_lbls)
    train_neg = len(valid_train_lbls) - train_pos
    results["training_class_balance"] = {
        "positive_count": train_pos,
        "negative_count": train_neg,
        "is_binary": bool(train_pos > 0 and train_neg > 0)
    }

    # B0: Always Accept Baseline
    b0 = AlwaysAcceptBaseline()
    b0_test_decisions = [b0.decide(r) for r in test_recs]
    b0_cov = float(np.mean(b0_test_decisions))
    b0_risk = compute_evidence_selection_risk(b0_test_decisions, valid_test_lbls)
    b0_t1, b0_t2 = compute_type_errors(b0_test_decisions, valid_test_lbls)

    results["model_evaluations"]["B0_AlwaysAccept"] = {
        "evaluation_type": "full_coverage_risk_baseline",
        "achieved_coverage": round(b0_cov, 4),
        "accepted_count": int(np.sum(b0_test_decisions)),
        "evidence_selection_risk": round(b0_risk, 4) if b0_risk is not None else None,
        "type1_error_insufficient_accepted": round(b0_t1, 4) if b0_t1 is not None else None,
        "type2_error_sufficient_rejected": round(b0_t2, 4) if b0_t2 is not None else None
    }

    # B2: Historical Gate Baseline (discrete binary operating point)
    b2 = FrozenHistoricalGateBaseline()
    b2_test_decisions = [b2.decide(r) for r in test_recs]
    b2_cov = float(np.mean(b2_test_decisions))
    b2_risk = compute_evidence_selection_risk(b2_test_decisions, valid_test_lbls)
    b2_t1, b2_t2 = compute_type_errors(b2_test_decisions, valid_test_lbls)

    results["model_evaluations"]["B2_FrozenHistoricalGate"] = {
        "evaluation_type": "discrete_binary_decision_baseline",
        "achieved_coverage": round(b2_cov, 4),
        "accepted_count": int(np.sum(b2_test_decisions)),
        "evidence_selection_risk": round(b2_risk, 4) if b2_risk is not None else None,
        "type1_error_insufficient_accepted": round(b2_t1, 4) if b2_t1 is not None else None,
        "type2_error_sufficient_rejected": round(b2_t2, 4) if b2_t2 is not None else None
    }

    # Ranking Models
    b_diag = LengthDiagnosticBaseline()
    b_diag.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    b_diag.fit_validation_thresholds(val_recs)

    b1 = ValidationTunedMaxThreshold()
    b1.fit_validation_thresholds(val_recs, valid_val_lbls)

    m1 = LogisticGatingModel(name="M1_Relevance_Raw", feature_mode="relevance_only", deduplicate=False)
    m1.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m1.fit_validation_thresholds(val_recs)

    m2 = LogisticGatingModel(name="M2_Relevance_Dedup", feature_mode="relevance_only", deduplicate=True)
    m2.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m2.fit_validation_thresholds(val_recs)

    m3 = LogisticGatingModel(name="M3_Full_Raw", feature_mode="combined", deduplicate=False)
    m3.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m3.fit_validation_thresholds(val_recs)

    m4 = LogisticGatingModel(name="M4_Full_Dedup", feature_mode="combined", deduplicate=True)
    m4.fit(train_recs, valid_train_lbls, val_recs, valid_val_lbls)
    m4.fit_validation_thresholds(val_recs)

    ranking_models = {
        "B_Diag_Length": b_diag,
        "B1_ValidationTunedThreshold": b1,
        "M1_Relevance_Raw": m1,
        "M2_Relevance_Dedup": m2,
        "M3_Full_Raw": m3,
        "M4_Full_Dedup": m4
    }

    scores_dict: Dict[str, List[float]] = {
        "B0_AlwaysAccept": [1.0] * len(test_recs),
        "B2_FrozenHistoricalGate": [float(d) for d in b2_test_decisions]
    }
    preds_dict: Dict[str, List[int]] = {
        "B0_AlwaysAccept": b0_test_decisions,
        "B2_FrozenHistoricalGate": b2_test_decisions
    }

    target_points = [0.50, 0.75, 0.90]

    for name, m in ranking_models.items():
        test_scores = [float(m.predict_score(r)) for r in test_recs]
        scores_dict[name] = test_scores
        preds_dict[name] = [m.decide(r) for r in test_recs]

        curve = compute_risk_coverage_curve(test_scores, valid_test_lbls)
        matched = evaluate_at_matched_coverage(curve, target_points)

        # Evaluate deployable operating points using frozen validation thresholds
        deployable_ops = {}
        for cov in target_points:
            cov_key = f"target_cov_{cov:.2f}"
            thresh = m.operating_thresholds.get(cov, m.operating_thresholds.get(cov_key))
            if thresh is not None:
                # Apply frozen validation threshold to test scores
                op_decisions = [1 if s >= thresh else 0 for s in test_scores]
                op_cov = float(np.mean(op_decisions))
                op_acc_count = int(np.sum(op_decisions))
                op_risk = compute_evidence_selection_risk(op_decisions, valid_test_lbls)
                op_t1, op_t2 = compute_type_errors(op_decisions, valid_test_lbls)

                deployable_ops[cov_key] = {
                    "frozen_validation_threshold": round(thresh, 4),
                    "achieved_test_coverage": round(op_cov, 4),
                    "accepted_count": op_acc_count,
                    "evidence_selection_risk": round(op_risk, 4) if op_risk is not None else None,
                    "type1_error_insufficient_accepted": round(op_t1, 4) if op_t1 is not None else None,
                    "type2_error_sufficient_rejected": round(op_t2, 4) if op_t2 is not None else None
                }
                # Store for bootstrap at this specific operating point
                scores_dict[f"{name}@{cov_key}"] = test_scores
                preds_dict[f"{name}@{cov_key}"] = op_decisions
            else:
                deployable_ops[cov_key] = {
                    "frozen_validation_threshold": None,
                    "achieved_test_coverage": 0.0,
                    "accepted_count": 0,
                    "evidence_selection_risk": None,
                    "type1_error_insufficient_accepted": None,
                    "type2_error_sufficient_rejected": None
                }

        results["model_evaluations"][name] = {
            "evaluation_type": "continuous_ranking_model",
            "aurc": curve["aurc"],
            "is_all_tied": curve["is_all_tied"],
            "matched_coverage_descriptive": matched,
            "deployable_operating_points": deployable_ops
        }

    # Clustered Bootstrap across connected paper components
    boot_res = cluster_bootstrap_evaluation(
        cluster_ids=test_cids,
        predictions_dict=preds_dict,
        scores_dict=scores_dict,
        labels=valid_test_lbls,
        n_bootstrap=n_bootstrap
    )
    results["cluster_bootstrap"] = boot_res
    results["total_elapsed_sec"] = round(time.time() - t_start, 3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(_to_json_serializable(results), fp, indent=2)

    print(f"Selective evaluation written to {output_path} (Status: {status})")
    return results


if __name__ == "__main__":
    pilot_json_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "metadata", "pilot_dataset_12families.json"
    ))
    if os.path.exists(pilot_json_path):
        fams = load_pilot_dataset_from_json(pilot_json_path)
    else:
        fams = build_corpus_grounded_12_pilot()
    run_selective_evaluation_pipeline(fams)
