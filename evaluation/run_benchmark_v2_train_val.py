"""
Benchmark v2: Dedicated Train / Validation Runner & Protocol Freeze.
Strictly implements Protocol Amendment v1.1.

Loads ONLY:
- 240 authoritative training contexts (data/metadata/benchmark_v2_training_contexts_240.json)
- 240 authoritative training semantic annotations (data/metadata/benchmark_v2_training_semantic_annotations_240.json)
- 60 authoritative training question specs (data/metadata/benchmark_v2_training_questions_60families.json)
- 96 authoritative validation contexts (data/metadata/benchmark_v2_validation_contexts_96.json)
- 96 authoritative validation semantic annotations (data/metadata/benchmark_v2_validation_semantic_annotations_96.json)
- 24 authoritative validation question specs (data/metadata/benchmark_v2_validation_questions_24families.json)

Rejects all proxy provenance, mismatched hashes, nonfinite scores, and invalid labels.
Preserves passage order, duplicate occurrences, source IDs, section labels, and actual reranker scores.
Recomputes token counts rather than trusting stored totals.
Fits: B0, B1, B2 (discrete), M1, M2, M3, M4, and Shortcut Diagnostic.
Selects operating thresholds on the pooled validation set for target coverages 0.50, 0.75, 0.90.
Reports validation results separately for:
1. Controlled contexts (72 contexts)
2. Natural retrieval contexts (24 contexts)
3. Pooled mixture (96 contexts)

Freezes final evaluation specification and saves fully reproducible fitted model artifacts.
"""

import json
import hashlib
import sys
import os
import copy
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional, Sequence
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from backend.selective.context import (
    Passage, FinalContextRecord, format_context_payload, count_tokens, deduplicate_passages_exact
)
from backend.selective.features import (
    extract_relevance_features, extract_source_features, extract_features_from_record
)
from backend.selective.models import (
    AlwaysAcceptBaseline, ValidationTunedMaxThreshold, FrozenHistoricalGateBaseline,
    LogisticGatingModel, LengthDiagnosticBaseline
)
from backend.selective.metrics import (
    validate_metric_inputs, compute_evidence_selection_risk, compute_type_errors,
    compute_risk_coverage_curve
)


# ------------------------------------------------------------------------------
# 1. Strict Data Loading & Integrity Verification
# ------------------------------------------------------------------------------

def load_and_verify_split(split_name: str) -> Dict[str, Any]:
    """
    Loads contexts, annotations, and questions for a split ('train' or 'val').
    Executes all required integrity checks.
    """
    if split_name == "train":
        expected_n_ctx = 240
        expected_n_fam = 60
        id_prefix = "ctx_tr_"
        c_path = "data/metadata/benchmark_v2_training_contexts_240.json"
        a_path = "data/metadata/benchmark_v2_training_semantic_annotations_240.json"
        q_path = "data/metadata/benchmark_v2_training_questions_60families.json"
    elif split_name == "val":
        expected_n_ctx = 96
        expected_n_fam = 24
        id_prefix = "ctx_val_"
        c_path = "data/metadata/benchmark_v2_validation_contexts_96.json"
        a_path = "data/metadata/benchmark_v2_validation_semantic_annotations_96.json"
        q_path = "data/metadata/benchmark_v2_validation_questions_24families.json"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    print(f"\n--- Loading and Verifying {split_name.upper()} Dataset ---")
    print(f"  Contexts:    {c_path}")
    print(f"  Annotations: {a_path}")
    print(f"  Questions:   {q_path}")

    contexts = json.load(open(c_path, encoding="utf-8"))
    ann_wrapper = json.load(open(a_path, encoding="utf-8"))
    q_data = json.load(open(q_path, encoding="utf-8"))

    # 1. Check provenance status
    prov = str(ann_wrapper.get("provenance_status", ""))
    if not prov.startswith("INDIVIDUALLY_AUTHORED_SEMANTIC_ANNOTATIONS"):
        raise ValueError(f"Split {split_name} has invalid provenance: {prov}. Rejecting proxy labels!")
    annotations = ann_wrapper.get("annotations", ann_wrapper)

    # 2. Check counts
    if len(contexts) != expected_n_ctx:
        raise ValueError(f"Expected {expected_n_ctx} contexts, got {len(contexts)}")
    if len(annotations) != expected_n_ctx:
        raise ValueError(f"Expected {expected_n_ctx} annotations, got {len(annotations)}")
    
    questions = q_data if isinstance(q_data, list) else q_data.get("questions", [])
    if len(questions) != expected_n_fam:
        raise ValueError(f"Expected {expected_n_fam} questions, got {len(questions)}")
    q_map = {q["family_id"]: q for q in questions}

    # 3. Check sequencing and duplicate IDs
    ctx_ids = [c["opaque_context_id"] for c in contexts]
    ann_ids = [a["opaque_context_id"] for a in annotations]

    if len(set(ctx_ids)) != expected_n_ctx:
        raise ValueError(f"Duplicate context IDs in {split_name} contexts!")
    if len(set(ann_ids)) != expected_n_ctx:
        raise ValueError(f"Duplicate context IDs in {split_name} annotations!")
    if set(ctx_ids) != set(ann_ids):
        raise ValueError(f"Context IDs and Annotation IDs do not match in {split_name}!")

    for idx, cid in enumerate(ctx_ids, 1):
        expected_cid = f"{id_prefix}{idx:04d}"
        if cid != expected_cid:
            raise ValueError(f"Context ID out of order at index {idx}: got {cid}, expected {expected_cid}")

    ann_map = {a["opaque_context_id"]: a for a in annotations}

    # 4. Check conditions per family
    family_conditions = {}
    records = []
    labels = []
    meta_list = []

    for c in contexts:
        cid = c["opaque_context_id"]
        fid = c["family_id"]
        cond = c["variant"]
        pid = c["paper_id"]
        passages_raw = c["passages"]
        stored_payload_hash = c["payload_hash"]

        family_conditions.setdefault(fid, []).append(cond)

        # Question verification
        if fid not in q_map:
            raise ValueError(f"Family {fid} in context {cid} not found in questions!")
        q_text = (c.get("question") or q_map[fid]["question"]).strip()
        if q_map[fid]["question"].strip() != q_text:
            raise ValueError(f"Question text mismatch for family {fid} in context {cid}!")
        q_hash = hashlib.sha256(q_text.encode("utf-8")).hexdigest()

        # Annotation verification
        ann = ann_map[cid]
        ann_q_hash = ann.get("question_hash")
        if ann_q_hash != q_hash:
            raise ValueError(f"Question hash mismatch in annotation {cid}!")

        ann_payload_hash = ann.get("payload_hash")
        if ann_payload_hash and ann_payload_hash != stored_payload_hash:
            raise ValueError(f"Payload hash mismatch in annotation {cid}!")

        judgment = ann.get("judgment") or ann.get("semantic_judgment")
        if judgment not in ["sufficient", "insufficient"]:
            raise ValueError(f"Invalid judgment {judgment} in {cid}. Only binary permitted!")
        label = 1 if judgment == "sufficient" else 0
        labels.append(label)

        # Build passages with recomputed token counts and verified scores
        passages = []
        chunk_scores = {}
        for p in passages_raw:
            if "rerank_score" not in p or p["rerank_score"] is None:
                raise ValueError(f"Missing rerank_score in {cid} passage {p.get('chunk_id')}! Defaulting to 0.0 is prohibited.")
            score = float(p["rerank_score"])
            if not np.isfinite(score):
                raise ValueError(f"Non-finite reranker score {score} in {cid} passage {p.get('chunk_id')}")

            # Verify duplicate occurrences receive identical scores to their base chunk
            base_cid = p["chunk_id"].replace("_dup", "")
            if base_cid in chunk_scores:
                diff = abs(chunk_scores[base_cid] - score)
                if diff > 1e-9:
                    raise ValueError(f"Duplicate score mismatch in {cid} for {p['chunk_id']}: base={chunk_scores[base_cid]} vs dup={score}")
            else:
                chunk_scores[base_cid] = score

            raw_text = p["text"]
            # Recompute token counts rather than trusting stored values
            recomputed_tokens = count_tokens(raw_text)

            passages.append(Passage.from_dict({
                "chunk_id": p["chunk_id"],
                "paper_id": p["paper_id"],
                "section": p.get("section", "General"),
                "rerank_score": score,
                "token_count": recomputed_tokens,
                "text": raw_text
            }))

        # Verify payload serialization and hash
        reconstructed_payload = format_context_payload(passages)
        recomputed_payload_hash = hashlib.sha256(reconstructed_payload.encode("utf-8")).hexdigest()
        if recomputed_payload_hash != stored_payload_hash:
            raise ValueError(f"Recomputed payload hash mismatch in {cid}!")

        # Wrap in FinalContextRecord
        rec = FinalContextRecord(
            query=q_text,
            passages=tuple(passages),
            max_passages=len(passages),
            token_budget=2000,
            tokenizer_identity="cl100k_base",
            total_tokens=sum(p.token_count for p in passages),
            skipped_passages=(),
            truncation_policy="no_truncation",
            serialized_payload=reconstructed_payload,
            payload_hash=recomputed_payload_hash
        )
        # Attach context_id as attribute
        object.__setattr__(rec, "context_id", cid)
        records.append(rec)
        meta_list.append({
            "opaque_context_id": cid,
            "family_id": fid,
            "condition": cond,
            "paper_id": pid,
            "label": label,
            "judgment": judgment,
            "passage_count": len(passages),
            "recomputed_tokens": sum(p.token_count for p in passages)
        })

    # Verify each family has exactly the 4 expected conditions
    expected_conditions = {"supported_candidate", "relevant_replacement", "duplicate_replacement", "natural_retrieval"}
    for fid, cond_list in family_conditions.items():
        if set(cond_list) != expected_conditions or len(cond_list) != 4:
            raise ValueError(f"Family {fid} does not have the 4 expected conditions: {cond_list}")

    print(f"Verified {len(records)} records across {len(family_conditions)} families in {split_name}.")
    label_counts = Counter(labels)
    print(f"  Class 1 (Sufficient):   {label_counts[1]} ({label_counts[1]/len(labels)*100:.1f}%)")
    print(f"  Class 0 (Insufficient): {label_counts[0]} ({label_counts[0]/len(labels)*100:.1f}%)")

    scores_repr = [
        {
            "opaque_context_id": c["opaque_context_id"],
            "passage_scores": [(p["chunk_id"], round(float(p["rerank_score"]), 6)) for p in c["passages"]]
        }
        for c in contexts
    ]
    scores_hash = hashlib.sha256(json.dumps(scores_repr, sort_keys=True).encode("utf-8")).hexdigest()
    scoring_provenance = {
        "model": "BAAI/bge-reranker-base",
        "device": "cuda:0",
        "activation": "sigmoid",
        "max_length": 512,
        "precision": "fp32",
        "cache_policy": "query_text_config_exact_match (exact_score_match_for_duplicates)",
        "generating_script": (
            "scratch/build_v2_repaired_training_contexts.py"
            if split_name == "train"
            else "scratch/build_v2_repaired_val_contexts.py"
        ),
        "scores_hash": scores_hash
    }

    return {
        "records": records,
        "labels": labels,
        "meta": meta_list,
        "contexts_raw": contexts,
        "annotations_raw": annotations,
        "questions_raw": questions,
        "scores_hash": scores_hash,
        "scoring_provenance": scoring_provenance
    }


# ------------------------------------------------------------------------------
# 2. Model Configuration & Fitting Protocol
# ------------------------------------------------------------------------------

def get_model_feature_names(feature_mode: str) -> List[str]:
    rel_names = ["top1_score", "top2_score", "mean_score", "score_spread"]
    src_names = ["s_top_paper_second_passage", "s_other", "section_diversity", "section_mean_score"]
    if feature_mode == "relevance_only":
        return rel_names
    elif feature_mode == "combined":
        return rel_names + src_names
    elif feature_mode == "shortcut":
        return ["passage_count", "total_token_count"]
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")


def fit_all_models(
    train_records: List[FinalContextRecord],
    train_labels: List[int],
    val_records: List[FinalContextRecord],
    val_labels: List[int]
) -> Dict[str, Any]:
    """
    Fits B0, B1, B2 (discrete), M1, M2, M3, M4, and Shortcut Diagnostic.
    Strictly follows:
    - StandardScaler fit strictly on training data.
    - C grid: [0.01, 0.1, 1.0, 10.0, 100.0].
    - Deterministic smaller-C tie-breaking on validation log-loss.
    - Full-precision thresholds selected on pooled validation set for coverages [0.50, 0.75, 0.90].
    """
    print("\n=======================================================")
    print("FITTING MODELS & BASELINES UNDER FROZEN SPECIFICATION")
    print("=======================================================")
    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    target_coverages = [0.50, 0.75, 0.90]

    print(f"Hyperparameter Grid: C in {c_grid}")
    print(f"Selection Criterion: Validation Log-Loss on pooled validation set (96 contexts)")
    print(f"Tie Rule:            Ascending C order (prefer smaller C / stronger regularization on tie)")
    print(f"Target Coverages:    {target_coverages}")

    models = {}

    # B0: Always Accept
    b0 = AlwaysAcceptBaseline()
    models["B0_AlwaysAccept"] = {
        "instance": b0,
        "type": "baseline_always_accept",
        "description": "Baseline B0: Unconditional acceptance (Coverage = 1.0)"
    }

    # B1: Max Relevance
    b1 = ValidationTunedMaxThreshold()
    # Select full-precision validation thresholds
    val_scores_b1 = np.array([b1.predict_score(r) for r in val_records], dtype=np.float64)
    n_val = len(val_scores_b1)
    sorted_scores_b1 = np.sort(val_scores_b1)[::-1]
    b1_thresholds = {}
    for cov in target_coverages:
        k = max(1, min(n_val, int(round(cov * n_val))))
        th = float(sorted_scores_b1[k - 1])
        achieved = float(np.mean(val_scores_b1 >= th))
        b1_thresholds[cov] = {
            "threshold": th,
            "target_coverage": cov,
            "val_achieved_coverage": achieved,
            "val_deviation": abs(achieved - cov)
        }
        b1.operating_thresholds[cov] = th
    models["B1_MaxRelevance"] = {
        "instance": b1,
        "type": "baseline_max_relevance",
        "thresholds": b1_thresholds,
        "description": "Baseline B1: Maximum passage reranker score"
    }

    # B2: Frozen Historical Gate
    b2 = FrozenHistoricalGateBaseline()
    models["B2_HistoricalGate"] = {
        "instance": b2,
        "type": "baseline_discrete_heuristic",
        "description": "Baseline B2: Frozen historical 4-condition heuristic gate (discrete operating point)"
    }

    # Learned Models: M1, M2, M3, M4
    learned_configs = [
        ("M1_Relevance_Raw", "relevance_only", False, "M1: Relevance features, raw passages"),
        ("M2_Relevance_Dedup", "relevance_only", True, "M2: Relevance features, deduplicated passages"),
        ("M3_Full_Raw", "combined", False, "M3: Relevance + source/section structural features, raw passages"),
        ("M4_Full_Dedup", "combined", True, "M4: Relevance + source/section structural features, deduplicated passages")
    ]

    for model_name, feat_mode, dedup, desc in learned_configs:
        print(f"\n--- Fitting {model_name} ---")
        clf_model = LogisticGatingModel(
            name=model_name,
            feature_mode=feat_mode,
            deduplicate=dedup
        )

        # 1. Extract raw features
        X_train_raw = clf_model.extract_features(train_records)
        X_val_raw = clf_model.extract_features(val_records)
        y_train = np.array(train_labels, dtype=int)
        y_val = np.array(val_labels, dtype=int)

        # 2. Fit scaler STRICTLY on training data
        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        # 3. Deterministic C selection on validation log-loss
        best_c = c_grid[0]
        best_val_loss = float("inf")
        c_eval_records = []

        for c_cand in sorted(c_grid):
            lr = LogisticRegression(penalty="l2", C=c_cand, solver="liblinear", random_state=42)
            lr.fit(X_train_scaled, y_train)
            val_probs = lr.predict_proba(X_val_scaled)
            loss = float(log_loss(y_val, val_probs, labels=[0, 1]))
            c_eval_records.append({"C": c_cand, "val_log_loss": loss})

            if loss < best_val_loss - 1e-6:
                best_val_loss = loss
                best_c = c_cand

        print(f"  C selection log-loss: {c_eval_records}")
        print(f"  Selected C: {best_c} (val_log_loss: {best_val_loss:.6f})")

        # 4. Final fit with selected C
        final_lr = LogisticRegression(penalty="l2", C=best_c, solver="liblinear", random_state=42)
        final_lr.fit(X_train_scaled, y_train)

        clf_model.scaler = scaler
        clf_model.model = final_lr
        clf_model.C = best_c
        clf_model.is_fitted = True

        # 5. Select operating thresholds on pooled validation set
        val_scores = np.array([clf_model.predict_score(r) for r in val_records], dtype=np.float64)
        sorted_val_scores = np.sort(val_scores)[::-1]
        thresholds_dict = {}
        for cov in target_coverages:
            k = max(1, min(n_val, int(round(cov * n_val))))
            th = float(sorted_val_scores[k - 1])
            achieved = float(np.mean(val_scores >= th))
            thresholds_dict[cov] = {
                "threshold": th,
                "target_coverage": cov,
                "val_achieved_coverage": achieved,
                "val_deviation": abs(achieved - cov)
            }
            clf_model.operating_thresholds[cov] = th
            print(f"  Target Cov {cov:.2f} -> Threshold: {th:.6f} | Achieved: {achieved:.4f} (dev: {abs(achieved-cov):.4f})")

        feat_names = get_model_feature_names(feat_mode)
        models[model_name] = {
            "instance": clf_model,
            "type": "logistic_gating_model",
            "feature_mode": feat_mode,
            "deduplicate": dedup,
            "feature_names": feat_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "selected_C": best_c,
            "coefficients": final_lr.coef_[0].tolist(),
            "intercept": float(final_lr.intercept_[0]),
            "thresholds": thresholds_dict,
            "c_tuning_log": c_eval_records,
            "description": desc
        }

    # Shortcut Diagnostic Baseline (passage count, total token count)
    print("\n--- Fitting Shortcut Diagnostic Baseline ---")
    diag_model = LengthDiagnosticBaseline()
    X_train_diag = diag_model.extract_features(train_records)
    X_val_diag = diag_model.extract_features(val_records)
    y_train = np.array(train_labels, dtype=int)
    y_val = np.array(val_labels, dtype=int)

    diag_scaler = StandardScaler()
    diag_scaler.fit(X_train_diag)
    X_train_diag_scaled = diag_scaler.transform(X_train_diag)
    X_val_diag_scaled = diag_scaler.transform(X_val_diag)

    # C selection for shortcut diagnostic
    best_c_diag = c_grid[0]
    best_val_loss_diag = float("inf")
    c_eval_records_diag = []
    for c_cand in sorted(c_grid):
        lr_diag = LogisticRegression(penalty="l2", C=c_cand, solver="liblinear", random_state=42)
        lr_diag.fit(X_train_diag_scaled, y_train)
        val_probs = lr_diag.predict_proba(X_val_diag_scaled)
        loss = float(log_loss(y_val, val_probs, labels=[0, 1]))
        c_eval_records_diag.append({"C": c_cand, "val_log_loss": loss})
        if loss < best_val_loss_diag - 1e-6:
            best_val_loss_diag = loss
            best_c_diag = c_cand

    final_lr_diag = LogisticRegression(penalty="l2", C=best_c_diag, solver="liblinear", random_state=42)
    final_lr_diag.fit(X_train_diag_scaled, y_train)

    diag_model.scaler = diag_scaler
    diag_model.model = final_lr_diag
    diag_model.is_fitted = True

    val_scores_diag = np.array([diag_model.predict_score(r) for r in val_records], dtype=np.float64)
    sorted_val_scores_diag = np.sort(val_scores_diag)[::-1]
    diag_thresholds = {}
    for cov in target_coverages:
        k = max(1, min(n_val, int(round(cov * n_val))))
        th = float(sorted_val_scores_diag[k - 1])
        achieved = float(np.mean(val_scores_diag >= th))
        diag_thresholds[cov] = {
            "threshold": th,
            "target_coverage": cov,
            "val_achieved_coverage": achieved,
            "val_deviation": abs(achieved - cov)
        }
        diag_model.operating_thresholds[cov] = th

    models["Shortcut_LengthDiagnostic"] = {
        "instance": diag_model,
        "type": "diagnostic_shortcut_baseline",
        "feature_names": ["passage_count", "total_token_count"],
        "scaler_mean": diag_scaler.mean_.tolist(),
        "scaler_scale": diag_scaler.scale_.tolist(),
        "selected_C": best_c_diag,
        "coefficients": final_lr_diag.coef_[0].tolist(),
        "intercept": float(final_lr_diag.intercept_[0]),
        "thresholds": diag_thresholds,
        "c_tuning_log": c_eval_records_diag,
        "description": "Diagnostic Baseline: Passage count and token count shortcuts (clearly separate from proposed methods)"
    }

    return models


# ------------------------------------------------------------------------------
# 3. Validation Evaluation by Context Subset
# ------------------------------------------------------------------------------

def evaluate_subset(
    subset_name: str,
    records: List[FinalContextRecord],
    labels: List[int],
    models: Dict[str, Any],
    target_coverages: Sequence[float] = (0.50, 0.75, 0.90)
) -> Dict[str, Any]:
    """
    Evaluates all models on a specific validation subset (controlled, natural, or pooled).
    Computes:
    - Sample size & insufficient prevalence
    - Discrete tie-correct AURC
    - Performance at validation-frozen thresholds
    - Discrete operating point for B2
    """
    n = len(records)
    n_insuff = int(sum(1 for y in labels if y == 0))
    n_suff = int(sum(1 for y in labels if y == 1))
    prevalence_insuff = float(n_insuff / n) if n > 0 else 0.0

    subset_results = {
        "subset_name": subset_name,
        "sample_count": n,
        "sufficient_count": n_suff,
        "insufficient_count": n_insuff,
        "insufficient_prevalence": prevalence_insuff,
        "models": {}
    }

    # Evaluate each model
    for model_name, m_info in models.items():
        inst = m_info["instance"]
        m_type = m_info["type"]

        if m_type == "baseline_discrete_heuristic":
            # B2: Single discrete operating point
            decisions = np.array([inst.decide(r) for r in records], dtype=int)
            accepted = int(np.sum(decisions == 1))
            coverage = float(accepted / n) if n > 0 else 0.0
            risk = compute_evidence_selection_risk(decisions, labels)
            type1, type2 = compute_type_errors(decisions, labels)

            subset_results["models"][model_name] = {
                "type": m_type,
                "discrete_operating_point": {
                    "accepted_count": accepted,
                    "coverage": coverage,
                    "evidence_selection_risk": risk,
                    "type_1_error": type1,
                    "type_2_error": type2
                }
            }
            continue

        if m_type == "baseline_always_accept":
            # B0: Always accept
            scores = np.ones(n, dtype=np.float64)
            decisions = np.ones(n, dtype=int)
            curve = compute_risk_coverage_curve(scores, labels)
            risk = compute_evidence_selection_risk(decisions, labels)
            type1, type2 = compute_type_errors(decisions, labels)

            subset_results["models"][model_name] = {
                "type": m_type,
                "aurc": curve["aurc"],
                "coverage_1_0": {
                    "coverage": 1.0,
                    "evidence_selection_risk": risk,
                    "type_1_error": type1,
                    "type_2_error": type2
                }
            }
            continue

        # Continuous scoring models (B1, M1, M2, M3, M4, Shortcut)
        scores = np.array([inst.predict_score(r) for r in records], dtype=np.float64)
        curve = compute_risk_coverage_curve(scores, labels)

        # Evaluate at frozen validation thresholds
        threshold_evals = {}
        frozen_thresholds = m_info["thresholds"]

        for cov in target_coverages:
            th = frozen_thresholds[cov]["threshold"]
            # Decision rule: score >= th
            decisions = (scores >= th).astype(int)
            acc_count = int(np.sum(decisions == 1))
            achieved_cov = float(acc_count / n) if n > 0 else 0.0
            risk = compute_evidence_selection_risk(decisions, labels)
            type1, type2 = compute_type_errors(decisions, labels)

            threshold_evals[f"target_{cov:.2f}"] = {
                "frozen_threshold": th,
                "target_coverage": cov,
                "achieved_coverage": achieved_cov,
                "coverage_deviation": abs(achieved_cov - cov),
                "accepted_count": acc_count,
                "evidence_selection_risk": risk,
                "type_1_error": type1,
                "type_2_error": type2
            }

        subset_results["models"][model_name] = {
            "type": m_type,
            "aurc": curve["aurc"],
            "frozen_operating_points": threshold_evals
        }

    return subset_results


# ------------------------------------------------------------------------------
# 4. Main Runner & Evaluator
# ------------------------------------------------------------------------------

def main():
    print("=================================================================")
    print("BENCHMARK V2: DEDICATED TRAIN / VALIDATION RUNNER & PROTOCOL FREEZE")
    print("=================================================================")

    # 1. Load and strictly verify datasets
    train_data = load_and_verify_split("train")
    val_data = load_and_verify_split("val")

    # 2. Fit models on training data and tune on pooled validation set
    models = fit_all_models(
        train_records=train_data["records"],
        train_labels=train_data["labels"],
        val_records=val_data["records"],
        val_labels=val_data["labels"]
    )

    # 3. Partition validation records into subsets
    # - Controlled: supported_candidate, relevant_replacement, duplicate_replacement (72 contexts)
    # - Natural: natural_retrieval (24 contexts)
    # - Pooled: all 96 contexts
    val_controlled_idx = [i for i, m in enumerate(val_data["meta"]) if m["condition"] != "natural_retrieval"]
    val_natural_idx = [i for i, m in enumerate(val_data["meta"]) if m["condition"] == "natural_retrieval"]

    assert len(val_controlled_idx) == 72, f"Expected 72 controlled contexts, got {len(val_controlled_idx)}"
    assert len(val_natural_idx) == 24, f"Expected 24 natural contexts, got {len(val_natural_idx)}"

    ctrl_records = [val_data["records"][i] for i in val_controlled_idx]
    ctrl_labels = [val_data["labels"][i] for i in val_controlled_idx]

    nat_records = [val_data["records"][i] for i in val_natural_idx]
    nat_labels = [val_data["labels"][i] for i in val_natural_idx]

    # 4. Evaluate each subset
    print("\n=======================================================")
    print("VALIDATION EVALUATION BY CONTEXT SUBSET")
    print("=======================================================")

    ctrl_eval = evaluate_subset("Controlled Contexts", ctrl_records, ctrl_labels, models)
    nat_eval = evaluate_subset("Natural Retrieval Contexts", nat_records, nat_labels, models)
    pool_eval = evaluate_subset("Pooled Validation Mixture", val_data["records"], val_data["labels"], models)

    # Print summary tables
    for res in [ctrl_eval, nat_eval, pool_eval]:
        sub_name = res["subset_name"]
        print(f"\n--- {sub_name.upper()} (N = {res['sample_count']}, Insufficient Prevalence = {res['insufficient_prevalence']:.1%}) ---")
        print(f"{'Model':<25s} | {'AURC':<8s} | {'Cov@0.50':<8s} {'Risk@0.50':<9s} | {'Cov@0.75':<8s} {'Risk@0.75':<9s} | {'Cov@0.90':<8s} {'Risk@0.90':<9s}")
        print("-" * 95)
        for m_name, m_res in res["models"].items():
            if m_res["type"] == "baseline_discrete_heuristic":
                d_pt = m_res["discrete_operating_point"]
                cov_str = f"{d_pt['coverage']:.3f}"
                r_str = f"{d_pt['evidence_selection_risk']:.3f}" if d_pt['evidence_selection_risk'] is not None else "null"
                print(f"{m_name:<25s} | {'N/A':<8s} | Discrete Operating Point: Coverage={cov_str}, Risk={r_str}, Type I={d_pt['type_1_error']:.3f}, Type II={d_pt['type_2_error']:.3f}")
                continue
            if m_res["type"] == "baseline_always_accept":
                aurc_str = f"{m_res['aurc']:.4f}" if m_res['aurc'] is not None else "N/A"
                r_str = f"{m_res['coverage_1_0']['evidence_selection_risk']:.3f}"
                print(f"{m_name:<25s} | {aurc_str:<8s} | Always Accept (Cov=1.000, Risk={r_str})")
                continue

            aurc_str = f"{m_res['aurc']:.4f}"
            pts = m_res["frozen_operating_points"]
            c50 = f"{pts['target_0.50']['achieved_coverage']:.3f}"
            r50 = f"{pts['target_0.50']['evidence_selection_risk']:.3f}" if pts['target_0.50']['evidence_selection_risk'] is not None else "null"
            c75 = f"{pts['target_0.75']['achieved_coverage']:.3f}"
            r75 = f"{pts['target_0.75']['evidence_selection_risk']:.3f}" if pts['target_0.75']['evidence_selection_risk'] is not None else "null"
            c90 = f"{pts['target_0.90']['achieved_coverage']:.3f}"
            r90 = f"{pts['target_0.90']['evidence_selection_risk']:.3f}" if pts['target_0.90']['evidence_selection_risk'] is not None else "null"
            print(f"{m_name:<25s} | {aurc_str:<8s} | {c50:<8s} {r50:<9s} | {c75:<8s} {r75:<9s} | {c90:<8s} {r90:<9s}")

    # 5. Extract per-context predictions on validation records
    val_predictions = {}
    for idx, r in enumerate(val_data["records"]):
        cid = r.context_id
        val_predictions[cid] = {
            "opaque_context_id": cid,
            "family_id": val_data["meta"][idx]["family_id"],
            "condition": val_data["meta"][idx]["condition"],
            "paper_id": val_data["meta"][idx]["paper_id"],
            "label": val_data["labels"][idx],
            "scores": {
                "B0_AlwaysAccept": 1.0,
                "B1_MaxRelevance": float(models["B1_MaxRelevance"]["instance"].predict_score(r)),
                "B2_HistoricalGate": int(models["B2_HistoricalGate"]["instance"].decide(r)),
                "M1_Relevance_Raw": float(models["M1_Relevance_Raw"]["instance"].predict_score(r)),
                "M2_Relevance_Dedup": float(models["M2_Relevance_Dedup"]["instance"].predict_score(r)),
                "M3_Full_Raw": float(models["M3_Full_Raw"]["instance"].predict_score(r)),
                "M4_Full_Dedup": float(models["M4_Full_Dedup"]["instance"].predict_score(r)),
                "Shortcut_LengthDiagnostic": float(models["Shortcut_LengthDiagnostic"]["instance"].predict_score(r))
            }
        }

    # 6. Save Fitted Model Artifacts
    # Build serializable model dictionary
    serializable_models = {
        "metadata": {
            "protocol": "research_protocol_v1_amendment_v1.1",
            "date": "2026-11-04",
            "training_dataset_sha256": hashlib.sha256(open("data/metadata/benchmark_v2_training_contexts_240.json", "rb").read()).hexdigest(),
            "validation_dataset_sha256": hashlib.sha256(open("data/metadata/benchmark_v2_validation_contexts_96.json", "rb").read()).hexdigest(),
            "training_scores_hash": train_data["scores_hash"],
            "validation_scores_hash": val_data["scores_hash"],
            "scoring_provenance": val_data["scoring_provenance"],
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "random_seed": 42,
            "reproducible_command": "python evaluation/run_benchmark_v2_train_val.py"
        },
        "models": {}
    }

    for m_name, m_info in models.items():
        entry = {
            "name": m_name,
            "type": m_info["type"],
            "description": m_info["description"]
        }
        if "feature_names" in m_info:
            entry["feature_names"] = m_info["feature_names"]
        if "feature_mode" in m_info:
            entry["feature_mode"] = m_info["feature_mode"]
        if "deduplicate" in m_info:
            entry["deduplicate"] = m_info["deduplicate"]
        if "scaler_mean" in m_info:
            entry["scaler_mean"] = m_info["scaler_mean"]
            entry["scaler_scale"] = m_info["scaler_scale"]
        if "selected_C" in m_info:
            entry["selected_C"] = m_info["selected_C"]
            entry["coefficients"] = m_info["coefficients"]
            entry["intercept"] = m_info["intercept"]
            entry["c_tuning_log"] = m_info.get("c_tuning_log", [])
        if "thresholds" in m_info:
            entry["thresholds"] = m_info["thresholds"]
        serializable_models["models"][m_name] = entry

    fitted_models_path = "data/metadata/benchmark_v2_fitted_models.json"
    with open(fitted_models_path, "w", encoding="utf-8") as f:
        json.dump(serializable_models, f, indent=2, ensure_ascii=False)
    print(f"\nSaved fitted models to {fitted_models_path}")

    val_preds_path = "data/metadata/benchmark_v2_validation_predictions.json"
    with open(val_preds_path, "w", encoding="utf-8") as f:
        json.dump(val_predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved validation predictions to {val_preds_path}")

    # 7. Freeze Final Evaluation Specification
    eval_spec = {
        "specification_title": "Benchmark v2 Final Evaluation Protocol Specification",
        "protocol_version": "research_protocol_v1_amendment_v1.1",
        "status": "FROZEN_BEFORE_V2_TEST_CONSTRUCTION_POST_STAGE4_EXPOSURE",
        "date_frozen": "2026-11-04",
        "scoring_provenance": {
            "model_identity": "BAAI/bge-reranker-base",
            "scoring_configuration": "CUDA fp32, sigmoid activation, duplicate-exact cached",
            "training_scores_hash": train_data["scores_hash"],
            "validation_scores_hash": val_data["scores_hash"]
        },
        "test_partition_status": {
            "total_papers": 48,
            "target_test_families": 36,
            "target_controlled_contexts": 108,
            "target_natural_contexts": 36,
            "target_total_contexts": 144,
            "status": "NO_ADDITIONAL_TEST_CONTENT_ACCESS_DURING_REPAIR",
            "historical_disclosure": "Test partition papers were exposed during the archived invalid Stage 4 proxy experiment. No additional test-content access occurred during this repair or benchmark rebuild. Frozen prior to new v2 test construction."
        },
        "hypotheses_and_comparisons": {
            "primary_hypothesis": "M4 (Full Deduplicated) achieves lower discrete tie-correct AURC than M1 (Raw Relevance) on controlled contexts.",
            "primary_comparison": {
                "method_A": "M4_Full_Dedup",
                "method_B": "M1_Relevance_Raw",
                "metric": "discrete_expected_AURC",
                "eval_subset": "controlled_contexts_108"
            },
            "factorial_ablations": [
                {
                    "ablation": "Deduplication effect in relevance-only models",
                    "comparison": "M2_Relevance_Dedup vs M1_Relevance_Raw"
                },
                {
                    "ablation": "Structural features effect on raw passages",
                    "comparison": "M3_Full_Raw vs M1_Relevance_Raw"
                },
                {
                    "ablation": "Structural features effect on deduplicated passages",
                    "comparison": "M4_Full_Dedup vs M2_Relevance_Dedup"
                },
                {
                    "ablation": "Deduplication effect with structural features",
                    "comparison": "M4_Full_Dedup vs M3_Full_Raw"
                }
            ],
            "practical_generalization_check": {
                "subset": "natural_retrieval_contexts_36",
                "reporting": "Reported separately from controlled contexts as the practical generalization check."
            },
            "property_checks": {
                "duplicate_invariance": "Evaluated separately from evidence-removal experiments to measure sensitivity to redundant passage duplication."
            }
        },
        "statistical_testing_protocol": {
            "bootstrap_method": "paired_cluster_bootstrap",
            "cluster_unit": "primary_source_paper_id",
            "cluster_policy": "All conditions and question families associated with a primary source paper are kept together in each resample.",
            "num_resamples": 2000,
            "random_seed": 20261104,
            "reported_metrics": [
                "observed_paired_difference (Metric_A - Metric_B)",
                "95_percent_bootstrap_percentile_interval [q_0.025, q_0.975]"
            ],
            "interpretation_rule": "Report the observed difference and bootstrap percentile interval as uncertainty estimates. An interval containing zero is compatible with differences in either direction and does not establish equivalence.",
            "dependence_disclosure": "Shared distractor passages across families can create residual inter-cluster dependence."
        },
        "threshold_evaluations": {
            "target_coverages": [0.50, 0.75, 0.90],
            "threshold_source": "frozen_validation_selected_thresholds",
            "reporting": "Both descriptive matched-coverage comparisons and deployable validation-frozen operating point performance must be reported."
        },
        "validation_tuning_summary": {
            "controlled_aurc": {m: ctrl_eval["models"][m].get("aurc") for m in ctrl_eval["models"] if "aurc" in ctrl_eval["models"][m]},
            "natural_aurc": {m: nat_eval["models"][m].get("aurc") for m in nat_eval["models"] if "aurc" in nat_eval["models"][m]},
            "pooled_aurc": {m: pool_eval["models"][m].get("aurc") for m in pool_eval["models"] if "aurc" in pool_eval["models"][m]},
            "selected_C_values": {m: models[m].get("selected_C") for m in models if "selected_C" in models[m]}
        }
    }

    eval_spec_path = "data/metadata/benchmark_v2_frozen_evaluation_specification.json"
    with open(eval_spec_path, "w", encoding="utf-8") as f:
        json.dump(eval_spec, f, indent=2, ensure_ascii=False)
    print(f"Saved frozen evaluation specification to {eval_spec_path}")

    # 8. Verification: Reload saved models and assert exact match
    print("\n--- Verifying Reloading and Exact Reproducibility ---")
    reloaded_json = json.load(open(fitted_models_path, encoding="utf-8"))
    for m_name, m_data in reloaded_json["models"].items():
        if m_data["type"] not in ["logistic_gating_model", "diagnostic_shortcut_baseline"]:
            continue
        # Reconstruct model from saved parameters
        recon_scaler = StandardScaler()
        recon_scaler.mean_ = np.array(m_data["scaler_mean"], dtype=np.float64)
        recon_scaler.scale_ = np.array(m_data["scaler_scale"], dtype=np.float64)

        recon_lr = LogisticRegression(penalty="l2", C=m_data["selected_C"], solver="liblinear", random_state=42)
        recon_lr.classes_ = np.array([0, 1])
        recon_lr.coef_ = np.array([m_data["coefficients"]], dtype=np.float64)
        recon_lr.intercept_ = np.array([m_data["intercept"]], dtype=np.float64)

        orig_inst = models[m_name]["instance"]

        # Check predictions on all 96 validation records
        for r in val_data["records"]:
            orig_score = orig_inst.predict_score(r)

            # Score via reconstructed
            if m_data["type"] == "diagnostic_shortcut_baseline":
                X_raw = np.array([[float(len(r.passages)), float(sum(p.token_count for p in r.passages))]], dtype=np.float64)
            else:
                X_raw = extract_features_from_record(r, feature_mode=m_data["feature_mode"], deduplicate=m_data["deduplicate"]).reshape(1, -1)
            X_sc = recon_scaler.transform(X_raw)
            recon_score = float(recon_lr.predict_proba(X_sc)[0, 1])

            diff = abs(orig_score - recon_score)
            assert diff < 1e-12, f"Reconstruction divergence in {m_name} on {r.context_id}: {diff}"

    print("Reproducibility verification PASSED: Reconstructed models match original predictions within 1e-12 tolerance!")


if __name__ == "__main__":
    main()
