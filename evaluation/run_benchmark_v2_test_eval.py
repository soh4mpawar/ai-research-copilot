"""
Benchmark v2: Dedicated Test Evaluation Runner & Frozen Model Verification.
Strictly implements Research Protocol Amendment v1.1.

Zero Model Fitting:
- Loads frozen model parameters, scalers, coefficients, intercepts, and thresholds
  from data/metadata/benchmark_v2_fitted_models.json (SHA-256: 6047be23e2985957f31a596f0a3837c977fc6cd83b812f6d9924fadd98617689).
- Verifies model reconstruction against saved validation predictions before test evaluation.
- Strictly evaluates the 144 authoritative test contexts against the 144 consolidated semantic annotations.
- Executes 2,000 paired cluster bootstrap resamples (seed 20261104, clustered by primary source paper_id).
- Reports controlled contexts (108), natural retrieval contexts (36), pooled mixture (144),
  and secondary condition/question-type subgroups.
"""

import json
import hashlib
import sys
import os
import time
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional, Sequence
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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
    compute_risk_coverage_curve, evaluate_at_matched_coverage
)

# ------------------------------------------------------------------------------
# 1. Authoritative Hashes and Prerequisite Verification
# ------------------------------------------------------------------------------

EXPECTED_HASHES = {
    "test_contexts": (
        "data/metadata/benchmark_v2_test_contexts_144.json",
        "a6b4cdf4b493d1d930d556238b1c0f273529175f3c1e773ae00f6b94402cd230"
    ),
    "test_questions": (
        "data/metadata/benchmark_v2_test_questions_36families.json",
        "a9c3f59ba6bc82ce8504c48346c4beaec3da2996ac30545ac88b4770df13a141"
    ),
    "test_annotations": (
        "data/metadata/benchmark_v2_test_semantic_annotations_144.json",
        "fcd815fc2f7d9c9bc23a9293462a41527c94a7da7df2dc34e7c58a76f2bf16f4"
    ),
    "fitted_models": (
        "data/metadata/benchmark_v2_fitted_models.json",
        "6047be23e2985957f31a596f0a3837c977fc6cd83b812f6d9924fadd98617689"
    ),
    "evaluation_spec": (
        "data/metadata/benchmark_v2_frozen_evaluation_specification.json",
        "71c1ee7e9b732f2f2d978fe771ec3e3bf786d7eb631fbb91fea0c63071b0bf42"
    )
}

def verify_all_input_hashes():
    print("\n=======================================================")
    print("STEP 1: VERIFYING AUTHORITATIVE INPUT HASHES")
    print("=======================================================")
    for key, (path, exp_hash) in EXPECTED_HASHES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")
        actual_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if actual_hash != exp_hash:
            raise ValueError(
                f"HASH MISMATCH for {key} ({path})!\n"
                f"  Actual:   {actual_hash}\n"
                f"  Expected: {exp_hash}"
            )
        print(f"  [PASS] {key:<18s}: {actual_hash}")
    print("All 5 authoritative input hashes verified successfully.")


# ------------------------------------------------------------------------------
# 2. Model Restoration & Reconstruction Verification (Zero Fitting)
# ------------------------------------------------------------------------------

def restore_fitted_models(models_json_path: str) -> Dict[str, Any]:
    """
    Restores models, scalers, coefficients, intercepts, and frozen thresholds
    strictly from the saved JSON artifact. ZERO fitting occurs.
    """
    models_data = json.load(open(models_json_path, encoding="utf-8"))
    reconstructed = {}

    for m_name, m_info in models_data["models"].items():
        m_type = m_info["type"]
        if m_type == "baseline_always_accept":
            reconstructed[m_name] = {
                "name": m_name,
                "type": m_type,
                "description": m_info["description"]
            }
        elif m_type == "baseline_max_relevance":
            b1 = ValidationTunedMaxThreshold()
            thresholds = {}
            for cov_str, th_dict in m_info["thresholds"].items():
                cov = float(cov_str)
                th = float(th_dict["threshold"])
                thresholds[cov] = th_dict
                b1.operating_thresholds[cov] = th
            reconstructed[m_name] = {
                "name": m_name,
                "type": m_type,
                "instance": b1,
                "thresholds": thresholds,
                "description": m_info["description"]
            }
        elif m_type == "baseline_discrete_heuristic":
            b2 = FrozenHistoricalGateBaseline()
            reconstructed[m_name] = {
                "name": m_name,
                "type": m_type,
                "instance": b2,
                "description": m_info["description"]
            }
        elif m_type in ["logistic_gating_model", "diagnostic_shortcut_baseline"]:
            scaler = StandardScaler()
            scaler.mean_ = np.array(m_info["scaler_mean"], dtype=np.float64)
            scaler.scale_ = np.array(m_info["scaler_scale"], dtype=np.float64)

            lr = LogisticRegression(penalty="l2", C=m_info["selected_C"], solver="liblinear", random_state=42)
            lr.classes_ = np.array([0, 1])
            lr.coef_ = np.array([m_info["coefficients"]], dtype=np.float64)
            lr.intercept_ = np.array([m_info["intercept"]], dtype=np.float64)

            thresholds = {}
            for cov_str, th_dict in m_info["thresholds"].items():
                cov = float(cov_str)
                thresholds[cov] = th_dict

            reconstructed[m_name] = {
                "name": m_name,
                "type": m_type,
                "scaler": scaler,
                "model": lr,
                "feature_names": m_info["feature_names"],
                "feature_mode": m_info.get("feature_mode"),
                "deduplicate": m_info.get("deduplicate"),
                "selected_C": m_info["selected_C"],
                "thresholds": thresholds,
                "description": m_info["description"]
            }
        else:
            raise ValueError(f"Unknown model type: {m_type}")

    return reconstructed


def verify_model_reconstruction_against_validation(
    models: Dict[str, Any],
    val_contexts_path: str = "data/metadata/benchmark_v2_validation_contexts_96.json",
    val_preds_path: str = "data/metadata/benchmark_v2_validation_predictions.json"
):
    print("\n=======================================================")
    print("STEP 2: VERIFYING MODEL RECONSTRUCTION ON VALIDATION SET")
    print("=======================================================")
    val_contexts = json.load(open(val_contexts_path, encoding="utf-8"))
    val_preds = json.load(open(val_preds_path, encoding="utf-8"))

    # Build validation records
    val_records = []
    for c in val_contexts:
        cid = c["opaque_context_id"]
        passages = []
        for p in c["passages"]:
            passages.append(Passage.from_dict({
                "chunk_id": p["chunk_id"],
                "paper_id": p["paper_id"],
                "section": p.get("section", "General"),
                "rerank_score": float(p["rerank_score"]),
                "token_count": count_tokens(p["text"]),
                "text": p["text"]
            }))
        payload = format_context_payload(passages)
        rec = FinalContextRecord(
            query=c.get("question", ""),
            passages=tuple(passages),
            max_passages=len(passages),
            token_budget=2000,
            tokenizer_identity="cl100k_base",
            total_tokens=sum(p.token_count for p in passages),
            skipped_passages=(),
            truncation_policy="no_truncation",
            serialized_payload=payload,
            payload_hash=c["payload_hash"]
        )
        object.__setattr__(rec, "context_id", cid)
        val_records.append(rec)

    max_diffs = {m: 0.0 for m in models}
    for r in val_records:
        cid = r.context_id
        saved_scores = val_preds[cid]["scores"]
        for m_name, m_info in models.items():
            pred = predict_model_score(m_info, r)
            diff = abs(pred - saved_scores[m_name])
            if diff > max_diffs[m_name]:
                max_diffs[m_name] = diff

    for m_name, max_d in max_diffs.items():
        print(f"  {m_name:<25s}: max absolute difference = {max_d:.2e}")
        if max_d >= 1e-6:
            raise ValueError(f"Reconstructed model {m_name} diverged from saved validation predictions (diff={max_d})!")
    print("Model reconstruction verification PASSED with zero divergence.")


def predict_model_score(model_info: Dict[str, Any], record: FinalContextRecord) -> float:
    """Predicts gating score for a record using restored model."""
    m_type = model_info["type"]
    if m_type == "baseline_always_accept":
        return 1.0
    elif m_type == "baseline_max_relevance":
        return float(model_info["instance"].predict_score(record))
    elif m_type == "baseline_discrete_heuristic":
        return float(model_info["instance"].decide(record))
    elif m_type == "diagnostic_shortcut_baseline":
        X_raw = np.array([[float(len(record.passages)), float(sum(p.token_count for p in record.passages))]], dtype=np.float64)
        X_sc = model_info["scaler"].transform(X_raw)
        return float(model_info["model"].predict_proba(X_sc)[0, 1])
    elif m_type == "logistic_gating_model":
        X_raw = extract_features_from_record(
            record,
            feature_mode=model_info["feature_mode"],
            deduplicate=model_info["deduplicate"]
        ).reshape(1, -1)
        X_sc = model_info["scaler"].transform(X_raw)
        return float(model_info["model"].predict_proba(X_sc)[0, 1])
    else:
        raise ValueError(f"Unknown model type: {m_type}")


# ------------------------------------------------------------------------------
# 3. Test Data Loading and Strict Verification
# ------------------------------------------------------------------------------

def load_and_verify_test_data() -> Dict[str, Any]:
    print("\n=======================================================")
    print("STEP 3: LOADING & VERIFYING AUTHORITATIVE TEST DATASET")
    print("=======================================================")
    c_path = "data/metadata/benchmark_v2_test_contexts_144.json"
    a_path = "data/metadata/benchmark_v2_test_semantic_annotations_144.json"
    q_path = "data/metadata/benchmark_v2_test_questions_36families.json"
    m_path = "data/metadata/benchmark_v2_test_semantic_annotations_144_manifest.json"

    contexts = json.load(open(c_path, encoding="utf-8"))
    ann_data = json.load(open(a_path, encoding="utf-8"))
    annotations = ann_data if isinstance(ann_data, list) else ann_data.get("annotations", ann_data)
    questions_raw = json.load(open(q_path, encoding="utf-8"))
    manifest = json.load(open(m_path, encoding="utf-8"))

    questions = questions_raw if isinstance(questions_raw, list) else questions_raw.get("questions", [])

    if len(contexts) != 144:
        raise ValueError(f"Expected 144 contexts, got {len(contexts)}")
    if len(annotations) != 144:
        raise ValueError(f"Expected 144 annotations, got {len(annotations)}")
    if len(questions) != 36:
        raise ValueError(f"Expected 36 questions, got {len(questions)}")
    if len(manifest) != 144:
        raise ValueError(f"Expected 144 manifest items, got {len(manifest)}")

    q_map = {q["family_id"]: q for q in questions}
    ann_map = {a["opaque_context_id"]: a for a in annotations}

    ctx_ids = [c["opaque_context_id"] for c in contexts]
    if len(set(ctx_ids)) != 144:
        raise ValueError("Duplicate context IDs in test set!")
    for idx, cid in enumerate(ctx_ids, 1):
        expected_cid = f"ctx_te_{idx:04d}"
        if cid != expected_cid:
            raise ValueError(f"Context ID out of order at index {idx}: got {cid}, expected {expected_cid}")

    records = []
    labels = []
    meta_list = []
    family_conditions = {}
    paper_clusters = set()

    for c in contexts:
        cid = c["opaque_context_id"]
        fid = c["family_id"]
        cond = c["variant"]
        pid = c["paper_id"]
        stored_payload_hash = c["payload_hash"]

        paper_clusters.add(pid)
        family_conditions.setdefault(fid, []).append(cond)

        if fid not in q_map:
            raise ValueError(f"Family {fid} not in question specs!")
        q_spec = q_map[fid]
        q_text = (c.get("question") or q_spec["question"]).strip()
        q_hash = hashlib.sha256(q_text.encode("utf-8")).hexdigest()
        q_type = q_spec.get("question_type", "single_passage")

        ann = ann_map[cid]
        if ann.get("question_hash") != q_hash:
            raise ValueError(f"Question hash mismatch in annotation {cid}!")
        if ann.get("payload_hash") != stored_payload_hash:
            raise ValueError(f"Payload hash mismatch in annotation {cid}!")

        judgment = ann.get("judgment")
        if judgment not in ["sufficient", "insufficient"]:
            raise ValueError(f"Invalid judgment {judgment} in {cid}!")
        label = 1 if judgment == "sufficient" else 0
        labels.append(label)

        # Parse passages
        passages = []
        chunk_scores = {}
        for p in c["passages"]:
            if "rerank_score" not in p or p["rerank_score"] is None:
                raise ValueError(f"Missing rerank_score in {cid} passage {p.get('chunk_id')}")
            score = float(p["rerank_score"])
            if not np.isfinite(score):
                raise ValueError(f"Non-finite rerank_score {score} in {cid}")

            base_cid = p["chunk_id"].replace("_dup", "")
            if base_cid in chunk_scores:
                diff = abs(chunk_scores[base_cid] - score)
                if diff > 1e-9:
                    raise ValueError(f"Duplicate score mismatch in {cid} for {p['chunk_id']}: base={chunk_scores[base_cid]} vs dup={score}")
            else:
                chunk_scores[base_cid] = score

            raw_text = p["text"]
            recomputed_tokens = count_tokens(raw_text)

            passages.append(Passage.from_dict({
                "chunk_id": p["chunk_id"],
                "paper_id": p["paper_id"],
                "section": p.get("section", "General"),
                "rerank_score": score,
                "token_count": recomputed_tokens,
                "text": raw_text
            }))

        reconstructed_payload = format_context_payload(passages)
        recomputed_payload_hash = hashlib.sha256(reconstructed_payload.encode("utf-8")).hexdigest()
        if recomputed_payload_hash != stored_payload_hash:
            raise ValueError(f"Recomputed payload hash mismatch in {cid}!")

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
        object.__setattr__(rec, "context_id", cid)
        records.append(rec)

        meta_list.append({
            "opaque_context_id": cid,
            "family_id": fid,
            "condition": cond,
            "paper_id": pid,
            "question_type": q_type,
            "label": label,
            "judgment": judgment,
            "passage_count": len(passages),
            "recomputed_tokens": sum(p.token_count for p in passages)
        })

    # Verify 4 conditions per family
    expected_conditions = {"supported_candidate", "relevant_replacement", "duplicate_replacement", "natural_retrieval"}
    for fid, cond_list in family_conditions.items():
        if set(cond_list) != expected_conditions or len(cond_list) != 4:
            raise ValueError(f"Family {fid} does not have exactly the 4 expected conditions: {cond_list}")

    print(f"Loaded {len(records)} test contexts across {len(family_conditions)} families.")
    print(f"Primary source paper clusters (paper_id): {len(paper_clusters)}")
    label_counts = Counter(labels)
    print(f"  Class 1 (Sufficient):   {label_counts[1]} ({label_counts[1]/len(labels)*100:.1f}%)")
    print(f"  Class 0 (Insufficient): {label_counts[0]} ({label_counts[0]/len(labels)*100:.1f}%)")

    return {
        "records": records,
        "labels": labels,
        "meta": meta_list,
        "paper_clusters": sorted(list(paper_clusters))
    }


# ------------------------------------------------------------------------------
# 4. Subset Evaluation Protocol
# ------------------------------------------------------------------------------

def evaluate_subset(
    subset_name: str,
    records: List[FinalContextRecord],
    labels: List[int],
    models: Dict[str, Any],
    scores_dict: Dict[str, List[float]],
    target_coverages: Sequence[float] = (0.50, 0.75, 0.90)
) -> Dict[str, Any]:
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

    for m_name, m_info in models.items():
        m_type = m_info["type"]
        scores = np.array(scores_dict[m_name], dtype=np.float64)

        if m_type == "baseline_discrete_heuristic":
            decisions = scores.astype(int)
            accepted = int(np.sum(decisions == 1))
            coverage = float(accepted / n) if n > 0 else 0.0
            risk = compute_evidence_selection_risk(decisions, labels)
            type1, type2 = compute_type_errors(decisions, labels)

            subset_results["models"][m_name] = {
                "type": m_type,
                "discrete_operating_point": {
                    "accepted_count": accepted,
                    "coverage": round(coverage, 4),
                    "evidence_selection_risk": round(risk, 4) if risk is not None else None,
                    "type_1_error": round(type1, 4) if type1 is not None else None,
                    "type_2_error": round(type2, 4) if type2 is not None else None
                }
            }
            continue

        if m_type == "baseline_always_accept":
            curve = compute_risk_coverage_curve(scores, labels)
            decisions = np.ones(n, dtype=int)
            risk = compute_evidence_selection_risk(decisions, labels)
            type1, type2 = compute_type_errors(decisions, labels)

            subset_results["models"][m_name] = {
                "type": m_type,
                "aurc": curve["aurc"],
                "coverage_1_0": {
                    "coverage": 1.0,
                    "evidence_selection_risk": round(risk, 4) if risk is not None else None,
                    "type_1_error": round(type1, 4) if type1 is not None else None,
                    "type_2_error": round(type2, 4) if type2 is not None else None
                }
            }
            continue

        # Continuous scoring models
        curve = compute_risk_coverage_curve(scores, labels)
        matched_eval = evaluate_at_matched_coverage(curve, target_coverages)

        # Frozen validation thresholds evaluation
        frozen_thresholds_eval = {}
        for cov in target_coverages:
            th_val = m_info["thresholds"][cov]["threshold"]
            decisions = (scores >= th_val).astype(int)
            acc_count = int(np.sum(decisions == 1))
            achieved_cov = float(acc_count / n) if n > 0 else 0.0
            risk = compute_evidence_selection_risk(decisions, labels)
            type1, type2 = compute_type_errors(decisions, labels)

            frozen_thresholds_eval[f"target_{cov:.2f}"] = {
                "frozen_threshold": round(th_val, 6),
                "target_coverage": cov,
                "achieved_coverage": round(achieved_cov, 4),
                "coverage_deviation": round(abs(achieved_cov - cov), 4),
                "accepted_count": acc_count,
                "evidence_selection_risk": round(risk, 4) if risk is not None else None,
                "type_1_error": round(type1, 4) if type1 is not None else None,
                "type_2_error": round(type2, 4) if type2 is not None else None
            }

        subset_results["models"][m_name] = {
            "type": m_type,
            "aurc": curve["aurc"],
            "frozen_operating_points": frozen_thresholds_eval,
            "descriptive_matched_coverage": matched_eval,
            "curve_points_count": len(curve["points"]),
            "unique_score_levels": curve["num_unique_score_levels"]
        }

    return subset_results


# ------------------------------------------------------------------------------
# 5. Paired Cluster Bootstrap
# ------------------------------------------------------------------------------

def run_paired_cluster_bootstrap(
    cluster_ids: Sequence[str],
    scores_dict: Dict[str, Sequence[float]],
    labels: Sequence[int],
    n_bootstrap: int = 2000,
    random_seed: int = 20261104,
    observed_aurcs: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Executes paired cluster bootstrap resamples.
    Clusters are defined by primary_source_paper_id (paper_id).
    All rows belonging to a sampled cluster are selected together.
    Identical resampled cluster draws are evaluated across all models.
    """
    cluster_arr = np.asarray(cluster_ids)
    labels_arr = np.asarray(labels)
    unique_clusters = np.unique(cluster_arr)
    n_clusters = len(unique_clusters)

    rng = np.random.RandomState(random_seed)
    score_models = [m for m in scores_dict if m != "B2_HistoricalGate"]

    boot_aurcs: Dict[str, List[float]] = {m: [] for m in score_models}

    for b in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = []
        for c in sampled_clusters:
            idx = np.where(cluster_arr == c)[0]
            sampled_indices.extend(idx)

        b_labels = labels_arr[sampled_indices]
        for m in score_models:
            b_scores = np.asarray(scores_dict[m])[sampled_indices]
            b_curve = compute_risk_coverage_curve(b_scores, b_labels)
            boot_aurcs[m].append(b_curve["aurc"])

    # Compute marginal CIs
    model_cis = {}
    for m in score_models:
        vals = boot_aurcs[m]
        model_cis[m] = {
            "observed_aurc": observed_aurcs.get(m) if observed_aurcs else None,
            "bootstrap_mean_aurc": round(float(np.mean(vals)), 4),
            "bootstrap_std_error": round(float(np.std(vals)), 4),
            "aurc_ci_95": [
                round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4)
            ]
        }

    # Compute planned paired differences
    # Sign convention: Delta = AURC_A - AURC_B. Negative favors Method A.
    comparisons = [
        ("M4_Full_Dedup", "M1_Relevance_Raw", "Primary Hypothesis (M4 vs M1)"),
        ("M2_Relevance_Dedup", "M1_Relevance_Raw", "Factorial Ablation: Dedup in relevance-only"),
        ("M3_Full_Raw", "M1_Relevance_Raw", "Factorial Ablation: Structure on raw passages"),
        ("M4_Full_Dedup", "M2_Relevance_Dedup", "Factorial Ablation: Structure on dedup passages"),
        ("M4_Full_Dedup", "M3_Full_Raw", "Factorial Ablation: Dedup with structural features"),
        ("B1_MaxRelevance", "M1_Relevance_Raw", "Baseline Comparison: B1 vs M1"),
        ("M4_Full_Dedup", "B1_MaxRelevance", "Method Comparison: M4 vs B1"),
        ("Shortcut_LengthDiagnostic", "M4_Full_Dedup", "Diagnostic Comparison: Length vs M4")
    ]

    paired_results = {}
    for m_a, m_b, label in comparisons:
        if m_a not in score_models or m_b not in score_models:
            continue
        diffs = [boot_aurcs[m_a][b] - boot_aurcs[m_b][b] for b in range(n_bootstrap)]
        obs_a = observed_aurcs.get(m_a) if observed_aurcs else None
        obs_b = observed_aurcs.get(m_b) if observed_aurcs else None
        obs_delta = round(float(obs_a - obs_b), 4) if (obs_a is not None and obs_b is not None) else None

        ci_low = round(float(np.percentile(diffs, 2.5)), 4)
        ci_high = round(float(np.percentile(diffs, 97.5)), 4)

        paired_results[f"{m_a}_minus_{m_b}"] = {
            "method_A": m_a,
            "method_B": m_b,
            "label": label,
            "observed_delta": obs_delta,
            "bootstrap_mean_delta": round(float(np.mean(diffs)), 4),
            "bootstrap_std_error": round(float(np.std(diffs)), 4),
            "delta_ci_95": [ci_low, ci_high],
            "favors_method_A": bool(ci_high < 0.0),
            "favors_method_B": bool(ci_low > 0.0),
            "contains_zero": bool(ci_low <= 0.0 <= ci_high)
        }

    return {
        "num_clusters": n_clusters,
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
        "models": model_cis,
        "paired_differences": paired_results
    }


# ------------------------------------------------------------------------------
# 6. Duplicate Invariance Property Check
# ------------------------------------------------------------------------------

def evaluate_duplicate_invariance(
    supported_records: List[FinalContextRecord],
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates sensitivity to redundant passage duplication on supported_candidate contexts.
    For each context:
    - Injects an exact duplicate of the highest scoring passage.
    - Measures absolute score shift: |score(C_dup) - score(C)|.
    - Deduplicated models (M2, M4) must demonstrate exact mathematical invariance (|diff| == 0.0).
    """
    results = {}
    n = len(supported_records)

    for m_name, m_info in models.items():
        if m_info["type"] in ["baseline_always_accept", "baseline_discrete_heuristic"]:
            continue

        diffs = []
        for r in supported_records:
            # Score original
            orig_score = predict_model_score(m_info, r)

            # Find top passage
            passages = list(r.passages)
            top_p = max(passages, key=lambda p: p.score)

            # Create duplicate passage
            dup_p = Passage.from_dict({
                "chunk_id": f"{top_p.passage_id}_dup_inject",
                "paper_id": top_p.paper_id,
                "section": top_p.section,
                "rerank_score": top_p.score,
                "token_count": top_p.token_count,
                "text": top_p.text
            })
            # Inject duplicate at end
            injected_passages = tuple(passages + [dup_p])
            payload_inj = format_context_payload(injected_passages)
            rec_inj = FinalContextRecord(
                query=r.query,
                passages=injected_passages,
                max_passages=len(injected_passages),
                token_budget=2500,
                tokenizer_identity="cl100k_base",
                total_tokens=sum(p.token_count for p in injected_passages),
                skipped_passages=(),
                truncation_policy="no_truncation",
                serialized_payload=payload_inj,
                payload_hash=hashlib.sha256(payload_inj.encode("utf-8")).hexdigest()
            )

            # Score injected
            inj_score = predict_model_score(m_info, rec_inj)
            diff = abs(inj_score - orig_score)
            diffs.append(diff)

        diffs_arr = np.array(diffs, dtype=np.float64)
        results[m_name] = {
            "mean_abs_score_shift": round(float(np.mean(diffs_arr)), 6),
            "max_abs_score_shift": round(float(np.max(diffs_arr)), 6),
            "std_abs_score_shift": round(float(np.std(diffs_arr)), 6),
            "is_strictly_invariant": bool(np.max(diffs_arr) < 1e-9)
        }

    return results


# ------------------------------------------------------------------------------
# 7. Risk-Coverage Curve Visualizations
# ------------------------------------------------------------------------------

def generate_risk_coverage_plots(
    eval_results: Dict[str, Any],
    svg_path: str = "data/metadata/risk_coverage_curve_v2_test.svg",
    html_path: str = "data/metadata/risk_coverage_curve_v2_test.html"
):
    print("\nGenerating Risk-Coverage curve plots...")
    ctrl_eval = eval_results["subsets"]["controlled_contexts_108"]

    # Build curves for plotting
    curves_dict = {}
    for m_name, m_res in ctrl_eval["models"].items():
        if m_res["type"] == "baseline_discrete_heuristic":
            d_pt = m_res["discrete_operating_point"]
            curves_dict[m_name] = {
                "type": "discrete",
                "coverage": d_pt["coverage"],
                "risk": d_pt["evidence_selection_risk"]
            }
        elif m_res["type"] == "baseline_always_accept":
            curves_dict[m_name] = {
                "type": "horizontal",
                "aurc": m_res["aurc"],
                "risk": m_res["coverage_1_0"]["evidence_selection_risk"]
            }
        else:
            # Recompute points for plot
            scores = eval_results["scores_by_subset"]["controlled_contexts_108"][m_name]
            labels = eval_results["labels_by_subset"]["controlled_contexts_108"]
            curve = compute_risk_coverage_curve(scores, labels)
            curves_dict[m_name] = {
                "type": "continuous",
                "aurc": curve["aurc"],
                "points": curve["points"]
            }

    styles = {
        "B0_AlwaysAccept": ("#95a5a6", "dot", "B0 Always Accept (Full-Coverage Floor)"),
        "B1_MaxRelevance": ("#34495e", "dash", "B1 Max Relevance Score"),
        "B2_HistoricalGate": ("#e67e22", "none", "B2 Frozen Historical Gate"),
        "Shortcut_LengthDiagnostic": ("#7f8c8d", "dash", "Shortcut Diagnostic (Length)"),
        "M1_Relevance_Raw": ("#e74c3c", "solid", "M1 Relevance (Raw)"),
        "M2_Relevance_Dedup": ("#e67e22", "solid", "M2 Relevance (Dedup)"),
        "M3_Full_Raw": ("#2980b9", "solid", "M3 Full Structure (Raw)"),
        "M4_Full_Dedup": ("#27ae60", "solid", "M4 Full Structure (Dedup)")
    }

    # Plotly HTML
    fig = go.Figure()
    title = "Selective Answering: Risk vs. Coverage (Benchmark v2 Test Set - 108 Controlled Contexts)"

    for m_name, curve in curves_dict.items():
        if m_name not in styles:
            continue
        color, lstyle, label = styles[m_name]
        if curve["type"] == "horizontal":
            r_val = curve["risk"]
            fig.add_trace(go.Scatter(
                x=[0.0, 1.0], y=[r_val, r_val],
                mode="lines",
                line=dict(color=color, dash="dot", width=2),
                name=f"{label} (Risk={r_val:.3f})"
            ))
        elif curve["type"] == "discrete":
            c_val = curve["coverage"]
            r_val = curve["risk"]
            fig.add_trace(go.Scatter(
                x=[c_val], y=[r_val],
                mode="markers",
                marker=dict(color=color, symbol="square", size=12),
                name=f"{label} (Cov={c_val:.2f}, Risk={r_val:.3f})"
            ))
        else:
            covs = [pt["coverage"] for pt in curve["points"]]
            risks = [pt["expected_risk"] for pt in curve["points"]]
            aurc_val = curve["aurc"]
            fig.add_trace(go.Scatter(
                x=covs, y=risks,
                mode="lines",
                line=dict(color=color, dash=lstyle, width=2.5),
                name=f"{label} (AURC={aurc_val:.4f})"
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Arial")),
        xaxis=dict(title="Coverage (Fraction of Contexts Accepted)", range=[0.0, 1.02], dtick=0.2),
        yaxis=dict(title="Evidence Selection Risk (Fraction Accepted Insufficient)", range=[-0.02, 1.02], dtick=0.2),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
        template="plotly_white",
        width=880,
        height=560
    )
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    fig.write_html(html_path)
    print(f"Saved interactive HTML plot to {html_path}")

    # Standalone SVG
    os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="600" viewBox="0 0 920 600" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="460" y="32" font-size="15" font-weight="bold" text-anchor="middle" fill="#2c3e50">{title}</text>',
        f'<g transform="translate(80, 50)">',
        f'  <line x1="0" y1="450" x2="520" y2="450" stroke="#333" stroke-width="1.5" />',
        f'  <line x1="0" y1="0" x2="0" y2="450" stroke="#333" stroke-width="1.5" />'
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x_pos = tick * 520
        y_pos = 450 - (tick * 450)
        svg_lines.append(f'  <line x1="{x_pos}" y1="0" x2="{x_pos}" y2="450" stroke="#eee" stroke-width="1" />')
        svg_lines.append(f'  <line x1="0" y1="{y_pos}" x2="520" y2="{y_pos}" stroke="#eee" stroke-width="1" />')
        svg_lines.append(f'  <text x="{x_pos}" y="470" font-size="11" text-anchor="middle" fill="#666">{tick:.1f}</text>')
        svg_lines.append(f'  <text x="-12" y="{y_pos+4}" font-size="11" text-anchor="end" fill="#666">{tick:.1f}</text>')

    svg_lines.append(f'  <text x="260" y="505" font-size="12" font-weight="bold" text-anchor="middle" fill="#2c3e50">Coverage (Fraction of Contexts Accepted)</text>')
    svg_lines.append(f'  <text x="-40" y="225" font-size="12" font-weight="bold" text-anchor="middle" fill="#2c3e50" transform="rotate(-90 -40 225)">Evidence Selection Risk</text>')

    for m_name, curve in curves_dict.items():
        if m_name not in styles:
            continue
        color, lstyle, label = styles[m_name]
        if curve["type"] == "horizontal":
            r_y = 450 - (curve["risk"] * 450)
            svg_lines.append(f'  <line x1="0" y1="{r_y}" x2="520" y2="{r_y}" stroke="{color}" stroke-dasharray="4,4" stroke-width="2" />')
        elif curve["type"] == "discrete":
            c_x = curve["coverage"] * 520
            r_y = 450 - (curve["risk"] * 450)
            svg_lines.append(f'  <rect x="{c_x-5}" y="{r_y-5}" width="10" height="10" fill="{color}" stroke="#fff" stroke-width="1.5" />')
        else:
            pts = curve.get("points", [])
            if pts:
                d_str = " ".join([f'{"M" if i==0 else "L"} {p["coverage"]*520:.1f} {450-p["expected_risk"]*450:.1f}' for i, p in enumerate(pts)])
                dash_attr = ' stroke-dasharray="6,4"' if lstyle == "dash" else ""
                svg_lines.append(f'  <path d="{d_str}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr} />')

    # Legend
    svg_lines.append(f'  <g transform="translate(545, 20)">')
    svg_lines.append(f'    <rect x="0" y="0" width="280" height="250" fill="#fdfdfd" stroke="#ddd" rx="4" />')
    l_idx = 0
    legend_order = [
        "M4_Full_Dedup", "M3_Full_Raw", "M2_Relevance_Dedup", "M1_Relevance_Raw",
        "B1_MaxRelevance", "Shortcut_LengthDiagnostic", "B2_HistoricalGate", "B0_AlwaysAccept"
    ]
    for m_name in legend_order:
        if m_name not in curves_dict or m_name not in styles:
            continue
        color, lstyle, label = styles[m_name]
        y_l = 25 + l_idx * 28
        curve = curves_dict[m_name]
        if curve["type"] == "discrete":
            svg_lines.append(f'    <rect x="10" y="{y_l-4}" width="8" height="8" fill="{color}" />')
            svg_lines.append(f'    <text x="26" y="{y_l+3}" font-size="10" fill="#333">B2 Gate (Cov={curve["coverage"]:.2f}, Risk={curve["risk"]:.3f})</text>')
        elif curve["type"] == "horizontal":
            svg_lines.append(f'    <line x1="8" y1="{y_l}" x2="22" y2="{y_l}" stroke="{color}" stroke-dasharray="3,3" stroke-width="2" />')
            svg_lines.append(f'    <text x="26" y="{y_l+3}" font-size="10" fill="#333">B0 Always Accept (Risk={curve["risk"]:.3f})</text>')
        else:
            dash_attr = ' stroke-dasharray="4,3"' if lstyle == "dash" else ""
            svg_lines.append(f'    <line x1="8" y1="{y_l}" x2="22" y2="{y_l}" stroke="{color}" stroke-width="2"{dash_attr} />')
            svg_lines.append(f'    <text x="26" y="{y_l+3}" font-size="10" fill="#333">{m_name} (AURC={curve["aurc"]:.4f})</text>')
        l_idx += 1
    svg_lines.append(f'  </g>')
    svg_lines.append(f'</g>')
    svg_lines.append(f'</svg>')

    with open(svg_path, "w", encoding="utf-8") as f_svg:
        f_svg.write("\n".join(svg_lines))
    print(f"Saved standalone SVG plot to {svg_path}")


def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return to_json_serializable(obj.tolist())
    return obj


# ------------------------------------------------------------------------------
# 8. Main Execution Flow
# ------------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=================================================================")
    print("BENCHMARK V2: DEDICATED FROZEN-MODEL TEST EVALUATION RUNNER")
    print("=================================================================")

    # 1. Verify Hashes
    verify_all_input_hashes()

    # 2. Restore Models (Zero Fitting)
    models = restore_fitted_models("data/metadata/benchmark_v2_fitted_models.json")
    verify_model_reconstruction_against_validation(models)

    # 3. Load Test Data
    test_data = load_and_verify_test_data()
    test_records = test_data["records"]
    test_labels = test_data["labels"]
    test_meta = test_data["meta"]

    # 4. Generate Predictions for all 144 contexts
    print("\n=======================================================")
    print("STEP 4: COMPUTING TEST PREDICTIONS (ZERO FITTING)")
    print("=======================================================")
    test_predictions = {}
    all_scores = {m: [] for m in models}

    for idx, r in enumerate(test_records):
        cid = r.context_id
        meta = test_meta[idx]
        ctx_scores = {}
        for m_name, m_info in models.items():
            sc = predict_model_score(m_info, r)
            ctx_scores[m_name] = sc
            all_scores[m_name].append(sc)

        test_predictions[cid] = {
            "opaque_context_id": cid,
            "family_id": meta["family_id"],
            "condition": meta["condition"],
            "paper_id": meta["paper_id"],
            "question_type": meta["question_type"],
            "label": meta["label"],
            "judgment": meta["judgment"],
            "scores": ctx_scores
        }

    out_preds_path = "data/metadata/benchmark_v2_test_predictions.json"
    with open(out_preds_path, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved 144 test predictions to {out_preds_path}")

    # 5. Define Subsets
    # Controlled: 108 contexts (not natural_retrieval)
    # Natural: 36 contexts (natural_retrieval)
    # Pooled: 144 contexts
    ctrl_indices = [i for i, m in enumerate(test_meta) if m["condition"] != "natural_retrieval"]
    nat_indices = [i for i, m in enumerate(test_meta) if m["condition"] == "natural_retrieval"]
    supp_indices = [i for i, m in enumerate(test_meta) if m["condition"] == "supported_candidate"]
    rel_indices = [i for i, m in enumerate(test_meta) if m["condition"] == "relevant_replacement"]
    dup_indices = [i for i, m in enumerate(test_meta) if m["condition"] == "duplicate_replacement"]
    single_indices = [i for i, m in enumerate(test_meta) if m["question_type"] == "single_passage"]
    multi_indices = [i for i, m in enumerate(test_meta) if m["question_type"] == "multi_passage"]

    ctrl_single_indices = [i for i in ctrl_indices if test_meta[i]["question_type"] == "single_passage"]
    ctrl_multi_indices = [i for i in ctrl_indices if test_meta[i]["question_type"] == "multi_passage"]

    subsets_def = {
        "controlled_contexts_108": ctrl_indices,
        "natural_retrieval_contexts_36": nat_indices,
        "pooled_test_mixture_144": list(range(144)),
        "supported_candidate_36": supp_indices,
        "relevant_replacement_36": rel_indices,
        "duplicate_replacement_36": dup_indices,
        "single_passage_100": single_indices,
        "multi_passage_44": multi_indices,
        "controlled_single_passage_75": ctrl_single_indices,
        "controlled_multi_passage_33": ctrl_multi_indices
    }

    # 6. Evaluate Each Subset
    print("\n=======================================================")
    print("STEP 5: EVALUATING SUBSETS UNDER FROZEN SPECIFICATION")
    print("=======================================================")
    eval_results = {
        "metadata": {
            "evaluation_title": "Benchmark v2 Final Test Evaluation Results",
            "protocol": "research_protocol_v1_amendment_v1.1",
            "date": "2026-11-04",
            "test_contexts_hash": EXPECTED_HASHES["test_contexts"][1],
            "test_annotations_hash": EXPECTED_HASHES["test_annotations"][1],
            "fitted_models_hash": EXPECTED_HASHES["fitted_models"][1],
            "eval_spec_hash": EXPECTED_HASHES["evaluation_spec"][1],
            "total_test_contexts": 144,
            "total_test_families": 36,
            "total_paper_clusters": len(test_data["paper_clusters"])
        },
        "subsets": {},
        "scores_by_subset": {},
        "labels_by_subset": {}
    }

    for s_name, indices in subsets_def.items():
        s_records = [test_records[i] for i in indices]
        s_labels = [test_labels[i] for i in indices]
        s_scores = {m: [all_scores[m][i] for i in indices] for m in models}

        res = evaluate_subset(s_name, s_records, s_labels, models, s_scores)
        eval_results["subsets"][s_name] = res
        eval_results["scores_by_subset"][s_name] = s_scores
        eval_results["labels_by_subset"][s_name] = s_labels

        # Print main subsets
        if s_name in ["controlled_contexts_108", "natural_retrieval_contexts_36", "pooled_test_mixture_144"]:
            print(f"\n--- {s_name.upper()} (N = {res['sample_count']}, Insufficient Prevalence = {res['insufficient_prevalence']:.1%}) ---")
            print(f"{'Model':<25s} | {'AURC':<8s} | {'Cov@0.50':<8s} {'Risk@0.50':<9s} | {'Cov@0.75':<8s} {'Risk@0.75':<9s} | {'Cov@0.90':<8s} {'Risk@0.90':<9s}")
            print("-" * 95)
            for m_name, m_res in res["models"].items():
                if m_res["type"] == "baseline_discrete_heuristic":
                    d_pt = m_res["discrete_operating_point"]
                    cov_str = f"{d_pt['coverage']:.3f}"
                    r_str = f"{d_pt['evidence_selection_risk']:.3f}" if d_pt['evidence_selection_risk'] is not None else "null"
                    print(f"{m_name:<25s} | {'N/A':<8s} | Discrete Operating Point: Coverage={cov_str}, Risk={r_str}, Type I={d_pt['type_1_error']}, Type II={d_pt['type_2_error']}")
                    continue
                if m_res["type"] == "baseline_always_accept":
                    aurc_str = f"{m_res['aurc']:.4f}" if m_res['aurc'] is not None else "N/A"
                    r_str = f"{m_res['coverage_1_0']['evidence_selection_risk']:.3f}" if m_res['coverage_1_0']['evidence_selection_risk'] is not None else "null"
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

    # 7. Paired Cluster Bootstrap
    print("\n=======================================================")
    print("STEP 6: PAIRED CLUSTER BOOTSTRAP (2,000 RESAMPLES, SEED 20261104)")
    print("=======================================================")
    # Primary bootstrap on Controlled Contexts (108)
    ctrl_pids = [test_meta[i]["paper_id"] for i in ctrl_indices]
    ctrl_scores = eval_results["scores_by_subset"]["controlled_contexts_108"]
    ctrl_labels = eval_results["labels_by_subset"]["controlled_contexts_108"]
    ctrl_aurcs = {m: eval_results["subsets"]["controlled_contexts_108"]["models"][m].get("aurc") for m in ctrl_scores}

    print("\nRunning Bootstrap on Controlled Contexts (108 contexts, 36 paper clusters)...")
    boot_ctrl = run_paired_cluster_bootstrap(
        cluster_ids=ctrl_pids,
        scores_dict=ctrl_scores,
        labels=ctrl_labels,
        n_bootstrap=2000,
        random_seed=20261104,
        observed_aurcs=ctrl_aurcs
    )

    # Secondary bootstrap on Natural Retrieval Contexts (36)
    nat_pids = [test_meta[i]["paper_id"] for i in nat_indices]
    nat_scores = eval_results["scores_by_subset"]["natural_retrieval_contexts_36"]
    nat_labels = eval_results["labels_by_subset"]["natural_retrieval_contexts_36"]
    nat_aurcs = {m: eval_results["subsets"]["natural_retrieval_contexts_36"]["models"][m].get("aurc") for m in nat_scores}

    print("Running Bootstrap on Natural Retrieval Contexts (36 contexts, 36 paper clusters)...")
    boot_nat = run_paired_cluster_bootstrap(
        cluster_ids=nat_pids,
        scores_dict=nat_scores,
        labels=nat_labels,
        n_bootstrap=2000,
        random_seed=20261104,
        observed_aurcs=nat_aurcs
    )

    # Bootstrap on Pooled Mixture (144)
    pool_pids = [test_meta[i]["paper_id"] for i in range(144)]
    pool_scores = eval_results["scores_by_subset"]["pooled_test_mixture_144"]
    pool_labels = eval_results["labels_by_subset"]["pooled_test_mixture_144"]
    pool_aurcs = {m: eval_results["subsets"]["pooled_test_mixture_144"]["models"][m].get("aurc") for m in pool_scores}

    print("Running Bootstrap on Pooled Test Mixture (144 contexts, 36 paper clusters)...")
    boot_pool = run_paired_cluster_bootstrap(
        cluster_ids=pool_pids,
        scores_dict=pool_scores,
        labels=pool_labels,
        n_bootstrap=2000,
        random_seed=20261104,
        observed_aurcs=pool_aurcs
    )

    bootstrap_results = {
        "controlled_contexts": boot_ctrl,
        "natural_retrieval_contexts": boot_nat,
        "pooled_mixture": boot_pool
    }

    # Print Primary & Ablation Comparisons on Controlled Contexts
    print("\n--- CONTROLLED CONTEXTS PAIRED COMPARISONS (Method A - Method B; Negative favors Method A) ---")
    print(f"{'Comparison':<30s} | {'Obs Delta':<10s} | {'Boot Mean':<10s} | {'Boot SE':<8s} | {'95% Percentile Interval':<24s} | {'Outcome'}")
    print("-" * 105)
    for pair_key, p_data in boot_ctrl["paired_differences"].items():
        obs_d = f"{p_data['observed_delta']:.4f}" if p_data['observed_delta'] is not None else "N/A"
        bm_d = f"{p_data['bootstrap_mean_delta']:.4f}"
        b_se = f"{p_data['bootstrap_std_error']:.4f}"
        ci_str = f"[{p_data['delta_ci_95'][0]:.4f}, {p_data['delta_ci_95'][1]:.4f}]"
        if p_data["favors_method_A"]:
            outcome = f"Favors {p_data['method_A']}"
        elif p_data["favors_method_B"]:
            outcome = f"Favors {p_data['method_B']}"
        else:
            outcome = "Contains 0 (Inconclusive / Overlap)"
        print(f"{pair_key:<30s} | {obs_d:<10s} | {bm_d:<10s} | {b_se:<8s} | {ci_str:<24s} | {outcome}")

    # 8. Duplicate Invariance Property Check
    print("\n=======================================================")
    print("STEP 7: DUPLICATE INVARIANCE PROPERTY CHECK")
    print("=======================================================")
    supp_records = [test_records[i] for i in supp_indices]
    dup_invariance = evaluate_duplicate_invariance(supp_records, models)
    eval_results["duplicate_invariance_property_check"] = dup_invariance

    print(f"{'Model':<25s} | {'Strictly Invariant?':<20s} | {'Mean Abs Shift':<16s} | {'Max Abs Shift':<14s}")
    print("-" * 80)
    for m_name, d_res in dup_invariance.items():
        inv_str = "YES (Identical 0.0)" if d_res["is_strictly_invariant"] else "NO (Sensitive to dups)"
        print(f"{m_name:<25s} | {inv_str:<20s} | {d_res['mean_abs_score_shift']:.6f}         | {d_res['max_abs_score_shift']:.6f}")

    # 9. Save Machine-Readable Results
    out_eval_path = "data/metadata/benchmark_v2_test_eval_results.json"
    with open(out_eval_path, "w", encoding="utf-8") as f:
        # Don't dump huge raw score arrays inside results summary
        save_eval = {
            "metadata": eval_results["metadata"],
            "subsets": eval_results["subsets"],
            "duplicate_invariance_property_check": eval_results["duplicate_invariance_property_check"]
        }
        json.dump(to_json_serializable(save_eval), f, indent=2, ensure_ascii=False)
    print(f"\nSaved test evaluation results summary to {out_eval_path}")

    out_boot_path = "data/metadata/benchmark_v2_test_bootstrap_results.json"
    with open(out_boot_path, "w", encoding="utf-8") as f:
        json.dump(to_json_serializable(bootstrap_results), f, indent=2, ensure_ascii=False)
    print(f"Saved test bootstrap results to {out_boot_path}")

    # 10. Generate Visualizations
    generate_risk_coverage_plots(eval_results)

    elapsed = time.time() - t_start
    print(f"\nEvaluation and bootstrap completed in {elapsed:.2f} seconds.")
    print("=================================================================")


if __name__ == "__main__":
    main()
