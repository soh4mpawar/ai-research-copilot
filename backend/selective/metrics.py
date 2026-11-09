"""
Evaluation Metrics & Risk-Coverage Analytics for Selective Scientific Question Answering.

Mathematical Conventions:
1. All-Tied Scorer Consistency:
   - For an all-tied scorer with score S across n items (fraction E/n insufficient):
     Under uniform random tie ordering, expected cumulative errors in prefix k is k * (E/n).
     Expected risk at every accepted count k in {1, ..., n} is exactly E/n.
     Discrete expected AURC is the mean of risks across k in {1, ..., n}, exactly E/n.
2. Undefined Cases:
   - If no items accepted (k=0), risk is undefined (None / null), not 0.0.
   - If class 0 (insufficient) is missing, Type I error is undefined (None / null).
   - If class 1 (sufficient) is missing, Type II error is undefined (None / null).
3. Attainable vs. Descriptive Operating Points:
   - Points at exact threshold boundaries are marked as attainable deterministic operating points.
   - Intermediate tie-group points are labeled descriptive randomized points.
4. Clustered Bootstrap & Paired Differences:
   - Resamples source-connected paper/family cluster components with replacement.
   - Computes paired method differences on identical resampled clusters.
"""

from typing import Sequence, Dict, Any, List, Tuple, Optional, Union
import numpy as np


def validate_metric_inputs(
    scores_or_decisions: Sequence[Union[float, int]],
    labels: Sequence[int],
    allow_pending: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate input array shapes, types, and domains."""
    if len(scores_or_decisions) != len(labels):
        raise ValueError(
            f"Length mismatch: {len(scores_or_decisions)} scores/decisions vs {len(labels)} labels."
        )

    arr = np.asarray(scores_or_decisions)
    lbl = np.asarray(labels)

    if not np.all(np.isfinite(arr)):
        raise ValueError("Scores/decisions contain non-finite values (NaN or Inf).")

    # Check for pending labels (-1 or None)
    if not allow_pending:
        invalid_mask = ~np.isin(lbl, [0, 1])
        if np.any(invalid_mask):
            raise ValueError(
                f"Labels contain invalid or pending values: {lbl[invalid_mask].tolist()}. "
                "Only verified binary labels {0, 1} are permitted for research metrics."
            )

    return arr, lbl


def compute_evidence_selection_risk(
    decisions: Sequence[int],
    sufficiency_labels: Sequence[int]
) -> Optional[float]:
    """
    Evidence Selection Risk:
    Fraction of accepted contexts where evidence is insufficient (Y_suff == 0).
    Returns None (undefined) if no contexts were accepted.
    """
    decisions_arr, labels_arr = validate_metric_inputs(decisions, sufficiency_labels)
    accepted_mask = (decisions_arr == 1)
    num_accepted = int(np.sum(accepted_mask))

    if num_accepted == 0:
        return None  # Undefined when coverage is zero

    insufficient_accepted = int(np.sum((labels_arr[accepted_mask] == 0)))
    return float(insufficient_accepted / num_accepted)


def compute_answer_risk(
    decisions: Sequence[int],
    answer_correctness_labels: Optional[Sequence[int]]
) -> Optional[float]:
    """
    Answer Risk:
    Fraction of accepted generated answers judged incorrect or unsupported (Y_correct == 0).
    Returns None if correctness labels are absent or no contexts were accepted.
    """
    if answer_correctness_labels is None or len(answer_correctness_labels) == 0:
        return None

    decisions_arr, labels_arr = validate_metric_inputs(decisions, answer_correctness_labels)
    accepted_mask = (decisions_arr == 1)
    num_accepted = int(np.sum(accepted_mask))

    if num_accepted == 0:
        return None

    incorrect_accepted = int(np.sum((labels_arr[accepted_mask] == 0)))
    return float(incorrect_accepted / num_accepted)


def compute_type_errors(
    decisions: Sequence[int],
    sufficiency_labels: Sequence[int]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Type I (FPR on insufficient) and Type II (FNR on sufficient) error rates.
    Returns None for the corresponding error if that label class is absent in the sample.
    """
    decisions_arr, labels_arr = validate_metric_inputs(decisions, sufficiency_labels)

    insufficient_mask = (labels_arr == 0)
    sufficient_mask = (labels_arr == 1)

    type_1 = (
        float(np.mean(decisions_arr[insufficient_mask] == 1))
        if np.any(insufficient_mask) else None
    )
    type_2 = (
        float(np.mean(decisions_arr[sufficient_mask] == 0))
        if np.any(sufficient_mask) else None
    )

    return type_1, type_2


def compute_risk_coverage_curve(
    scores: Sequence[float],
    sufficiency_labels: Sequence[int]
) -> Dict[str, Any]:
    """
    Compute discrete risk-coverage curve and expected AURC under uniform random tie ordering.

    Algorithm:
    1. Group samples by distinct score levels in descending order: S_1 > S_2 > ... > S_M.
    2. Each group m has size n_m and E_m insufficient contexts (labels == 0).
    3. For any accepted count k in {1, ..., n}:
       Let level m be the boundary group such that C_{m-1} < k <= C_m.
       Expected cumulative errors = cumulative_errors_{m-1} + (k - C_{m-1}) * (E_m / n_m).
       Expected risk at k = Expected cumulative errors / k.
    4. Discrete expected AURC is the arithmetic mean of expected risks for k in {1, ..., n}.
    5. Returns attainable deterministic thresholds and intermediate randomized operating points.
    """
    scores_arr, labels_arr = validate_metric_inputs(scores, sufficiency_labels)
    n = len(scores_arr)

    if n == 0:
        return {
            "aurc": None,
            "sample_size": 0,
            "points": [],
            "attainable_thresholds": [],
            "expected_risks_by_k": {},
            "coverages_by_k": {}
        }

    # Group by unique score descending
    unique_scores, group_indices = np.unique(scores_arr, return_inverse=True)
    # Sort descending
    desc_order = np.argsort(unique_scores)[::-1]
    sorted_unique_scores = unique_scores[desc_order]

    # Precompute group sizes and error counts
    group_sizes = []
    group_errors = []
    for s_idx in desc_order:
        mask = (group_indices == s_idx)
        group_sizes.append(int(np.sum(mask)))
        group_errors.append(int(np.sum(labels_arr[mask] == 0)))

    M = len(sorted_unique_scores)
    cum_counts = np.cumsum(group_sizes)
    cum_errors = np.cumsum(group_errors)

    # Attainable deterministic threshold records
    attainable_thresholds = []
    for m in range(M):
        thresh = float(sorted_unique_scores[m])
        c_m = int(cum_counts[m])
        e_m = int(cum_errors[m])
        cov = float(c_m / n)
        risk = float(e_m / c_m)
        attainable_thresholds.append({
            "score_threshold": thresh,
            "accepted_count": c_m,
            "coverage": round(cov, 4),
            "risk": round(risk, 4),
            "is_attainable_deterministic": True
        })

    # Expected risk for each integer count k in 1..n
    points = []
    expected_risks_by_k = {}
    coverages_by_k = {}
    risks_list = []

    m = 0
    for k in range(1, n + 1):
        while k > cum_counts[m]:
            m += 1

        prev_count = cum_counts[m - 1] if m > 0 else 0
        prev_errors = cum_errors[m - 1] if m > 0 else 0

        r_in_group = k - prev_count
        group_size = group_sizes[m]
        group_errs = group_errors[m]

        expected_errors = prev_errors + r_in_group * (group_errs / group_size)
        risk_k = float(expected_errors / k)
        cov_k = float(k / n)

        is_attainable = (k == cum_counts[m])

        expected_risks_by_k[k] = risk_k
        coverages_by_k[k] = cov_k
        risks_list.append(risk_k)

        points.append({
            "k": k,
            "coverage": round(cov_k, 4),
            "expected_risk": round(risk_k, 4),
            "is_attainable_deterministic": is_attainable,
            "boundary_score": float(sorted_unique_scores[m])
        })

    # Discrete expected AURC
    aurc = float(np.mean(risks_list))

    return {
        "aurc": round(aurc, 6),
        "sample_size": n,
        "is_all_tied": (M == 1),
        "num_unique_score_levels": M,
        "expected_risks_by_k": expected_risks_by_k,
        "coverages_by_k": coverages_by_k,
        "attainable_thresholds": attainable_thresholds,
        "points": points
    }


def evaluate_at_matched_coverage(
    curve: Dict[str, Any],
    target_coverages: Sequence[float] = (0.50, 0.75, 0.90)
) -> Dict[float, Dict[str, Any]]:
    """
    Descriptive operating points at target coverage levels under uniform random tie ordering.
    Finds exact nearest integer accepted count k = round(target * n).
    Returns expected risk and flags whether this point is deterministically attainable or randomized.
    """
    n = curve.get("sample_size", 0)
    if n == 0 or curve.get("aurc") is None:
        return {cov: {"expected_risk": None, "k": 0, "is_attainable": False} for cov in target_coverages}

    matched = {}
    for cov in target_coverages:
        k = int(round(cov * n))
        if k == 0:
            matched[cov] = {
                "expected_risk": None,  # Zero accepted items -> undefined risk
                "k": 0,
                "achieved_coverage": 0.0,
                "is_attainable": True
            }
        else:
            k = min(k, n)
            matched[cov] = {
                "expected_risk": round(curve["expected_risks_by_k"][k], 4),
                "k": k,
                "achieved_coverage": round(curve["coverages_by_k"][k], 4),
                "is_attainable": curve["points"][k - 1]["is_attainable_deterministic"]
            }

    return matched


def cluster_bootstrap_evaluation(
    cluster_ids: Sequence[str],
    predictions_dict: Dict[str, Sequence[int]],
    scores_dict: Dict[str, Sequence[float]],
    labels: Sequence[int],
    n_bootstrap: int = 1000,
    random_state: int = 42,
    observed_aurcs: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Cluster-Bootstrap Uncertainty Estimation across Connected Source Components.
    Resamples clusters with replacement, preserving intra-paper and intra-family correlations.
    Evaluates all methods on identical resampled clusters to compute paired method differences.

    Parameters:
    - cluster_ids: Connected component ID (paper-family cluster) per instance.
    - predictions_dict: Model name -> binary decisions array.
    - scores_dict: Model name -> continuous scores array.
    - labels: Ground truth sufficiency labels {0, 1}.
    """
    cluster_arr = np.asarray(cluster_ids)
    labels_arr = np.asarray(labels)
    unique_clusters = np.unique(cluster_arr)
    n_clusters = len(unique_clusters)

    # Empty or insufficient bootstrap clusters
    if n_clusters < 2:
        empty_res = {}
        for m in scores_dict:
            empty_res[m] = {
                "aurc_ci_95": [None, None],
                "risk_ci_95": [None, None],
                "type1_ci_95": [None, None],
                "type2_ci_95": [None, None]
            }
        return {
            "num_clusters": n_clusters,
            "models": empty_res,
            "paired_differences": {}
        }

    rng = np.random.RandomState(random_state)
    score_models = list(scores_dict.keys())
    decision_models = list(predictions_dict.keys())
    all_models = list(dict.fromkeys(score_models + decision_models))

    boot_aurcs: Dict[str, List[Optional[float]]] = {m: [] for m in all_models}
    boot_risks: Dict[str, List[Optional[float]]] = {m: [] for m in all_models}
    boot_type1: Dict[str, List[Optional[float]]] = {m: [] for m in all_models}

    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        # Gather all indices matching sampled clusters
        sampled_indices = []
        for c in sampled_clusters:
            idx = np.where(cluster_arr == c)[0]
            sampled_indices.extend(idx)

        b_labels = labels_arr[sampled_indices]

        # Evaluate continuous curve models
        for m in score_models:
            b_scores = np.asarray(scores_dict[m])[sampled_indices]
            b_curve = compute_risk_coverage_curve(b_scores, b_labels)
            boot_aurcs[m].append(b_curve["aurc"])

        # Evaluate discrete decision rules
        for m in decision_models:
            b_decisions = np.asarray(predictions_dict[m])[sampled_indices]
            r = compute_evidence_selection_risk(b_decisions, b_labels)
            t1, _ = compute_type_errors(b_decisions, b_labels)
            boot_risks[m].append(r)
            boot_type1[m].append(t1)

    model_cis = {}
    for m in all_models:
        valid_aurcs = [x for x in boot_aurcs[m] if x is not None]
        aurc_ci = (
            [round(float(np.percentile(valid_aurcs, 2.5)), 4), round(float(np.percentile(valid_aurcs, 97.5)), 4)]
            if len(valid_aurcs) >= 10 else [None, None]
        )
        valid_risks = [x for x in boot_risks[m] if x is not None]
        risk_ci = (
            [round(float(np.percentile(valid_risks, 2.5)), 4), round(float(np.percentile(valid_risks, 97.5)), 4)]
            if len(valid_risks) >= 10 else [None, None]
        )
        valid_t1 = [x for x in boot_type1[m] if x is not None]
        t1_ci = (
            [round(float(np.percentile(valid_t1, 2.5)), 4), round(float(np.percentile(valid_t1, 97.5)), 4)]
            if len(valid_t1) >= 10 else [None, None]
        )

        model_cis[m] = {
            "aurc_ci_95": aurc_ci,
            "risk_ci_95": risk_ci,
            "type1_ci_95": t1_ci
        }

    # Paired differences: Delta(AURC) = AURC(M_i) - AURC(M_j)
    paired_diffs = {}
    for i in range(len(score_models)):
        for j in range(i + 1, len(score_models)):
            m_a = score_models[i]
            m_b = score_models[j]
            
            diffs_forward = [
                boot_aurcs[m_a][b] - boot_aurcs[m_b][b]
                for b in range(n_bootstrap)
                if boot_aurcs[m_a][b] is not None and boot_aurcs[m_b][b] is not None
            ]
            if len(diffs_forward) >= 10:
                pair_ci_f = [
                    round(float(np.percentile(diffs_forward, 2.5)), 4),
                    round(float(np.percentile(diffs_forward, 97.5)), 4)
                ]
                pair_ci_r = [
                    round(float(np.percentile([-d for d in diffs_forward], 2.5)), 4),
                    round(float(np.percentile([-d for d in diffs_forward], 97.5)), 4)
                ]
            else:
                pair_ci_f = [None, None]
                pair_ci_r = [None, None]
                
            obs_delta_f = None
            obs_delta_r = None
            if observed_aurcs and m_a in observed_aurcs and m_b in observed_aurcs:
                if observed_aurcs[m_a] is not None and observed_aurcs[m_b] is not None:
                    obs_delta_f = round(float(observed_aurcs[m_a] - observed_aurcs[m_b]), 4)
                    obs_delta_r = round(float(observed_aurcs[m_b] - observed_aurcs[m_a]), 4)
                    
            paired_diffs[f"{m_a}_minus_{m_b}"] = {
                "observed_delta": obs_delta_f,
                "bootstrap_mean_delta": round(float(np.mean(diffs_forward)), 4) if diffs_forward else None,
                "aurc_delta_ci_95": pair_ci_f
            }
            paired_diffs[f"{m_b}_minus_{m_a}"] = {
                "observed_delta": obs_delta_r,
                "bootstrap_mean_delta": round(float(np.mean([-d for d in diffs_forward])), 4) if diffs_forward else None,
                "aurc_delta_ci_95": pair_ci_r
            }

    return {
        "num_clusters": n_clusters,
        "n_bootstrap": n_bootstrap,
        "models": model_cis,
        "paired_differences": paired_diffs
    }
