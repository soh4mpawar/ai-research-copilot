"""
Focused Unit & Regression Tests for Evaluation Functions.
Tests:
1. Model reconstruction: Restored weights from benchmark_v2_fitted_models.json match saved validation predictions with zero divergence.
2. All-tied discrete AURC: Analytical property check that all-tied scores yield AURC exactly equal to insufficient prevalence E / N.
3. Zero-acceptance null risk: compute_evidence_selection_risk returns None (not 0.0) when accepted count is 0.
4. Paired bootstrap repeated clusters preserved: Sampling a cluster with multiplicity k replicates all associated rows exactly k times.
5. Identical resamples across models: The exact same cluster draws and sample indices evaluate all models in every bootstrap iteration.
"""

import unittest
import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from backend.selective.context import Passage, FinalContextRecord, format_context_payload, count_tokens
from backend.selective.features import extract_features_from_record
from backend.selective.models import (
    AlwaysAcceptBaseline, ValidationTunedMaxThreshold, FrozenHistoricalGateBaseline,
    LogisticGatingModel, LengthDiagnosticBaseline
)
from backend.selective.metrics import (
    compute_evidence_selection_risk, compute_type_errors,
    compute_risk_coverage_curve, evaluate_at_matched_coverage
)
from evaluation.run_benchmark_v2_test_eval import (
    restore_fitted_models,
    predict_model_score,
    run_paired_cluster_bootstrap
)
from unittest.mock import patch, MagicMock


class TestEvaluationFunctions(unittest.TestCase):

    def test_1_model_reconstruction_zero_divergence(self):
        """Verifies restored model parameters yield exact validation predictions."""
        models_path = "data/metadata/benchmark_v2_fitted_models.json"
        val_ctx_path = "data/metadata/benchmark_v2_validation_contexts_96.json"
        val_preds_path = "data/metadata/benchmark_v2_validation_predictions.json"

        self.assertTrue(os.path.exists(models_path), f"Missing {models_path}")
        self.assertTrue(os.path.exists(val_ctx_path), f"Missing {val_ctx_path}")
        self.assertTrue(os.path.exists(val_preds_path), f"Missing {val_preds_path}")

        models = restore_fitted_models(models_path)
        with open(val_ctx_path, encoding="utf-8") as f:
            val_contexts = json.load(f)
        with open(val_preds_path, encoding="utf-8") as f:
            val_preds = json.load(f)

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

        for r in val_records:
            cid = r.context_id
            saved_scores = val_preds[cid]["scores"]
            for m_name, m_info in models.items():
                pred = predict_model_score(m_info, r)
                diff = abs(pred - saved_scores[m_name])
                self.assertLess(
                    diff, 1e-6,
                    f"Divergence in {m_name} on {cid}: reconstructed={pred}, saved={saved_scores[m_name]}, diff={diff}"
                )

    def test_2_all_tied_discrete_aurc_equals_prevalence(self):
        """Analytical property: for all-tied scores, discrete expected AURC must equal E / N."""
        test_cases = [
            (10, 3),   # N=10, 3 insufficient -> prevalence 0.3
            (20, 10),  # N=20, 10 insufficient -> prevalence 0.5
            (36, 3),   # N=36, 3 insufficient -> prevalence 3/36 = 0.083333
            (100, 25), # N=100, 25 insufficient -> prevalence 0.25
            (108, 72)  # N=108, 72 insufficient -> prevalence 72/108 = 0.666667
        ]
        for n, e in test_cases:
            # All items receive identical score 0.5
            scores = [0.5] * n
            # e items have label 0 (insufficient), n-e items have label 1 (sufficient)
            labels = [0] * e + [1] * (n - e)

            curve = compute_risk_coverage_curve(scores, labels)
            expected_prevalence = round(float(e / n), 6)
            self.assertTrue(curve["is_all_tied"], f"Failed is_all_tied flag for N={n}")
            self.assertEqual(curve["num_unique_score_levels"], 1)
            self.assertAlmostEqual(
                curve["aurc"], expected_prevalence, places=5,
                msg=f"All-tied AURC {curve['aurc']} != expected prevalence {expected_prevalence} for N={n}, E={e}"
            )

    def test_3_zero_acceptance_returns_none_risk(self):
        """Undefined risk convention: when accepted count k = 0, risk must be None, not 0.0."""
        # 10 items, all decisions = 0 (rejected)
        decisions = [0] * 10
        labels = [0] * 5 + [1] * 5

        risk = compute_evidence_selection_risk(decisions, labels)
        self.assertIsNone(
            risk,
            f"Zero acceptance must yield None risk, got {risk}"
        )

        # When at least 1 item accepted, risk is a valid float
        decisions_some = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        risk_some = compute_evidence_selection_risk(decisions_some, labels)
        self.assertIsNotNone(risk_some)
        self.assertEqual(risk_some, 1.0)  # item 0 was label 0

    def test_4_paired_bootstrap_preserves_repeated_clusters(self):
        """Cluster resampling with replacement in production run_paired_cluster_bootstrap
        must replicate ALL instances of sampled clusters with exact multiplicity."""
        # Define 4 clusters with varying instance counts
        # p1: 2 rows (indices 0, 1)
        # p2: 1 row  (index 2)
        # p3: 3 rows (indices 3, 4, 5)
        # p4: 1 row  (index 6)
        cluster_ids = ["p1", "p1", "p2", "p3", "p3", "p3", "p4"]
        labels = [1, 0, 1, 0, 1, 0, 1]
        scores_dict = {
            "M1_Relevance_Raw": [10.0, 11.0, 20.0, 30.0, 31.0, 32.0, 40.0],
            "M4_Full_Dedup": [1.0, 1.1, 2.0, 3.0, 3.1, 3.2, 4.0]
        }

        # Inject a deterministic cluster draw:
        # p3 sampled twice, p1 sampled once, p2 sampled once, p4 sampled zero times
        deterministic_clusters = np.array(["p3", "p1", "p3", "p2"])

        captured_inputs = []

        def spy_curve(scores, labels_arg):
            captured_inputs.append((list(scores), list(labels_arg)))
            return compute_risk_coverage_curve(scores, labels_arg)

        mock_rng = MagicMock()
        mock_rng.choice.return_value = deterministic_clusters

        with patch("evaluation.run_benchmark_v2_test_eval.np.random.RandomState", return_value=mock_rng), \
             patch("evaluation.run_benchmark_v2_test_eval.compute_risk_coverage_curve", side_effect=spy_curve):
            result = run_paired_cluster_bootstrap(
                cluster_ids=cluster_ids,
                scores_dict=scores_dict,
                labels=labels,
                n_bootstrap=1,
                random_seed=42
            )

        # Production runner must have evaluated score models via compute_risk_coverage_curve
        self.assertGreaterEqual(len(captured_inputs), 2)

        # Expected sampled index sequence for ["p3", "p1", "p3", "p2"]:
        # p3 -> indices [3, 4, 5] (scores [30.0, 31.0, 32.0])
        # p1 -> indices [0, 1]    (scores [10.0, 11.0])
        # p3 -> indices [3, 4, 5] (scores [30.0, 31.0, 32.0])
        # p2 -> index   [2]       (scores [20.0])
        expected_m1_scores = [30.0, 31.0, 32.0, 10.0, 11.0, 30.0, 31.0, 32.0, 20.0]
        expected_m4_scores = [3.0, 3.1, 3.2, 1.0, 1.1, 3.0, 3.1, 3.2, 2.0]

        m1_calls = [c for c in captured_inputs if c[0] == expected_m1_scores]
        m4_calls = [c for c in captured_inputs if c[0] == expected_m4_scores]

        self.assertEqual(len(m1_calls), 1, f"Production runner did not evaluate expected M1 resample: {captured_inputs}")
        self.assertEqual(len(m4_calls), 1, f"Production runner did not evaluate expected M4 resample: {captured_inputs}")

        # Verify repeated-cluster multiplicity:
        # p3 instances (originally 3) appeared twice -> exactly 6 rows
        # p1 instances (originally 2) appeared once  -> exactly 2 rows
        # p2 instances (originally 1) appeared once  -> exactly 1 row
        # p4 instances (originally 1) appeared 0 times -> exactly 0 rows
        evaluated_m1 = m1_calls[0][0]
        self.assertEqual(len(evaluated_m1), 9)
        self.assertEqual(evaluated_m1.count(30.0), 2)
        self.assertEqual(evaluated_m1.count(31.0), 2)
        self.assertEqual(evaluated_m1.count(32.0), 2)
        self.assertEqual(evaluated_m1.count(10.0), 1)
        self.assertEqual(evaluated_m1.count(11.0), 1)
        self.assertEqual(evaluated_m1.count(20.0), 1)
        self.assertEqual(evaluated_m1.count(40.0), 0)

    def test_5_paired_bootstrap_identical_draws_across_models(self):
        """Production run_paired_cluster_bootstrap must evaluate all models on identical
        resampled indices and identical metric inputs for every bootstrap draw."""
        cluster_ids = [f"p_{i % 5}" for i in range(20)]
        labels = [i % 2 for i in range(20)]
        scores_dict = {
            "M1_Relevance_Raw": [float(i) for i in range(20)],
            "M4_Full_Dedup": [float(100 + i) for i in range(20)]
        }

        captured_per_iteration = {}

        def spy_curve(scores, labels_arg):
            if scores[0] >= 100:
                m_name = "M4_Full_Dedup"
                orig_indices = [int(s - 100) for s in scores]
            else:
                m_name = "M1_Relevance_Raw"
                orig_indices = [int(s) for s in scores]

            call_idx = len(captured_per_iteration)
            captured_per_iteration[call_idx] = {
                "model": m_name,
                "indices": orig_indices,
                "labels": list(labels_arg)
            }
            return compute_risk_coverage_curve(scores, labels_arg)

        n_boot = 5
        with patch("evaluation.run_benchmark_v2_test_eval.compute_risk_coverage_curve", side_effect=spy_curve):
            result = run_paired_cluster_bootstrap(
                cluster_ids=cluster_ids,
                scores_dict=scores_dict,
                labels=labels,
                n_bootstrap=n_boot,
                random_seed=20261104
            )

        # 2 models * n_boot iterations = 2 * n_boot calls
        self.assertEqual(len(captured_per_iteration), 2 * n_boot)

        for b in range(n_boot):
            call_a = captured_per_iteration[2 * b]
            call_b = captured_per_iteration[2 * b + 1]

            # Verify identical resampled indices across both models
            self.assertEqual(call_a["indices"], call_b["indices"],
                             f"Iteration {b}: Resampled indices differ across models!")

            # Verify identical labels passed to both models
            self.assertEqual(call_a["labels"], call_b["labels"],
                             f"Iteration {b}: Labels differ across models!")

        # Verify paired differences in result dictionary
        pair_key = "M4_Full_Dedup_minus_M1_Relevance_Raw"
        self.assertIn(pair_key, result["paired_differences"])
        diff_info = result["paired_differences"][pair_key]
        self.assertIn("bootstrap_mean_delta", diff_info)
        self.assertIn("delta_ci_95", diff_info)


if __name__ == "__main__":
    unittest.main()
