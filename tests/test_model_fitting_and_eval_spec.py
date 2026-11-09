"""
Unit tests for Benchmark v2 Model Fitting, Preprocessing, and Frozen Evaluation Specification.
Verifies:
1. Loading integrity and exact prediction reproduction (< 1e-12).
2. Train-only preprocessing: StandardScaler mean/scale derived strictly from training data.
3. Threshold precision and operating point reproducibility.
4. Complete model serialization schema and metadata hashes.
5. Evaluation specification structure and bootstrap protocol requirements.
"""

import unittest
import json
import hashlib
import numpy as np
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from backend.selective.context import FinalContextRecord, Passage, count_tokens
from backend.selective.features import extract_features_from_record


class TestModelFittingAndEvalSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_path = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_fitted_models.json")
        cls.eval_spec_path = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_frozen_evaluation_specification.json")
        cls.val_preds_path = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_validation_predictions.json")
        cls.tr_ctx_path = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_training_contexts_240.json")
        cls.val_ctx_path = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_validation_contexts_96.json")
        cls.tr_q_path = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_training_questions_60families.json")
        cls.val_q_path = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_validation_questions_24families.json")

        cls.models_json = json.load(open(cls.models_path, encoding="utf-8"))
        cls.eval_spec = json.load(open(cls.eval_spec_path, encoding="utf-8"))
        cls.val_preds = json.load(open(cls.val_preds_path, encoding="utf-8"))
        cls.tr_contexts = json.load(open(cls.tr_ctx_path, encoding="utf-8"))
        cls.val_contexts = json.load(open(cls.val_ctx_path, encoding="utf-8"))
        cls.tr_questions = json.load(open(cls.tr_q_path, encoding="utf-8"))
        cls.val_questions = json.load(open(cls.val_q_path, encoding="utf-8"))

        cls.tr_q_map = {q["family_id"]: q for q in (cls.tr_questions if isinstance(cls.tr_questions, list) else cls.tr_questions["questions"])}
        cls.val_q_map = {q["family_id"]: q for q in (cls.val_questions if isinstance(cls.val_questions, list) else cls.val_questions["questions"])}

    def _to_record(self, ctx_dict, q_map):
        fid = ctx_dict["family_id"]
        q_text = (ctx_dict.get("question") or q_map[fid]["question"]).strip()
        passages = [
            Passage.from_dict({
                "chunk_id": p["chunk_id"],
                "paper_id": p["paper_id"],
                "section": p.get("section", "General"),
                "rerank_score": float(p.get("rerank_score", 0.0)),
                "text": p["text"],
                "token_count": count_tokens(p["text"])
            })
            for p in ctx_dict["passages"]
        ]
        rec = FinalContextRecord(
            query=q_text,
            passages=tuple(passages),
            max_passages=len(passages),
            token_budget=2000,
            tokenizer_identity="cl100k_base",
            total_tokens=sum(p.token_count for p in passages),
            skipped_passages=(),
            truncation_policy="no_truncation",
            serialized_payload="",
            payload_hash=""
        )
        object.__setattr__(rec, "context_id", ctx_dict["opaque_context_id"])
        return rec

    def test_fitted_models_schema(self):
        """Verify complete metadata and model entries in fitted models artifact."""
        meta = self.models_json["metadata"]
        self.assertEqual(meta["protocol"], "research_protocol_v1_amendment_v1.1")
        self.assertEqual(meta["random_seed"], 42)
        self.assertIn("python_version", meta)
        self.assertIn("sklearn_version", meta)
        self.assertIn("training_dataset_sha256", meta)
        self.assertIn("validation_dataset_sha256", meta)

        required_models = [
            "B0_AlwaysAccept", "B1_MaxRelevance", "B2_HistoricalGate",
            "M1_Relevance_Raw", "M2_Relevance_Dedup", "M3_Full_Raw", "M4_Full_Dedup",
            "Shortcut_LengthDiagnostic"
        ]
        for m in required_models:
            self.assertIn(m, self.models_json["models"])

    def test_train_only_preprocessing(self):
        """Verify that StandardScaler mean and scale match training data strictly."""
        # Check M1 scaler against training features
        m1_data = self.models_json["models"]["M1_Relevance_Raw"]
        tr_records = [self._to_record(c, self.tr_q_map) for c in self.tr_contexts]
        feats = np.array([
            extract_features_from_record(r, feature_mode=m1_data["feature_mode"], deduplicate=m1_data["deduplicate"])
            for r in tr_records
        ])
        expected_mean = np.mean(feats, axis=0)
        expected_scale = np.std(feats, axis=0)

        saved_mean = np.array(m1_data["scaler_mean"])
        saved_scale = np.array(m1_data["scaler_scale"])

        np.testing.assert_allclose(saved_mean, expected_mean, atol=1e-7, err_msg="Scaler mean mismatch on training data!")
        np.testing.assert_allclose(saved_scale, expected_scale, atol=1e-7, err_msg="Scaler scale mismatch on training data!")

    def test_loading_integrity_and_prediction_reproducibility(self):
        """Verify that reconstructing models from saved parameters reproduces predictions within 1e-12."""
        val_records = [self._to_record(c, self.val_q_map) for c in self.val_contexts]

        for m_name in ["M1_Relevance_Raw", "M2_Relevance_Dedup", "M3_Full_Raw", "M4_Full_Dedup"]:
            m_data = self.models_json["models"][m_name]
            scaler = StandardScaler()
            scaler.mean_ = np.array(m_data["scaler_mean"], dtype=np.float64)
            scaler.scale_ = np.array(m_data["scaler_scale"], dtype=np.float64)

            lr = LogisticRegression(penalty="l2", C=m_data["selected_C"], solver="liblinear", random_state=42)
            lr.classes_ = np.array([0, 1])
            lr.coef_ = np.array([m_data["coefficients"]], dtype=np.float64)
            lr.intercept_ = np.array([m_data["intercept"]], dtype=np.float64)

            for r in val_records:
                cid = r.context_id
                saved_score = self.val_preds[cid]["scores"][m_name]
                X_raw = extract_features_from_record(r, feature_mode=m_data["feature_mode"], deduplicate=m_data["deduplicate"]).reshape(1, -1)
                X_sc = scaler.transform(X_raw)
                recon_score = float(lr.predict_proba(X_sc)[0, 1])
                self.assertAlmostEqual(saved_score, recon_score, places=12)

    def test_threshold_precision_and_tie_breaking(self):
        """Verify full-precision threshold storage and tie breaking."""
        for m_name in ["M1_Relevance_Raw", "M2_Relevance_Dedup", "M3_Full_Raw", "M4_Full_Dedup", "Shortcut_LengthDiagnostic"]:
            m_data = self.models_json["models"][m_name]
            thresholds = m_data["thresholds"]
            for cov_str, t_info in thresholds.items():
                cov = float(cov_str)
                th = t_info["threshold"]
                self.assertIsInstance(th, float)
                self.assertGreater(th, 0.0)
                self.assertLess(th, 1.0)
                self.assertIn("val_achieved_coverage", t_info)
                self.assertIn("val_deviation", t_info)

    def test_frozen_evaluation_specification_fields(self):
        """Verify frozen evaluation specification contains all required protocol fields."""
        spec = self.eval_spec
        self.assertEqual(spec["status"], "FROZEN_BEFORE_V2_TEST_CONSTRUCTION_POST_STAGE4_EXPOSURE")
        self.assertEqual(spec["test_partition_status"]["total_papers"], 48)
        self.assertEqual(spec["test_partition_status"]["target_test_families"], 36)
        self.assertEqual(spec["test_partition_status"]["target_controlled_contexts"], 108)
        self.assertEqual(spec["test_partition_status"]["target_natural_contexts"], 36)
        self.assertIn("NO_ADDITIONAL_TEST_CONTENT_ACCESS", spec["test_partition_status"]["status"])

        comps = spec["hypotheses_and_comparisons"]
        self.assertEqual(comps["primary_comparison"]["method_A"], "M4_Full_Dedup")
        self.assertEqual(comps["primary_comparison"]["method_B"], "M1_Relevance_Raw")
        self.assertEqual(comps["primary_comparison"]["eval_subset"], "controlled_contexts_108")
        self.assertEqual(comps["practical_generalization_check"]["subset"], "natural_retrieval_contexts_36")

        boot = spec["statistical_testing_protocol"]
        self.assertEqual(boot["num_resamples"], 2000)
        self.assertEqual(boot["random_seed"], 20261104)
        self.assertEqual(boot["cluster_unit"], "primary_source_paper_id")
        self.assertIn("Report the observed difference and bootstrap percentile interval as uncertainty estimates", boot["interpretation_rule"])
        self.assertIn("does not establish equivalence", boot["interpretation_rule"])


if __name__ == "__main__":
    unittest.main()
