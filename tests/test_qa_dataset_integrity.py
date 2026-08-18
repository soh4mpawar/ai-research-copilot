"""
Regression Test Suite: Synthetic QA Dataset Integrity & Ground-Truth Alignment Scanner.
Audits the 40-sample benchmark dataset, asserts schema and content completeness,
and cross-encoder-scores every (question, ground_truth_passage) pair to flag potential
lexical divergence or hallucinated ground truths for developer review.
"""

import unittest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.reranking.reranker import CrossEncoderReranker


class TestQADatasetIntegrity(unittest.TestCase):
    """Scan and audit all 40 QA pairs in draft_qa_dataset.json for ground-truth alignment."""

    @classmethod
    def setUpClass(cls):
        cls.reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "metadata", "draft_qa_dataset.json"))
        with open(dataset_path, "r", encoding="utf-8") as f:
            cls.qa_dataset = json.load(f)

    def test_schema_and_completeness(self):
        """Assert that all 40 samples have complete required fields, non-empty questions, and non-empty ground truths."""
        self.assertEqual(len(self.qa_dataset), 40, "Evaluation dataset must contain exactly 40 samples")

        for sample in self.qa_dataset:
            sid = sample.get("id")
            self.assertIsNotNone(sid, "Sample must have an 'id'")
            self.assertTrue(sample.get("question", "").strip(), f"Sample {sid} question must not be empty")
            self.assertGreater(len(sample.get("question", "")), 10, f"Sample {sid} question too short")
            
            ground_truth = sample.get("ground_truth_passage", sample.get("ground_truth_answer", "")).strip()
            self.assertTrue(ground_truth, f"Sample {sid} ground truth must not be empty")
            self.assertGreater(len(ground_truth), 15, f"Sample {sid} ground truth too short")

            paper_id = sample.get("source_paper_id", "")
            self.assertTrue(paper_id, f"Sample {sid} must have a 'source_paper_id'")

    def test_cross_encoder_ground_truth_alignment_audit(self):
        """Cross-encoder audit to log alignment scores and flag potential phrasing mismatches."""
        if not self.reranker.model:
            self.skipTest("Cross-encoder model not loaded")

        pairs = []
        sample_ids = []
        for sample in self.qa_dataset:
            pairs.append([
                sample.get("question", ""),
                sample.get("ground_truth_passage", sample.get("ground_truth_answer", ""))
            ])
            sample_ids.append(sample.get("id"))

        raw_scores = self.reranker.model.predict(pairs)
        aligned_count = 0
        flagged_samples = []

        for sid, score, sample in zip(sample_ids, raw_scores, self.qa_dataset):
            score_val = float(score)
            if score_val >= 0.35:
                aligned_count += 1
            else:
                flagged_samples.append({
                    "id": sid,
                    "paper_id": sample.get("source_paper_id"),
                    "score": score_val,
                    "question": sample.get("question")
                })

        print(f"\n[QA Dataset Integrity Audit] {aligned_count}/40 ({aligned_count/40*100:.1f}%) direct high-alignment pairs (score >= 0.35).")
        if flagged_samples:
            print(f"[Audit Notice] {len(flagged_samples)} sample(s) flagged for lexical/phrasing divergence:")
            for item in flagged_samples:
                print(f"  • {item['id']} ({item['paper_id']}) - Score: {item['score']:.4f} | Q: {item['question'][:80]}...")

        # Assert that all 40 samples were audited successfully without runtime errors
        self.assertEqual(len(raw_scores), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
