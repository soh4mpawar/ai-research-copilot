"""
Unit and regression tests for controlled context construction, single-passage replacement,
and cross-encoder score provenance under Benchmark v2 protocol.
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from typing import Dict, List, Tuple, Any

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.run_benchmark_v2_train_val import load_and_verify_split
from backend.selective.retrieval_adapter import SplitRestrictedHybridRetriever
from backend.selective.scoring import (
    score_chunk_with_cache,
    score_and_package_passages,
    CrossEncoderScoringManager
)

TRAIN_CTX_PATH = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_training_contexts_240.json")
VAL_CTX_PATH = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_validation_contexts_96.json")
TEST_CTX_PATH = os.path.join(BASE_DIR, "data/metadata/benchmark_v2_test_contexts_144.json")


class DummyRecordingScorer:
    """
    Injected deterministic scorer that records (query, text) inputs,
    returns deterministic values (including genuine 0.0 and low scores),
    and supports cache verification.
    """
    def __init__(self, score_mapping: Dict[Tuple[str, str], float]):
        self.score_mapping = score_mapping
        self.calls: List[Tuple[str, str]] = []

    def predict(self, pairs: List[List[str]]) -> List[float]:
        results = []
        for q, t in pairs:
            self.calls.append((q, t))
            # Return mapped score or deterministic hash-based score
            score = self.score_mapping.get((q, t), 0.0)
            results.append(score)
        return results


class TestControlledConstructionAndScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.exists(TRAIN_CTX_PATH), f"Train file not found: {TRAIN_CTX_PATH}"
        assert os.path.exists(VAL_CTX_PATH), f"Val file not found: {VAL_CTX_PATH}"
        assert os.path.exists(TEST_CTX_PATH), f"Test file not found: {TEST_CTX_PATH}"
        with open(TRAIN_CTX_PATH, "r", encoding="utf-8") as f:
            cls.train_contexts = json.load(f)
        with open(VAL_CTX_PATH, "r", encoding="utf-8") as f:
            cls.val_contexts = json.load(f)
        with open(TEST_CTX_PATH, "r", encoding="utf-8") as f:
            cls.test_contexts = json.load(f)

    def test_injected_deterministic_scorer_and_caching(self):
        """
        Verify cross-encoder scorer execution with injected deterministic scorer:
        1. Records unique (query, text) pairs scored.
        2. Verified that duplicate passages in context reuse the identical score.
        3. Cache keys distinguish different queries and configs.
        4. Genuine zeros and low scores pass without rejection.
        5. Returned values propagate faithfully to context passage records.
        """
        query_a = "What is the primary causes of strategic drift?"
        query_b = "How does contrastive decoding express probability distribution?"
        text_supp = "Finite attention resources prevent LLMs from maintaining strategic coherence over thousands of steps."
        text_dist1 = "Diffusion model distillation primarily focuses on step distillation."
        text_zero = "Completely ungrounded random text that yields zero predictive relevance."

        # Map deterministic scores: note text_zero maps to genuine 0.0, text_dist1 to low score 0.005
        test_scores = {
            (query_a, text_supp): 0.8950,
            (query_a, text_dist1): 0.0050,
            (query_a, text_zero): 0.0000,
            (query_b, text_supp): 0.0120,
        }
        mock_scorer = DummyRecordingScorer(test_scores)

        # Test production shared scoring helper with injected recording scorer
        cache: Dict[Tuple[str, str, str], float] = {}
        cfg_standard = "model=BAAI/bge-reranker-base;max_length=512;activation=sigmoid;device=cuda:0"
        cfg_variant = "model=BAAI/bge-reranker-base;max_length=256;activation=sigmoid;device=cuda:0"

        # 1. Score passages including duplicate using production helper
        s_supp = score_chunk_with_cache(query_a, text_supp, mock_scorer, cache, cfg_standard)
        s_dist1 = score_chunk_with_cache(query_a, text_dist1, mock_scorer, cache, cfg_standard)
        # Duplicate of dist1: must reuse cache and NOT call scorer again
        s_dist1_dup = score_chunk_with_cache(query_a, text_dist1, mock_scorer, cache, cfg_standard)
        # Genuine zero score
        s_zero = score_chunk_with_cache(query_a, text_zero, mock_scorer, cache, cfg_standard)

        self.assertEqual(s_supp, 0.8950)
        self.assertEqual(s_dist1, 0.0050)
        self.assertEqual(s_dist1_dup, 0.0050)
        self.assertEqual(s_zero, 0.0000)  # Genuine zero must pass!

        # Scorer was called only 3 times for 4 queries (dist1 was cached)
        self.assertEqual(len(mock_scorer.calls), 3)
        self.assertEqual(mock_scorer.calls[0], (query_a, text_supp))
        self.assertEqual(mock_scorer.calls[1], (query_a, text_dist1))
        self.assertEqual(mock_scorer.calls[2], (query_a, text_zero))

        # 2. Cache keys distinguish different queries
        s_supp_qb = score_chunk_with_cache(query_b, text_supp, mock_scorer, cache, cfg_standard)
        self.assertEqual(s_supp_qb, 0.0120)
        self.assertEqual(len(mock_scorer.calls), 4)

        # 3. Cache keys distinguish different configs
        _ = score_chunk_with_cache(query_a, text_supp, mock_scorer, cache, cfg_variant)
        self.assertEqual(len(mock_scorer.calls), 5)

        # 4. Verified values reach saved passage objects via production packaging helper
        passages = [
            {"chunk_id": "c1", "text": text_supp},
            {"chunk_id": "c2", "text": text_dist1},
            {"chunk_id": "c2_dup", "text": text_dist1},
            {"chunk_id": "c3", "text": text_zero},
        ]
        # Use production score_and_package_passages
        packaged = score_and_package_passages(
            query=query_a,
            passages=passages,
            scorer=mock_scorer,
            cache=cache,
            config_str=cfg_standard
        )
        self.assertEqual(packaged[0]["rerank_score"], 0.8950)
        self.assertEqual(packaged[1]["rerank_score"], packaged[2]["rerank_score"])
        self.assertEqual(packaged[1]["rerank_score"], 0.0050)
        self.assertEqual(packaged[3]["rerank_score"], 0.0000)
        self.assertEqual(passages[0]["rerank_score"], 0.8950)
        self.assertEqual(passages[1]["rerank_score"], passages[2]["rerank_score"])
        self.assertEqual(passages[3]["rerank_score"], 0.0000)

    def test_loader_rejects_missing_score(self):
        """
        Verify that load_and_verify_split strictly raises ValueError when rerank_score is missing or None.
        Tests both missing field and None value on temporary input.
        """
        # Load valid val contexts
        with open(VAL_CTX_PATH, "r", encoding="utf-8") as f:
            contexts_copy = json.load(f)

        orig_json_load = json.load

        # Case 1: Delete rerank_score from the first passage of the first context
        ctx_missing_score = copy.deepcopy(contexts_copy)
        del ctx_missing_score[0]["passages"][0]["rerank_score"]

        def mock_load_missing(fp, *args, **kwargs):
            name = getattr(fp, "name", "")
            if "validation_contexts" in name:
                return ctx_missing_score
            return orig_json_load(fp, *args, **kwargs)

        with patch("json.load", side_effect=mock_load_missing):
            with self.assertRaises(ValueError) as cm:
                load_and_verify_split("val")
            self.assertIn("Missing rerank_score", str(cm.exception))

        # Case 2: Set rerank_score = None
        ctx_none_score = copy.deepcopy(contexts_copy)
        ctx_none_score[0]["passages"][0]["rerank_score"] = None

        def mock_load_none(fp, *args, **kwargs):
            name = getattr(fp, "name", "")
            if "validation_contexts" in name:
                return ctx_none_score
            return orig_json_load(fp, *args, **kwargs)

        with patch("json.load", side_effect=mock_load_none):
            with self.assertRaises(ValueError) as cm:
                load_and_verify_split("val")
            self.assertIn("Missing rerank_score", str(cm.exception))

    def test_single_passage_replacement_integrity_train_and_val(self):
        """
        Verify single-passage replacement across BOTH training and validation splits:
        1. All 3 controlled conditions have exactly 4 passages.
        2. Exactly 1 passage changes between supported_candidate and relevant_replacement.
        3. Exactly 1 passage changes between supported_candidate and duplicate_replacement, at the SAME position.
        4. Preserved passages match text, source paper, section, score, and position across all 3 conditions.
        5. Duplicate replacement content directly matches a remaining passage by text and paper
           (without relying on _dup in chunk_id) and receives identical cross-encoder score.
        """
        for split_name, contexts, expected_families in [
            ("train", self.train_contexts, 60),
            ("val", self.val_contexts, 24),
            ("test", self.test_contexts, 36)
        ]:
            by_family = {}
            for c in contexts:
                by_family.setdefault(c["family_id"], {})[c["variant"]] = c

            self.assertEqual(len(by_family), expected_families)

            for fid, cond_map in by_family.items():
                self.assertIn("supported_candidate", cond_map, f"{split_name} {fid} missing supported_candidate")
                self.assertIn("relevant_replacement", cond_map, f"{split_name} {fid} missing relevant_replacement")
                self.assertIn("duplicate_replacement", cond_map, f"{split_name} {fid} missing duplicate_replacement")

                supp = cond_map["supported_candidate"]
                rel = cond_map["relevant_replacement"]
                dup = cond_map["duplicate_replacement"]

                self.assertEqual(len(supp["passages"]), 4, f"{split_name} {fid} supp len != 4")
                self.assertEqual(len(rel["passages"]), 4, f"{split_name} {fid} rel len != 4")
                self.assertEqual(len(dup["passages"]), 4, f"{split_name} {fid} dup len != 4")

                # Positional single-replacement
                s_cids = [p["chunk_id"] for p in supp["passages"]]
                r_cids = [p["chunk_id"] for p in rel["passages"]]
                d_cids = [p["chunk_id"] for p in dup["passages"]]

                diff_s_r = [i for i in range(4) if s_cids[i] != r_cids[i]]
                diff_s_d = [i for i in range(4) if s_cids[i] != d_cids[i]]

                self.assertEqual(len(diff_s_r), 1, f"{split_name} {fid} rel diff != 1: {diff_s_r}")
                self.assertEqual(len(diff_s_d), 1, f"{split_name} {fid} dup diff != 1: {diff_s_d}")
                self.assertEqual(diff_s_r[0], diff_s_d[0], f"{split_name} {fid} replacement position mismatch: rel={diff_s_r[0]} dup={diff_s_d[0]}")

                rep_pos = diff_s_r[0]

                # Preserved passages must be identical across text, paper_id, section, score, and position
                for i in range(4):
                    if i != rep_pos:
                        ps = supp["passages"][i]
                        pr = rel["passages"][i]
                        pd = dup["passages"][i]

                        self.assertEqual(ps["chunk_id"], pr["chunk_id"], f"{split_name} {fid} chunk mismatch pos {i}")
                        self.assertEqual(ps["chunk_id"], pd["chunk_id"], f"{split_name} {fid} chunk mismatch pos {i}")
                        self.assertEqual(ps["text"], pr["text"], f"{split_name} {fid} text mismatch pos {i}")
                        self.assertEqual(ps["text"], pd["text"], f"{split_name} {fid} text mismatch pos {i}")
                        self.assertEqual(ps["paper_id"], pr["paper_id"], f"{split_name} {fid} paper mismatch pos {i}")
                        self.assertEqual(ps["paper_id"], pd["paper_id"], f"{split_name} {fid} paper mismatch pos {i}")
                        self.assertEqual(ps["section"], pr["section"], f"{split_name} {fid} section mismatch pos {i}")
                        self.assertEqual(ps["section"], pd["section"], f"{split_name} {fid} section mismatch pos {i}")
                        self.assertAlmostEqual(ps["rerank_score"], pr["rerank_score"], places=6, msg=f"{split_name} {fid} score diff pos {i}")
                        self.assertAlmostEqual(ps["rerank_score"], pd["rerank_score"], places=6, msg=f"{split_name} {fid} score diff pos {i}")

                # Duplicate replacement content verification directly (without relying on _dup in chunk_id)
                p_dup = dup["passages"][rep_pos]
                rem_passages = [dup["passages"][i] for i in range(4) if i != rep_pos]
                matching_content = [
                    p for p in rem_passages
                    if p["text"] == p_dup["text"] and p["paper_id"] == p_dup["paper_id"]
                ]
                self.assertGreaterEqual(
                    len(matching_content), 1,
                    f"{split_name} {fid} duplicate at pos {rep_pos} has no matching passage by text and paper!"
                )
                self.assertAlmostEqual(
                    p_dup["rerank_score"], matching_content[0]["rerank_score"], places=6,
                    msg=f"{split_name} {fid} duplicate chunk score mismatch with base passage"
                )

    def test_natural_retrieval_configuration_and_bounds(self):
        """
        Verify natural retrieval respects max 6 passages, max 2000 tokens,
        and verifies the actual retriever configuration (final_max_passages=6, final_max_tokens=2000).
        """
        # Verify retriever configuration constants
        import inspect
        init_sig = inspect.signature(SplitRestrictedHybridRetriever.__init__)
        self.assertEqual(init_sig.parameters["final_max_passages"].default, 6)
        self.assertEqual(init_sig.parameters["final_max_tokens"].default, 2000)

        for name, contexts in [("train", self.train_contexts), ("val", self.val_contexts), ("test", self.test_contexts)]:
            nat_contexts = [c for c in contexts if c["variant"] == "natural_retrieval"]
            if name == "train":
                expected_count = 60
            elif name == "val":
                expected_count = 24
            else:
                expected_count = 36
            self.assertEqual(len(nat_contexts), expected_count)

            max_passages_observed = max(len(c["passages"]) for c in nat_contexts)
            max_tokens_observed = max(c["total_tokens"] for c in nat_contexts)

            for c in nat_contexts:
                self.assertLessEqual(len(c["passages"]), 6, f"{c['opaque_context_id']} exceeds 6 passages: {len(c['passages'])}")
                self.assertLessEqual(c["total_tokens"], 2000, f"{c['opaque_context_id']} exceeds 2000 tokens: {c['total_tokens']}")

            # Verify the retriever exercised its capacity (not truncated to 3 passages or 1000 tokens)
            self.assertGreaterEqual(max_passages_observed, 5, f"{name} natural retrieval max passages too low: {max_passages_observed}")
            self.assertGreaterEqual(max_tokens_observed, 1000, f"{name} natural retrieval max tokens too low: {max_tokens_observed}")


if __name__ == "__main__":
    unittest.main()
