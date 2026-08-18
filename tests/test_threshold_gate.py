"""
Regression Test Suite: Source-Grounded Coherence Gate & Reranker Calibration (FR-11, PRD §7.4).
Ensures that:
1. Out-of-domain queries (including adversarial single-chunk lexical hooks like weather, bicycle tires,
   and sports references) are strictly blocked by the gate (0 false positives across 25 OOD queries).
2. In-domain queries (including narrow single-fact hyperparameter queries and deep-focus single-paper queries)
   reliably pass the gate with clustered source support (0 false negatives).
3. Unit tests verify same-paper dual support, deep topic focus, multi-paper consensus, and isolated spike rejection.
"""

import unittest
import sys
import os
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.retrieval.threshold_gate import (
    RelevanceThresholdGate,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_SECONDARY_THRESHOLD,
    DEFAULT_DEEP_FOCUS_THRESHOLD
)
from backend.reranking.reranker import CrossEncoderReranker
from backend.pipeline import ResearchPipelineOrchestrator


class TestThresholdGateLogic(unittest.TestCase):
    """Unit tests for RelevanceThresholdGate decision rules."""

    def setUp(self):
        self.gate = RelevanceThresholdGate(
            threshold=0.35,
            secondary_threshold=0.15,
            deep_focus_threshold=0.85,
            min_deep_focus_candidates=3
        )

    def test_same_paper_dual_support_passes(self):
        """Pass if same paper has primary chunk >= 0.35 and secondary chunk >= 0.15."""
        chunks = [
            {"paper_id": "paper_A", "rerank_score": 0.88, "text": "Chunk 1 from Paper A"},
            {"paper_id": "paper_A", "rerank_score": 0.42, "text": "Chunk 2 from Paper A"},
            {"paper_id": "paper_B", "rerank_score": 0.05, "text": "Chunk 1 from Paper B"},
        ]
        passed, valid, msg = self.gate.evaluate_chunks(chunks)
        self.assertTrue(passed, "Same-paper dual reinforcement must pass the gate.")
        self.assertEqual(len(valid), 2)
        self.assertIn("[OK]", msg)

    def test_isolated_single_chunk_spike_fails(self):
        """Reject if top chunk is >= 0.35 but no secondary chunk from that paper >= 0.15 exists."""
        chunks = [
            {"paper_id": "paper_A", "rerank_score": 0.94, "text": "Isolated semantic hook chunk"},
            {"paper_id": "paper_B", "rerank_score": 0.08, "text": "Unrelated chunk B"},
            {"paper_id": "paper_C", "rerank_score": 0.02, "text": "Unrelated chunk C"},
        ]
        passed, valid, msg = self.gate.evaluate_chunks(chunks)
        self.assertFalse(passed, "Isolated single-chunk spike without source support must be blocked.")
        self.assertEqual(len(valid), 0)
        self.assertIn("[WARN]", msg)

    def test_same_paper_deep_focus_passes(self):
        """Pass if paper has top chunk >= 0.85 and >= 3 candidates in the pool."""
        chunks = [
            {"paper_id": "paper_A", "rerank_score": 0.95, "text": "Abstract of Paper A"},
            {"paper_id": "paper_A", "rerank_score": 0.09, "text": "Section 1 of Paper A"},
            {"paper_id": "paper_A", "rerank_score": 0.06, "text": "Section 2 of Paper A"},
            {"paper_id": "paper_B", "rerank_score": 0.03, "text": "Unrelated chunk B"},
        ]
        passed, valid, msg = self.gate.evaluate_chunks(chunks)
        self.assertTrue(passed, "Deep topic focus with high score (>=0.85) and >=3 paper candidates must pass.")
        self.assertIn("[OK]", msg)

    def test_multi_paper_consensus_passes(self):
        """Pass if at least two distinct papers each have a chunk >= 0.35."""
        chunks = [
            {"paper_id": "paper_A", "rerank_score": 0.72, "text": "Finding in Paper A"},
            {"paper_id": "paper_B", "rerank_score": 0.65, "text": "Finding in Paper B"},
            {"paper_id": "paper_C", "rerank_score": 0.05, "text": "Unrelated chunk C"},
        ]
        passed, valid, msg = self.gate.evaluate_chunks(chunks)
        self.assertTrue(passed, "Multi-paper cross-source consensus must pass the gate.")
        self.assertEqual(len(valid), 2)
        self.assertIn("[OK]", msg)

    def test_below_baseline_floor_fails(self):
        """Reject if all candidate scores are strictly < 0.35."""
        chunks = [
            {"paper_id": "paper_A", "rerank_score": 0.12, "text": "Noisy chunk 1"},
            {"paper_id": "paper_A", "rerank_score": 0.09, "text": "Noisy chunk 2"},
        ]
        passed, valid, msg = self.gate.evaluate_chunks(chunks)
        self.assertFalse(passed, "Scores below floor threshold must fail gate.")
        self.assertEqual(len(valid), 0)

    def test_empty_chunks_fails(self):
        """Reject empty candidate list."""
        passed, valid, msg = self.gate.evaluate_chunks([])
        self.assertFalse(passed, "Empty chunks must fail gate.")
        self.assertEqual(len(valid), 0)


class TestThresholdGateEndToEnd(unittest.TestCase):
    """End-to-end integration regression tests against the ingested 200-paper corpus."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ResearchPipelineOrchestrator()
        cls.gate = RelevanceThresholdGate()

    def _query_and_evaluate(self, query: str) -> bool:
        dense = self.orchestrator.vector_store.search_dense(query, top_k=25)
        sparse = self.orchestrator.bm25_retriever.search_sparse(query, top_k=25)
        fused = self.orchestrator.fusion_retriever.fuse_results(dense, sparse, top_k=30)
        graph = self.orchestrator.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
        pool = list(fused)
        if graph:
            pool.extend(graph)
        reranked = self.orchestrator.reranker.rerank_chunks(query, pool, top_k=12)
        passed, _, _ = self.gate.evaluate_chunks(reranked)
        return passed

    def test_narrow_in_domain_queries_pass(self):
        """Assert 5 narrow single-fact in-domain questions pass the gate."""
        narrow_queries = [
            ("VGG Batch Size", "What batch size was used for training ConvNets in paper 1409.1556?"),
            ("BERT Masking Rate", "What percentage of tokens are masked in BERT masked language modeling?"),
            ("Swin Resolution", "What input image resolution is used for Swin Transformer pre-training?"),
            ("AdamW Beta2", "What beta2 hyperparameter value is used for Adam in Attention Is All You Need?"),
            ("ResNet Stride", "What stride is used in downsampling convolutional layers in ResNet?"),
        ]
        for name, q in narrow_queries:
            with self.subTest(query_name=name):
                passed = self._query_and_evaluate(q)
                self.assertTrue(passed, f"Narrow in-domain query '{name}' must pass gate.")

    def test_broad_in_domain_queries_pass(self):
        """Assert broad in-domain queries pass the gate."""
        broad_queries = [
            ("Transformer", "How does self-attention work in the Transformer architecture?"),
            ("ResNet", "How do residual connections address vanishing gradients in deep networks?"),
            ("VGG Architecture", "How does the architecture of VGGNet use very small 3x3 convolution filters?"),
        ]
        for name, q in broad_queries:
            with self.subTest(query_name=name):
                passed = self._query_and_evaluate(q)
                self.assertTrue(passed, f"Broad in-domain query '{name}' must pass gate.")

    def test_adversarial_out_of_domain_queries_blocked(self):
        """Assert 23 adversarial out-of-domain queries (including lexical spikes) are blocked."""
        adversarial_ood_queries = [
            ("Weather Forecast", "What's the weather going to be like tomorrow?"),
            ("Bicycle Tire", "How do I fix a flat bicycle tire?"),
            ("Champions League", "Who won the UEFA Champions League last season?"),
            ("Lasagna Recipe", "How to bake a classic meat lasagna with ricotta cheese and tomato sauce?"),
            ("Tax Filing", "What is the deadline for filing federal income taxes in the United States?"),
            ("Capital of Australia", "What is the capital city of Australia?"),
            ("Poem", "Write a rhyming poem about a cat and a dog becoming best friends."),
            ("Workout Plan", "What is the best 5-day workout split for building muscle hypertrophy?"),
            ("Sci-Fi Movies", "What are the best sci-fi movies released in the 1990s?"),
            ("Gravity Spacetime", "What is Einstein's general theory of relativity and spacetime curvature?"),
            ("Bread Algorithm", "What is the algorithm for kneading sourdough bread dough?"),
            ("Photosynthesis", "How does chlorophyll in plant leaves absorb light energy during photosynthesis?"),
            ("Honda Oil Change", "How do you change the engine oil on a 2018 Honda Civic?"),
            ("Texas Hold'em", "What are the complete rules and hand rankings in Texas Hold'em poker?"),
            ("Roman Empire", "What caused the fall of the Western Roman Empire in 476 AD?"),
            ("Paris Vacation", "What are the top 10 tourist attractions to visit in Paris France?"),
            ("Airplane Lift", "How does Bernoulli's principle explain aerodynamic lift on airplane wings?"),
            ("Volcano Magma", "What causes volcanic eruptions and magma formation inside active volcanoes?"),
            ("Zillow Scraper", "How do I write a web scraper in Python to extract real estate prices from Zillow?"),
            ("Acoustic Guitar", "How do you tune an acoustic guitar using standard EADGBE tuning?"),
            ("Human Digestion", "How does the human digestive system absorb nutrients in the small intestine?"),
            ("Japan Inflation", "What economic policies did the Bank of Japan use to fight deflation?"),
            ("Elden Ring Plot", "What is the main storyline and lore of the video game Elden Ring?"),
        ]
        for name, q in adversarial_ood_queries:
            with self.subTest(ood_name=name):
                passed = self._query_and_evaluate(q)
                self.assertFalse(passed, f"Adversarial OOD query '{name}' must be blocked by the gate.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
