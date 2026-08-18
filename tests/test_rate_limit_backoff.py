"""
Regression Test Suite: Rate Limit Retry & Exponential Backoff.
Asserts that transient 429 rate limits from LLM APIs trigger automatic retries
rather than silently degrading to offline-fallback on the first error.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.generation.point_qa import GroundedPointQAEngine


class TestRateLimitBackoff(unittest.TestCase):
    """Test retry-with-backoff behavior on 429 rate limit exceptions."""

    def setUp(self):
        self.qa_engine = GroundedPointQAEngine()
        self.mock_chunks = [
            {
                "chunk_id": "c1",
                "paper_id": "1706.03762",
                "paper_title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "section": "Methodology",
                "text": "The Transformer uses multi-head self-attention.",
                "rerank_score": 0.88,
                "score": 0.88
            },
            {
                "chunk_id": "c2",
                "paper_id": "1706.03762",
                "paper_title": "Attention Is All You Need",
                "authors": "Vaswani et al.",
                "section": "Architecture",
                "text": "Multi-head attention allows the model to attend to information.",
                "rerank_score": 0.65,
                "score": 0.65
            }
        ]

    def test_retry_succeeds_after_transient_429(self):
        """Simulate 429 Rate Limit on first 2 attempts, success on attempt 3."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Multi-head attention allows the model to attend to information [1]."

        # Call sequence: 429 error, 429 error, success
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 ResourceExhausted: Rate limit exceeded")
            return mock_response

        mock_client.models.generate_content.side_effect = side_effect
        self.qa_engine.client = mock_client
        self.qa_engine.fallback_models = ["gemini-3.5-flash-lite"]

        with patch("time.sleep", return_value=None) as mock_sleep:
            result = self.qa_engine.generate_point_qa(
                query="How does multi-head attention work?",
                reranked_chunks=self.mock_chunks
            )

            # Asserts
            self.assertEqual(call_count, 3, "Engine should have attempted 3 calls before succeeding")
            self.assertNotEqual(result.generator_model, "Google/offline-fallback", "Should NOT degrade to offline-fallback when retry succeeds")
            self.assertIn("Multi-head attention", result.answer)
            self.assertEqual(mock_sleep.call_count, 2, "Should have slept between failed attempts for backoff")

    def test_graceful_fallback_when_all_retries_exhausted(self):
        """Simulate persistent 429 errors across all retry attempts."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 Permanent Quota Exhausted")

        self.qa_engine.client = mock_client
        self.qa_engine.fallback_models = ["gemini-3.5-flash-lite"]

        with patch("time.sleep", return_value=None):
            result = self.qa_engine.generate_point_qa(
                query="How does multi-head attention work?",
                reranked_chunks=self.mock_chunks
            )

            # Asserts
            self.assertEqual(result.generator_model, "Google/offline-fallback", "Should fall back gracefully only after all retries are exhausted")
            self.assertTrue(len(result.answer) > 0, "Fallback answer should contain synthesized context")


if __name__ == "__main__":
    unittest.main(verbosity=2)
