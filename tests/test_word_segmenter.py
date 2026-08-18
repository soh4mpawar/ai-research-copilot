"""
Regression Test Suite: Viterbi Word Segmenter & Proper Noun Guardrail.
Asserts that:
(a) Known glued words / ligature artifacts are correctly split.
(b) Known proper nouns, model names, and scientific domain terms are NEVER split.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.word_segmenter import ViterbiWordSegmenter


class TestWordSegmenter(unittest.TestCase):
    """Test word segmentation correctness, boundary penalties, and proper noun protection."""

    @classmethod
    def setUpClass(cls):
        cls.segmenter = ViterbiWordSegmenter()

    def test_glued_words_are_split(self):
        """Known glued word concatenations from raw PDF streams must be split."""
        cases = [
            ("BUTNOTRELEASABLE", "BUT NOT RELEASABLE"),
            ("SILENTGATE", "SILENT GATE"),
            ("BOUNDEDLINEARRELEASE", "BOUNDED LINEAR RELEASE"),
            ("failtouse", "fail to use"),
            ("modelsrepresent", "models represent"),
            ("atfindingstructure", "at finding structure"),
            ("describedIn", "described In"),
            ("evaluatedOn", "evaluated On"),
            ("computedBy", "computed By"),
            ("shownIn", "shown In"),
        ]

        for glued, expected in cases:
            with self.subTest(glued=glued):
                norm = self.segmenter.normalize_text(glued)
                self.assertEqual(norm, expected, f"Failed to split '{glued}'. Got '{norm}', expected '{expected}'")

    def test_proper_nouns_and_models_are_preserved(self):
        """Model names, CamelCase proper nouns, and scientific domain terms must NOT be split."""
        protected_terms = [
            "PolyCam",
            "UniCon-Former",
            "ResNet",
            "ImageNet",
            "ConvNet",
            "TensorFlow",
            "PyTorch",
            "ChatGPT",
            "BioBERT",
            "Docling",
            "GraphRAG",
            "arXiv",
            "GloVe",
            "FitzGerald",
            "inversion",
            "autoencoder",
            "hyperparameter",
            "backpropagation",
            "eigenvalue",
            "convolutional",
            "regularization",
            "tokenization",
        ]

        for term in protected_terms:
            with self.subTest(term=term):
                norm = self.segmenter.normalize_text(term)
                self.assertEqual(norm, term, f"Over-segmentation error: protected term '{term}' was erroneously split into '{norm}'")

    def test_full_sentence_normalization(self):
        """Test full paragraph string with mixed proper nouns and glued ligatures."""
        raw_sentence = (
            "We evaluate PolyCam and ResNet models as describedIn the paper. "
            "LOCATED BUTNOTRELEASABLE: SILENTGATEINVERSION ANDBOUNDEDLINEARRELEASE. "
            "The modelsrepresent latent structure that they failtouse during inversion."
        )
        expected_sentence = (
            "We evaluate PolyCam and ResNet models as described In the paper. "
            "LOCATED BUT NOT RELEASABLE: SILENT GATE INVERSION AND BOUNDED LINEAR RELEASE. "
            "The models represent latent structure that they fail to use during inversion."
        )

        norm = self.segmenter.normalize_text(raw_sentence)
        self.assertEqual(norm, expected_sentence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
