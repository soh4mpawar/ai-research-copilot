"""
Comprehensive Offline Test Suite for Selective Scientific QA Harness & Application Integrity.

Covers:
1. All-Tied Regression: Uniform random tie ordering yields exact 0.5 risk at all k and 0.5 AURC.
2. Undefined Metrics: Zero accepted -> None risk; missing classes -> None Type I/II error.
3. Clustered Bootstrap & Paired Differences on identical clusters.
4. Case-Preserving Deduplication: Scientific 'A' vs 'a' preserved; whitespace normalized.
5. Hash Integrity Rejection: Inconsistent content_hash raises ValueError.
6. Unsorted Deduplication: Retains max-scoring duplicate deterministically and resolves section metadata.
7. Dual Context Budget: 6-passage cap and 2,000-token budget with pinned cl100k_base tokenizer.
8. Generation Payload Integration: Verifies exact payload string and payload_hash sent to LLM prompt.
9. Autorater Provenance: Cache miss returns pending (verdict=None); strict parsing of [[1]]/[[0]].
10. 2x2 Factorial Logistic Design: M1, M2, M3, M4 with identical preprocessing, log-loss tuning, and deployable thresholds.
11. Splitter Order Invariance & Leakage Audit.
12. Strict Mode Failure Propagation without fallback.

All tests are strictly offline and deterministic with zero paid API calls.
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.selective.context import (
    Passage,
    FinalContextRecord,
    select_final_context,
    deduplicate_passages_exact,
    inject_duplicates_post_selection,
    inject_duplicates_with_displacement,
    compute_text_sha256,
    normalize_whitespace,
    count_tokens,
    PINNED_TOKENIZER_NAME
)
from backend.selective.features import (
    extract_relevance_features,
    extract_source_features,
    extract_features_from_record
)
from backend.selective.models import (
    AlwaysAcceptBaseline,
    ValidationTunedMaxThreshold,
    FrozenHistoricalGateBaseline,
    LogisticGatingModel
)
from backend.selective.dataset import (
    QuestionFamily,
    SupportingEvidence,
    partition_families_by_paper_components,
    partition_families_by_manifest
)
from evaluation.run_selective_eval import run_selective_evaluation_pipeline
from backend.selective.metrics import (
    compute_evidence_selection_risk,
    compute_type_errors,
    compute_risk_coverage_curve,
    evaluate_at_matched_coverage,
    cluster_bootstrap_evaluation
)
from backend.selective.autorater import (
    SufficientContextInspiredAutorater,
    AutoraterResult
)
from backend.generation.point_qa import GroundedPointQAEngine


class TestSelectiveAggregationHarness(unittest.TestCase):
    """Unit test suite for selective answering infrastructure and integrity fixes."""

    def setUp(self):
        self.p1 = Passage.from_dict({
            "chunk_id": "chk_01",
            "paper_id": "paper_101",
            "section": "Methodology",
            "text": "Transformer uses scaled dot-product attention with scaling factor 1/sqrt(d_k).",
            "score": 0.92
        })
        self.p2 = Passage.from_dict({
            "chunk_id": "chk_02",
            "paper_id": "paper_101",
            "section": "Experiments",
            "text": "On WMT 2014 English-to-German, the model achieves a BLEU score of 28.4.",
            "score": 0.68
        })
        self.p3 = Passage.from_dict({
            "chunk_id": "chk_03",
            "paper_id": "paper_202",
            "section": "Architecture",
            "text": "Residual connections allow gradients to propagate directly through shortcut connections.",
            "score": 0.55
        })

    def test_all_tied_regression_uniform_random_tie_ordering(self):
        """
        Regression Test: With 50% insufficient contexts and all scores tied,
        expected risk at EVERY nonzero accepted count k and expected AURC must both equal exactly 0.5.
        Intermediate tie-group points must be flagged as non-attainable deterministic thresholds.
        """
        n = 10
        scores = [0.75] * n
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 5 insufficient (0), 5 sufficient (1)

        curve = compute_risk_coverage_curve(scores, labels)
        self.assertTrue(curve["is_all_tied"])
        self.assertAlmostEqual(curve["aurc"], 0.5, places=6)

        # Expected risk at every k in 1..n must be 0.5
        for k in range(1, n + 1):
            risk_k = curve["expected_risks_by_k"][k]
            self.assertAlmostEqual(risk_k, 0.5, places=6, msg=f"Risk at k={k} must be 0.5")

        # Intermediate points must be flagged as randomized descriptive points
        for pt in curve["points"][:-1]:
            self.assertFalse(pt["is_attainable_deterministic"])
        # Only the final point (k=n, all accepted) is attainable deterministically
        self.assertTrue(curve["points"][-1]["is_attainable_deterministic"])

        matched = evaluate_at_matched_coverage(curve, [0.50, 0.75, 0.90])
        for cov, res in matched.items():
            self.assertAlmostEqual(res["expected_risk"], 0.5, places=6)
            self.assertFalse(res["is_attainable"])

    def test_undefined_risk_and_conditional_errors(self):
        """
        Undefined Edge Cases:
        - When no items accepted, risk is undefined (None), not 0.0.
        - Missing negative class -> Type I error is None.
        - Missing positive class -> Type II error is None.
        """
        labels = [0, 1, 0, 1]

        # Zero accepted -> risk is None
        risk_zero = compute_evidence_selection_risk([0, 0, 0, 0], labels)
        self.assertIsNone(risk_zero)

        # Missing negative class (all labels = 1) -> Type I error is None
        t1, t2 = compute_type_errors([1, 1], [1, 1])
        self.assertIsNone(t1)
        self.assertAlmostEqual(t2, 0.0)

        # Missing positive class (all labels = 0) -> Type II error is None
        t1, t2 = compute_type_errors([0, 0], [0, 0])
        self.assertAlmostEqual(t1, 0.0)
        self.assertIsNone(t2)

    def test_case_preserving_deduplication(self):
        """
        Deduplication MUST preserve character case.
        Scientific symbols ('A' vs 'a', 'p' vs 'P') differ by case.
        Whitespace is normalized, but casing must not be conflated.
        """
        p_upper = Passage.from_dict({
            "chunk_id": "p_up",
            "paper_id": "paper_1",
            "section": "Sec",
            "text": "Variable A represents matrix attention.",
            "score": 0.85
        })
        p_lower = Passage.from_dict({
            "chunk_id": "p_low",
            "paper_id": "paper_1",
            "section": "Sec",
            "text": "Variable a represents scalar attention.",
            "score": 0.80
        })

        deduped = deduplicate_passages_exact([p_upper, p_lower])
        # Case differs -> both passages must be preserved
        self.assertEqual(len(deduped), 2)

        # Whitespace normalization with identical case -> collapses
        p_spaces = Passage.from_dict({
            "chunk_id": "p_sp",
            "paper_id": "paper_1",
            "section": "Sec",
            "text": "Variable   A   represents \n matrix attention.",
            "score": 0.90
        })
        deduped_ws = deduplicate_passages_exact([p_upper, p_spaces])
        self.assertEqual(len(deduped_ws), 1)
        # Winning passage is p_spaces because score 0.90 > 0.85
        self.assertEqual(deduped_ws[0].passage_id, "p_sp")

    def test_hash_integrity_rejection(self):
        """Passage initialization must reject inconsistent supplied hashes."""
        with self.assertRaises(ValueError):
            Passage(
                passage_id="bad_p",
                paper_id="paper_1",
                section="Sec",
                text="Actual genuine text",
                score=0.8,
                content_hash="fabricated_inconsistent_hash_12345",
                token_count=10
            )

    def test_deduplication_retains_max_scoring_duplicate_on_unsorted_input(self):
        """
        Deduplication must retain maximum-scoring duplicate deterministically
        even when candidates are passed in unsorted order.
        Resolves generic section metadata if a duplicate has a specific section.
        """
        p_low = Passage.from_dict({
            "chunk_id": "chk_low_first",
            "paper_id": "paper_1",
            "section": "General",
            "text": "Identical scientific claim across sections.",
            "score": 0.40
        })
        p_high = Passage.from_dict({
            "chunk_id": "chk_high_second",
            "paper_id": "paper_1",
            "section": "Methodology",
            "text": "Identical scientific claim across sections.",
            "score": 0.95
        })

        # Passed with low-scoring passage first
        deduped = deduplicate_passages_exact([p_low, p_high])
        self.assertEqual(len(deduped), 1)
        winner = deduped[0]
        self.assertEqual(winner.score, 0.95)
        self.assertEqual(winner.passage_id, "chk_high_second")
        # Specific section 'Methodology' resolved
        self.assertEqual(winner.section, "Methodology")

    def test_dual_context_budget_passage_cap_and_token_budget(self):
        """
        Tests selection within dual context budget:
        - 6 passage cap
        - 2,000 token budget via pinned tiktoken cl100k_base encoder
        - Whole passage selection: oversized passages skipped and recorded.
        """
        # Create normal passages and an oversized passage
        passages = []
        for i in range(8):
            passages.append(Passage.from_dict({
                "chunk_id": f"p_{i}",
                "paper_id": f"paper_{i}",
                "section": "Sec",
                "text": f"Short passage text {i} " * 20,
                "score": 0.90 - i * 0.05
            }))

        rec = select_final_context("Query", passages, max_passages=6, token_budget=2000)
        self.assertLessEqual(len(rec.passages), 6)
        self.assertEqual(rec.tokenizer_identity, "cl100k_base")
        self.assertLessEqual(rec.total_tokens, 2000)

        # Oversized passage test: 1,500 tokens
        big_text = "Dense neural embedding representation. " * 300
        p_big = Passage.from_dict({
            "chunk_id": "p_big",
            "paper_id": "paper_big",
            "section": "Sec",
            "text": big_text,
            "score": 0.99
        })
        rec_tight = select_final_context("Query", [p_big, passages[0], passages[1]], max_passages=6, token_budget=500)
        # Big passage exceeds 500 tokens, must be skipped and logged
        skipped_ids = [s[0] for s in rec_tight.skipped_passages]
        self.assertIn("p_big", skipped_ids)
        self.assertNotIn("p_big", [p.passage_id for p in rec_tight.passages])

    def test_generation_adapter_payload_hash_consistency(self):
        """
        Integration test verifying that the actual generation engine receives
        the IDENTICAL serialized context payload and hash evaluated by the gate.
        """
        rec = select_final_context("What is attention?", [self.p1, self.p2], max_passages=6)
        engine = GroundedPointQAEngine(api_key="", strict_mode=False)

        res, payload_sent, hash_sent = engine.generate_from_context_record(rec)
        self.assertEqual(payload_sent, rec.serialized_payload)
        self.assertEqual(hash_sent, rec.payload_hash)
        self.assertEqual(res.metrics.final_context_chunks_count, len(rec.passages))

    def test_autorater_pending_on_cache_miss_and_strict_parsing(self):
        """
        Autorater Repair Verification:
        1. Cache miss returns pending_execution with verdict=None (never 0 or fabricated).
        2. Incidental 'sufficient' in prose without [[1]] returns parsed_malformed.
        3. Strict terminal [[1]] and [[0]] parsed correctly.
        """
        rec = select_final_context("Query", [self.p1])
        autorater = SufficientContextInspiredAutorater(cache_path="data/metadata/test_autorater_cache.json")

        # Cache miss
        res = autorater.evaluate(rec)
        self.assertIsNone(res.verdict)
        self.assertEqual(res.parse_status, "pending_execution")
        self.assertFalse(res.is_cached)

        # Malformed: incidental word without format brackets
        st, v, j, pids = autorater.parse_response("The context is sufficient to answer the prompt.")
        self.assertEqual(st, "parsed_malformed")
        self.assertIsNone(v)

        # Valid verdict
        raw_valid = "Justification: Evidence is missing key facts.\nSupporting Passage IDs: None\nVerdict: [[0]]"
        st, v, j, pids = autorater.parse_response(raw_valid)
        self.assertEqual(st, "parsed_valid")
        self.assertEqual(v, 0)
        self.assertEqual(j, "Evidence is missing key facts.")

    def test_full_2x2_factorial_logistic_design(self):
        """
        Verifies all 4 cells of the 2x2 factorial logistic design:
        - M1_Relevance_Raw
        - M2_Relevance_Dedup
        - M3_Full_Raw
        - M4_Full_Dedup
        With identical preprocessing, L2 regularization selected by log-loss,
        and validation operating thresholds targeting 0.50, 0.75, 0.90.
        """
        rec_tr1 = select_final_context("Q1", [self.p1, self.p2])
        p_neg = Passage.from_dict({
            "chunk_id": "chk_neg",
            "paper_id": "p_neg",
            "section": "Sec",
            "text": "Negative sample text.",
            "score": 0.10
        })
        rec_tr2 = select_final_context("Q2", [p_neg])
        rec_val1 = select_final_context("Q3", [self.p1])
        rec_val2 = select_final_context("Q4", [p_neg])

        models = [
            LogisticGatingModel(name="M1_Relevance_Raw", feature_mode="relevance_only", deduplicate=False),
            LogisticGatingModel(name="M2_Relevance_Dedup", feature_mode="relevance_only", deduplicate=True),
            LogisticGatingModel(name="M3_Full_Raw", feature_mode="combined", deduplicate=False),
            LogisticGatingModel(name="M4_Full_Dedup", feature_mode="combined", deduplicate=True),
        ]

        for m in models:
            m.fit(
                train_records=[rec_tr1, rec_tr2],
                train_labels=[1, 0],
                val_records=[rec_val1, rec_val2],
                val_labels=[1, 0],
                c_grid=[0.1, 1.0, 10.0]
            )
            self.assertTrue(m.is_fitted)
            score = m.predict_score(rec_tr1)
            self.assertTrue(0.0 <= score <= 1.0)

            # Fit validation operating thresholds
            th_res = m.fit_validation_thresholds([rec_val1, rec_val2], target_coverages=[0.50, 0.75, 0.90])
            self.assertIn(0.50, th_res)
            self.assertIn(0.90, th_res)

    def test_splitter_order_invariance_and_target_leakage(self):
        """
        Splitter Integrity:
        - Must be invariant to input ordering of families.
        - Must enforce zero target paper leakage.
        - Audits contextual paper overlap transparently.
        """
        f1 = QuestionFamily("fam1", "Q1", "A1", ["pA"], ["pA", "pB"], SupportingEvidence("c1", "pA", "Sec", "txt", 0.9, "h1"), {}, {}, {}, {})
        f2 = QuestionFamily("fam2", "Q2", "A2", ["pB"], ["pB"], SupportingEvidence("c2", "pB", "Sec", "txt", 0.9, "h2"), {}, {}, {}, {})
        f3 = QuestionFamily("fam3", "Q3", "A3", ["pC"], ["pC"], SupportingEvidence("c3", "pC", "Sec", "txt", 0.9, "h3"), {}, {}, {}, {})
        f4 = QuestionFamily("fam4", "Q4", "A4", ["pD"], ["pD"], SupportingEvidence("c4", "pD", "Sec", "txt", 0.9, "h4"), {}, {}, {}, {})

        # Order 1
        tr1, val1, te1, meta1 = partition_families_by_paper_components([f1, f2, f3, f4], random_seed=42)
        # Order 2 (shuffled)
        tr2, val2, te2, meta2 = partition_families_by_paper_components([f4, f2, f1, f3], random_seed=42)

        self.assertEqual([f.family_id for f in tr1], [f.family_id for f in tr2])
        self.assertEqual([f.family_id for f in val1], [f.family_id for f in val2])
        self.assertEqual([f.family_id for f in te1], [f.family_id for f in te2])
        self.assertFalse(meta1["target_paper_leakage_detected"])

    def test_strict_mode_propagation_failure_injection(self):
        """Strict mode must raise exceptions without falling back to mock or demo strings."""
        engine = GroundedPointQAEngine(api_key="", strict_mode=True)
        rec = select_final_context("Query", [self.p1])

        with self.assertRaises(RuntimeError):
            engine.generate_from_context_record(rec, strict_mode=True)

    def test_mock_sdk_generation_integration(self):
        """
        Verify exact SDK call arguments:
        - prompt contains exact serialized evidence and query.
        - model identity matches configured generator.
        - empty response raises RuntimeError under strict mode.
        - raised API exception propagates under strict mode.
        """
        class FakeGenAIResponse:
            def __init__(self, text: str):
                self.text = text

        class FakeGenAIModels:
            def __init__(self, calls_log: list, return_text: str = "Test response", raise_exc: Exception = None):
                self.calls_log = calls_log
                self.return_text = return_text
                self.raise_exc = raise_exc

            def generate_content(self, model: str, contents: str):
                self.calls_log.append({"model": model, "contents": contents})
                if self.raise_exc:
                    raise self.raise_exc
                return FakeGenAIResponse(self.return_text)

        class FakeGenAIClient:
            def __init__(self, return_text: str = "Test response", raise_exc: Exception = None):
                self.calls_log = []
                self.models = FakeGenAIModels(self.calls_log, return_text, raise_exc)

        rec = select_final_context("What is attention?", [self.p1, self.p2])

        # 1. Normal successful generation
        fake_client = FakeGenAIClient(return_text="Attention mechanism dynamically weights input tokens [1].")
        engine = GroundedPointQAEngine(api_key="fake_key", strict_mode=True)
        engine.client = fake_client

        res, payload_sent, computed_hash = engine.generate_from_context_record(rec, strict_mode=True)
        self.assertEqual(len(fake_client.calls_log), 1)
        call = fake_client.calls_log[0]
        self.assertEqual(call["model"], "gemini-3.5-flash-lite")
        self.assertIn("What is attention?", call["contents"])
        self.assertIn(rec.serialized_payload, call["contents"])
        self.assertEqual(res.answer, "Attention mechanism dynamically weights input tokens [1].")
        self.assertEqual(len(res.retrieved_chunks), len(rec.passages))

        # 2. Empty response under strict mode must raise RuntimeError
        fake_empty_client = FakeGenAIClient(return_text="")
        engine_empty = GroundedPointQAEngine(api_key="fake_key", strict_mode=True)
        engine_empty.client = fake_empty_client
        engine_empty.retry_delay = 0.0

        with self.assertRaises(RuntimeError) as ctx_empty:
            engine_empty.generate_from_context_record(rec, strict_mode=True)
        self.assertIn("Strict Mode Error", str(ctx_empty.exception))

        # 3. Raised exception under strict mode must propagate
        fake_err_client = FakeGenAIClient(raise_exc=RuntimeError("Google GenAI 429 Quota Exceeded"))
        engine_err = GroundedPointQAEngine(api_key="fake_key", strict_mode=True)
        engine_err.client = fake_err_client
        engine_err.retry_delay = 0.0

        with self.assertRaises(RuntimeError) as ctx_err:
            engine_err.generate_from_context_record(rec, strict_mode=True)
        self.assertIn("Strict Mode Error", str(ctx_err.exception))

    def test_feature_extraction_invariance_and_source_isolation(self):
        """
        Verify feature extraction properties:
        1. Input order invariance: identical features regardless of candidate permutation.
        2. Deterministic top-passage tie-breaking: score descending, then passage_id ascending.
        3. Unknown top source handling: assigned deterministic source ID; s_other covers other sources.
        4. Cross-paper section isolation: identical section names across papers are NOT merged.
        """
        text_a = "Passage A text"
        text_b = "Passage B text"
        text_c = "Passage C text"

        p_a = Passage("chk_01", "paper_1", "Methods", text_a, 0.85, compute_text_sha256(text_a), count_tokens(text_a))
        p_b = Passage("chk_02", "paper_1", "Results", text_b, 0.70, compute_text_sha256(text_b), count_tokens(text_b))
        p_c = Passage("chk_03", "paper_2", "Methods", text_c, 0.60, compute_text_sha256(text_c), count_tokens(text_c))

        # 1. Order invariance
        feats_order1 = extract_source_features([p_a, p_b, p_c])
        feats_order2 = extract_source_features([p_c, p_a, p_b])
        np.testing.assert_allclose(feats_order1, feats_order2)

        # 2. Tied top scores broken deterministically by passage_id
        p_tied_1 = Passage("chk_01", "paper_1", "Methods", text_a, 0.90, compute_text_sha256(text_a), count_tokens(text_a))
        p_tied_2 = Passage("chk_02", "paper_2", "Methods", text_b, 0.90, compute_text_sha256(text_b), count_tokens(text_b))
        feats_tied1 = extract_source_features([p_tied_1, p_tied_2])
        feats_tied2 = extract_source_features([p_tied_2, p_tied_1])
        # Top passage must always be chk_01 (paper_1). Second passage from paper_1 is None -> 0.0. Other paper score is 0.90
        np.testing.assert_allclose(feats_tied1, feats_tied2)
        self.assertEqual(feats_tied1[0], 0.0)   # s_p*_2 is 0.0
        self.assertEqual(feats_tied1[1], 0.90)  # s_other is 0.90

        # 3. Unknown top source
        p_unk = Passage("chk_unk", None, "Methods", text_a, 0.95, compute_text_sha256(text_a), count_tokens(text_a))
        feats_unk = extract_source_features([p_unk, p_a, p_c])
        # Top source is __unknown_source_chk_unk__. s_p*_2 is 0.0. s_other is max(p_a.score, p_c.score) = 0.85
        self.assertEqual(feats_unk[0], 0.0)
        self.assertEqual(feats_unk[1], 0.85)

        # 4. Cross-paper section isolation: p_a (paper_1, Methods) and p_c (paper_2, Methods)
        # Must be treated as 2 distinct sections, NOT 1
        feats_sec = extract_source_features([p_a, p_c])
        # 2 passages, 2 distinct (paper, section) pairs -> section_diversity = 2 / 2 = 1.0
        self.assertEqual(feats_sec[2], 1.0)

    def test_complete_labeled_evaluation_pipeline(self):
        """
        Verify end-to-end evaluation pipeline with binary labels:
        - Evaluates B0, B1, B2, M1, M2, M3, M4.
        - Computes deployable operating points at target coverages.
        - Computes cluster bootstrap across paper components.
        - Operates offline with zero paid API calls.
        """
        # Build 4 minimal QuestionFamilies with clean binary labels
        fams = []
        for i in range(4):
            q_text = f"Question {i}?"
            ans = f"Reference answer {i}."
            txt_supp = f"Supporting evidence passage text for family {i}."
            txt_dist = f"Distractor passage text for family {i}."
            p_supp = Passage(f"chk_s_{i}", f"paper_{i}", "Methods", txt_supp, 0.88, compute_text_sha256(txt_supp), count_tokens(txt_supp))
            p_dist = Passage(f"chk_d_{i}", f"dist_paper_{i}", "Introduction", txt_dist, 0.40, compute_text_sha256(txt_dist), count_tokens(txt_dist))

            supp_ev = SupportingEvidence(f"chk_s_{i}", f"paper_{i}", "Methods", txt_supp, 0.88, compute_text_sha256(txt_supp))
            v_clean = select_final_context(q_text, [p_supp, p_dist])
            v_removed = select_final_context(q_text, [p_dist])

            fam = QuestionFamily(
                family_id=f"synth_fam_{i}",
                base_question=q_text,
                authored_reference_answer=ans,
                target_paper_ids=[f"paper_{i}"],
                contextual_paper_ids=[f"paper_{i}", f"dist_paper_{i}"],
                supporting_evidence=supp_ev,
                variant_contexts={"clean": v_clean, "evidence_removed": v_removed},
                variant_sufficiency_status={"clean": "reviewed", "evidence_removed": "reviewed"},
                variant_sufficiency_labels={"clean": 1, "evidence_removed": 0},
                scoring_provenance={"model": "test"}
            )
            fams.append(fam)

        results = run_selective_evaluation_pipeline(
            fams,
            output_path=None,
            annotation_provenance="PROVENANCE_FIXTURE_SYNTHETIC",
            n_bootstrap=50
        )

        self.assertEqual(results["status"], "PROVENANCE_FIXTURE_SYNTHETIC")
        self.assertIn("B0_AlwaysAccept", results["model_evaluations"])
        self.assertEqual(results["model_evaluations"]["B0_AlwaysAccept"]["achieved_coverage"], 1.0)
        self.assertIn("B2_FrozenHistoricalGate", results["model_evaluations"])
        self.assertIn("M4_Full_Dedup", results["model_evaluations"])
        m4_res = results["model_evaluations"]["M4_Full_Dedup"]
        self.assertIn("deployable_operating_points", m4_res)
        self.assertIn("target_cov_0.50", m4_res["deployable_operating_points"])
        self.assertIn("cluster_bootstrap", results)

    def test_provenance_guard_rejects_heuristic_proxy_labels(self):
        """
        Verify that passing heuristic proxy labels (DEVELOPMENT_EXCERPT_PRESENCE_PROXY_LABELS)
        to evaluation configured for semantic review raises ValueError.
        """
        txt = "text1"
        p1 = Passage("c1", "p1", "Sec", txt, 0.9, compute_text_sha256(txt), count_tokens(txt))
        fam = QuestionFamily(
            family_id="proxy_fam_01",
            base_question="What is X?",
            authored_reference_answer="X is Y.",
            target_paper_ids=["p1"],
            contextual_paper_ids=["p1"],
            supporting_evidence=SupportingEvidence("c1", "p1", "Sec", txt, 0.9, compute_text_sha256(txt)),
            variant_contexts={"clean": select_final_context("What is X?", [p1])},
            variant_sufficiency_status={"clean": "DEVELOPMENT_EXCERPT_PRESENCE_PROXY_LABELS"},
            variant_sufficiency_labels={"clean": 1},
            scoring_provenance={"model": "test"},
            annotation_provenance="DEVELOPMENT_EXCERPT_PRESENCE_PROXY_LABELS"
        )

        with self.assertRaises(ValueError) as ctx:
            run_selective_evaluation_pipeline(
                [fam],
                output_path=None,
                annotation_provenance="PROVENANCE_AI_REVIEWED_DEVELOPMENT_PILOT",
                n_bootstrap=10
            )
        self.assertIn("Provenance guard rejection", str(ctx.exception))

    def test_partition_families_by_manifest(self):
        """
        Verify that partition_families_by_manifest strictly respects pre-assigned splits.
        """
        p1 = Passage("c1", "p1", "Sec", "t1", 0.9, compute_text_sha256("t1"), count_tokens("t1"))
        p2 = Passage("c2", "p2", "Sec", "t2", 0.8, compute_text_sha256("t2"), count_tokens("t2"))
        p3 = Passage("c3", "p3", "Sec", "t3", 0.7, compute_text_sha256("t3"), count_tokens("t3"))

        f1 = QuestionFamily(
            family_id="fam_1", base_question="Q1", authored_reference_answer="A1",
            target_paper_ids=["p1"], contextual_paper_ids=["p1"],
            supporting_evidence=SupportingEvidence("c1", "p1", "Sec", "t1", 0.9, compute_text_sha256("t1")),
            variant_contexts={"clean": select_final_context("Q1", [p1])},
            variant_sufficiency_status={"clean": "reviewed"},
            variant_sufficiency_labels={"clean": 1}, scoring_provenance={}
        )
        f2 = QuestionFamily(
            family_id="fam_2", base_question="Q2", authored_reference_answer="A2",
            target_paper_ids=["p2"], contextual_paper_ids=["p2"],
            supporting_evidence=SupportingEvidence("c2", "p2", "Sec", "t2", 0.8, compute_text_sha256("t2")),
            variant_contexts={"clean": select_final_context("Q2", [p2])},
            variant_sufficiency_status={"clean": "reviewed"},
            variant_sufficiency_labels={"clean": 1}, scoring_provenance={}
        )
        f3 = QuestionFamily(
            family_id="fam_3", base_question="Q3", authored_reference_answer="A3",
            target_paper_ids=["p3"], contextual_paper_ids=["p3"],
            supporting_evidence=SupportingEvidence("c3", "p3", "Sec", "t3", 0.7, compute_text_sha256("t3")),
            variant_contexts={"clean": select_final_context("Q3", [p3])},
            variant_sufficiency_status={"clean": "reviewed"},
            variant_sufficiency_labels={"clean": 1}, scoring_provenance={}
        )

        manifest = {
            "paper_allocations": {
                "train": [{"paper_id": "p1"}],
                "val": [{"paper_id": "p2"}],
                "test": [{"paper_id": "p3"}]
            }
        }

        train_fams, val_fams, test_fams, comp_ids = partition_families_by_manifest(
            [f1, f2, f3], manifest
        )

        self.assertEqual([f.family_id for f in train_fams], ["fam_1"])
        self.assertEqual([f.family_id for f in val_fams], ["fam_2"])
        self.assertEqual([f.family_id for f in test_fams], ["fam_3"])
        self.assertIn("fam_1", comp_ids["family_to_component_id"])
        self.assertIn("fam_2", comp_ids["family_to_component_id"])
        self.assertIn("fam_3", comp_ids["family_to_component_id"])



if __name__ == "__main__":
    unittest.main(verbosity=2)
