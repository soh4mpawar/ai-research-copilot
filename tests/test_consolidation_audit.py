"""
Unit tests for the strict consolidation audit in scratch/consolidate_training_benchmark_v2.py.
Verifies that all failure modes (duplicate IDs, duplicate tuples, missing annotations,
mismatched hashes, out-of-split papers, bad quotes, semantic anomalies) strictly raise exceptions.
"""

import unittest
import copy
import json
import hashlib
import os
import sys

# Ensure backend and scratch are in path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from evaluation.consolidation_audits.consolidate_training_benchmark_v2 import run_consolidation_audit
from evaluation.consolidation_audits.consolidate_validation_benchmark_v2 import run_validation_consolidation_audit
from evaluation.consolidation_audits.consolidate_test_benchmark_v2_repaired import run_test_consolidation_audit


class TestConsolidationAuditFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load real valid data once for testing mutations
        cls.split_manifest = json.load(open(os.path.join(BASE_DIR, "data/metadata/full_benchmark_split_manifest.json"), encoding="utf-8"))
        
        q_data = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_training_questions_60families.json"), encoding="utf-8"))
        cls.questions = q_data if isinstance(q_data, list) else q_data["questions"]

        cls.contexts = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_training_contexts_240.json"), encoding="utf-8"))
        ann_d = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_training_semantic_annotations_240.json"), encoding="utf-8"))
        cls.annotations = ann_d["annotations"] if isinstance(ann_d, dict) else ann_d

        cls.batch01_packet = json.load(open(os.path.join(BASE_DIR, "data/metadata/audit_archive/benchmark_v2_batch01_shuffled_blind_packet.json"), encoding="utf-8"))
        cls.batch01_mapping = json.load(open(os.path.join(BASE_DIR, "data/metadata/audit_archive/benchmark_v2_batch01_shuffled_blind_mapping.json"), encoding="utf-8"))

    def test_baseline_passes(self):
        """Confirm valid dataset passes audit with zero exceptions."""
        result = run_consolidation_audit(
            contexts=copy.deepcopy(self.contexts),
            annotations=copy.deepcopy(self.annotations),
            questions=copy.deepcopy(self.questions),
            split_manifest=copy.deepcopy(self.split_manifest),
            batch01_packet=copy.deepcopy(self.batch01_packet),
            batch01_mapping=copy.deepcopy(self.batch01_mapping),
            chroma_lookup=None,
            dry_run=True
        )
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["total_records"], 240)
        self.assertEqual(result["total_families"], 60)

    def test_duplicate_context_id_fails(self):
        """Audit must fail when duplicate context ID is present."""
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[1]["opaque_context_id"] = bad_contexts[0]["opaque_context_id"]
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("Duplicate context IDs detected", str(ctx.exception))

    def test_duplicate_family_condition_tuple_fails(self):
        """Audit must fail when duplicate (family_id, variant) tuple is present."""
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[1]["variant"] = bad_contexts[0]["variant"]  # Now family has duplicate condition
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("Duplicate (family_id, condition) tuples detected", str(ctx.exception))

    def test_duplicate_annotation_id_fails(self):
        """Audit must fail when duplicate annotation ID is present."""
        bad_ann = copy.deepcopy(self.annotations)
        bad_ann[1]["opaque_context_id"] = bad_ann[0]["opaque_context_id"]
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("Duplicate annotation IDs detected", str(ctx.exception))

    def test_missing_annotation_fails(self):
        """Audit must fail when an annotation record is missing."""
        bad_ann = copy.deepcopy(self.annotations)[:-1]
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("Expected 240 annotations", str(ctx.exception))

    def test_unexpected_annotation_fails(self):
        """Audit must fail when an annotation ID is not in context set."""
        bad_ann = copy.deepcopy(self.annotations)
        bad_ann[-1]["opaque_context_id"] = "ctx_tr_9999"
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("Mismatched context and annotation sets", str(ctx.exception))

    def test_mismatched_question_hash_fails(self):
        """Audit must fail when annotation question hash differs from authoritative question hash."""
        bad_ann = copy.deepcopy(self.annotations)
        # Modify Batch 02 annotation (index 50)
        bad_ann[50]["question_hash"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("question_hash mismatch", str(ctx.exception))

    def test_mismatched_payload_hash_fails(self):
        """Audit must fail when annotation payload hash differs from authoritative payload hash."""
        bad_ann = copy.deepcopy(self.annotations)
        bad_ann[50]["payload_hash"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("payload_hash mismatch", str(ctx.exception))

    def test_out_of_split_paper_fails(self):
        """Audit must fail when a passage references a paper outside the train split."""
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[50]["passages"][0]["paper_id"] = "9999.99999"
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("not in train paper allocations", str(ctx.exception))

    def test_bad_quotation_text_fails(self):
        """Audit must fail when a supporting quotation is not found in the passage."""
        bad_ann = copy.deepcopy(self.annotations)
        # Find a sufficient annotation in Batch 02
        for a in bad_ann[48:]:
            if a["judgment"] == "sufficient" and a.get("supporting_quotations"):
                a["supporting_quotations"][0]["quote"] = "THIS STRING DEFINITELY DOES NOT EXIST IN PASSAGE 12345"
                break
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("not found in passage", str(ctx.exception))

    def test_bad_quotation_passage_index_fails(self):
        """Audit must fail when a supporting quotation references a non-existent passage index."""
        bad_ann = copy.deepcopy(self.annotations)
        for a in bad_ann[48:]:
            if a["judgment"] == "sufficient" and a.get("supporting_quotations"):
                a["supporting_quotations"][0]["passage_index"] = 99
                break
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("passage_index 99 out of range", str(ctx.exception))

    def test_insufficient_missing_info_empty_fails(self):
        """Audit must fail when an insufficient record has empty missing_information."""
        bad_ann = copy.deepcopy(self.annotations)
        for a in bad_ann[48:]:
            if a["judgment"] == "insufficient":
                a["missing_information"] = ""
                break
        with self.assertRaises(ValueError) as ctx:
            run_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions),
                self.split_manifest, self.batch01_packet, self.batch01_mapping
            )
        self.assertIn("marked insufficient but has empty missing_information", str(ctx.exception))



class TestValidationConsolidationAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_manifest = json.load(open(os.path.join(BASE_DIR, "data/metadata/full_benchmark_split_manifest.json"), encoding="utf-8"))
        cls.questions = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_validation_questions_24families.json"), encoding="utf-8"))
        cls.contexts = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_validation_contexts_96.json"), encoding="utf-8"))
        ann_data = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_validation_semantic_annotations_96.json"), encoding="utf-8"))
        cls.annotations = ann_data["annotations"] if isinstance(ann_data, dict) else ann_data

    def test_valid_validation_consolidation_passes(self):
        """The frozen validation dataset must pass all strict audit checks."""
        report = run_validation_consolidation_audit(
            contexts=copy.deepcopy(self.contexts),
            annotations=copy.deepcopy(self.annotations),
            questions=copy.deepcopy(self.questions),
            split_manifest=self.split_manifest,
            dry_run=True
        )
        self.assertEqual(report["audit_status"], "PASSED")
        self.assertEqual(report["total_contexts"], 96)
        self.assertEqual(report["total_families"], 24)
        self.assertEqual(report["judgment_counts"]["sufficient"], 47)
        self.assertEqual(report["judgment_counts"]["insufficient"], 49)

    def test_val_duplicate_context_id_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[1]["opaque_context_id"] = bad_contexts[0]["opaque_context_id"]
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Duplicate context IDs", str(ctx.exception))

    def test_val_duplicate_tuple_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[1]["family_id"] = bad_contexts[0]["family_id"]
        bad_contexts[1]["variant"] = bad_contexts[0]["variant"]
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Duplicate (family_id, condition) tuples", str(ctx.exception))

    def test_val_mismatched_annotation_id_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        bad_ann[0]["opaque_context_id"] = "ctx_val_9999"
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Mismatched context and annotation sets", str(ctx.exception))

    def test_val_mismatched_question_hash_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        bad_ann[0]["question_hash"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Question hash mismatch", str(ctx.exception))

    def test_val_mismatched_payload_hash_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[0]["payload_hash"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Payload hash mismatch", str(ctx.exception))

    def test_val_out_of_split_paper_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        # Introduce a training paper into validation context
        bad_contexts[0]["passages"][0]["paper_id"] = "2608.11829"  # Training paper
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("not in validation pool", str(ctx.exception))

    def test_val_bad_quotation_text_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        for a in bad_ann:
            if a.get("judgment") == "sufficient" and a.get("supporting_quotations"):
                a["supporting_quotations"][0]["quote"] = "THIS STRING DOES NOT EXIST"
                break
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("not found in passage", str(ctx.exception))

    def test_val_empty_missing_info_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        for a in bad_ann:
            if a.get("judgment") == "insufficient":
                a["missing_information"] = ""
                break
        with self.assertRaises(ValueError) as ctx:
            run_validation_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("empty missing information", str(ctx.exception))


class TestTestConsolidationAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_manifest = json.load(open(os.path.join(BASE_DIR, "data/metadata/full_benchmark_split_manifest.json"), encoding="utf-8"))
        cls.questions = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_test_questions_36families.json"), encoding="utf-8"))
        cls.contexts = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_test_contexts_144.json"), encoding="utf-8"))
        ann_data = json.load(open(os.path.join(BASE_DIR, "data/metadata/benchmark_v2_test_semantic_annotations_144.json"), encoding="utf-8"))
        cls.annotations = ann_data["annotations"] if isinstance(ann_data, dict) else ann_data

    def test_valid_test_consolidation_passes(self):
        """The frozen test dataset must pass all strict audit checks."""
        report = run_test_consolidation_audit(
            contexts=copy.deepcopy(self.contexts),
            annotations=copy.deepcopy(self.annotations),
            questions=copy.deepcopy(self.questions),
            split_manifest=self.split_manifest,
            dry_run=True
        )
        self.assertEqual(report["audit_status"], "PASSED")
        self.assertEqual(report["total_contexts"], 144)
        self.assertEqual(report["total_families"], 36)
        self.assertEqual(report["judgment_counts"]["sufficient"], 69)
        self.assertEqual(report["judgment_counts"]["insufficient"], 75)

    def test_test_duplicate_context_id_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[1]["opaque_context_id"] = bad_contexts[0]["opaque_context_id"]
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Duplicate context IDs", str(ctx.exception))

    def test_test_duplicate_tuple_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[1]["family_id"] = bad_contexts[0]["family_id"]
        bad_contexts[1]["variant"] = bad_contexts[0]["variant"]
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Duplicate (family_id, condition) tuples", str(ctx.exception))

    def test_test_mismatched_annotation_id_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        bad_ann[0]["opaque_context_id"] = "ctx_te_9999"
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Mismatched context and annotation sets", str(ctx.exception))

    def test_test_mismatched_question_hash_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        bad_ann[0]["question_hash"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Question hash mismatch", str(ctx.exception))

    def test_test_mismatched_payload_hash_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        bad_contexts[0]["payload_hash"] = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("Payload hash mismatch", str(ctx.exception))

    def test_test_out_of_split_paper_fails(self):
        bad_contexts = copy.deepcopy(self.contexts)
        # Introduce a training paper into test context
        bad_contexts[0]["passages"][0]["paper_id"] = "2608.11829"  # Training paper
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                bad_contexts, copy.deepcopy(self.annotations), copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("not in test pool", str(ctx.exception))

    def test_test_bad_quotation_text_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        for a in bad_ann:
            if a.get("judgment") == "sufficient" and a.get("verbatim_quotes"):
                a["verbatim_quotes"][0] = "THIS STRING DOES NOT EXIST"
                break
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("not found in passage payload", str(ctx.exception))

    def test_test_empty_missing_info_fails(self):
        bad_ann = copy.deepcopy(self.annotations)
        for a in bad_ann:
            if a.get("judgment") == "insufficient":
                a["missing_information"] = ""
                break
        with self.assertRaises(ValueError) as ctx:
            run_test_consolidation_audit(
                copy.deepcopy(self.contexts), bad_ann, copy.deepcopy(self.questions), self.split_manifest
            )
        self.assertIn("empty missing_information", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
