"""
Authoritative consolidation and strict verification of Test Benchmark v2 semantic annotations.

Loads:
1. Authored semantic reviews (Batch 01, Batch 02, Batch 03: 144 records total)
2. Blind shuffled mapping: benchmark_v2_test_shuffled_blind_mapping.json
3. Verified test contexts: benchmark_v2_test_contexts_144.json
4. Verified test specs: benchmark_v2_test_questions_36families.json
5. Archived proxy annotations: for changed vs unchanged judgment comparison

Performs:
- Full join on review_id with 4-field verification (family_id, variant, question_hash, payload_hash)
- Verbatim quote verification against formatted context payload
- Strict pre-checks: non-empty answers, no missing fields, sequence continuity
- Multiset & positional single-replacement audit across all 36 families
- ChromaDB chunk text fidelity verification
- Saves authoritative benchmark_v2_test_semantic_annotations_144.json and benchmark_v2_test_hash_manifest.json
"""

import json
import hashlib
import sys
import os
from collections import Counter
from typing import Dict, List, Any

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from backend.selective.context import Passage, format_context_payload, count_tokens
from backend.ingestion.vector_store import VectorStore
from typing import Optional


def run_test_consolidation_audit(
    contexts: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    split_manifest: Dict[str, Any],
    chroma_lookup: Optional[Dict[str, str]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    test_alloc = split_manifest["paper_allocations"]["test"]
    test_paper_ids = set(p["paper_id"] if isinstance(p, dict) else p for p in test_alloc)
    if len(test_paper_ids) != 48:
        raise ValueError(f"Expected 48 test papers, got {len(test_paper_ids)}")

    train_alloc = split_manifest["paper_allocations"]["train"]
    train_paper_ids = set(p["paper_id"] if isinstance(p, dict) else p for p in train_alloc)
    val_alloc = split_manifest["paper_allocations"]["val"]
    val_paper_ids = set(p["paper_id"] if isinstance(p, dict) else p for p in val_alloc)

    if len(test_paper_ids & train_paper_ids) != 0:
        raise ValueError("Test overlaps with training!")
    if len(test_paper_ids & val_paper_ids) != 0:
        raise ValueError("Test overlaps with validation!")

    if len(contexts) != 144:
        raise ValueError(f"Expected 144 contexts, got {len(contexts)}")
    if len(annotations) != 144:
        raise ValueError(f"Expected 144 annotations, got {len(annotations)}")

    context_ids = [c.get("opaque_context_id") for c in contexts]
    id_counts = Counter(context_ids)
    dup_ids = [cid for cid, count in id_counts.items() if count > 1]
    if dup_ids:
        raise ValueError(f"Duplicate context IDs found in contexts: {dup_ids}")

    tuples = [(c.get("family_id"), c.get("variant")) for c in contexts]
    tuple_counts = Counter(tuples)
    dup_tuples = [t for t, count in tuple_counts.items() if count > 1]
    if dup_tuples:
        raise ValueError(f"Duplicate (family_id, condition) tuples found in contexts: {dup_tuples}")

    ann_ids = [a.get("opaque_context_id") for a in annotations]
    ann_id_counts = Counter(ann_ids)
    dup_ann_ids = [aid for aid, count in ann_id_counts.items() if count > 1]
    if dup_ann_ids:
        raise ValueError(f"Duplicate context IDs found in annotations: {dup_ann_ids}")

    if set(context_ids) != set(ann_ids):
        raise ValueError("Mismatched context and annotation sets")

    for idx, cid in enumerate(context_ids, 1):
        expected_cid = f"ctx_te_{idx:04d}"
        if cid != expected_cid:
            raise ValueError(f"Context ID sequence broken at index {idx}: expected {expected_cid}, got {cid}")

    q_map = {q["family_id"]: q for q in questions}
    if len(q_map) != 36:
        raise ValueError(f"Expected 36 question families, got {len(q_map)}")

    ctx_map = {c["opaque_context_id"]: c for c in contexts}
    ann_map = {a["opaque_context_id"]: a for a in annotations}

    for cid in context_ids:
        c = ctx_map[cid]
        a = ann_map[cid]

        fid = c.get("family_id")
        if fid not in q_map:
            raise ValueError(f"Family {fid} not in question specifications")

        if a.get("family_id") != fid:
            raise ValueError(f"Annotation family_id mismatch for {cid}: {a.get('family_id')} vs {fid}")
        if a.get("variant") != c.get("variant"):
            raise ValueError(f"Annotation variant mismatch for {cid}: {a.get('variant')} vs {c.get('variant')}")

        calc_q_hash = hashlib.sha256(c["question"].encode("utf-8")).hexdigest()
        if a.get("question_hash") != calc_q_hash:
            raise ValueError(f"Question hash mismatch for {cid}")

        # Passages check
        passages = c.get("passages", [])
        if c.get("variant") == "natural_retrieval":
            if len(passages) > 6:
                raise ValueError(f"Natural retrieval context {cid} exceeds 6 passages: {len(passages)}")
        else:
            if len(passages) > 4:
                raise ValueError(f"Controlled context {cid} exceeds 4 passages: {len(passages)}")

        if c.get("total_tokens", 0) > 2000:
            raise ValueError(f"Context {cid} exceeds 2000 tokens: {c.get('total_tokens')}")

        for p in passages:
            pid = p.get("paper_id")
            if pid not in test_paper_ids:
                raise ValueError(f"Paper {pid} in context {cid} not in test pool")
            sec = p.get("section", "").lower()
            if "reference" in sec or sec == "ref" or "bibliography" in sec:
                raise ValueError(f"Boilerplate/reference section found: {sec} in context {cid}")
            score = p.get("rerank_score")
            if score is None or not np.isfinite(score):
                raise ValueError(f"Non-finite score in {cid} chunk {p.get('chunk_id')}")
            if chroma_lookup:
                base_id = p["chunk_id"].replace("_dup", "")
                if base_id in chroma_lookup:
                    if p["text"] != chroma_lookup[base_id]:
                        raise ValueError(f"ChromaDB text mismatch for {base_id}")

        passages_obj = [Passage.from_dict(p) for p in c["passages"]]
        calc_payload = format_context_payload(passages_obj)
        calc_p_hash = hashlib.sha256(calc_payload.encode("utf-8")).hexdigest()
        if a.get("payload_hash") != calc_p_hash:
            raise ValueError(f"Payload hash mismatch for {cid}")
        if c.get("payload_hash") != calc_p_hash:
            raise ValueError(f"Payload hash mismatch for {cid}")

        # Quotation / semantic verification
        j = a.get("judgment")
        if j == "sufficient":
            if not a.get("derivable_answer"):
                raise ValueError(f"Record {cid} marked sufficient but has empty derivable_answer")
            quotes = a.get("verbatim_quotes", [])
            if not quotes:
                raise ValueError(f"Record {cid} marked sufficient but has empty verbatim_quotes")
            for q in quotes:
                if q not in calc_payload:
                    raise ValueError(f"Quote in {cid} not found in passage payload: {repr(q[:60])}")
        elif j == "insufficient":
            if not a.get("missing_information"):
                raise ValueError(f"Record {cid} marked insufficient but has empty missing_information")
        elif j == "ambiguous":
            pass
        else:
            raise ValueError(f"Unknown judgment {j} in {cid}")

        if not a.get("annotation_rationale"):
            raise ValueError(f"Record {cid} missing annotation_rationale")

    # Positional single-replacement audit
    contexts_by_family = {}
    for c in contexts:
        contexts_by_family.setdefault(c["family_id"], {})[c["variant"]] = c

    for fid, cond_map in contexts_by_family.items():
        c_supp = cond_map["supported_candidate"]
        c_rel = cond_map["relevant_replacement"]
        c_dup = cond_map["duplicate_replacement"]

        p_supp = c_supp["passages"]
        p_rel = c_rel["passages"]
        p_dup = c_dup["passages"]

        diff_supp_rel = [i for i in range(4) if p_supp[i]["chunk_id"] != p_rel[i]["chunk_id"]]
        diff_supp_dup = [i for i in range(4) if p_supp[i]["chunk_id"] != p_dup[i]["chunk_id"]]

        if len(diff_supp_rel) != 1:
            raise ValueError(f"Family {fid} rel differences: {len(diff_supp_rel)} != 1")
        if len(diff_supp_dup) != 1:
            raise ValueError(f"Family {fid} dup differences: {len(diff_supp_dup)} != 1")

        rep_pos = diff_supp_rel[0]
        if diff_supp_dup[0] != rep_pos:
            raise ValueError(f"Family {fid} replacement position mismatch: rel={rep_pos} vs dup={diff_supp_dup[0]}")

        for i in range(4):
            if i != rep_pos:
                for fld in ["chunk_id", "text", "paper_id", "section", "rerank_score"]:
                    if p_supp[i][fld] != p_rel[i][fld]:
                        raise ValueError(f"Field {fld} mismatch at pos {i} in {fid}")
                    if p_supp[i][fld] != p_dup[i][fld]:
                        raise ValueError(f"Field {fld} mismatch at pos {i} in {fid}")

        dup_passage = p_dup[rep_pos]
        remaining_passages = [p_dup[i] for i in range(4) if i != rep_pos]
        matching_rem = [p for p in remaining_passages if p["text"] == dup_passage["text"]]
        if not matching_rem:
            raise ValueError(f"Duplicate passage in {fid} does not match any remaining passage")
        if dup_passage["rerank_score"] != matching_rem[0]["rerank_score"]:
            raise ValueError(f"Score mismatch on duplicate in {fid}")

    return {
        "audit_status": "PASSED",
        "total_families": len(q_map),
        "total_contexts": len(contexts),
        "total_annotations": len(annotations),
        "judgment_counts": dict(Counter(a["judgment"] for a in annotations)),
        "by_condition": {cond: dict(Counter(a["judgment"] for a in annotations if a["variant"] == cond)) for cond in ["supported_candidate", "relevant_replacement", "duplicate_replacement", "natural_retrieval"]}
    }


def run_repaired_test_consolidation():
    print("=== Starting Repaired Test Benchmark v2 Consolidation & Audit ===")

    # 1. Load Split Manifest and verify test paper pool (48 papers)
    manifest_path = "data/metadata/full_benchmark_split_manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    test_alloc = manifest["paper_allocations"]["test"]
    test_paper_ids = set(p["paper_id"] if isinstance(p, dict) else p for p in test_alloc)
    assert len(test_paper_ids) == 48, f"Expected 48 test papers, got {len(test_paper_ids)}"
    print(f"Loaded {len(test_paper_ids)} test papers from split manifest.")

    # 2. Load Questions Specs & Contexts
    specs_file = "data/metadata/benchmark_v2_test_questions_36families.json"
    contexts_file = "data/metadata/benchmark_v2_test_contexts_144.json"
    all_specs = json.load(open(specs_file, encoding="utf-8"))
    all_contexts = json.load(open(contexts_file, encoding="utf-8"))
    assert len(all_specs) == 36, f"Expected 36 specs, got {len(all_specs)}"
    assert len(all_contexts) == 144, f"Expected 144 contexts, got {len(all_contexts)}"

    # 3. Load Shuffled Blind Mapping
    mapping_file = "data/metadata/benchmark_v2_test_shuffled_blind_mapping.json"
    mapping = json.load(open(mapping_file, encoding="utf-8"))
    assert len(mapping) == 144, f"Expected 144 mapping records, got {len(mapping)}"

    # 4. Load Authored Reviews (Batches 01, 02, 03)
    b1_reviews = json.load(open("scratch/benchmark_v2_test_batch01_semantic_reviews.json", encoding="utf-8"))
    b2_reviews = json.load(open("scratch/benchmark_v2_test_batch02_semantic_reviews.json", encoding="utf-8"))
    b3_reviews = json.load(open("scratch/benchmark_v2_test_batch03_semantic_reviews.json", encoding="utf-8"))
    all_reviews = b1_reviews + b2_reviews + b3_reviews
    assert len(all_reviews) == 144, f"Expected 144 reviews, got {len(all_reviews)}"
    review_ids = [r["review_id"] for r in all_reviews]
    assert len(set(review_ids)) == 144, "Duplicate review IDs found!"

    # 5. Load Archived Proxy Annotations for Comparative Auditing
    archived_proxy_path = "data/archived_superseded/test_annotations_proxy_condition_derived/benchmark_v2_test_semantic_annotations_144_proxy.json"
    archived_proxy = json.load(open(archived_proxy_path, encoding="utf-8"))
    proxy_by_cid = {r["opaque_context_id"]: r for r in archived_proxy}
    assert len(proxy_by_cid) == 144, "Proxy annotations count mismatch!"

    # 6. Map Reviews to Annotations with 4-Field Strict Match
    ctx_by_cid = {c["opaque_context_id"]: c for c in all_contexts}
    built_annotations = []
    
    for r in all_reviews:
        rid = r["review_id"]
        map_info = mapping[rid]
        cid = map_info["opaque_context_id"]
        ctx = ctx_by_cid[cid]

        fid = map_info["family_id"]
        variant = map_info["variant"]
        q_hash = map_info["question_hash"]
        p_hash = map_info["payload_hash"]

        # Strict 4-field check against context
        assert ctx["family_id"] == fid, f"Family mismatch in {rid}: {ctx['family_id']} vs {fid}"
        assert ctx["variant"] == variant, f"Variant mismatch in {rid}: {ctx['variant']} vs {variant}"
        calc_q_hash = hashlib.sha256(ctx["question"].encode("utf-8")).hexdigest()
        assert calc_q_hash == q_hash, f"Question hash mismatch in {rid}: {calc_q_hash} vs {q_hash}"
        assert ctx["payload_hash"] == p_hash, f"Payload hash mismatch in {rid}: {ctx['payload_hash']} vs {p_hash}"

        # Context payload verification for quotes
        passages_obj = [Passage.from_dict(p) for p in ctx["passages"]]
        calc_payload = format_context_payload(passages_obj)
        calc_p_hash = hashlib.sha256(calc_payload.encode("utf-8")).hexdigest()
        assert calc_p_hash == p_hash, f"Context payload serialization hash mismatch in {cid}!"

        # Extract quotations
        if r["judgment"] == "sufficient":
            assert r["derivable_answer"], f"Missing derivable answer in {rid}"
            assert len(r["quotations"]) > 0, f"No quotations for sufficient {rid}"
            verbatim_quotes = []
            for q in r["quotations"]:
                q_text = q["quote"]
                # Verify quote is in the formatted payload
                if q_text not in calc_payload:
                    raise ValueError(f"Quote in {rid} ({cid}) not in payload:\n{repr(q_text[:80])}...")
                verbatim_quotes.append(q_text)
            missing_info = None
        elif r["judgment"] == "insufficient":
            assert r["missing_information"], f"Missing missing_info in {rid}"
            verbatim_quotes = []
            missing_info = r["missing_information"]
        elif r["judgment"] == "ambiguous":
            verbatim_quotes = [q["quote"] for q in r.get("quotations", [])]
            missing_info = r.get("missing_information")
        else:
            raise ValueError(f"Unknown judgment {r['judgment']} in {rid}")

        assert r["rationale"], f"Missing rationale in {rid}"

        ann_record = {
            "opaque_context_id": cid,
            "family_id": fid,
            "variant": variant,
            "question": ctx["question"],
            "question_hash": q_hash,
            "payload_hash": p_hash,
            "judgment": r["judgment"],
            "derivable_answer": r["derivable_answer"],
            "verbatim_quotes": verbatim_quotes,
            "missing_information": missing_info,
            "annotation_rationale": r["rationale"],
            "annotator_metadata": {
                "annotator": "assistant_reading_displayed_passages",
                "protocol": "benchmark_v2_blind_reading",
                "exposure_disclosure": "authoring_and_annotation_by_same_ai_assistant_with_prior_exposure",
                "review_id": rid
            }
        }
        built_annotations.append(ann_record)

    # 7. Sort by opaque_context_id (ctx_te_0001 through ctx_te_0144)
    built_annotations.sort(key=lambda x: int(x["opaque_context_id"].split("_")[-1]))
    assert len(built_annotations) == 144, f"Expected 144 built annotations, got {len(built_annotations)}"
    for idx, a in enumerate(built_annotations, 1):
        expected_cid = f"ctx_te_{idx:04d}"
        assert a["opaque_context_id"] == expected_cid, f"Ordering error: got {a['opaque_context_id']}, expected {expected_cid}"

    # 8. Compare with Archived Proxy Annotations
    changed_count = 0
    unchanged_count = 0
    changed_records = []
    
    for a in built_annotations:
        cid = a["opaque_context_id"]
        proxy_rec = proxy_by_cid[cid]
        new_j = a["judgment"]
        old_j = proxy_rec["judgment"]
        if new_j != old_j:
            changed_count += 1
            changed_records.append({
                "opaque_context_id": cid,
                "family_id": a["family_id"],
                "variant": a["variant"],
                "old_judgment": old_j,
                "new_judgment": new_j,
                "rationale": a["annotation_rationale"]
            })
        else:
            unchanged_count += 1

    print("\n=== Comparison with Archived Proxy Annotations ===")
    print(f"Total Records: {len(built_annotations)}")
    print(f"Unchanged Judgments: {unchanged_count} ({unchanged_count / 144 * 100:.1f}%)")
    print(f"Changed Judgments: {changed_count} ({changed_count / 144 * 100:.1f}%)")
    if changed_records:
        print("Details of changed judgments:")
        for cr in changed_records:
            print(f"  [{cr['opaque_context_id']}] {cr['family_id']} ({cr['variant']}): {cr['old_judgment']} -> {cr['new_judgment']}")

    # 9. Verify ChromaDB chunk text fidelity
    print("\nConnecting to ChromaDB to verify chunk text fidelity...")
    vs = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
    all_chunk_ids = set()
    for c in all_contexts:
        for p in c["passages"]:
            base_cid = p["chunk_id"].replace("_dup", "")
            all_chunk_ids.add(base_cid)
    
    chroma_res = vs.collection.get(ids=list(all_chunk_ids), include=["documents", "metadatas"])
    chroma_docs = dict(zip(chroma_res["ids"], chroma_res["documents"]))
    assert len(chroma_docs) == len(all_chunk_ids), "Missing chunks in ChromaDB!"

    for c in all_contexts:
        for p in c["passages"]:
            base_id = p["chunk_id"].replace("_dup", "")
            assert p["text"] == chroma_docs[base_id], f"Chroma text mismatch in {p['chunk_id']}"
            assert p["paper_id"] in test_paper_ids, f"Paper {p['paper_id']} not in test pool"
            assert np.isfinite(p["rerank_score"]), f"Non-finite score in {p['chunk_id']}"

    print("ChromaDB chunk fidelity verified.")

    # 10. Positional Single-Replacement Check Across All 36 Families
    print("Auditing positional single replacement across all 36 families...")
    contexts_by_family = {}
    for c in all_contexts:
        fid = c["family_id"]
        contexts_by_family.setdefault(fid, {})[c["variant"]] = c

    for fid, cond_map in contexts_by_family.items():
        c_supp = cond_map["supported_candidate"]
        c_rel = cond_map["relevant_replacement"]
        c_dup = cond_map["duplicate_replacement"]

        p_supp = c_supp["passages"]
        p_rel = c_rel["passages"]
        p_dup = c_dup["passages"]

        assert len(p_supp) == 4 and len(p_rel) == 4 and len(p_dup) == 4, f"Length error in {fid}"

        diff_supp_rel = [i for i in range(4) if p_supp[i]["chunk_id"] != p_rel[i]["chunk_id"]]
        diff_supp_dup = [i for i in range(4) if p_supp[i]["chunk_id"] != p_dup[i]["chunk_id"]]

        assert len(diff_supp_rel) == 1, f"Family {fid} rel differences: {len(diff_supp_rel)}"
        assert len(diff_supp_dup) == 1, f"Family {fid} dup differences: {len(diff_supp_dup)}"

        rep_pos = diff_supp_rel[0]
        assert diff_supp_dup[0] == rep_pos, f"Family {fid} replacement position mismatch"

        for i in range(4):
            if i != rep_pos:
                for fld in ["chunk_id", "text", "paper_id", "section", "rerank_score"]:
                    assert p_supp[i][fld] == p_rel[i][fld], f"Field {fld} mismatch at pos {i} in {fid}"
                    assert p_supp[i][fld] == p_dup[i][fld], f"Field {fld} mismatch at pos {i} in {fid}"

        dup_passage = p_dup[rep_pos]
        remaining_passages = [p_dup[i] for i in range(4) if i != rep_pos]
        matching_rem = [p for p in remaining_passages if p["text"] == dup_passage["text"]]
        assert len(matching_rem) > 0, f"Duplicate passage in {fid} does not match any remaining passage"
        assert dup_passage["rerank_score"] == matching_rem[0]["rerank_score"], f"Score mismatch on duplicate in {fid}"

    print("Positional single-replacement audit passed across all 36 families.")

    # 11. Compute Summary Counts and Condition Breakdown
    counts_by_cond = Counter(c["variant"] for c in all_contexts)
    counts_by_judgment = Counter(a["judgment"] for a in built_annotations)
    counts_by_type = Counter(s["question_type"] for s in all_specs)
    
    cond_judgment_breakdown = {}
    for a in built_annotations:
        v = a["variant"]
        j = a["judgment"]
        cond_judgment_breakdown.setdefault(v, Counter())[j] += 1

    print("\n=== Final Test Benchmark v2 Summary Counts ===")
    print(f"Total Families: {len(all_specs)}")
    print(f"Question Types: {dict(counts_by_type)}")
    print(f"Total Contexts: {len(all_contexts)}")
    print(f"Total Annotations: {len(built_annotations)}")
    print(f"Judgments Overall: {dict(counts_by_judgment)}")
    print(f"Ambiguous Cases: {counts_by_judgment.get('ambiguous', 0)}")
    print(f"Breakdown by Condition:")
    for cond, j_counts in cond_judgment_breakdown.items():
        print(f"  - {cond}: {dict(j_counts)}")

    # 12. Save Authoritative Consolidated Files
    out_annotations = "data/metadata/benchmark_v2_test_semantic_annotations_144.json"
    out_manifest = "data/metadata/benchmark_v2_test_hash_manifest.json"

    with open(out_annotations, "w", encoding="utf-8") as f:
        json.dump(built_annotations, f, indent=2, ensure_ascii=False)
    print(f"\nSaved authoritative annotations to {out_annotations}")

    manifest_records = []
    for ctx, ann in zip(all_contexts, built_annotations):
        assert ctx["opaque_context_id"] == ann["opaque_context_id"]
        manifest_records.append({
            "opaque_context_id": ctx["opaque_context_id"],
            "family_id": ctx["family_id"],
            "paper_id": ctx["paper_id"],
            "variant": ctx["variant"],
            "question_hash": ann["question_hash"],
            "payload_hash": ann["payload_hash"],
            "judgment": ann["judgment"],
            "total_tokens": ctx["total_tokens"],
            "passage_count": len(ctx["passages"]),
            "scoring_provenance": ctx["scoring_provenance"],
            "review_id": ann["annotator_metadata"]["review_id"]
        })

    manifest_data = {
        "benchmark_version": "v2",
        "split": "test",
        "status": "CONSOLIDATED_AND_FROZEN_PRIOR_TO_EVALUATION",
        "attribution_notes": "Test content was accessed during authorized test construction. Gating models and evaluation protocol were frozen prior to this rebuild. Authored and annotated by AI assistant with prior exposure; condition-derived proxy labels superseded.",
        "counts": {
            "families": len(all_specs),
            "contexts": len(all_contexts),
            "annotations": len(built_annotations),
            "by_condition": dict(counts_by_cond),
            "by_judgment": dict(counts_by_judgment),
            "by_question_type": dict(counts_by_type),
            "breakdown": {k: dict(v) for k, v in cond_judgment_breakdown.items()}
        },
        "records": manifest_records
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"Saved authoritative manifest to {out_manifest}")

    # Compute and display file SHA-256 hashes
    print("\n=== Consolidated File Hashes (SHA-256) ===")
    for path in [
        "data/metadata/benchmark_v2_test_questions_36families.json",
        "data/metadata/benchmark_v2_test_contexts_144.json",
        out_annotations,
        out_manifest
    ]:
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"  {path}: {h}")

    print("\nREPAIRED TEST CONSOLIDATION & AUDIT FULLY PASSED!")

if __name__ == "__main__":
    run_repaired_test_consolidation()
