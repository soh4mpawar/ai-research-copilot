"""
Dataset Machinery & Group-Split Engine for Selective Scientific Question Answering.

Design & Protocol:
1. Real Corpus Grounding:
   - Passages loaded directly through read-only calls to ChromaDB.
   - Real chunk IDs, paper IDs, exact text, content hashes, and section metadata.
   - Every supporting excerpt verified as an exact substring of the source chunk.
2. Real Scoring:
   - Cross-encoder reranker (BAAI/bge-reranker-base) scores computed directly. Fallbacks disabled.
3. True Evidence Removal:
   - Supporting chunk is omitted; context contains only real distractor chunks.
   - NO invented contradictory statements.
4. Separate Real Distractors:
   - Distractor condition uses two distinct real corpus chunks, never a second copy.
5. Pending Annotations:
   - Sufficiency labels remain explicitly 'pending' (verdict=None, status='pending').
   - Classifiers must NOT be trained on unverified pending labels.
6. Connected-Component Group Splitter:
   - Namespaced graph nodes: 'family:{id}', 'target_paper:{pid}', 'context_paper:{pid}'.
   - Fully reproducible: sorted node ordering, seeded deterministic partition, order-invariant.
   - Explicit reporting: target paper leakage, contextual paper overlap, exact-text overlap.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Tuple, Set, Optional, Sequence, Union
import json
import os
import random
import hashlib
import networkx as nx

from backend.selective.context import (
    Passage,
    FinalContextRecord,
    select_final_context,
    inject_duplicates_post_selection,
    compute_text_sha256
)


@dataclass
class SupportingEvidence:
    """Verified supporting evidence excerpt from real corpus chunk."""
    chunk_id: str
    paper_id: str
    section: str
    verified_excerpt: str
    cross_encoder_score: float
    content_hash: str


@dataclass
class QuestionFamily:
    """
    Independent question family containing a base scientific question, authored reference answer,
    supporting evidence passages, and 4 controlled experimental context variants.
    """
    family_id: str
    base_question: str
    authored_reference_answer: str
    target_paper_ids: List[str]
    contextual_paper_ids: List[str]
    supporting_evidence: SupportingEvidence
    # Controlled context variants: 'clean', 'duplicate', 'distractor', 'evidence_removed'
    variant_contexts: Dict[str, FinalContextRecord]
    # Sufficiency status: 'pending' (unless reviewed independently against full context)
    variant_sufficiency_status: Dict[str, str]
    # Reference sufficiency labels: None if pending, 1 (sufficient), 0 (insufficient)
    variant_sufficiency_labels: Dict[str, Optional[int]]
    scoring_provenance: Dict[str, Any]
    annotation_provenance: Dict[str, Any] = field(default_factory=dict)


def partition_families_by_paper_components(
    families: Sequence[QuestionFamily],
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
    random_seed: int = 42
) -> Tuple[List[QuestionFamily], List[QuestionFamily], List[QuestionFamily], Dict[str, Any]]:
    """
    Graph-based connected component splitter.
    Primary grouping is formed by all supporting/target papers for a question family.
    Contextual papers (including distractors) are tracked separately and audited for cross-split overlap.

    Guarantees:
    1. Input-order invariance: families are sorted by family_id before graph construction.
    2. Namespaced graph node IDs: 'family:{id}', 'target_paper:{pid}'.
    3. Reproducible component sorting and seeded partition.
    4. Comprehensive split audit: reports target paper leakage, contextual paper overlap,
       and exact-text overlap.
    """
    # 1. Enforce input order invariance
    sorted_fams = sorted(families, key=lambda f: f.family_id)

    # 2. Build primary grouping bipartite graph on target papers
    G = nx.Graph()
    fam_map = {}
    for fam in sorted_fams:
        f_node = f"family:{fam.family_id}"
        fam_map[f_node] = fam
        G.add_node(f_node, node_type="family")

        for pid in fam.target_paper_ids:
            p_node = f"target_paper:{pid}"
            G.add_node(p_node, node_type="target_paper")
            G.add_edge(f_node, p_node)

    # 3. Connected components with deterministic sorting
    raw_components = list(nx.connected_components(G))
    family_components: List[List[QuestionFamily]] = []

    for comp in raw_components:
        # Sort component nodes lexicographically
        comp_nodes = sorted(list(comp))
        fams_in_comp = [fam_map[n] for n in comp_nodes if n in fam_map]
        if fams_in_comp:
            family_components.append(fams_in_comp)

    # Sort components deterministically by size desc, then family_id of first member asc
    family_components.sort(key=lambda c: (-len(c), c[0].family_id))

    # Stable connected-component identifiers for cluster bootstrap and audits
    family_to_component_id: Dict[str, str] = {}
    for c_idx, comp in enumerate(family_components):
        target_pids = sorted({pid for f in comp for pid in f.target_paper_ids})
        pid_str = "_".join(target_pids) if target_pids else comp[0].family_id
        comp_id = f"comp_{c_idx:03d}_{pid_str}"
        for fam in comp:
            family_to_component_id[fam.family_id] = comp_id

    # 4. Deterministic seeded allocation
    rng = random.Random(random_seed)
    # Shuffle component order deterministically
    component_indices = list(range(len(family_components)))
    rng.shuffle(component_indices)

    train_families: List[QuestionFamily] = []
    val_families: List[QuestionFamily] = []
    test_families: List[QuestionFamily] = []

    total_fams = len(sorted_fams)
    target_train = max(1, int(round(total_fams * train_ratio)))
    target_val = max(1, int(round(total_fams * val_ratio)))

    for idx in component_indices:
        comp = family_components[idx]
        if len(train_families) + len(comp) <= target_train or not train_families:
            train_families.extend(comp)
        elif len(val_families) + len(comp) <= target_val or not val_families:
            val_families.extend(comp)
        else:
            test_families.extend(comp)

    # 5. Overlap and Leakage Audit
    train_target_papers = {pid for f in train_families for pid in f.target_paper_ids}
    val_target_papers = {pid for f in val_families for pid in f.target_paper_ids}
    test_target_papers = {pid for f in test_families for pid in f.target_paper_ids}

    train_context_papers = {pid for f in train_families for pid in f.contextual_paper_ids}
    val_context_papers = {pid for f in val_families for pid in f.contextual_paper_ids}
    test_context_papers = {pid for f in test_families for pid in f.contextual_paper_ids}

    target_leakage_tv = train_target_papers.intersection(val_target_papers)
    target_leakage_tt = train_target_papers.intersection(test_target_papers)
    target_leakage_vt = val_target_papers.intersection(test_target_papers)

    context_overlap_tv = train_context_papers.intersection(val_context_papers)
    context_overlap_tt = train_context_papers.intersection(test_context_papers)
    context_overlap_vt = val_context_papers.intersection(test_context_papers)

    # Exact passage text overlap
    def get_texts(fams):
        texts = set()
        for f in fams:
            for vrec in f.variant_contexts.values():
                for p in vrec.passages:
                    texts.add(p.content_hash)
        return texts

    train_texts = get_texts(train_families)
    val_texts = get_texts(val_families)
    test_texts = get_texts(test_families)

    text_overlap_tv = train_texts.intersection(val_texts)
    text_overlap_tt = train_texts.intersection(test_texts)
    text_overlap_vt = val_texts.intersection(test_texts)

    split_metadata = {
        "num_connected_components": len(family_components),
        "train_families_count": len(train_families),
        "val_families_count": len(val_families),
        "test_families_count": len(test_families),
        "target_paper_leakage_detected": bool(target_leakage_tv or target_leakage_tt or target_leakage_vt),
        "target_paper_leakage_details": {
            "train_val": list(target_leakage_tv),
            "train_test": list(target_leakage_tt),
            "val_test": list(target_leakage_vt)
        },
        "contextual_paper_overlap": {
            "train_val": list(context_overlap_tv),
            "train_test": list(context_overlap_tt),
            "val_test": list(context_overlap_vt)
        },
        "exact_text_overlap_count": {
            "train_val": len(text_overlap_tv),
            "train_test": len(text_overlap_tt),
            "val_test": len(text_overlap_vt)
        },
        "family_to_component_id": family_to_component_id
    }

    return train_families, val_families, test_families, split_metadata


def partition_families_by_manifest(
    families: Sequence[QuestionFamily],
    split_manifest: Union[Dict[str, Any], str]
) -> Tuple[List[QuestionFamily], List[QuestionFamily], List[QuestionFamily], Dict[str, Any]]:
    """
    Partition question families using an externally supplied, frozen split manifest.
    Guarantees that pre-computed assignments are strictly respected rather than recomputed.
    """
    if isinstance(split_manifest, str):
        with open(split_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    else:
        manifest_data = split_manifest

    def _extract_pids(split_entry):
        if not split_entry:
            return set()
        if isinstance(split_entry, dict):
            pids = split_entry.get("paper_ids", [])
        elif isinstance(split_entry, list):
            pids = split_entry
        else:
            return set()
        res = set()
        for item in pids:
            if isinstance(item, str):
                res.add(item)
            elif isinstance(item, dict) and "paper_id" in item:
                res.add(item["paper_id"])
        return res

    alloc = manifest_data.get("split_allocation") or manifest_data.get("paper_allocations") or {}
    train_paper_set = _extract_pids(alloc.get("train"))
    val_paper_set = _extract_pids(alloc.get("val"))
    test_paper_set = _extract_pids(alloc.get("test"))

    train_families = []
    val_families = []
    test_families = []
    family_to_component_id = {}

    for fam in families:
        primary_paper = fam.target_paper_ids[0] if fam.target_paper_ids else None
        if primary_paper in train_paper_set:
            train_families.append(fam)
            family_to_component_id[fam.family_id] = f"manifest_train_{primary_paper}"
        elif primary_paper in val_paper_set:
            val_families.append(fam)
            family_to_component_id[fam.family_id] = f"manifest_val_{primary_paper}"
        elif primary_paper in test_paper_set:
            test_families.append(fam)
            family_to_component_id[fam.family_id] = f"manifest_test_{primary_paper}"
        else:
            fam_splits = manifest_data.get("family_split_allocation", {})
            if fam.family_id in fam_splits.get("train", []):
                train_families.append(fam)
                family_to_component_id[fam.family_id] = f"manifest_train_{fam.family_id}"
            elif fam.family_id in fam_splits.get("val", []):
                val_families.append(fam)
                family_to_component_id[fam.family_id] = f"manifest_val_{fam.family_id}"
            elif fam.family_id in fam_splits.get("test", []):
                test_families.append(fam)
                family_to_component_id[fam.family_id] = f"manifest_test_{fam.family_id}"
            else:
                raise ValueError(f"QuestionFamily '{fam.family_id}' (paper '{primary_paper}') not found in supplied manifest!")

    split_metadata = {
        "manifest_loaded": True,
        "train_families_count": len(train_families),
        "val_families_count": len(val_families),
        "test_families_count": len(test_families),
        "family_to_component_id": family_to_component_id,
        "split_manifest_metadata": {
            k: v for k, v in manifest_data.items() if k not in ("split_allocation", "family_split_allocation")
        }
    }
    return train_families, val_families, test_families, split_metadata


def build_corpus_grounded_12_pilot() -> List[QuestionFamily]:
    """
    Build the replacement development pilot of 12 question families grounded in real ChromaDB chunks.
    Scored using actual CrossEncoderReranker (BAAI/bge-reranker-base) without fallbacks.
    Sufficiency labels remain 'pending'.
    """
    from backend.ingestion.vector_store import VectorStore
    from backend.reranking.reranker import CrossEncoderReranker

    vs = VectorStore()
    reranker = CrossEncoderReranker()
    if reranker.model is None:
        raise RuntimeError("Real CrossEncoder model failed to load. Fallbacks are disabled in research pilot.")

    # 12 real corpus specifications
    pilot_specs = [
        {
            "id": "real_pilot_01",
            "paper_id": "1706.03762",
            "support_cid": "1706.03762_sec_009",
            "distract_cid": "1706.03762_sec_005",
            "other_cid": "1512.03385_sec_002",
            "other_cid_2": "1409.1556_sec_002",
            "question": "Why is the scaling factor 1/sqrt(d_k) applied in Scaled Dot-Product Attention?",
            "authored_answer": "To prevent dot products from growing large in magnitude for large values of d_k, which pushes softmax into regions with extremely small gradients.",
            "excerpt": "dot products grow large in magnitude, pushing the soft max function into regions where it has extremely small gradients"
        },
        {
            "id": "real_pilot_02",
            "paper_id": "1512.03385",
            "support_cid": "1512.03385_sec_003",
            "distract_cid": "1512.03385_sec_002",
            "other_cid": "1409.1556_sec_003",
            "other_cid_2": "1706.03762_sec_004",
            "question": "What constructed solution indicates that a deeper residual network should produce no higher training error than its shallower counterpart?",
            "authored_answer": "The added layers are identity mapping, and the other layers are copied from the learned shallower model.",
            "excerpt": "the added layers are identity mapping, and the other layers are copied from the learned shallower model"
        },
        {
            "id": "real_pilot_03",
            "paper_id": "1810.04805",
            "support_cid": "1810.04805_sec_012",
            "distract_cid": "1810.04805_sec_002",
            "other_cid": "1706.03762_sec_003",
            "other_cid_2": "2005.11401_sec_002",
            "question": "What token replacement distribution is used when masking tokens in BERT MLM pre-training?",
            "authored_answer": "80% of the time replaced with [MASK], 10% of the time with a random token, and 10% of the time with the unchanged token.",
            "excerpt": "the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged"
        },
        {
            "id": "real_pilot_04",
            "paper_id": "2005.11401",
            "support_cid": "2005.11401_sec_007",
            "distract_cid": "2005.11401_sec_002",
            "other_cid": "2004.04906_sec_002",
            "other_cid_2": "1810.04805_sec_003",
            "question": "How do RAG models treat retrieved documents mathematically when generating output sequences?",
            "authored_answer": "It treats the retrieved document as a single latent variable that is marginalized to get the seq2seq probability via a top-K approximation.",
            "excerpt": "treats the retrieved document as a single latent variable that is marginalized to get the seq2seq probability"
        },
        {
            "id": "real_pilot_05",
            "paper_id": "2004.04906",
            "support_cid": "2004.04906_sec_038",
            "distract_cid": "2004.04906_sec_002",
            "other_cid": "1810.04805_sec_004",
            "other_cid_2": "2005.11401_sec_003",
            "question": "On the Natural Questions benchmark, how does DPR's retrieval accuracy compare to BM25?",
            "authored_answer": "DPR achieves top-20 passage retrieval accuracy of 78.1% compared to BM25 where the top scoring passage is frequently irrelevant.",
            "excerpt": "top scoring passage from BM25 is irrelevant, even though keywords such as England and Ireland appear multiple times. In comparison, DPR is able to return Top-1 Top-5 Top-20 Top-100 Gold 44.9 66.8 78.1"
        },
        {
            "id": "real_pilot_06",
            "paper_id": "2010.11929",
            "support_cid": "2010.11929_sec_003",
            "distract_cid": "2010.11929_sec_002",
            "other_cid": "2103.14030_sec_002",
            "other_cid_2": "1512.03385_sec_004",
            "question": "Why does Vision Transformer yield modest accuracies when trained on mid-sized datasets without strong regularization compared to ResNets?",
            "authored_answer": "Transformers lack inductive biases inherent to CNNs, such as translation equivariance and locality, and do not generalize well without massive pre-training.",
            "excerpt": "Transformers lack some of the inductive biases"
        },
        {
            "id": "real_pilot_07",
            "paper_id": "2103.14030",
            "support_cid": "2103.14030_sec_014",
            "distract_cid": "2103.14030_sec_002",
            "other_cid": "2010.11929_sec_004",
            "other_cid_2": "1409.1556_sec_004",
            "question": "How does shifted window partitioning in Swin Transformer introduce cross-window connections?",
            "authored_answer": "By alternating between regular window partitioning and shifted window partitioning in successive blocks while maintaining non-overlapping window computation.",
            "excerpt": "Shifted window partitioning in successive blocks The window-based self-attention module lacks connections across windows"
        },
        {
            "id": "real_pilot_08",
            "paper_id": "1409.1556",
            "support_cid": "1409.1556_sec_005",
            "distract_cid": "1409.1556_sec_002",
            "other_cid": "1512.03385_sec_005",
            "other_cid_2": "1703.06870_sec_003",
            "question": "What convolution stride is fixed in VGGNet to preserve spatial resolution after convolution?",
            "authored_answer": "The convolution stride is fixed to 1 pixel with 1 pixel spatial padding for 3x3 conv layers.",
            "excerpt": "convolution stride is"
        },
        {
            "id": "real_pilot_09",
            "paper_id": "1703.06870",
            "support_cid": "1703.06870_sec_011",
            "distract_cid": "1703.06870_sec_002",
            "other_cid": "1409.1556_sec_006",
            "other_cid_2": "1905.11946_sec_003",
            "question": "How does RoIPool quantize floating-number RoIs and why does it introduce spatial misalignment?",
            "authored_answer": "RoIPool quantizes floating-number RoIs to discrete feature map granularity and spatial bins, introducing misalignments between RoI and features.",
            "excerpt": "RoIPool"
        },
        {
            "id": "real_pilot_10",
            "paper_id": "1905.11946",
            "support_cid": "1905.11946_sec_011",
            "distract_cid": "1905.11946_sec_002",
            "other_cid": "1512.03385_sec_006",
            "other_cid_2": "2103.14030_sec_003",
            "question": "What are the common dimensions along which ConvNets are scaled up in prior work according to EfficientNet?",
            "authored_answer": "Depth, width, and image resolution.",
            "excerpt": "scale up ConvNets by their depth"
        },
        {
            "id": "real_pilot_11",
            "paper_id": "2608.11822",
            "support_cid": "2608.11822_sec_002",
            "distract_cid": "2608.11822_sec_003",
            "other_cid": "2608.11830_sec_003",
            "other_cid_2": "1706.03762_sec_006",
            "question": "In paper 2608.11822, what failure occurs when linear release fails despite located representation?",
            "authored_answer": "The detector is OOD-inverted and the entire family of linear release directions at that site and resolution is bounded away from sufficiency.",
            "excerpt": "detector is OOD-inverted"
        },
        {
            "id": "real_pilot_12",
            "paper_id": "2608.11830",
            "support_cid": "2608.11830_sec_002",
            "distract_cid": "2608.11830_sec_003",
            "other_cid": "2608.11822_sec_003",
            "other_cid_2": "2005.11401_sec_004",
            "question": "What approach is highlighted in paper 2608.11830 for reducing environmental impact while preserving clinical performance in therapeutic LLMs?",
            "authored_answer": "Dynamic model selection, including model cascading.",
            "excerpt": "dynamic model selection, including model cascading"
        }
    ]

    def fetch_chunk(cid: str) -> Dict[str, Any]:
        res = vs.collection.get(ids=[cid], include=["metadatas", "documents"])
        if not res["ids"]:
            raise ValueError(f"Chunk {cid} not found in ChromaDB collection.")
        meta = res["metadatas"][0] if res["metadatas"] else {}
        return {
            "chunk_id": cid,
            "paper_id": meta.get("paper_id"),
            "section": meta.get("section", "General"),
            "text": res["documents"][0]
        }

    families: List[QuestionFamily] = []

    for spec in pilot_specs:
        q = spec["question"]
        s_data = fetch_chunk(spec["support_cid"])
        d_data = fetch_chunk(spec["distract_cid"])
        o_data = fetch_chunk(spec["other_cid"])
        o2_data = fetch_chunk(spec["other_cid_2"])

        # Verification: excerpt must occur in supporting chunk text (whitespace-normalized)
        from backend.selective.context import normalize_whitespace
        clean_text_norm = normalize_whitespace(s_data["text"])
        excerpt_norm = normalize_whitespace(spec["excerpt"])
        assert excerpt_norm.lower() in clean_text_norm.lower(), (
            f"Grounding failure: Excerpt '{spec['excerpt']}' not found in chunk '{spec['support_cid']}'."
        )

        # Cross-encoder scoring (no fallbacks)
        pairs = [
            [q, s_data["text"]],
            [q, d_data["text"]],
            [q, o_data["text"]],
            [q, o2_data["text"]]
        ]
        scores = reranker.model.predict(pairs)
        s_score = round(float(scores[0]), 4)
        d_score = round(float(scores[1]), 4)
        o_score = round(float(scores[2]), 4)
        o2_score = round(float(scores[3]), 4)

        p_support = Passage.from_dict({
            "chunk_id": s_data["chunk_id"],
            "paper_id": s_data["paper_id"],
            "section": s_data["section"],
            "text": s_data["text"],
            "score": s_score
        })
        p_distract = Passage.from_dict({
            "chunk_id": d_data["chunk_id"],
            "paper_id": d_data["paper_id"],
            "section": d_data["section"],
            "text": d_data["text"],
            "score": d_score
        })
        p_other = Passage.from_dict({
            "chunk_id": o_data["chunk_id"],
            "paper_id": o_data["paper_id"],
            "section": o_data["section"],
            "text": o_data["text"],
            "score": o_score
        })
        p_other_2 = Passage.from_dict({
            "chunk_id": o2_data["chunk_id"],
            "paper_id": o2_data["paper_id"],
            "section": o2_data["section"],
            "text": o2_data["text"],
            "score": o2_score
        })

        # 1. Clean context: support + intra-paper distractor + cross-paper distractor
        clean_rec = select_final_context(q, [p_support, p_distract, p_other])

        # 2. Duplicate context: clean context with exact post-selection duplicate injected
        dup_rec = inject_duplicates_post_selection(clean_rec, source_idx=0)

        # 3. Distractor context: support + TWO DISTINCT real distractors
        distract_rec = select_final_context(q, [p_support, p_distract, p_other, p_other_2])

        # 4. Evidence Removed context: supporting passage is OMITTED (removed), leaving only real distractors
        removed_rec = select_final_context(q, [p_distract, p_other, p_other_2])

        fam = QuestionFamily(
            family_id=spec["id"],
            base_question=q,
            authored_reference_answer=spec["authored_answer"],
            target_paper_ids=[spec["paper_id"]],
            contextual_paper_ids=sorted(list(set([spec["paper_id"], o_data["paper_id"], o2_data["paper_id"]]))),
            supporting_evidence=SupportingEvidence(
                chunk_id=s_data["chunk_id"],
                paper_id=s_data["paper_id"],
                section=s_data["section"],
                verified_excerpt=spec["excerpt"],
                cross_encoder_score=s_score,
                content_hash=p_support.content_hash
            ),
            variant_contexts={
                "clean": clean_rec,
                "duplicate": dup_rec,
                "distractor": distract_rec,
                "evidence_removed": removed_rec
            },
            variant_sufficiency_status={
                "clean": "pending",
                "duplicate": "pending",
                "distractor": "pending",
                "evidence_removed": "pending"
            },
            variant_sufficiency_labels={
                "clean": None,
                "duplicate": None,
                "distractor": None,
                "evidence_removed": None
            },
            scoring_provenance={
                "cross_encoder_model": reranker.model_name,
                "score_config": "cross_encoder_sigmoid_probability",
                "fallbacks_disabled": True
            }
        )
        families.append(fam)

    return families


def save_pilot_manifest(
    families: Sequence[QuestionFamily],
    output_path: str = "ai-research-copilot/data/metadata/pilot_dataset_12families.json"
):
    """Serialize the verified pilot dataset manifest to disk."""
    records = []
    for f in families:
        rec = {
            "family_id": f.family_id,
            "base_question": f.base_question,
            "authored_reference_answer": f.authored_reference_answer,
            "target_paper_ids": f.target_paper_ids,
            "contextual_paper_ids": f.contextual_paper_ids,
            "supporting_evidence": asdict(f.supporting_evidence),
            "variant_sufficiency_status": f.variant_sufficiency_status,
            "variant_sufficiency_labels": f.variant_sufficiency_labels,
            "scoring_provenance": f.scoring_provenance,
            "annotation_provenance": f.annotation_provenance,
            "variants": {}
        }
        for vname, vrec in f.variant_contexts.items():
            rec["variants"][vname] = {
                "num_passages": len(vrec.passages),
                "total_tokens": vrec.total_tokens,
                "payload_hash": vrec.payload_hash,
                "passages": [
                    {
                        "passage_id": p.passage_id,
                        "paper_id": p.paper_id,
                        "section": p.section,
                        "score": p.score,
                        "content_hash": p.content_hash,
                        "token_count": p.token_count,
                        "text": p.text
                    }
                    for p in vrec.passages
                ]
            }
        records.append(rec)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)
    print(f"Serialized genuine corpus pilot dataset to {output_path}")


save_pilot_dataset_to_json = save_pilot_manifest


def load_pilot_dataset_from_json(json_path: str) -> List[QuestionFamily]:
    """
    Load saved QuestionFamily records from JSON without querying ChromaDB or loading models.
    Reconstructs exact Passage and FinalContextRecord instances.
    """
    from backend.selective.context import format_context_payload, PINNED_TOKENIZER_NAME

    with open(json_path, "r", encoding="utf-8") as fp:
        raw_list = json.load(fp)

    families: List[QuestionFamily] = []
    for item in raw_list:
        s_data = item["supporting_evidence"]
        supp_ev = SupportingEvidence(
            chunk_id=s_data["chunk_id"],
            paper_id=s_data["paper_id"],
            section=s_data["section"],
            verified_excerpt=s_data["verified_excerpt"],
            cross_encoder_score=float(s_data["cross_encoder_score"]),
            content_hash=s_data["content_hash"]
        )

        q_text = item["base_question"]
        variant_contexts: Dict[str, FinalContextRecord] = {}
        raw_variants = item.get("variant_contexts") or item.get("variants") or {}
        for vname, vdata in raw_variants.items():
            passages = [Passage.from_dict(p) for p in vdata["passages"]]
            p_tuple = tuple(passages)
            total_tokens = sum(p.token_count for p in p_tuple)
            payload = format_context_payload(p_tuple)
            p_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            v_rec = FinalContextRecord(
                query=q_text,
                passages=p_tuple,
                max_passages=6,
                token_budget=2000,
                tokenizer_identity=PINNED_TOKENIZER_NAME,
                total_tokens=total_tokens,
                skipped_passages=(),
                truncation_policy="top_k_score_desc_id_asc_token_budget_2000",
                serialized_payload=payload,
                payload_hash=p_hash
            )
            variant_contexts[vname] = v_rec

        fam = QuestionFamily(
            family_id=item["family_id"],
            base_question=q_text,
            authored_reference_answer=item["authored_reference_answer"],
            target_paper_ids=list(item["target_paper_ids"]),
            contextual_paper_ids=list(item["contextual_paper_ids"]),
            supporting_evidence=supp_ev,
            variant_contexts=variant_contexts,
            variant_sufficiency_status=dict(item.get("variant_sufficiency_status", {})),
            variant_sufficiency_labels=dict(item.get("variant_sufficiency_labels", {})),
            scoring_provenance=dict(item.get("scoring_provenance", {})),
            annotation_provenance=item.get("annotation_provenance", {})
        )
        families.append(fam)

    return families

