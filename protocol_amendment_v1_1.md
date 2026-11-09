# Protocol Amendment v1.1: Development Specification for Training Benchmark Construction

**Document Identifier**: `protocol_amendment_v1_1.md`  
**Base Protocol**: `research_protocol_v1.md`  
**Status**: Active Development Amendment  
**Date**: 2026-11-04  
**Author**: Antigravity Assistant (under research direction of ChatGPT)  
**Scope**: Construction and Semantic Annotation of the 60 Training Question Families (240 Contexts)  

---

## 1. Context and Rationale

During initial pilot construction of Benchmark v2 Batch 01 (12 families, 48 contexts), two structural issues were identified:
1. **Retrieval Implementation**: The initial scratch context builder used an unranked lexical token-set overlap heuristic rather than the system's actual hybrid retrieval stack (Dense Vector + BM25 + Reciprocal Rank Fusion + Cross-Encoder Reranking).
2. **Distractor Quality & Duplicate Boilerplate**: Random distractor sampling from other papers occasionally selected administrative boilerplate (grant acknowledgments, ethics statements, or disconnected computer vision text), and duplicate replacement sometimes duplicated this boilerplate rather than substantive scientific evidence.
3. **Question Type Verification**: Several questions designed as multi-passage cross-section synthesis were found to be completely answerable from a single comprehensive passage (e.g., an extensive introduction or architecture section).

This amendment defines formal operational rules for correcting Batch 01 and authoring the remaining 48 training families (Batches 02–05, families 13–60).

---

## 2. Formal Protocol Amendments

### Amendment 1: Empirical Question Classification Policy
- Every authored question must record both `intended_question_type` and `verified_question_type`.
- If semantic review confirms that any single supplied passage is sufficient to answer the complete question (as observed in `v2_train_006`, `v2_train_010`, and `v2_train_012`), the question MUST be empirically reclassified as `single_passage` (`is_multi_passage = False`).
- No artificial 50/50 split will be forced, and no multi-passage necessity claim will be made when a single passage is factually sufficient.

### Amendment 2: Standardized Context Sizing & Final-Context Rules
- **Natural Retrieval Policy**: Natural retrieval contexts adhere to the system's standard production policy: up to 6 passages and $\le 2,000$ tokens, selected by `select_final_context` from the cross-encoder reranked candidates. Natural retrieval is NOT forced down to 4 passages.
- **Controlled Perturbation Conditions**: The three controlled conditions (`supported_candidate`, `relevant_replacement`, `duplicate_replacement`) standardize to 4 passages and $\le 2,000$ tokens per context to enable controlled comparison of evidence perturbation within each family.
- Controlled conditions and natural retrieval results will be evaluated and reported as separate experimental strata.

### Amendment 3: Genuine Hybrid Retrieval Pipeline Specification
All natural retrieval contexts must be generated using the project's actual retrieval stack, restricted to the assigned training-paper pool:
1. **Pool Restriction**: Both dense and sparse retrieval are strictly restricted to the assigned training papers (`paper_allocations['train']` in `full_benchmark_split_manifest.json`) prior to candidate ranking. Global retrieval with post-hoc filtering is prohibited.
2. **Dense Stream**: ChromaDB cosine query using `nomic-ai/nomic-embed-text-v1.5` dense embeddings (or exact cosine similarity over stored document embeddings) restricted to the training paper IDs.
3. **Sparse Stream**: Sparse BM25 keyword index constructed via `bm25s` over only the corpus chunks belonging to the training paper pool.
4. **Fusion**: Reciprocal Rank Fusion (`RRFFusionRetriever`, $k=60$) merging dense and sparse ranks within the pool.
5. **Cross-Encoder Reranking**: `BAAI/bge-reranker-base` joint query-passage sequence classification executed on CUDA.
6. **No Oracle Injection**: The retrieval engine receives only the question string; no gold chunk IDs, answers, or hints are provided. Citation graph expansion is explicitly disabled.
7. **Traceability**: All component ranks (`dense_rank`, `bm25_rank`), fused ranks (`rrf_rank`), cross-encoder scores (`rerank_score`), execution latencies, and payload SHA-256 hashes must be preserved in context metadata.

### Amendment 4: Topical Replacement and Substantive Duplication Rules
- **Candidate Pool for Replacements**: Replacement passages must be selected from actual candidate chunks retrieved or ranked from the training split.
- **Exclusion of Boilerplate**: Acknowledgments, author contribution statements, ethical disclosures, grant numbers, empty fragments, and references are strictly excluded from being selected as replacement or duplicate evidence.
- **Topical Relationship Requirement**: Replacement passages should discuss the same method, model family, task, dataset, or scientific concept, while omitting the specific detail requested by the prompt. If a topical replacement is unavailable in the training pool, the condition must be reported as unavailable rather than filled with arbitrary text.
- **Substantive Duplication**: In `duplicate_replacement`, an existing substantive scientific passage (such as the primary supporting chunk or a topical candidate) must be duplicated—never boilerplate.

### Amendment 5: Deterministic Shuffling and Blind Annotation Protocol
- Opaque context identifiers (e.g. `ctx_tr_0001` through `ctx_tr_0240`) must be assigned via deterministic pseudorandom shuffling across families and conditions.
- Packets must not be grouped in predictable 4-condition blocks or reveal paper/family identities in their IDs.
- Semantic reviews must be individually authored from the blind packets, recording `judgment` (`sufficient`, `insufficient`, or `ambiguous`), `derivable_answer`, `supporting_quotations` (programmatically verified exact substrings), `missing_information`, and `rationale`. Programmatic excerpt-presence heuristic labeling is prohibited.
- Prior exposure (the assistant serving as author and annotator) remains an explicitly disclosed research limitation.
