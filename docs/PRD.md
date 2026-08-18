Product Requirements Document
AI Research Copilot
A Retrieval-Augmented System for Automated Scientific Literature Analysis

# 1. Overview

The volume of scientific publications — particularly in AI, machine learning, computer vision, and natural language processing — has grown to the point where researchers and students cannot realistically read enough papers to stay current. Existing academic search engines surface titles, abstracts, and metadata, but leave the work of reading, comparing, and synthesizing across papers entirely to the user.
Emerging AI-driven tools narrow this gap only partially: Semantic Scholar’s TLDR feature [7] generates single-sentence paper summaries but does not perform cross-paper synthesis or answer free-form questions, while Elicit [8] and Consensus [9] answer research questions from indexed paper databases but operate as closed, hosted systems without exposing their retrieval architecture or evaluation methodology.
The AI Research Copilot is a Retrieval-Augmented Generation (RAG) system that retrieves relevant academic papers, extracts their key contributions, and synthesizes that information into grounded summaries, answers, and literature reviews — reducing the manual effort required to conduct a literature review while keeping every generated claim traceable to a retrieved source.
Product type: Locally-deployed research assistant with a web-based (Streamlit) interface, backed by a hybrid dense + sparse retrieval pipeline and a hosted LLM for generation
Primary users: Researchers and students conducting literature reviews or exploratory research in NLP and computer vision
Core value proposition: Ask a research question in natural language and receive a synthesized, citation-grounded answer drawn from a curated corpus of papers, instead of manually reading and cross-referencing dozens of PDFs

# 2. Problem Statement

The continuously increasing number of published research papers creates an inefficient process for researchers and students trying to locate and analyze relevant literature. Traditional academic search engines are effective at locating papers but do not provide deep insights, cross-paper comparisons, or emerging-trend analysis.
As a result, users must read through multiple papers before understanding the key ideas and contributions within a research domain — a slow process that delays research progress. General-purpose LLMs used without retrieval compound this problem: they generate answers purely from fixed training data, cannot verify claims against current sources, and may produce outdated, incomplete, or unsupported answers.
What's needed is a system that automatically retrieves relevant papers, extracts their key contributions, and synthesizes that information into concise, grounded summaries and cross-paper comparisons.

# 3. Goals and Objectives

Objectives 1–5 and 7 constitute the core, guaranteed deliverable. Objective 7 is realized as a single-configuration RAGAS [5] evaluation (completed pipeline plus a non-RAG baseline for comparison). Objective 6 (GraphRAG) and an optional multi-configuration RAGAS ablation under Objective 7 are extensions pursued only as time and hardware permit. Objective numbers are labeled explicitly on each bullet below in Sections 3.1–3.2; Objective 6 is listed in Section 3.2 (Extension) rather than between Objectives 5 and 7 in Section 3.1, since it is schedule-permitting rather than guaranteed.

## 3.1 Core Objectives (guaranteed)

Develop an AI-based research assistant using Retrieval-Augmented Generation. (Objective 1)
Retrieve relevant research papers using a Hybrid Retrieval Engine combining dense vectors, sparse keyword search, and a cross-encoder re-ranker. (Objective 2)
Summarize and analyze research papers automatically. (Objective 3)
Generate literature reviews by synthesizing across multiple sources. (Objective 4)
Provide a UI that allows users to perform research and explore topics of interest. (Objective 5)
Quantitatively evaluate the system's accuracy and reliability using the RAGAS framework (faithfulness, context precision, context recall, answer relevance), including a non-RAG baseline comparison. (Objective 7, single-configuration)

## 3.2 Extension Objectives (schedule-permitting)

Visually depict and query citation relationships between papers using Neo4j Community Edition to enable Graph-based Retrieval-Augmented Generation (GraphRAG). (Objective 6)
Extend the single-configuration RAGAS evaluation to a multi-configuration ablation (e.g., with/without reranker, with/without GraphRAG). (Objective 7, extension variant)

# 4. Target Users and Use Cases


## 4.1 Primary Persona

Graduate student / researcher: Conducting a literature review in NLP or computer vision; needs to quickly understand the state of the art, identify key contributions across papers, and locate supporting evidence without reading every paper end-to-end.

## 4.2 Key Use Cases

Ask a free-form research question and receive a synthesized, citation-grounded answer drawn from the ingested corpus.
Request a summary of a specific paper's contributions, methodology, or results.
Request a literature review on a sub-topic that synthesizes findings across multiple papers.
Explore how papers relate to one another via citation links (extension: GraphRAG).
Receive an explicit "no sufficiently relevant context found" response rather than a hallucinated answer when the corpus does not cover the question.

# 5. Scope


## 5.1 In Scope (Core)

Batch ingestion of ~200 papers from arXiv across two related subfields (NLP and computer vision), using seed-paper snowball sampling to ensure citation density (Section 7.1.1), and respecting arXiv API rate limits and bulk-download terms.
PDF-to-structured-Markdown conversion via Docling, including section-aware parsing (abstract, introduction, methodology, results, etc.).
Chunking and dense-vector embedding (nomic-embed-text) stored in a local ChromaDB instance.
Sparse keyword retrieval via BM25 (bm25s library, LlamaIndex's BM25Retriever).
Fusion of dense and sparse candidates via Reciprocal Rank Fusion (LlamaIndex QueryFusionRetriever).
Cross-encoder re-ranking of fused candidates (bge-reranker-base).
Answer and summary generation via a hosted LLM (Gemini 2.5 Flash).
Streamlit web interface for query input and review of generated summaries.
Explicit handling of low-text / unparseable PDFs and low-relevance queries (see Section 7.4).
Single-configuration RAGAS evaluation (faithfulness, context precision, context recall, answer relevance) plus a non-RAG baseline comparison.

## 5.2 In Scope (Extension — schedule permitting)

Neo4j-based citation graph construction primarily from structured reference/citation data retrieved via the Semantic Scholar API during corpus construction (Section 7.1.1), with fuzzy title/DOI matching against Docling-extracted text as a fallback, relying on the snowball-sampled corpus to ensure a sufficiently connected graph.
Registration of the citation graph as an additional candidate source in the fusion stage (GraphRAG-inspired retrieval).
Real-time visual exploration of the citation graph within the UI.
Multi-configuration RAGAS ablation (e.g., reranker on/off, GraphRAG on/off).

## 5.3 Out of Scope

Live, per-query arXiv search that pulls in papers not already present in the local index (stretch goal only, not a core or extension requirement).
Fully offline / locally-hosted generation (e.g., Llama 3 [6] via Ollama [12]) — noted as a viable fallback architecture, not built for this project.
Multi-user, concurrent, or production-grade deployment; the system targets a single local machine with 8 GB GPU VRAM.
Full citation-network coverage beyond the ingested corpus (only citation links between two ingested papers are captured).

# 6. Functional Requirements


## 6.1 Non-Functional Requirements

Additional non-functional requirements:
Storage: ~2-4 GB disk space.
Startup time: 30-60 seconds.
API cost budget: minimal/free-tier.
Recoverability: batch ingestion resumes from last success.
System RAM: minimum 8 GB for the core pipeline (ChromaDB persistence plus query-time retrieval and reranking). If the GraphRAG extension is built, the minimum rises to 16 GB (32 GB recommended) to support Neo4j and ChromaDB maintaining persistent memory/connections concurrently during query execution. (Docling’s layout models are loaded only during batch ingestion and unloaded prior to query-time operations; distinct from the 8 GB GPU VRAM budget above.)

# 7. Technical Approach / Architecture

The system targets development and testing on a local machine with 8 GB of GPU VRAM, which constrains model choice and motivates offloading generation to a hosted API while keeping embedding and reranking local and lightweight.

## 7.1 Pipeline Stages

Ingestion: papers pulled from arXiv in a one-time or periodically refreshed batch step (not live per-query), respecting the arXiv API's minimum 3-second request delay and bulk-download terms. Demo corpus: ~200 papers across NLP and computer vision, built via seed-paper snowball sampling (see below) rather than random batch sampling.
Parsing: Docling converts PDFs (including multi-column text, tables, and formulas) into structured Markdown; heading structure is used to separate standard sections.
Chunking & embedding: within each section-aware unit, a single boundary-aware splitting step bounds text to a target chunk size while treating math blocks as atomic, so equations are never severed; resulting chunks are embedded with nomic-embed-text and stored in ChromaDB.
Hybrid retrieval: dense search (ChromaDB) runs alongside sparse search (BM25 via bm25s / LlamaIndex BM25Retriever) at query time.
Fusion: LlamaIndex QueryFusionRetriever merges and deduplicates candidates via Reciprocal Rank Fusion.
Reranking: bge-reranker-base cross-encoder sorts the fused list by relevance.
Generation: reranked context is passed to Gemini 2.5 Flash (hosted), which has a context window of roughly 1M tokens — large enough to pass multiple full papers directly for literature-review synthesis when the retrieved set is small.
Interface: Streamlit web application for query input and result review.

## 7.1.1 Corpus Construction — Snowball Sampling

A random batch of ~200 papers drawn independently across two subfields is unlikely to contain many mutual citations, which would leave the Neo4j citation graph sparse and largely disconnected, undermining the GraphRAG extension. To avoid this, the corpus is built by snowball sampling from two separate seed sets: 3–5 seminal NLP papers (e.g., Vaswani et al. [2], Lewis et al. [1]) and 3–5 seminal computer-vision papers (e.g., He et al. [10], Dosovitskiy et al. [11]).
Two refinements keep this approach both topically coherent and large enough to reach the ~200-paper target:
Subfield filtering: a candidate paper is added to the corpus only if its arXiv category (e.g., cs.CL/cs.LG for NLP, cs.CV for computer vision) matches one of the two target subfields, discarding off-topic papers (e.g., pure math or systems papers) that get pulled in incidentally through a seed paper's reference list.
Forward and backward snowballing: backward snowballing alone (pulling from each paper's own reference list) tends to converge toward a shrinking set of older, foundational papers and may not reach ~200 unique papers from only 6–10 seeds. This is supplemented with forward snowballing — using a citation index such as the Semantic Scholar API to also pull papers that cite each seed/candidate paper — so the corpus can grow outward as well as backward. To prevent exponential explosion from highly-cited seed papers, forward snowballing is strictly capped at 15 new candidate papers per source paper, applied at every hop (not only the original seeds). Additionally, a hard ceiling of 300 total Semantic Scholar API calls is enforced for the entire corpus-construction phase, after which snowballing stops and the corpus is finalized at whatever size it has reached, subject to the 150-paper GraphRAG viability threshold defined below, with the descoping consequence of falling short of it described in Section 10.
This combination is intended to produce a corpus with a dense web of in-subfield citation edges, so the citation-graph construction step (FR-14) has a realistic chance of connecting a meaningful share of ingested papers, rather than yielding mostly isolated nodes or drifting outside the two target subfields.
The 150-paper figure referenced as the GraphRAG viability threshold throughout this document (Section 10; Appendix A) is not an arbitrary corpus-size cutoff. Corpus connectivity, not just corpus size, is what determines whether FR-15's one-hop graph traversal returns useful candidates: at ~200 papers built via dense snowball sampling, the corpus is expected to have most papers holding at least one in-corpus citation edge, giving graph traversal a realistic chance of surfacing extra candidates for a typical query. Below ~150 papers, a much larger share of papers is expected to end up as isolated nodes with no in-corpus citation edges, causing FR-15's traversal to return empty candidate sets for most queries and fall back to dense/sparse fusion alone on nearly every query rather than merely degrading. Directly after ingestion, the actual fraction of papers with at least one in-corpus citation edge will be computed and used as the descoping check, rather than corpus size alone; the 150-paper figure is carried through this document as the size at which that connectivity fraction is expected to become too low to be worth attempting.

## 7.1.2 Sequential Pipeline Execution for VRAM Management

Docling's layout analysis and table-structure recognition can themselves load vision models into GPU memory. Running Docling parsing, nomic-embed-text embedding, and bge-reranker-base reranking concurrently risks exceeding the 8 GB local VRAM budget. To avoid this, the ingestion-time and query-time stages are kept sequential rather than concurrent: during ingestion, Docling's models are loaded, used to parse the full batch, and unloaded before the embedding model is loaded to vectorize the resulting chunks. At query time, only the embedding and reranking models need to be resident, since Docling is not invoked again. This keeps peak VRAM usage well under budget without requiring smaller or quantized models.

## 7.2 Component Summary


## 7.3 Architecture Rationale

LlamaIndex is used over LangChain for the core pipeline because it is comparatively better optimized for indexing- and retrieval-centric workloads, which fits a document-heavy research assistant.
ChromaDB is chosen over FAISS [3] for easier setup and built-in persistence, appropriate for a locally deployed, single-user tool, trading away some of FAISS's raw in-memory search speed.
Gemini 2.5 Flash is accessed as a hosted service specifically so the local 8 GB VRAM budget is dedicated entirely to embedding and reranking; this trades a small amount of latency and an external API dependency for development velocity and a stronger model than could realistically run locally. Llama 3 [6] via Ollama [12] remains a documented fallback for a fully offline or production-grade deployment.
The citation graph is GraphRAG-inspired rather than a direct implementation of Edge et al.'s [4] community-summarization method: it performs citation-graph traversal, using Neo4j specifically to map citation relationships, rather than summarizing over community structure. A free-text query has no direct representation in the citation graph, so graph traversal is not seeded by parsing the query itself (e.g., via NER); instead, the dense/sparse RRF-fused candidates already retrieved for that query are used as seed nodes, and the graph is traversed outward from them (FR-15). This keeps the graph a genuinely query-aware candidate source without requiring a separate query-to-node mapping component.
The UI surfaces the exact retrieved chunks behind each answer (FR-18) rather than only the final generated text, so groundedness can be verified visually during review or demo rather than taken on faith.
UI-level caching (FR-19) is used specifically to protect against redundant API calls during a live demo (e.g., an accidental page refresh), not as a general performance optimization.

## 7.4 Failure Modes and Handling

Scanned / image-only PDFs with no text layer: OCR (Docling's do_ocr fallback) is attempted first; only papers whose OCR output still falls below a minimum confidence/quality threshold are flagged and excluded from the corpus, rather than ingested as empty or unreliable content.
No sufficiently relevant retrieved context: the system surfaces an explicit "no sufficiently relevant context found" message rather than forwarding empty/noisy context to the generator, to reduce ungrounded hallucination on out-of-corpus questions.
Partial ungroundedness despite reasonable context: expected to occur at some non-zero rate; this is exactly what the faithfulness metric in the evaluation plan is designed to quantify, not something the pipeline is assumed to eliminate.
Empty graph-candidate result on niche queries: if citation-graph traversal for a given query returns no matching nodes/edges, the system silently bypasses the graph candidate source and falls back to dense/sparse RRF fusion alone (FR-15), rather than raising an error or blocking the response.
Oversized atomic block reranking degradation: math blocks kept whole under FR-3 can still exceed bge-reranker-base’s 512-token combined query+chunk limit even when they fit within the embedding model’s input window. bge-reranker-base truncates rather than erroring in this case, so the block is cross-encoded on partial content only; this can artificially suppress its rank and is a documented limitation rather than a solved problem.

# 8. Success Metrics and Evaluation Plan

A held-out test set of approximately 30–50 question-answer pairs will be assembled from a subset of the ingested papers, with ground-truth source passages for each question. An LLM will first draft a larger candidate list of questions and answers from the Docling-parsed Markdown; the team will manually review, correct, and filter these to the final set, discarding ambiguous, unanswerable, or near-verbatim items.
To mitigate confirmation bias (the same team both builds the system and curates the test set), question drafting and filtering will be completed before final tuning of retrieval and reranking, and at least 20% of the final questions will be reviewed by someone outside the immediate development team.
RAGAS defaults to an OpenAI model as its LLM judge unless explicitly reconfigured; using that default would incur an unbudgeted OpenAI API cost not assumed elsewhere in this PRD. Simply reconfiguring the judge to Gemini would avoid that cost but introduces a different problem: since Gemini 2.5 Flash is also the generation model, using it as both generator and judge risks self-preference bias, where a judge tends to score outputs from its own model family more favorably. To avoid both problems, RAGAS will be explicitly configured with a judge LLM from a different model family than the generator — e.g., a low-cost model such as GPT-4o-mini or Claude Haiku (FR-20). If team budget does not permit a separate-provider judge, Gemini will be used as a documented fallback judge, and this will be reported as an explicit limitation on the faithfulness and answer-relevance scores rather than left unstated.

## 8.1 Metrics


## 8.2 Baseline Comparison

The same held-out QA set will be run against a non-RAG baseline (Gemini 2.5 Flash answering directly from parametric knowledge, no retrieved context), scored with answer relevance (RAGAS) and a manual factual-accuracy check. The comparison is intended to demonstrate the marginal contribution of the retrieval pipeline, not to evaluate the pipeline in isolation.

## 8.3 Validation Sequence

Interim: lightweight manual spot-check against approximately 10-15 QA pairs, run ahead of the full evaluation. This pass is also used to empirically calibrate FR-11's relevance-score threshold rather than hardcoding an arbitrary cutoff. A single fixed relevance threshold may not generalize equally across all query types (e.g., broad conceptual vs. narrow factual queries); this is a known limitation disclosed in reported results rather than something the calibration pass is assumed to fully eliminate.
Core: single-configuration RAGAS evaluation of the completed pipeline plus the non-RAG baseline comparison (guaranteed deliverable).
Optional extension: multi-configuration RAGAS ablation (e.g., with/without reranker or GraphRAG) if time and hardware permit.

## 8.4 Evaluation Coverage by Generation Mode

FR-8 defines two generation modes: point-question answering over retrieved chunks, and literature-review synthesis, which may pass multiple full papers directly into Gemini's long context rather than discrete retrieved chunks. The held-out QA set and RAGAS metrics above are scoped to the point-QA mode, where discrete retrieved chunks make context precision, context recall, and faithfulness well-defined. Literature-review outputs are not scored with the same automated metrics, since context precision assumes a discrete, rankable set of retrieved chunks that does not apply cleanly when multiple full papers are passed as context. Instead, this mode is checked with a smaller number of manual spot-checks (3–5 review questions), assessing coverage of key papers and absence of unsupported claims. Extending quantitative RAGAS-style scoring to the literature-review mode is treated as an optional stretch, not a core evaluation requirement.

# 9. Timeline and Milestones

The project is sequenced so the core deliverable is complete and independently demoable before any extension work begins, ensuring no single component's delay jeopardizes the overall project.
Week 10 has two competing claims on it — spillover buffer for the Week 9 evaluation, and the window for extension work — and completing the core evaluation always takes priority over starting extension work within that week.

# 10. Risks and Mitigations


# 11. Assumptions and Dependencies


## 11.1 Assumptions

The demo corpus (~200 papers from NLP and computer vision) is representative enough to exercise cross-paper synthesis and citation-graph construction.
A hosted Gemini API connection will be available and within budget for the duration of development and demo.
The development machine provides 8 GB of GPU VRAM, sufficient for the local embedding and reranking models.
The system operates over a pre-ingested, periodically refreshed corpus rather than live per-query retrieval.

## 11.2 Dependencies

External: arXiv API/bulk-download access (subject to arXiv's rate limits and terms); Google Gemini API availability; Semantic Scholar API (or equivalent citation index) for forward-citation snowball sampling during core corpus construction (Section 7.1.1) — used to build the shared corpus consumed by both the core pipeline and the optional GraphRAG extension.
Libraries/frameworks: Docling, LlamaIndex (BM25Retriever, QueryFusionRetriever), ChromaDB, bm25s, RAGAS, Streamlit, Neo4j Community Edition (extension).
Models: nomic-embed-text (embedding), bge-reranker-base (reranking), Gemini 2.5 Flash (generation), a separate-provider judge model for RAGAS (e.g., GPT-4o-mini or Claude Haiku) or documented Gemini fallback (FR-20).

# 12. References

[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," Advances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
[2] A. Vaswani et al., "Attention Is All You Need," Advances in Neural Information Processing Systems, vol. 30, 2017.
[3] J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2021.
[4] D. Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization," arXiv preprint arXiv:2404.16130, 2024.
[5] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," arXiv preprint arXiv:2309.15217, 2023.
[6] A. Grattafiori et al., "The Llama 3 Herd of Models," arXiv preprint arXiv:2407.21783, 2024.
[7] I. Cachola, K. Lo, A. Cohan, and D. S. Weld, "TLDR: Extreme Summarization of Scientific Documents," in Findings of ACL: EMNLP 2020, arXiv preprint arXiv:2004.15011, 2020.
[8] Elicit, Inc., "Elicit: The AI Research Assistant." Available: https://elicit.com.
[9] Consensus, "Consensus: AI Search Engine for Research." Available: https://consensus.app.
[10] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," arXiv preprint arXiv:1512.03385, 2015 (also in Proc. IEEE CVPR, 2016, pp. 770–778).
[11] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," arXiv preprint arXiv:2010.11929, 2020.
[12] Ollama, "Ollama: Run Large Language Models Locally." Available: https://ollama.com.
Appendix A: Document Revision History
v1.7 (current): Splits the System RAM non-functional requirement (Section 6.1) into an 8 GB core-pipeline minimum and a 16–32 GB minimum that only applies if the GraphRAG extension (Neo4j) is built, since the prior single 16 GB figure was justified entirely by an optional component; adds a definition of the 150-paper GraphRAG viability threshold (Section 7.1.1), which was previously used four times as a decision trigger without ever being derived, and points the Section 10 risk-table and prior-revision-history mentions back to that definition instead of restating the bare number; names streamlit-agraph (with pyvis as a fallback) as the implementation for FR-16's citation-graph visualization and adds a corresponding Graph UI row to the Section 7.2 Component Summary table, closing a gap where FR-16 was the only stack component without a named library; and corrects the Section 10 OCR risk/mitigation pair, which had justified sequencing Tesseract OCR after Docling's parsing pass on GPU-VRAM-contention grounds even though Tesseract runs on CPU and does not compete for VRAM — the mitigation is retained but re-justified on pipeline-simplicity and resumability grounds instead.
v1.6: Fixes FR-12’s opening clause, which still omitted context recall after the v1.5 sync; replaces FR-12’s baseline-comparison target, which claimed all four RAGAS metrics would exceed the non-RAG baseline even though three of them (faithfulness, context precision, context recall) are undefined for that baseline, with a comparison scoped to answer relevance plus a qualitative reference to the Section 8.2 manual check; removes Docling from the System RAM non-functional requirement’s concurrency justification (Section 6.1), aligning it with Section 7.1.2’s sequential-execution design, in which Docling is unloaded before query-time; and clarifies, in the Section 10 risk table, that the backward-only snowball fallback (~100-150 papers) will likely fall under the 150-paper GraphRAG viability threshold rather than leaving that consequence implicit.
v1.5: Backfills "context recall" into the earlier definitional mentions of the RAGAS metric set (Section 3.1 Objective 7, Section 5.1 Scope, and the Section 7.2 Component Summary table) so they match FR-12 and the Section 8.1 Metrics table, which already listed it as a fourth core metric; syncs the header table and this revision history to the v1.5 filename.
v1.4: Lowers target chunk size to 250–350 tokens to stay within bge-reranker-base's 512-token combined query+chunk limit; clarifies GraphRAG paper-to-chunk retrieval via pre-computed ChromaDB chunks; documents BM25 tokenization behavior on LaTeX/math content; adds RAGAS Context Recall as a fourth core metric; adds Semantic Scholar exponential-backoff requirement; and adds the 150-paper corpus-size contingency to the Synopsis for consistency with the PRD.
v1.3: Adds explicit objective numbering, a query-to-graph seeding mechanism for FR-15, an oversized-atomic-chunk fallback for FR-2/FR-3, a documented reranker-threshold generalization limitation, a Semantic Scholar API availability risk, and a corpus-version cache-invalidation clause for FR-19.
v1.2: Initial detailed PRD.
v1.0-v1.1: Early drafts.


| Field | Detail |
| --- | --- |
| Document Owner | Soham Pawar |
| Team / Department | CSE Core, Computer Science and Engineering |
| Minor Specialization | Artificial Intelligence / Machine Learning (Computational Intelligence) |
| Faculty Guide | Dr. Adarsh Rag |
| Document Version | v1.7 |
| Status | Draft for review |
| Target Submission Date | 15 November 2026 |




| ID | Requirement | Priority |
| --- | --- | --- |
| FR-1 | The system shall ingest PDFs from a pre-collected arXiv corpus and convert them to structured Markdown via Docling, preserving section headings. | Must (core) |
| FR-2 | The system shall parse converted Markdown into standard paper sections (abstract, introduction, methodology, results, etc.) for section-aware chunking, and further split any section exceeding a maximum chunk size (target ~250–350 tokens, chosen so that a chunk plus a typical query stays within bge-reranker-base's 512-token combined input limit) into smaller, size-bounded sub-chunks so that long sections do not dilute sparse (BM25) term-frequency statistics relative to shorter chunks from other papers. | Must (core) |
| FR-3 | The system shall chunk section-aware text using boundary-aware splitting that applies FR-2's size bound while treating math blocks (fenced and LaTeX) as atomic units, ensuring equations are never split across chunk boundaries. Oversized atomic blocks (e.g., long derivations) shall be kept whole up to a documented ceiling and flagged for review if they exceed the embedding model's input window. Note that even blocks within that window may still exceed bge-reranker-base's separate 512-token combined query+chunk limit; bge-reranker-base truncates rather than erroring in that case, so such blocks are scored on partial content during reranking and may be under-ranked as a result (see Section 7.4). Dense vector embeddings shall be generated using nomic-embed-text and stored in ChromaDB. | Must (core) |
| FR-4 | The system shall support sparse keyword retrieval over the same corpus using BM25 (bm25s). Because standard BM25 tokenization strips or mangles LaTeX commands and mathematical symbols, sparse keyword search over math-block content is expected to be unreliable; equation and formula matching therefore relies primarily on dense (nomic-embed-text) retrieval rather than BM25, and this is a documented limitation rather than a solved problem. | Must (core) |
| FR-5 | The system shall fuse dense and sparse retrieval results using Reciprocal Rank Fusion without manual weight tuning. | Must (core) |
| FR-6 | The system shall re-rank fused candidates using a cross-encoder (bge-reranker-base) before passing context to generation. | Must (core) |
| FR-7 | The system shall generate answers and summaries via Gemini 2.5 Flash, grounded in retrieved context. Acceptance criterion: every factual claim in the generated output must be attributable to at least one chunk present in the retrieved context passed to the generator; groundedness is measured quantitatively by the RAGAS faithfulness metric (FR-12). | Must (core) |
| FR-8 | The system shall support literature-review generation by retrieving the top-10 chunks per sub-topic (where sub-topics are identified by the user's query or automatically decomposed by the LLM into constituent research questions) across relevant papers. When the total retrieved context for a literature-review query spans fewer than 5 papers, full papers may be passed directly within the LLM's context window instead of discrete chunks. Acceptance criterion: generated literature reviews must reference at least 3 distinct source papers and contain no unsupported claims, as verified by manual spot-check (Section 8.4). | Must (core) |
| FR-9 | The system shall provide a Streamlit UI for entering queries and viewing generated summaries/answers. | Must (core) |
| FR-10 | The system shall enable OCR as a fallback (Docling's do_ocr option, e.g. via Tesseract) for papers with no extractable text layer (e.g., scanned image-only PDFs), and shall flag and exclude only those papers whose OCR-extracted text still falls below a minimum quality threshold (defined as fewer than 100 extractable words or a Tesseract mean-confidence score below 60%), rather than ingesting empty or unreliable content. | Must (core) |
| FR-11 | The system shall return an explicit "no sufficiently relevant context found" message when no retrieved chunk exceeds a minimum relevance threshold, instead of forwarding empty/noisy context to the generator. The relevance threshold shall be calibrated empirically during the interim validation pass (Section 8.3) and applied to normalized relevance scores, ensuring comparability across queries. | Must (core) |
| FR-12 | The system shall support a RAGAS evaluation pass producing faithfulness, context precision, context recall, and answer relevance scores against a held-out QA set. Target: the RAG pipeline should achieve a faithfulness score above 0.7, context precision above 0.6, context recall above 0.6, and answer relevance above 0.7. Answer relevance must meaningfully exceed the non-RAG baseline’s answer-relevance score. Faithfulness, context precision, and context recall are undefined for the non-RAG baseline and therefore have no baseline value to exceed; the RAG pipeline’s factual grounding relative to the baseline is instead assessed qualitatively via the manual factual-accuracy comparison in Section 8.2, not as a direct numeric comparison (FR-13). | Must (core) |
| FR-13 | The system shall support a non-RAG baseline run (direct LLM answers, no retrieval) over the same QA set for comparison. | Must (core) |
| FR-14 | The system shall build a citation graph in Neo4j primarily from the structured reference/citation data (with DOI/arXiv IDs) already retrieved from the Semantic Scholar API during corpus construction (Section 7.1.1), matching against the ingested corpus by ID rather than string; fuzzy title/DOI matching against Docling-extracted reference text is used only as a fallback for papers Semantic Scholar does not cover. | Should (extension) |
| FR-15 | The system shall register the citation graph as an optional additional candidate source in the fusion/rerank stage. Graph traversal shall be seeded from the top-N papers already returned by the dense/sparse RRF-fused candidate list for that query (not by parsing the free-text query into a graph node directly, which the citation graph has no schema for); the graph is then traversed outward one hop from those seed papers to surface additional, citation-connected papers as extra candidates. If graph traversal for a given query returns no matching nodes/edges (e.g., none of the top-N seed papers are in the citation graph), the system shall bypass the graph candidate source and fall back to dense/sparse RRF fusion alone, without raising an error or blocking the response. When a paper is retrieved via graph traversal, its pre-computed chunks already stored in ChromaDB (from the ingestion-time embedding step) are fetched directly and added to the candidate pool for reranking — no new embedding or retrieval call is made against that paper at query time. Because Reciprocal Rank Fusion (RRF) calculates the final score as a sum over available ranked lists, dropping an empty stream does not require special-casing or skew the remaining math. | Should (extension) |
| FR-16 | The UI shall support visual exploration of the citation graph using streamlit-agraph (or pyvis, embedded via st.components.v1.html, if node count exceeds streamlit-agraph's practical rendering limit), querying Neo4j directly for the subgraph around a selected paper. | Could (extension) |
| FR-17 | The system shall support a multi-configuration RAGAS ablation (e.g., reranker or GraphRAG on/off). | Could (extension) |
| FR-18 | The Streamlit UI shall include an expander/panel that displays the exact retrieved chunks (with source paper) used to produce each generated answer, for traceability. | Must (core) |
| FR-19 | The UI layer shall cache heavyweight model and connection objects (embedding model, reranker, database client, retriever) as application-scoped singletons, and cache serializable query results separately, so that page refreshes during a live demo do not re-trigger redundant API calls. Cached query results shall be keyed by both the query string and a corpus-version identifier (e.g., a timestamp of the last ingestion run), so that a corpus refresh invalidates previously cached answers instead of silently serving stale results. | Must (core) |
| FR-20 | The RAGAS evaluation shall use a judge LLM from a different model family/provider than the generation model (Gemini) — e.g., a low-cost model such as GPT-4o-mini or Claude Haiku — to avoid both RAGAS's unbudgeted OpenAI default and self-preference bias from judging Gemini outputs with Gemini; if no separate-provider budget is available, Gemini shall be used as a documented fallback judge with the limitation disclosed in reported results. | Must (core) |




| Category | Requirement |
| --- | --- |
| Performance / Latency | For point-QA queries (FR-8's chunk-retrieval mode), end-to-end response time (retrieval + fusion + rerank + generation) should stay within an interactive range for a synchronous single-user demo — target under ~15 seconds, dominated by the hosted Gemini call. Literature-review queries that pass multiple full papers into Gemini's long context (FR-8's full-paper mode) are exempt from this target and are expected to take substantially longer given the larger payload; this should be communicated to the user in the UI (e.g., a progress indicator) rather than left unexplained. |
| Data licensing & redistribution | Ingested PDFs and derived Markdown are stored and used locally for research/demo purposes only, consistent with arXiv's terms of use; the corpus is not redistributed or published as a standalone dataset. |
| Data privacy | The system does not collect or persist personally identifiable user data beyond the query text needed to serve a response within a session; no user accounts, tracking, or analytics are implemented. |
| Reliability | Anticipated failure modes (unparseable PDFs, low-relevance queries, an empty graph-candidate result on niche queries) degrade gracefully with explicit user-facing messaging or silent fallback (FR-10, FR-11, FR-15) rather than crashing or silently returning ungrounded output. |
| Portability | The system is designed and tested for a single local machine with 8 GB GPU VRAM; it is explicitly not designed for multi-user concurrency or horizontal scaling (Section 5.3). |




| Component | Technology | Notes |
| --- | --- | --- |
| PDF parsing | Docling | Layout-aware; preserves headings for section-aware chunking |
| Dense embeddings | nomic-embed-text | Lightweight; runs within local 8 GB VRAM budget |
| Vector store | ChromaDB (local) | Chosen for easy setup and built-in persistence over FAISS for this single-user deployment |
| Sparse retrieval | bm25s via LlamaIndex BM25Retriever | Sparse-matrix based; faster/more memory-efficient than older BM25 implementations |
| Fusion | LlamaIndex QueryFusionRetriever (RRF) | Combines dense + sparse scores without manual weight tuning |
| Reranking | bge-reranker-base | Cross-encoder; local, within VRAM budget |
| Generation | Gemini 2.5 Flash (hosted API) | Offloads generation from local GPU; ~1M-token context window |
| Citation graph (extension) | Neo4j Community Edition | Built primarily via the Semantic Scholar API (DOI/arXiv ID matching); fuzzy title/DOI matching against Docling-extracted text is a fallback only |
| Graph UI (extension) | streamlit-agraph (pyvis fallback) | Renders the Neo4j subgraph around a selected paper inline in Streamlit (FR-16) |
| UI | Streamlit | Web app for queries, summaries, and (extension) graph exploration |
| Evaluation | RAGAS | Faithfulness, context precision, context recall, answer relevance |




| Metric | Definition | Applies To |
| --- | --- | --- |
| Faithfulness | Whether generated claims are supported by retrieved context | RAG pipeline only |
| Context precision | Whether retrieved chunks are relevant to the question | RAG pipeline only |
| Answer relevance | Whether the generated answer addresses the question | RAG pipeline and non-RAG baseline |
| Context Recall | Whether all ground-truth facts required to answer the question were successfully retrieved (guards against high Context Precision masking incomplete retrieval) | RAG pipeline only |
| Factual accuracy (manual) | Manual check of correctness, used in place of faithfulness/context precision/context recall for the baseline, which are undefined without retrieval | Non-RAG baseline only |




| Weeks | Milestone | Deliverable Type |
| --- | --- | --- |
| 1–3 | Ingestion, chunking, and a working dense-retrieval baseline (Docling, ChromaDB) | Core |
| 4–6 | Sparse retrieval, RRF fusion, reranking, Gemini API integration, and interim manual spot-check; core hybrid pipeline complete and independently demoable by end of week 6 | Core |
| 7–8 | Streamlit user interface built around the validated pipeline | Core |
| 9 | Full RAGAS evaluation on the completed pipeline plus non-RAG baseline comparison and manual factual-accuracy grading (may spill into week 10) | Core |
| 10 | Buffer and polish. If Week 9's RAGAS evaluation and baseline grading are unfinished, they take priority and continue into Week 10; the Neo4j/GraphRAG layer and optional RAGAS ablation are attempted only once the core evaluation is confirmed complete, with descoping remaining an option if time runs out. If schedule permits beyond those two, an optional LLM-as-a-judge automated metric for literature reviews (scoring source coverage and synthesis quality) may also be attempted, using a separate-provider judge per the FR-20 mitigation; this is lower priority than GraphRAG and the RAGAS ablation, and the manual spot-check in Section 8.4 remains the fallback if it is not reached | Extension |




| Risk | Mitigation |
| --- | --- |
| 8 GB local VRAM limits which models can run locally | Offload generation to hosted Gemini 2.5 Flash; keep only lightweight embedding (nomic-embed-text) and reranking (bge-reranker-base) local |
| Hosted-API dependency for generation (latency, availability, cost) | Document Llama 3 [6] via Ollama [12] as a viable local fallback for offline/production deployment |
| Some PDFs have no extractable text layer (e.g., scanned, image-only); OCR must be invoked selectively rather than run on the full corpus, and re-running Docling's layout pipeline on the same document twice would complicate the ingestion pipeline's control flow | Tesseract-based OCR runs on CPU and does not compete for GPU VRAM; it is still run only for papers lacking a text layer, sequenced after the main Docling parsing pass (not for VRAM reasons, but to avoid invoking Docling's layout pipeline twice on the same document and to keep ingestion resumable); flag and exclude only papers whose OCR output still falls below a minimum confidence threshold |
| Ungrounded / hallucinated answers on out-of-corpus questions | Enforce a minimum relevance threshold; return explicit "no sufficiently relevant context found" message when unmet |
| Confirmation bias in the team-curated evaluation QA set | Finalize question drafting/filtering before final retrieval tuning; have ≥20% of questions reviewed externally |
| GraphRAG extension or ablation delays the core deliverable | Treat GraphRAG and the multi-configuration ablation as extensions attempted only after the core pipeline is validated and on schedule; project remains complete and gradeable without them |
| arXiv API rate limits or bulk-download terms during ingestion | Use one-time/periodically refreshed batch ingestion (not live per-query fetching), respecting the minimum 3-second request delay |
| Semantic Scholar API rate limits, downtime, or incomplete coverage during forward-citation snowballing — this is now a core-path dependency (Section 7.1.1), not just an extension risk, since it is used to reach the ~200-paper corpus target | Cache Semantic Scholar responses locally during ingestion so a later run does not re-query already-resolved papers; if the target of ~200 papers is not reached via forward snowballing within a reasonable retry budget, fall back to backward-only snowballing plus a larger seed set (expanding from 6-10 to 15-20 seeds), accepting a smaller corpus of approximately 100-150 papers; because this fallback range sits almost entirely below the 150-paper GraphRAG viability threshold defined in Section 7.1.1, invoking this fallback will likely trigger the descoping of the GraphRAG extension (Objective 6) rather than demonstrating it with a sparse graph. All Semantic Scholar API calls during corpus construction shall use exponential backoff on rate-limit (HTTP 429) responses, with a conservative default request rate assumed at or below the API's published unauthenticated limit; this rate should be verified against Semantic Scholar's current documentation before implementation, as published limits are subject to change. |
| Independent random corpus sampling would yield a sparse, largely disconnected citation graph (making GraphRAG hard to demonstrate); snowball sampling itself risks drifting outside the two target subfields or falling short of ~200 papers if done backward-only | Build the corpus via seed-paper snowball sampling instead of independent random sampling; filter candidates by arXiv category to stay in-subfield; supplement backward reference-list snowballing with forward citation lookups (e.g., Semantic Scholar API) to reach the target size (Section 7.1.1) |
| Concurrent GPU use by Docling's layout models, embedding, and reranking could exceed the 8 GB VRAM budget | Run pipeline stages sequentially rather than concurrently: unload Docling's models after ingestion-time parsing before loading the embedding model; only embedding and reranking models are resident at query time (Section 7.1.2) |
| RAGAS defaults to an OpenAI LLM judge, risking unbudgeted API costs or rate limits | Explicitly configure RAGAS with a specified, budgeted judge LLM before running the evaluation pass (FR-20) |
| Using Gemini as both the generation model and the RAGAS judge risks self-preference bias in faithfulness/answer-relevance scores | Use a judge LLM from a different model family (e.g., GPT-4o-mini or Claude Haiku) where budget allows; otherwise use Gemini as a documented fallback judge and disclose the limitation in reported results (FR-20) |
| RAGAS metrics (context precision in particular) are not well-defined for the literature-review generation mode, which can pass full papers instead of discrete chunks | Scope automated RAGAS scoring to the point-QA mode; validate literature-review outputs separately via manual spot-checks (Section 8.4) |
| If the optional LLM-as-a-judge metric for literature reviews (Week 10 extension) is attempted, it risks self-preference bias if evaluating Gemini's outputs with Gemini | Use a judge LLM from a separate provider (e.g., GPT-4o-mini or Claude Haiku), mirroring the FR-20 mitigation; retain the manual spot-check (Section 8.4) as fallback if this extension is not reached |

