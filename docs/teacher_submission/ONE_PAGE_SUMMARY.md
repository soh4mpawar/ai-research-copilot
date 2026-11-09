# AI Research Copilot: Executive Summary & Project Contribution

**Team Members:** Akanksha Jaiswal (230905106), Mohd. Uzair Qureshi (220905158), Soham Pawar (230905312) — CSE Core  
**Minor Specialization:** Artificial Intelligence / Machine Learning (Computational Intelligence)  
**Department Offering:** Computer Science and Engineering | **Name of Guide:** Dr. Adarsh Rag  
**Submission Date:** 15 November 2026 | **Commercial API Spend:** **$0.00**

---

### 1. Problem Statement
> Retrieval-augmented scientific question-answering systems must decide whether retrieved passages contain enough evidence to answer a question. High relevance scores and repeated passages do not necessarily establish that all requested information is available. This project investigates whether exact deduplication and paper/section features improve evidence-sufficiency ranking over relevance-only baselines, while measuring acceptance coverage and the proportion of accepted contexts judged insufficient.

### 2. System Architecture vs. Evaluated Components
- **Software Deliverable:** End-to-end literature research assistant with ChromaDB vector search (nomic-embed-text), BM25 keyword retrieval, cross-encoder reranking (BAAI/bge-reranker-base), in-memory NetworkX DiGraph citation traversal with PyVis visualization, and an interactive Streamlit UI.
- **Experimental Scope:** The scientific study strictly isolates and evaluates the evidence-sufficiency gating models (M1–M4 and baselines) on frozen context records. No unverified claims of generation accuracy or hallucination reduction are made without generation experiments.

### 3. Controlled Benchmark Design
- **Corpus:** 160 eligible papers (7,368 text chunks); 40 historical development papers quarantined.
- **Split Partitioning:** 60 Train families (240 contexts), 24 Validation families (96 contexts), 36 Held-Out Test families (144 contexts across 36 paper clusters).
- **Four Conditions:** `supported_candidate`, `relevant_replacement`, and `duplicate_replacement` (exactly 4 passages per context); `natural_retrieval` (allows up to 6 passages within a 2,000-token budget).
- **Zero-Fitting Protocol:** Models, scalers, and validation thresholds were completely frozen before new v2 test construction, while disclosing historical test-paper exposure during an earlier invalid Stage 4 run. Uncertainty was evaluated via 2,000 paired cluster bootstrap resamples (seed `20261104`, clustered by primary paper ID).

### 4. Key Empirical Findings
1. **Primary Hypothesis (M4 < M1): NOT SUPPORTED**  
   On 108 controlled test contexts, M4 achieved discrete AURC = **0.4400** vs. M1 = **0.4329** ($\Delta = \mathbf0.0071$, 95% CI: **$[-0.0503, +0.0679]$**; negative favors M4). The hypothesized improvement is not supported. The interval includes differences favoring either method; this study does not establish equivalence. Baseline **B1 (Max Relevance)** achieved the lowest observed controlled-context AURC (**0.4049**).
2. **Length Diagnostic Shortcut:** Achieved discrete AURC = **0.6798** (compared to all-tied prevalence benchmark B0 = 0.6667), establishing only that this specific fitted diagnostic performed poorly; it does not prove the absence of shortcuts in the benchmark.
3. **Descriptive Multi-Passage Subgroup (11 Families):** Deduplicated models exhibited lower absolute discrete AURC (M2=0.3508, M4=0.3803) than raw models (M1=0.6894, M3=0.6900). This is a descriptive subgroup observation on 11 families and does not establish causal claims.
4. **Natural Retrieval Generalization (36 Contexts):** M4 achieved the lowest observed AURC (**0.0501**). However, with only 3 insufficient contexts, the 95% CI spans zero ($[-0.1176, +0.0160]$).
5. **Duplicate Invariance Property:** Under explicit operating assumptions, M2 and M4 confer bounded duplicate invariance (observed **0.000000** score change under tested post-selection duplication on fixed contexts), while raw models suffer $>0.31$ score distortion.

### 5. Methodological Disclosures
- Models and thresholds were frozen prior to new v2 test construction, disclosing historical test-paper exposure during earlier invalid Stage 4 run.
- Annotations were adjudicated via an untruncated AI assistant reading protocol (196 verified verbatim quotes), not independent human gold raters.
- Controlled contexts have 66.7% synthetic insufficient prevalence, compared to 8.3% in natural retrieval.
- Commercial API spend: strictly **$0.00**.
