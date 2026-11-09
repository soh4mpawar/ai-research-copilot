# AI Research Copilot: Project Presentation Outline

**Presentation Title:** Evaluating Duplicate-Aware Evidence Aggregation for Selective Scientific Question Answering  
**Team Members:** Akanksha Jaiswal (230905106), Mohd. Uzair Qureshi (220905158), Soham Pawar (230905312) — CSE Core  
**Minor Specialization:** Artificial Intelligence / Machine Learning (Computational Intelligence)  
**Department Offering:** Computer Science and Engineering | **Name of Guide:** Dr. Adarsh Rag  
**Date of Submission:** 15 November 2026  
**Presentation Duration:** ~12–15 Minutes (8 Slides)

---

### Slide 1: Title & Executive Overview (1 min)
- **Project Title:** AI Research Copilot: Evaluating Duplicate-Aware Evidence Aggregation for Selective Scientific Question Answering.
- **Team & Affiliation:** Department of Computer Science and Engineering.
- **Core Framing:** An applied research platform and empirical benchmark evaluating when a scientific RAG system should answer vs. abstain.
- **Key Takeaway:** An honest, rigorous empirical study presenting an **unsupported primary hypothesis** alongside valuable architectural insights on deduplication and multi-passage evidence.

---

### Slide 2: Problem Statement & Motivation (2 mins)
- **The Core Problem:** RAG systems answer unconditionally, even when evidence is missing.
- **The Relevance Illusion:** Cross-encoders assign high relevance (>0.90) to passages that discuss the topic but omit the specific formula, parameter, or empirical finding needed.
- **The Redundancy Problem:** Retrieved passages from the same paper often repeat high-level claims, crowding out complementary evidence.
- **Research Question:** Does aggregating distinct evidence by paper/section and exact deduplication improve evidence-sufficiency ranking over raw relevance baselines at matched coverage?

---

### Slide 3: Software Implementation vs. Experimental Scope (1.5 mins)
- **Software Deliverable (AI Research Copilot):**
  - Ingestion: Docling / pypdf with section regex parsing.
  - Retrieval: Dense ChromaDB (nomic-embed-text) + Sparse BM25 + Cross-Encoder (BAAI/bge-reranker-base).
  - Graph Extension: In-memory NetworkX DiGraph citation traversal with PyVis visualization.
  - Interface: Multi-tab interactive Streamlit research assistant.
- **Clear Demarcation:** The software provides the retrieval pipeline; the scientific experiment rigorously isolates and evaluates the selective evidence-sufficiency gate (M1–M4). No unsubstantiated generation accuracy claims are made.

---

### Slide 4: Mathematical Methodology (2.5 mins)
- **Immutable Context Representation:** Context $C = (p_1, ..., p_K)$ with text, rerank score $s_i \in [0, 1]$, source paper ID, section name, and token count.
- **Exact Within-Paper Deduplication:** Collapses identical whitespace-normalized text within the same paper to the highest-scoring representative.
- **2x2 Factorial Feature Space:**
  - 4D Relevance: Top-1 score, Top-2 score, Mean score, Score spread.
  - 4D Structural: Top-paper second score, Other-paper max score, Section entropy, Section mean score.
  - Configurations: M1 (Raw Relevance), M2 (Dedup Relevance), M3 (Raw Full), M4 (Dedup Full).
- **Metric Formulation:** Tie-correct discrete AURC under uniform random tie ordering, Coverage, and Evidence Selection Risk (undefined when coverage = 0).
- **Statistical Testing:** 2,000 paired cluster bootstrap resamples by primary paper cluster (seed `20261104`).

---

### Slide 5: Controlled Benchmark Construction (2 mins)
- **Corpus & Split Partitioning:** 160 eligible papers (7,368 chunks); 40 historical papers strictly quarantined.
  - Train: 60 families (240 contexts) | Val: 24 families (96 contexts) | Test: 36 families (144 contexts across 36 paper clusters).
- **Four Controlled Conditions:**
  1. `supported_candidate`: Gold evidence + distractors (exactly 4 passages).
  2. `relevant_replacement`: Essential evidence replaced by topically relevant non-supporting chunk (exactly 4 passages).
  3. `duplicate_replacement`: Essential evidence replaced by an exact duplicate distractor (exactly 4 passages).
  4. `natural_retrieval`: Realistic end-to-end hybrid retrieval (allows up to 6 passages within 2,000 tokens).
- **Semantic Adjudication:** 12-checkpoint untruncated AI assistant reading protocol (196 verified verbatim quotes, 0 failures).
- **Zero-Fitting Protocol:** Models, scalers, and operating thresholds frozen before new v2 test construction, disclosing historical test-paper exposure during earlier invalid Stage 4 run.

---

### Slide 6: Primary Results & Hypothesis Verdict (3 mins)
- **Primary Hypothesis:** $	ext{AURC}_{	ext{M4}} < 	ext{AURC}_{	ext{M1}}$ on 108 controlled contexts.
- **Verdict:** **NOT SUPPORTED**
  - M1 (Relevance Raw): $	ext{AURC} = 0.4329$
  - M4 (Full Dedup): $	ext{AURC} = 0.4400$
  - Observed Difference: $\Delta = \mathbf{+0.0071}$ *(negative favors M4)*
  - 95% Bootstrap Confidence Interval: **$[-0.0503, +0.0679]$** *(includes differences favoring either method; study does not establish equivalence)*
- **Baseline B1 (Max Relevance):** Achieved the lowest observed controlled-context AURC (**0.4049**).
- **Shortcut Length Diagnostic:** Achieved discrete AURC = **0.6798** (compared to all-tied prevalence baseline B0 = 0.6667), establishing only that this fitted diagnostic performed poorly.

---

### Slide 7: Nuanced Secondary Insights (2 mins)
- **1. Descriptive Multi-Passage Subgroup (11 Families):**
  - On single-passage questions ($N=75$), M1 (0.3743) and M4 (0.3388) perform comparably.
  - On multi-passage questions ($N=33$), raw models degrade severely ($	ext{M1} = 0.6894$, $	ext{M3} = 0.6900$), whereas deduplicated models maintain lower absolute discrete AURC ($	ext{M2} = 0.3508$, $	ext{M4} = 0.3803$). Deduplication prevents repeated distractors from deceiving multi-evidence synthesis.
- **2. Natural Retrieval Contexts ($N=36$):**
  - M4 achieved the lowest observed AURC (**0.0501** vs $	ext{B1}=0.0707$, $	ext{M1}=0.0861$). However, with only 3 insufficient contexts, the 95% CI spans zero ($[-0.1176, +0.0160]$).
- **3. Duplicate Invariance Property:**
  - Under explicit assumptions, M2 and M4 confer **bounded duplicate invariance** (observed $0.000000$ score change under post-selection duplication on fixed contexts), while raw models suffer $>0.31$ score distortion.

---

### Slide 8: Threats to Validity & Future Directions (1 min)
- **Methodological Limitations:**
  - AI assistant semantic annotations (not independent human gold raters).
  - Models and thresholds frozen before new v2 test construction, disclosing historical test-paper exposure during earlier invalid Stage 4 run.
  - High synthetic insufficient prevalence in controlled benchmark (66.7%) vs. natural retrieval (8.3%).
  - Commercial API spend: strictly **$0.00** (all models run locally on PyTorch CUDA).
- **Conclusion:** An evaluated system and controlled empirical comparison demonstrating that while structural deduplication does not outperform simple maximum relevance on overall controlled scientific contexts, it is critical for duplicate robustness and multi-passage evidence aggregation.
