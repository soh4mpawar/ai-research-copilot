# Controlled Benchmark Evaluation Protocol (v1.0 Frozen)
**Duplicate-Aware Source Aggregation for Selective Scientific Question Answering**

---

## 1. Executive Summary & Protocol Identification
- **Protocol Version**: `1.0_frozen`
- **Date**: November 3, 2026
- **Research Lead**: Directing Research Lead (ChatGPT)
- **Harness & Verification**: Implementation Agent (Antigravity)
- **Primary Research Question**:
  *Does aggregating distinct evidence by paper and section improve answer-or-abstain decisions over passage relevance thresholds, at matched answer coverage?*
- **Empirical Status**: Experimental hypothesis under controlled offline testing. No novelty, superiority, or production readiness claims are authorized. Negative results must be reported transparently.

---

## 2. Corpus Grounding & Strict Split Partitioning

### A. Subcorpus Scope & Exclusion of Development Data
- **Total Corpus**: 200 scientific papers in persistent ChromaDB collection `scientific_papers`.
- **Exclusions**: Exactly **40 historical development papers** (audited from `evaluation/ragas_benchmark_results_40samples.json` and the 12-family pilot) are completely excluded to eliminate prompt tuning and exploration leakage.
- **Eligible Corpus**: 160 papers, containing 7,368 real stored text chunks.
- **Citation Graph Structure**: Exactly 0 citation edges connect eligible papers to each other; all 59 observed external citations point to the excluded historical development/landmark papers.

### B. Group-Separated Split Allocation (`full_benchmark_split_manifest.json`)
Partitioned deterministically using pseudo-random allocation (random seed 42):
1. **Train Split**: 80 papers (50.0%), 3,434 ChromaDB chunks.
2. **Validation Split**: 32 papers (20.0%), 1,409 ChromaDB chunks.
3. **Held-Out Test Split**: 48 papers (30.0%), 2,525 ChromaDB chunks.

### C. 120 Question Family Selection
From the eligible papers, 120 independent question families are deterministically selected (1 family per distinct primary paper):
- **Train Families**: 60 families (from the first 60 train papers).
- **Validation Families**: 24 families (from the first 24 validation papers).
- **Held-Out Test Families**: 36 families (from the first 36 test papers).
- **Leakage Guarantees**: 0% target paper leakage, 0% contextual paper overlap, and 0% passage text overlap across splits.

---

## 3. Question Typology & Controlled Experimental Conditions

### A. Balanced Single-Passage vs. Multi-Passage Synthesis
To rigorously test source aggregation across document sections:
- **Single-Passage Families (50.0% / 60 families)**: The question is answered completely by a single focused passage from a specific section of the paper.
  - Train: 30 families | Val: 12 families | Test: 18 families.
- **Multi-Passage Families (50.0% / 60 families)**: The question requires synthesizing two distinct facts located in two separate sections of the same paper (e.g., method premise in Introduction/Methods and empirical metric in Experiments/Results). Neither passage alone is sufficient.
  - Train: 30 families | Val: 12 families | Test: 18 families.

### B. Four Controlled Context Conditions per Family (480 Total Contexts)
Every question family is evaluated across 4 standardized context conditions. To eliminate passage-count and length confounds, context passage count is held constant (4 passages, <=2,000 tokens):
1. **`supported_candidate`**:
   - Single-passage: 1 essential supporting passage + 3 topically relevant distractors from the split's papers.
   - Multi-passage: 2 essential supporting passages (from distinct sections) + 2 topically relevant distractors.
   - Expected status: Sufficient (1).
2. **`relevant_replacement`**:
   - Essential supporting passage is replaced with another topically relevant corpus distractor, maintaining passage count at 4.
   - Expected status: Insufficient (0).
3. **`duplicate_replacement`**:
   - Essential supporting passage is replaced with an exact duplicate of one of the distractors already present in the context, maintaining passage count at 4.
   - Directly tests whether gating models are deceived by redundant passage copies.
   - Expected status: Insufficient (0).
4. **`natural_retrieval`**:
   - Realistic end-to-end retrieval: Hybrid retrieval (BM25 + dense ChromaDB) followed by `BAAI/bge-reranker-base` cross-encoder reranking restricted strictly to the papers within that split.
   - Evaluated under realistic retrieval noise without artificial injection.
   - Status: Determined by blind semantic review.

---

## 4. Models, Baselines, & Diagnostic Controls

### A. Baselines
1. **`B0_AlwaysAccept`**: Full-coverage baseline that unconditionally accepts all contexts ($\text{Coverage} = 1.0$), establishing the full-coverage evidence selection risk floor.
2. **`B1_ValidationTunedThreshold`**: Standard max-relevance baseline that thresholds the top cross-encoder score $s_{(1)}$, tuned strictly on validation data.
3. **`B2_FrozenHistoricalGate`**: The historical 4-condition heuristic gate evaluated as a discrete binary operating point $\{0, 1\}$.
4. **`B_Diag_Length`**: Passage-count and token-count diagnostic logistic baseline. Evaluates whether context length or passage counts predict sufficiency, detecting construction shortcuts.

### B. 2×2 Factorial Logistic Design
- Feature space:
  - **Relevance Features (4D)**: $[s_{(1)}, s_{(2)}, \text{mean\_score}, \text{score\_spread}]$
  - **Structural Features (4D)**: $[s_{\text{top\_paper\_second\_passage}}, s_{\text{other}}, \text{section\_diversity}, \text{section\_mean\_score}]$
- Models:
  - **`M1_Relevance_Raw`**: 4 relevance features, raw passages.
  - **`M2_Relevance_Dedup`**: 4 relevance features, deduplicated passages.
  - **`M3_Full_Raw`**: 4 relevance + 4 structural features, raw passages.
  - **`M4_Full_Dedup`**: 4 relevance + 4 structural features, deduplicated passages.

### C. Training & Threshold Tuning Protocol
- **Preprocessing**: `StandardScaler` fit strictly on training features.
- **Regularization**: L2 logistic regression with $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ selected strictly by validation split log-loss (smaller $C$ chosen on ties).
- **Validation-Only Operating Points**: Decision thresholds $\tau_{0.50}, \tau_{0.75}, \tau_{0.90}$ tuned strictly on validation split to achieve target coverages $50\%$, $75\%$, and $90\%$. Frozen before viewing held-out test predictions.

---

## 5. Primary Comparisons & Hypothesis Testing

### A. Primary Hypotheses
1. **Structural Aggregation Benefit after Deduplication**:
   - Comparison: **`M4_Full_Dedup` vs. `M2_Relevance_Dedup`**
   - Tests whether adding paper and section structural features improves selective answering performance once duplicates are removed.
2. **Deduplication Benefit with Structural Features**:
   - Comparison: **`M4_Full_Dedup` vs. `M3_Full_Raw`**
   - Tests whether deduplication improves selective answering performance when structural aggregation features are utilized.
3. **Deduplication on Pure Relevance**:
   - Comparison: **`M2_Relevance_Dedup` vs. `M1_Relevance_Raw`**
   - Tests whether deduplication benefits simple passage relevance scoring.

### B. Statistical Significance & Clustered Bootstrap
- **Primary Metric**: Discrete Expected AURC (Area Under the Risk-Coverage Curve).
- **Secondary Metrics**: Evidence-selection risk and achieved coverage at deployable operating points ($\tau_{0.50}, \tau_{0.75}, \tau_{0.90}$).
- **Resampling**: 2,000 paired cluster bootstrap resamples across connected paper components.
- **Reporting**: Observed point estimates and bootstrap mean differences reported separately with 95% percentile bootstrap confidence intervals.
- **Disaggregation**:
  - By condition (`supported_candidate`, `relevant_replacement`, `duplicate_replacement`, `natural_retrieval`).
  - By question type (single-passage vs. multi-passage).
  - Controlled mixture vs. natural retrieval.

---

## 6. Blind Semantic Review & Integrity Controls
- **Blinded Annotation**: Annotations performed on exported review packets containing only opaque context IDs, natural language questions, and numbered passage texts.
- **Concealed Metadata**: Review packet omits candidate/source mappings, variant condition names, reference answers, cross-encoder scores, split names, and model predictions.
- **Derivability & Grounding**: Reviewer must supply a derivable answer, exact supporting substring quotation (programmatically asserted against passage text), missing facts, and uncertainty rationale.
- **Negative Result Allowance**: If M4 fails to outperform M2/M3 or if deduplication shows no empirical benefit, the result will be reported as an empirical negative result.

---

## 7. Frozen Code & Artifact Hashes (SHA-256)
- `data/metadata/full_benchmark_split_manifest.json`: `7f7005c89813266a2826cc7a707c3a6101fbcf158a3796ea97578d422f044b2c`
- `data/metadata/benchmark_120_question_specs.json`: `5995d911eb18606bc41b976a5d7b2e8a13aab24e03c1f892802ba67489bd9f68`
- `data/metadata/full_benchmark_120families_annotated.json`: `ebe3bc684e2db2b19b87ca06595839fb20747af1fda10019feefe7e967b2049b`
- `data/metadata/frozen_models_train_val.json`: `3e977e4cdcb5b7a5d80fc58451356557704ddc6945f72c2eb8657af8dcb9bbbd`
- `data/metadata/full_benchmark_eval_results.json`: `742bb98e97f5559683e7e8704fc8adcdccf71871042ce8822553ccc18443a9c1`
- `data/metadata/risk_coverage_curve_full_benchmark.svg`: `c5a04beb1a24a6da0d83ad721afc49e1fae22ed519351279e200e2530ad75923`
- `data/metadata/risk_coverage_curve_full_benchmark.html`: `d69fd4c485b4eacf9d51c0ea588deb676ac33f491dd27e771d15720951abf3de`
- `backend/selective/context.py`: `f1c7cfa73fbbc0d53dc4991e51f893ef118eb24398020f838d218926b6c6260d`
- `backend/selective/dataset.py`: `127813a0ea6ff27b4a7846d38e90b22b398b71021459d1268b8d57a5542de890`
- `backend/selective/features.py`: `3bfe3af7148d875847d5322230c1f6d27fcd739fc622b5122255051d5bb6d422`
- `backend/selective/models.py`: `38cff672b496703723cc97856254819f363cdd41cc8add57d71e7a3ad99ad0d2`
- `backend/selective/metrics.py`: `f534778f4a84127600ebe7b9995c0705093dd335ed852f235c2560e965c2de0e`
- `backend/selective/autorater.py`: `6d92916b2bbe616d8801d7bbe7fd99fc250e4f70f3ca8969cc1d87a1042cf734`
- `evaluation/run_selective_eval.py`: `b690a62349de1688bdd33d2a1f9811ec827b0a5b7966a021bd3c4454a20aa7d8`
- `evaluation/run_full_benchmark_eval.py`: `063cca8bcee9adb8971fa2bf81ab8dc93f34a9551cf4f46089382e9a3b2f9177`
