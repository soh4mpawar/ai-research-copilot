# Benchmark v2 Batch 01: Corrected Audit and Semantic Annotation Report

**Status**: Complete (First 12 Training Families, 48 Contexts - Corrected under Protocol Amendment v1.1)  
**Date**: 2026-11-04  
**Protocol**: Frozen `research_protocol_v1.md` amended by `protocol_amendment_v1_1.md`  
**Investigator**: Antigravity Assistant (under research direction of ChatGPT)

---

## 1. Executive Summary & Corrections Implemented

This document records the corrected construction, scoring, and context-by-context semantic review of **Benchmark v2 Batch 01**, comprising the first 12 training paper families from `full_benchmark_split_manifest.json`.

Per ChatGPT's instructions, three critical corrections were executed:

1. **Reclassification of Single-Passage Answerable Questions**: Families `v2_train_006` (fiction overclosure), `v2_train_010` (DPOD open-world), and `v2_train_012` (mACH visual grounding) were empirically reclassified from `multi_passage` to `single_passage` (`is_multi_passage = False`) because a single supplied passage was confirmed sufficient to answer the complete question. We do not force an artificial 50/50 split or claim multi-passage necessity when one passage factually suffices.
2. **Replacement of Lexical Overlap with Production Hybrid Pipeline**: The initial word-set overlap retrieval heuristic was quarantined in `data/metadata/audit_archive/batch01_initial/` and replaced by `SplitRestrictedHybridRetriever` (`backend/selective/retrieval_adapter.py`). This adapter executes dense retrieval (`nomic-embed-text-v1.5`), sparse keyword search (`bm25s`), Reciprocal Rank Fusion ($k=60$), and cross-encoder reranking (`BAAI/bge-reranker-base`), strictly restricted to the 80 training papers prior to ranking. Natural retrieval adheres to the standard production policy: up to 6 passages and $\le 2,000$ tokens.
3. **Topical Replacements & Substantive Duplicates**: Random boilerplate distractors (grant acknowledgments, ethics statements, disconnected computer vision fragments) were eliminated. Replacement passages were chosen from top-ranked substantive candidates from the same paper or topic, explaining their topical relationship. In `duplicate_replacement`, an existing substantive scientific chunk was duplicated, preserving the 4-passage / 2,000-token budget.
4. **Deterministic Shuffling & Re-Annotation**: All 48 contexts were deterministically shuffled across families and conditions to issue opaque IDs (`ctx_tr_0001` through `ctx_tr_0048`), eliminating predictable 4-condition blocks. Every context was personally re-read and semantically reviewed, with 100% of supporting quotations programmatically verified as exact substrings.

## 2. Provenance Disclosures & Ethical Boundaries

1. **Provisional AI-Assisted Annotations**: All semantic judgments in this batch were authored by Antigravity's language model performing direct semantic analysis of blind packets. They are **provisional research labels**, not human expert annotations, independent double-blind evaluations, or publication-ready ground truth.
2. **Prior Exposure Disclosure**: The same AI assistant authored the question specifications and annotated the blind context packets. Although blind packets concealed question IDs, paper IDs, chunk IDs, condition labels, split names, reference answers, and reranker scores, prior latent knowledge of the papers exists.
3. **Zero Paid APIs**: All embeddings, retrieval operations, and cross-encoder rerankings were executed locally using open-source PyTorch models on a local RTX 3070 Ti Laptop GPU.
4. **Corpus Immutability**: All source passages were retrieved from the read-only ChromaDB vector store (`data/chroma_db`). No synthetic text or fabricated scores were introduced.

## 3. The 12 Question Families Specification

### Family v2_train_001 (Paper: `2608.11829` | Verified Type: Single-Passage)

- **Intended Type**: `single_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: In on-policy distillation (OPD) for large language models, how does the pass@K performance of OPD-trained models change relative to pre-OPD base models as the sampling budget K increases, and what does problem-level solvability analysis at pass@1024 reveal?
- **Reference Answer**: As the sampling budget K increases, the pass@K advantage initially held by OPD-trained models gradually shifts to the pre-OPD base models, indicating that OPD improves sampling efficiency rather than expanding the capability boundary. Furthermore, a problem-level solvability analysis using pass@1024 reveals an asymmetry: OPD causes more previously solvable problems to become unsolvable than previously unsolvable problems to become solvable.
- **Supporting Chunks**: `2608.11829_sec_001`
- **Fact Explanations & Semantic Necessity**: Single passage supplies both the pass@K scaling shift and the pass@1024 solvability asymmetry.

### Family v2_train_002 (Paper: `2608.11889` | Verified Type: Multi-Passage)

- **Intended Type**: `multi_passage` | **Verified Type**: `multi_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: In DexterSQL, why does resolving schema ambiguity between columns like Patient.Diagnosis and Examination.Diagnosis require inspecting data distributions rather than schema definitions alone, and what execution accuracy (EX) does DexterSQL achieve on the BIRD-Dev benchmark using the open-weight GPT-OSS-120B model?
- **Reference Answer**: Resolving schema ambiguity between Patient.Diagnosis and Examination.Diagnosis requires inspecting data distributions because column names and schemas alone cannot differentiate their roles: Patient.Diagnosis contains one row per patient representing a single final diagnosis, whereas Examination.Diagnosis contains multiple preliminary rows per patient ID. On the BIRD-Dev benchmark, DexterSQL achieves an execution accuracy (EX) of 67.6% using the open-weight GPT-OSS-120B model, outperforming prior non-fine-tuned Text-to-SQL systems by at least 2.7%.
- **Supporting Chunks**: `2608.11889_sec_009, 2608.11889_sec_007`
- **Fact Explanations & Semantic Necessity**: Passage 2608.11889_sec_009 provides the structural data distribution reason (one final diagnosis row vs multiple examination rows per patient ID), while Passage 2608.11889_sec_007 provides the empirical BIRD-Dev execution accuracy (67.6% on GPT-OSS-120B).

### Family v2_train_003 (Paper: `2608.11919` | Verified Type: Multi-Passage)

- **Intended Type**: `multi_passage` | **Verified Type**: `multi_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: How does LazyTrain's Hybrid 8-bit operator optimize LLM training on limited hardware, and what sustained TFLOPS, throughput, and peak GPU memory does LazyTrain achieve in the Qwen3.6-27B H800 MetaMathQA run at batch size 72?
- **Reference Answer**: LazyTrain's Hybrid 8-bit operator couples 8-bit optimizer states with fast gradient clipping, where state compression reduces optimizer state memory and fast clipping counteracts the additional CPU-side update overhead. In the primary Qwen3.6-27B H800 MetaMathQA run at batch size 72, LazyTrain achieves 219.95 TFLOPS and 1361 tokens/s throughput, peaking at 68.84 GB of GPU memory while obtaining 95.42% exact match accuracy.
- **Supporting Chunks**: `2608.11919_sec_001, 2608.11919_sec_002`
- **Fact Explanations & Semantic Necessity**: Passage 2608.11919_sec_001 provides the Hybrid 8-bit operator coupling mechanism, while Passage 2608.11919_sec_002 provides the specific empirical metrics (219.95 TFLOPS, 1361 tokens/s, 68.84 GB peak GPU memory) on Qwen3.6-27B.

### Family v2_train_004 (Paper: `2608.11922` | Verified Type: Single-Passage)

- **Intended Type**: `single_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: What failure mode occurs when ranking candidate retrieval passages by raw respondent predictive entropy alone within a single question, and how does the LODESTAR method modify the prompt to create a directed entropy signal?
- **Reference Answer**: Within a single question, ranking candidate passages by raw respondent predictive entropy alone is unreliable because misleading passages systematically lower entropy, reading a misleading passage 30.3% of the time (worse than the random candidate baseline of 28.9%). LODESTAR modifies the prompt to [p; psi*; q] by inserting a learned short natural-language polarizer string psi* between the passage and the question, which steers entropy higher on misleading passages while keeping it low on supporting passages.
- **Supporting Chunks**: `2608.11922_sec_006`
- **Fact Explanations & Semantic Necessity**: Single passage 2608.11922_sec_006 contains the failure rate (30.3% vs 28.9%) and the learned string polarizer mechanism.

### Family v2_train_005 (Paper: `2608.12626` | Verified Type: Single-Passage)

- **Intended Type**: `single_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: What primary limitation causes strategic drift in LLM agents during long-horizon reasoning tasks, and how does the EpicStar framework address it during inference using its episodic and working memory components?
- **Reference Answer**: In long-horizon reasoning tasks, finite attention resources prevent LLMs from maintaining strategic coherence over thousands of steps, causing agents to overfit to local observations and produce inconsistent subgoals (strategic drift). EpicStar addresses this by maintaining a structured memory bank of successful past episodes alongside a working memory of short-term environmental changes, using a dynamic gating mechanism during inference to choose between executing a retrieved action directly or generating new reasoning through contextual fusion of past episodes and working memory.
- **Supporting Chunks**: `2608.12626_sec_001`
- **Fact Explanations & Semantic Necessity**: Single passage 2608.12626_sec_001 explains both the strategic drift mechanism and the EpicStar dual-memory inference architecture.

### Family v2_train_006 (Paper: `2608.12630` | Verified Type: Single-Passage)

- **Intended Type**: `multi_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Empirically reclassified from multi_passage to single_passage per ChatGPT decision: a single supplied passage was confirmed sufficient to answer the complete question.
- **Question**: In the study of AI-generated fiction, how do the authors define the corpus-level concept of 'overclosure' (including their specific definition of 'formal range'), and how do they distinguish between 'variance overclosure' and 'correlational overclosure'?
- **Reference Answer**: Overclosure describes a corpus-level condition where generated texts remain locally fluent and coherent but display an unusually restricted range across measured formal dimensions; 'formal range' specifically denotes the dispersion and coordination among measured stylistic features (excluding plot, characterization, or narrative structure). The authors distinguish 'variance overclosure' (where novels differ very little from one another with respect to individual stylistic features) from 'correlational overclosure', which describes coupled feature compression where an individual AI-generated novel may stylistically resemble human fiction, while a collection of AI-generated novels occupies a much narrower formal range.
- **Supporting Chunks**: `2608.12630_sec_007, 2608.12630_sec_002`
- **Fact Explanations & Semantic Necessity**: Passage 2608.12630_sec_007 defines corpus-level overclosure and clarifies formal range as stylistic dispersion, while Passage 2608.12630_sec_002 provides the conceptual distinction between variance and correlational overclosure.

### Family v2_train_007 (Paper: `2608.12652` | Verified Type: Single-Passage)

- **Intended Type**: `single_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: Why does training a linear activation probe on temporal benchmark splits (such as WikiMIA) fail to provide an unconfounded estimate of model memorization in benchmark contamination detection?
- **Reference Answer**: A probe trained to separate two item sets will exploit any feature that separates them. When member and non-member benchmark sets are split by publication date (as in WikiMIA), they systematically differ in topic, style, and temporal distribution shift. As a result, a probe without access to the target model can achieve a high AUC, and any activation probe evaluated on such a split reports the confounded sum of two quantities—memorization and distribution shift—rather than an isolated memorization signal.
- **Supporting Chunks**: `2608.12652_sec_005`
- **Fact Explanations & Semantic Necessity**: Single passage 2608.12652_sec_005 explains why temporal splits confound probe memorization with distribution shift.

### Family v2_train_008 (Paper: `2608.12698` | Verified Type: Multi-Passage)

- **Intended Type**: `multi_passage` | **Verified Type**: `multi_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: How does Class-Geometry Supervision (CGS) structurally depart from prior prototype-based few-shot object detectors regarding relational constraints among prototypes, and what do ablation studies reveal regarding meaningful visual geometry versus random geometry?
- **Reference Answer**: Prior prototype-based few-shot detectors optimize class prototypes independently from support examples without constraining the relational geometry among the prototypes themselves, whereas CGS explicitly supervises the prototype space by aligning pairwise distances among learned class representations with a target class-geometry matrix estimated from training data. Ablation studies show that meaningful visual geometry provides the most reliable performance gains in sample efficiency and detection, whereas random geometry can assist novel class separation but is less consistent for few-shot detection.
- **Supporting Chunks**: `2608.12698_sec_007, 2608.12698_sec_001`
- **Fact Explanations & Semantic Necessity**: Passage 2608.12698_sec_007 explains the relational prototype constraint departure, while Passage 2608.12698_sec_001 reports the empirical ablation comparison between visual and random geometry.

### Family v2_train_009 (Paper: `2608.12713` | Verified Type: Single-Passage)

- **Intended Type**: `single_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: What security vulnerability arises from using highly edit-robust LLM watermarks alone (termed piggyback spoofing), and how does the Cocktail watermarking scheme resolve it to support three-state detection?
- **Reference Answer**: When an LLM watermark is designed to be highly robust to edits, an adversary can alter critical content while retaining the watermark attribution signal, making the scheme endorse content the model did not generate (piggyback spoofing). Cocktail resolves this by co-embedding two complementary signals into each generated token—a robust signal using a short seeding window and a fragile signal using a long seeding window—sharing the same mechanism but with independent keys. At detection, their joint scores form a two-dimensional space supporting three states: Intact, Tampered, and No-Watermark.
- **Supporting Chunks**: `2608.12713_sec_001`
- **Fact Explanations & Semantic Necessity**: Single passage 2608.12713_sec_001 defines piggyback spoofing and specifies the co-embedded robust/fragile dual-signal Cocktail architecture.

### Family v2_train_010 (Paper: `2608.12714` | Verified Type: Single-Passage)

- **Intended Type**: `multi_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Empirically reclassified from multi_passage to single_passage per ChatGPT decision: a single supplied passage was confirmed sufficient to answer the complete question.
- **Question**: Why do conventional Open-World Object Detection (OWOD) and Sparsely Annotated Object Detection (SAOD) frameworks fail in sparsely annotated open environments, and how do the KTRM and DDTG modules in the DPOD framework resolve the dual ambiguity of unlabeled regions?
- **Reference Answer**: Conventional OWOD assumes densely annotated data (so unlabeled known objects obscure boundaries between known and unknown categories), while SAOD operates under a closed-world assumption where unseen novel categories are implicitly treated as background. In the DPOD framework, KTRM recovers missing known objects to stabilize pseudo-label learning and refine decision boundaries, while DDTG discovers additional unknown instances via cross-view semantic disagreement.
- **Supporting Chunks**: `2608.12714_sec_007, 2608.12714_sec_005`
- **Fact Explanations & Semantic Necessity**: Passage 2608.12714_sec_007 explains why prior OWOD and SAOD fail under sparse annotations, while Passage 2608.12714_sec_005 specifies the roles of KTRM (known recovery) and DDTG (unknown discovery).

### Family v2_train_011 (Paper: `2608.12717` | Verified Type: Single-Passage)

- **Intended Type**: `single_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Verified matches intended type.
- **Question**: In the PRISM framework for studying naming-error dissociations across transformers and post-stroke aphasia patients, how does the contrast operator differ between the LLM and cortical substrates, and what specific anatomical and layer clusters were recovered in the phonemic-favoring dissociation?
- **Reference Answer**: In PRISM, the contrast operator differs between substrates: it is a within-subject error-proportion difference for the LLM, whereas it is a between-subject correlation difference for the cortex. Both substrates recover a replicating phonemic-favoring dissociation, located in a deep layer cluster for the LLM and a frontal-perisylvian cortical cluster for post-stroke aphasia patients.
- **Supporting Chunks**: `2608.12717_sec_002`
- **Fact Explanations & Semantic Necessity**: Single passage 2608.12717_sec_002 specifies both the substrate contrast operator difference and the recovered phonemic-favoring cluster locations.

### Family v2_train_012 (Paper: `2608.12748` | Verified Type: Single-Passage)

- **Intended Type**: `multi_passage` | **Verified Type**: `single_passage`
- **Classification Note**: Empirically reclassified from multi_passage to single_passage per ChatGPT decision: a single supplied passage was confirmed sufficient to answer the complete question.
- **Question**: How does the visual grounding framework in paper 2608.12748 combine discriminative grounding with reconstructive regularization during training without incurring inference overhead, and how does the Modulated Attention-Contrastive Head (mACH) reformulate query-text interaction to accelerate multi-query inference?
- **Reference Answer**: The framework jointly supervises visual features during training with a discriminative Modulated Attention-Contrastive Head (mACH) and an auxiliary reconstructive JEPA stream, where the JEPA branch is removed after training, adding zero inference cost. To accelerate multi-query inference, mACH broadcasts flattened visual feature maps along the batch dimension (pairing each visual map with every referring expression in an expanded batch) so that a single visual forward pass interacts with all candidate expressions simultaneously.
- **Supporting Chunks**: `2608.12748_sec_007, 2608.12748_sec_008`
- **Fact Explanations & Semantic Necessity**: Passage 2608.12748_sec_007 explains the dual-objective training and inference-free JEPA removal, while Passage 2608.12748_sec_008 provides the mACH batch-broadcasting formula for multi-query inference.

## 4. Cross-Tabulation of Corrected Annotations

| Experimental Condition | Sufficient | Insufficient | Ambiguous | Total |
| :--- | :---: | :---: | :---: | :---: |
| `duplicate_replacement` | 3 | 9 | 0 | 12 |
| `natural_retrieval` | 11 | 1 | 0 | 12 |
| `relevant_replacement` | 4 | 8 | 0 | 12 |
| `supported_candidate` | 12 | 0 | 0 | 12 |
| **Total** | **30** | **18** | **0** | **48** |

## 5. Three Genuinely Topical Replacements & Semantic Judgments

Below are 3 representative examples of substantive, non-boilerplate topical replacements:

### Example 1: DexterSQL (`ctx_tr_0029`, Family `v2_train_002`, `relevant_replacement`)

- **Replacement Passage**: `2608.11889_sec_002` (Abstract of DexterSQL).
- **Topical Relationship**: Discusses the DexterSQL system and reports its BIRD-Dev execution accuracy of 67.6% using GPT-OSS-120B.
- **Semantic Outcome**: `SUFFICIENT`. Because chunk 1 (`2608.11889_sec_009`) supplied the `Patient.Diagnosis` vs `Examination.Diagnosis` data distribution distinction, and the replacement chunk supplied the 67.6% EX metric, the two passages together completely answered the question.
- **Derivable Answer**: Resolving schema ambiguity between Patient.Diagnosis and Examination.Diagnosis requires inspecting data distributions because column names and overlapping values do not clarify the distinction: Patient.Diagnosis has one row per patient (recording a single final diagnosis), whereas Examination.Diagnosis can have multiple rows per patient. On the BIRD-Dev benchmark, DexterSQL achieves an execution accuracy (EX) of 67.6% using the open-weight GPT-OSS-120B model.

### Example 2: LazyTrain (`ctx_tr_0022`, Family `v2_train_003`, `relevant_replacement`)

- **Replacement Passage**: `2608.11919_sec_039` (LazyTrain Methodology / Appendix Case Study).
- **Topical Relationship**: Discusses the Qwen3.6-27B run on MetaMathQA at batch size 72, sequence length 1024, and PySCIPOpt MILP scheduling.
- **Semantic Outcome**: `INSUFFICIENT`. While Passage 2 explained the Hybrid 8-bit operator, Passage 1 discussed the software stack and MILP scopes rather than the sustained TFLOPS (219.95), throughput (1361 tokens/s), and peak GPU memory (68.84 GB).
- **Missing Information**: The context lacks the sustained TFLOPS (219.95 TFLOPS), throughput (1361 tokens/s), and peak GPU memory (68.84 GB) achieved in the Qwen3.6-27B H800 MetaMathQA run at batch size 72.

### Example 3: Class-Geometry Supervision (`ctx_tr_0035`, Family `v2_train_008`, `relevant_replacement`)

- **Replacement Passage**: `2608.12698_sec_023` (CGS Methodology / Discussion).
- **Topical Relationship**: Discusses adapting the full detector and trading known-class discrimination for unknown discovery under CGS.
- **Semantic Outcome**: `INSUFFICIENT`. While Passage 2 explained the structural departure from prior prototype anchors, Passage 1 discussed full detector adaptation rather than the ablation findings comparing meaningful visual geometry against random geometry.
- **Missing Information**: The context lacks the ablation study findings comparing meaningful visual geometry versus random geometry.

## 6. Changed-Condition Mapping and Payload Hash Tracking

All 48 contexts were regenerated and re-scored with new payload hashes and shuffled opaque IDs. The full mapping is preserved in `scratch/benchmark_v2_batch01_shuffled_blind_mapping.json`.

Initial Batch 01 artifacts have been archived under `data/metadata/audit_archive/batch01_initial/` for complete experimental provenance.
