# Benchmark v2 Training Split Audit: 60 Families (240 Contexts)

**Protocol Version**: Protocol Amendment v1.1  
**Provenance Status**: `INDIVIDUALLY_AUTHORED_SEMANTIC_ANNOTATIONS`  
**Execution Date**: November 4, 2026  
**Status**: Frozen & Verified (Ready for ChatGPT Inspection)  

---

## 1. Executive Summary & Methodology

Under Protocol Amendment v1.1, the full 60-family training split (240 controlled contexts) for the research project **"Duplicate Aware Source Aggregation for Selective Scientific Question Answering"** has been completely generated, cleaned of reference/boilerplate content, individually annotated through blind semantic review, verified for quotation integrity, and consolidated with full hash manifest tracking.

### Crucial Methodological Safeguards Maintained
1. **Zero Programmatic Label Assignment**: No regex heuristics, substring presence checks, or condition-tag lookups were used to assign semantic sufficiency labels. Every single context was evaluated by directly reading the metadata-concealed question and passage text in shuffled blind packets.
2. **Deterministic Shuffling & Metadata Concealment**: Every batch (01 through 05) was deterministically shuffled with fixed seeds to generate opaque IDs (`ctx_tr_0001` through `ctx_tr_0240`), completely concealing family IDs, source paper IDs, condition tags, and candidate scoring.
3. **100% Verbatim Quotation Verification**: Every sufficient judgment is supported by character-level verbatim quotations strictly checked via Python assertions against the exact context passage text.
4. **Authoritative Hash-Based Joins**: Consolidation joins are audited and verified on `(family_id, condition, question_hash, payload_hash)`. Zero stale annotations, zero duplicate keys, and zero missing contexts.
5. **Zero Paid API Spend**: Total commercial API cost across all 60 families is **$0.00**.

---

## 2. Frozen Dataset Architecture & Provenance Ledger

| Batch | Families Covered | Context ID Range | Authored / Reused Status | Verified Quotations | Sufficient | Insufficient | Ambiguous |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Batch 01** | `v2_train_001` – `v2_train_012` | `ctx_tr_0001` – `ctx_tr_0048` | Reused (100% hash match) | 38 | 30 (62.5%) | 18 (37.5%) | 0 (0.0%) |
| **Batch 02** | `v2_train_013` – `v2_train_024` | `ctx_tr_0049` – `ctx_tr_0096` | Reused (100% hash match) | 35 | 27 (56.25%) | 21 (43.75%) | 0 (0.0%) |
| **Batch 03** | `v2_train_025` – `v2_train_036` | `ctx_tr_0097` – `ctx_tr_0144` | Newly Reviewed & Verified | 30 | 24 (50.0%) | 24 (50.0%) | 0 (0.0%) |
| **Batch 04** | `v2_train_037` – `v2_train_048` | `ctx_tr_0145` – `ctx_tr_0192` | Newly Reviewed & Verified | 42 | 24 (50.0%) | 24 (50.0%) | 0 (0.0%) |
| **Batch 05** | `v2_train_049` – `v2_train_060` | `ctx_tr_0193` – `ctx_tr_0240` | Newly Reviewed & Verified | 27 | 27 (56.25%) | 21 (43.75%) | 0 (0.0%) |
| **TOTAL** | **60 Families** | **ctx_tr_0001 – ctx_tr_0240** | **96 Reused + 144 New** | **172 Quotes** | **132 (55.0%)** | **108 (45.0%)** | **0 (0.0%)** |

---

## 3. Condition Breakdown & Empirical Distribution

Each of the 60 families contains exactly 4 controlled contexts representing the 4 experimental conditions (60 contexts per condition):

| Condition Variant | Total Contexts | Sufficient | Insufficient | Sufficiency Rate | Empirical Behavior & Findings |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`supported_candidate`** | 60 | 59 | 1 | 98.3% | Contains the designated gold evidence chunk(s). 59/60 provide complete factual answers; 1 context (`v2_train_019` in Batch 02) lacked complete cross-passage definition. |
| **`relevant_replacement`** | 60 | 9 | 51 | 15.0% | Designed to test topic distractor robustness. 85.0% correctly lack the required facts; in 9 cases, the substantive replacement chunk from the same paper (e.g., abstract or methodology overview) genuinely contained the required answer. |
| **`duplicate_replacement`** | 60 | 7 | 53 | 11.7% | Replaces supporting evidence with duplicate copies of other passages or replaces partial evidence. 88.3% are insufficient; 7 remain sufficient because single surviving passages were individually sufficient. |
| **`natural_retrieval`** | 60 | 57 | 3 | 95.0% | Retrieved via the frozen split-restricted hybrid pipeline (Dense + BM25s + RRF + BGE Cross-Encoder). High sufficiency (95.0%) confirms strong retrieval quality, with 3 natural retrieval failures where retrieved passages missed exact fine-grained details (e.g., longitudinal MRI registration tool). |
| **OVERALL TOTAL** | **240** | **132** | **108** | **55.0%** | **Natural, realistic distribution without artificial 50/50 forcing.** |

---

## 4. Six Representative Semantic Judgments (Batches 03, 04, 05)

### Representative Judgment 1 (Batch 03: `ctx_tr_0100` — Sufficient)
- **Question**: *In the study of LLM reasoning with code-derived formal schemas, what three downstream benchmarks were evaluated across the three evaluation categories?*
- **Judgment**: `sufficient`
- **Derivable Answer**: *Across the three evaluation categories, the three downstream benchmarks evaluated were: (1) SVAMP for complex mathematical reasoning, (2) GSM8K for multi-step grade school math reasoning, and (3) HumanEval for code generation.*
- **Supporting Quotation** (Passage 1): `"We evaluate our approach on three diverse downstream benchmarks spanning different reasoning categories: SVAMP [28] for complex mathematical reasoning, GSM8K [29] for multi-step grade school math, and HumanEval [30] for code generation."`
- **Rationale**: Passage 1 explicitly lists the three benchmarks and directly pairs each with its corresponding evaluation category.

### Representative Judgment 2 (Batch 03: `ctx_tr_0105` — Insufficient)
- **Question**: *What are the three structural components of the CMap model for learning in spatial time series?*
- **Judgment**: `insufficient`
- **Derivable Answer**: `""`
- **Missing Information**: *Context presents general Gaussian process regression background and algorithmic comparisons, but omits the three explicit structural components defining the CMap architecture (e.g., local kernel, global drift, spatial covariance mapping).*
- **Rationale**: Passages 1 and 2 discuss online learning and autocovariance matrices in related work without detailing the architectural decomposition of CMap. Passages 3 and 4 are unrelated distractors.

### Representative Judgment 3 (Batch 04: `ctx_tr_0152` — Sufficient)
- **Question**: *In the hybrid Concept Bottleneck Model, what is the overall training objective in Eq. 2 and what does each constituent loss term optimize?*
- **Judgment**: `sufficient`
- **Derivable Answer**: *In the hybrid Concept Bottleneck Model, the overall training objective in Eq. 2 is L = L_diag(D) + lambda_c L_concept(A) + lambda_d L_dist(U) + lambda_r L_reg. L_diag is the weighted binary cross-entropy for malignancy on all samples D; L_concept is binary cross-entropy on annotated concept samples A; L_dist aligns distributions on unannotated samples U; and L_reg regularizes parameters.*
- **Supporting Quotation** (Passage 4): `"L=L diag(D) +λcLconcept(A) +λdLdist(U) +λrLreg.(2)"` and `"Ldiag is the weighted binary cross-entropy for malignancy in all samples.Lconcept\nis binary cross-entropy with log its between zi and the image-level concept labels\nci, evaluated only for samples inA. Ldist is illustrated in Section 2.2."`
- **Rationale**: Passage 4 explicitly provides Eq. 2 and defines all constituent loss terms.

### Representative Judgment 4 (Batch 04: `ctx_tr_0151` — Insufficient)
- **Question**: *What are the four evaluation aspects and the scoring scale used in LigBench for evaluating research ideas generated by LLMs?*
- **Judgment**: `insufficient`
- **Derivable Answer**: `""`
- **Missing Information**: *Context discusses LLM idea evaluation motivation in duplicate Passages 1 and 2, but omits the four specific evaluation aspects (rating, contribution, soundness, novelty) and the 0-5 scoring scale used in LigBench.*
- **Rationale**: Passages 1 and 2 duplicate general benchmark motivation. Passages 3 and 4 discuss FineX action recognition and TRAPSBench epistemic calibration.

### Representative Judgment 5 (Batch 05: `ctx_tr_0214` — Sufficient)
- **Question**: *What registration tool was used in the longitudinal MRI preprocessing pipeline, and to what neurological convention were all volumes reoriented?*
- **Judgment**: `sufficient`
- **Derivable Answer**: *In the longitudinal MRI preprocessing pipeline, serial scans were registered to baseline using ANTs (Advanced Normalization Tools), and all volumes were reoriented to RAS+ neurological convention.*
- **Supporting Quotation** (Passage 1): `"Serial scans were registered to baseline using ANTs [2] (deformable for most cohorts, rigid for geometrically consistent acquisitions). All volumeswerereorientedtoRAS+neurological convention to ensure standardized\nanatomical presentation."`
- **Rationale**: Passage 1 explicitly documents that serial scans were registered using ANTs and all volumes were reoriented to RAS+ neurological convention.

### Representative Judgment 6 (Batch 05: `ctx_tr_0234` — Sufficient)
- **Question**: *In ProME's problem formulation, how is an endogenous environment construction distinguished from an exogenous construction?*
- **Judgment**: `sufficient`
- **Derivable Answer**: *In ProME's problem formulation, an exogenous environment construction reads partitions off a detached reference encoder trained beforehand, which reflects that encoder's geometry rather than the downstream representation, causing persistent environment–representation mismatch; in contrast, an endogenous construction draws the partition from an encoder checkpoint on the training trajectory that it later regularizes.*
- **Supporting Quotation** (Passage 1): `"To supply the missing assignments, a\nnatural strategy is to read them off a detached reference\nencoder𝜙 ref trained beforehand. However, a partition fixed\nby𝜙 ref reflects the geometry of that encoder, not that of\nthe representation trained downstream. Thisenvironment–\nrepresentation mismatch then persists for the rest of training. Wecallsuchaconstructionexogenous,and reserve endogenous for a partition drawn from a checkpoint on the trajectory\nthat it later regularizes."`
- **Rationale**: Passage 1 explicitly defines and contrasts exogenous and endogenous environment constructions in invariant representation learning.

---

## 5. Frozen Artifacts & Checksums

| Artifact Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `data/metadata/full_benchmark_split_manifest.json` | `7f7005c89813266a2826cc7a707c3a6101fbcf158a3796ea97578d422f044b2c` | Authoritative paper split allocations (80 train, 32 val, 48 test) |
| `data/metadata/benchmark_v2_training_questions_60families.json` | `53fc5b147320c2a145f3e1af4efc802d349644db5fdfaa942c5a46c393a1063a` | 60 individually validated question specifications |
| `data/metadata/benchmark_v2_batch01_semantic_annotations.json` | `8e7ef1dd5419e1d7994396ee24f98bfba03883ffeff0aa83459ae50ee7477177` | Batch 01 verified semantic annotations (48 contexts) |
| `data/metadata/benchmark_v2_batch02_semantic_annotations.json` | `ba7dc70e1b1bddcaa409c6a24ed498c1c336df1345f04521f6d10ae27cd41e71` | Batch 02 verified semantic annotations (48 contexts) |
| `data/metadata/benchmark_v2_batch03_semantic_annotations.json` | `e5186c8ff0a5ed9bf9ffe4edf177838ee675cbe0683357473f7425491be43844` | Batch 03 verified semantic annotations (48 contexts) |
| `data/metadata/benchmark_v2_batch04_semantic_annotations.json` | `13cdab741f35ee8b93635898aa74495e8ff7a776687339a33d2b0a66567d1eee` | Batch 04 verified semantic annotations (48 contexts) |
| `data/metadata/benchmark_v2_batch05_semantic_annotations.json` | `628ca9defbc135d559e3d84f4ae093225f4981c75043ce32cfdb0b71be22e5e0` | Batch 05 verified semantic annotations (48 contexts) |
| `data/metadata/benchmark_v2_training_contexts_240.json` | `16a74e1912fad9432252d75791b66a48959d5e409c1c483fa8ea1e64465e6566` | Consolidated 240 controlled contexts (reference-free) |
| `data/metadata/benchmark_v2_training_semantic_annotations_240.json` | `712f8e5f6c5dc9b4c5c3d2d49b5b4f98ac6358eaeefab4faadf04d7815fdfff8` | Consolidated 240 authoritative semantic reviews |
| `data/metadata/benchmark_v2_training_hash_manifest.json` | `51171f3055edcd2d20934acdf031b1033d126d6ae76d4761d7ed9a117c6921e8` | Full record-level join and hash audit manifest |

---

## 6. Test Suite Status

- **Unit Test Execution**: `python -m unittest discover tests`
- **Result**: `Ran 43 tests in 191.916s. OK (expected failures=1)`
- **Retrieval Adapter Test**: `tests/test_natural_retrieval_adapter.py` ran 7 focused tests including progressive retrieval expansion past reference blocks, all passing with zero errors.

---

## 7. Commitments & Boundaries

- **Commercial API Spend**: $0.00 spent.
- **Model Fitting**: Zero models fitted on the training split in this stage.
- **Evaluation / Test Set**: No validation or test split contexts constructed or evaluated.
- **Next Stage**: Awaiting ChatGPT inspection and explicit authorization before proceeding to model fitting or evaluation.
