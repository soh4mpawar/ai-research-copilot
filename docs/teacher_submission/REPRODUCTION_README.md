# AI Research Copilot: Reproduction Guide & Artifact Manifest

**Protocol:** Research Protocol v1 (Amendment v1.1)  
**Execution Environment:** Windows 11, Python 3.10, PyTorch 2.5.1+cu121, NVIDIA RTX 3070 Ti Laptop GPU  
**Commercial API Spend:** **$0.00** (Strictly zero paid APIs invoked)

---

## 1. Quick Reproduction Commands

All evaluation code is deterministic and runnable from the repository root:

```powershell
# Navigate to the project directory
cd ai-research-copilot

# 1. Run focused unit tests for evaluation functions (reconstruction, tie-handling, bootstrap)
python -m unittest tests.test_evaluation_functions -v

# 2. Run the dedicated frozen-model test evaluation runner
python evaluation/run_benchmark_v2_test_eval.py
```

Expected Execution Time: ~40 seconds on local CUDA GPU.

---

## 2. Authoritative Artifacts & SHA-256 Hashes

Prior to test evaluation, all input datasets and model weights are cryptographically verified against their authoritative SHA-256 hashes:

| Artifact Name | Path | Authoritative SHA-256 Hash | Description |
| :--- | :--- | :--- | :--- |
| **Test Contexts (144)** | `data/metadata/benchmark_v2_test_contexts_144.json` | `a6b4cdf4b493d1d930d556238b1c0f273529175f3c1e773ae00f6b94402cd230` | 144 test contexts across 36 families and 4 conditions |
| **Test Questions (36)** | `data/metadata/benchmark_v2_test_questions_36families.json` | `a9c3f59ba6bc82ce8504c48346c4beaec3da2996ac30545ac88b4770df13a141` | 36 question specs (25 single-passage, 11 multi-passage) |
| **Test Annotations (144)** | `data/metadata/benchmark_v2_test_semantic_annotations_144.json` | `fcd815fc2f7d9c9bc23a9293462a41527c94a7da7df2dc34e7c58a76f2bf16f4` | 144 consolidated semantic annotations (196 verified quotes) |
| **Fitted Models** | `data/metadata/benchmark_v2_fitted_models.json` | `6047be23e2985957f31a596f0a3837c977fc6cd83b812f6d9924fadd98617689` | Frozen scalers, coefficients, intercepts, and validation thresholds |
| **Evaluation Spec** | `data/metadata/benchmark_v2_frozen_evaluation_specification.json` | `71c1ee7e9b732f2f2d978fe771ec3e3bf786d7eb631fbb91fea0c63071b0bf42` | Frozen evaluation protocol, seed 20261104, cluster unit |
| **Training Contexts (240)**| `data/metadata/benchmark_v2_training_contexts_240.json` | `884f33b0086a5163cef6181065bafa1e8c4caba2ecb19f3267d59bc8a165b7eb` | 240 training contexts across 60 families |
| **Validation Contexts (96)**| `data/metadata/benchmark_v2_validation_contexts_96.json` | `832ea363dcb242dc60f5c7bb70f11ecc09d2a41e093c1835e3c8b4af853c6a61` | 96 validation contexts across 24 families |

---

## 3. Generated Machine-Readable Outputs

The test runner generates the following verified machine-readable evaluation outputs:

- **Context-Level Predictions (144):** `data/metadata/benchmark_v2_test_predictions.json`  
  Contains exact gating probabilities, decisions, and metadata for each context across all 8 models.
- **Evaluation Summary:** `data/metadata/benchmark_v2_test_eval_results.json`  
  Contains discrete AURC, operating points at validation-frozen thresholds (0.50, 0.75, 0.90), and duplicate invariance results.
- **Bootstrap Distributions (2,000 Resamples):** `data/metadata/benchmark_v2_test_bootstrap_results.json`  
  Contains paired bootstrap differences, standard errors, and 95% percentile intervals for all comparisons.
- **Risk-Coverage Plots:**
  - Standalone SVG: `data/metadata/risk_coverage_curve_v2_test.svg`
  - Interactive HTML: `data/metadata/risk_coverage_curve_v2_test.html`
  - High-Resolution PNG: `data/metadata/risk_coverage_curve_v2_test.png`

---

## 4. Key Empirical Results Reference Table

### Controlled Contexts ($N = 108$, Insufficient Prevalence = 66.7%)
| Model | Discrete AURC | Frozen Cov@0.50 (Risk) | Frozen Cov@0.75 (Risk) | Frozen Cov@0.90 (Risk) |
| :--- | :---: | :---: | :---: | :---: |
| **B0 (Always Accept)** | 0.6667 | 1.000 (0.667) | 1.000 (0.667) | 1.000 (0.667) |
| **B1 (Max Relevance)** | **0.4049** | 0.315 (0.294) | 0.472 (0.372) | 0.852 (0.609) |
| **B2 (Historical Gate)** | *N/A (Discrete)* | Cov=0.528 | Risk=0.404 | Type I=0.319, Type II=0.056 |
| **M1 (Relevance Raw)** | 0.4329 | 0.250 (0.259) | 0.491 (0.377) | 0.824 (0.596) |
| **M2 (Relevance Dedup)** | 0.4373 | 0.232 (0.520) | 0.481 (0.385) | 0.824 (0.596) |
| **M3 (Full Structure Raw)** | 0.4658 | 0.352 (0.395) | 0.491 (0.377) | 0.907 (0.633) |
| **M4 (Full Structure Dedup)**| 0.4400 | 0.222 (0.500) | 0.546 (0.424) | 0.722 (0.538) |
| **Shortcut (Length Diag.)** | 0.6798 | 0.056 (0.500) | 0.296 (0.656) | 0.759 (0.683) |

### Paired Cluster Bootstrap Comparisons (2,000 Resamples, Seed `20261104`)
| Comparison Pair | Observed $\Delta$ | Bootstrap Mean | Bootstrap SE | 95% Percentile CI | Hypothesis Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **M4 − M1 (Primary)** | **+0.0071** | **+0.0092** | **0.0305** | **$[-0.0503, +0.0679]$** | **Not Supported (Contains 0)** |
| **M2 − M1 (Ablation 1)** | +0.0044 | +0.0052 | 0.0283 | $[-0.0520, +0.0603]$ | Contains 0 |
| **M3 − M1 (Ablation 2)** | +0.0329 | +0.0352 | 0.0228 | $[-0.0058, +0.0846]$ | Contains 0 |
| **M4 − M2 (Ablation 3)** | +0.0027 | +0.0041 | 0.0117 | $[-0.0199, +0.0270]$ | Contains 0 |
| **M4 − M3 (Ablation 4)** | -0.0258 | -0.0260 | 0.0334 | $[-0.0939, +0.0363]$ | Contains 0 |
| **B1 − M1 (Baseline)** | -0.0281 | -0.0253 | 0.0243 | $[-0.0668, +0.0267]$ | Contains 0 |
| **Length − M4 (Shortcut)** | +0.2398 | +0.2410 | 0.0350 | $[+0.1722, +0.3081]$ | Favors M4 ($p < 0.001$) |
