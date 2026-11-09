# Scientific Corpus Redistribution Provenance & Licensing Disclosure

**Repository:** `ai-research-copilot`  
**Evaluation Scope:** Benchmark v2 (Evidence-Sufficiency Gate & Selective Answering Evaluation)  
**Date:** November 2026

---

## 1. Provenance of Evaluated Text Chunks

The evaluation benchmark in this repository evaluates evidence sufficiency across 160 scientific papers partitioned into:
- **Training pool:** 80 papers (240 contexts across 60 question families)
- **Validation pool:** 32 papers (96 contexts across 24 question families)
- **Test pool:** 48 papers (144 contexts across 36 question families)

All 160 papers are open-access research preprints retrieved from [arXiv.org](https://arxiv.org/) across Computer Science subject categories including Computation and Language (`cs.CL`), Computer Vision (`cs.CV`), Artificial Intelligence (`cs.AI`), Machine Learning (`cs.LG`), and Information Retrieval (`cs.IR`).

Complete bibliographic metadata for every evaluated paper (including arXiv preprint identifier, title, primary subject category, and split allocation) is documented in:
- `data/metadata/full_benchmark_split_manifest.json`
- `data/metadata/papers_corpus.json`

---

## 2. Nature of Included Data

The repository includes:
1. **Passage Excerpts:** Short text passages (~150–300 words each, formatted into contexts with at most 4 passages for controlled conditions and up to 6 passages within a 2,000-token ceiling for natural retrieval) serialized inside `data/metadata/benchmark_v2_*_contexts_*.json`.
2. **Cryptographic Checksums:** Exact SHA-256 hashes of question strings, passage payload concatenations, and model parameter vectors to ensure deterministic reproducibility.
3. **Verbatim Quotations:** Exact substring quotations identifying the specific sentences supporting answers to the evaluated questions.

**Full paper PDF files, complete unchunked document bodies, and binary vector database indices are intentionally excluded from the Git repository** (`data/pdfs/` and `data/chroma_db/` are ignored in `.gitignore`).

---

## 3. Licensing Disclosure & Material Limitations

### arXiv Licensing Structure
Authors submitting preprints to arXiv select from several license agreements:
- **arXiv.org Perpetual Non-Exclusive License:** The author grants arXiv an irrevocable, perpetual license to distribute the article, but retains full copyright without granting blanket redistribution rights to third parties.
- **Creative Commons Licenses:** Some submissions are distributed under CC BY 4.0, CC BY-SA 4.0, CC BY-NC-SA 4.0, or CC0 1.0.

### Redistribution Status & Honest Disclosure
- An exhaustive, individual copyright and license review verifying third-party redistribution permissions for all 160 arXiv source preprints was **not** conducted.
- We do **not** claim or imply that blanket redistribution approval or CC-BY licensing exists for all paper texts excerpted in the benchmark.
- The included passage excerpts are provided strictly for **non-commercial academic research reproducibility** under principles of fair use (U.S. Copyright Act, 17 U.S.C. § 107) and fair dealing (Indian Copyright Act, § 52(1)(a) for research and private study):
  - Purpose: Non-commercial empirical evaluation of duplicate-aware selective answering.
  - Amount: Only limited, localized text excerpts essential for cryptographic verification of sufficiency decisions are included; full documents are omitted.
  - Market Effect: Inclusion of evaluation excerpts does not substitute for the primary academic preprints hosted on arXiv.

---

## 4. Reproducible Direct Acquisition Alternative

For evaluators, researchers, or organizations operating under strict data governance policies that disallow evaluating bundled text excerpts without explicit per-document CC licenses, a fully reproducible alternative is provided to acquire the corpus directly from primary arXiv sources:

1. **Direct Download via arXiv Identifiers:**  
   Every paper's authoritative arXiv ID is cataloged in `data/metadata/full_benchmark_split_manifest.json`. Any paper can be retrieved directly:
   ```
   https://arxiv.org/abs/<arxiv_id>
   https://arxiv.org/pdf/<arxiv_id>.pdf
   ```

2. **Automated Corpus Ingestion:**  
   The repository includes automated acquisition and parsing scripts:
   ```powershell
   # Download genuine PDFs directly from arXiv
   python scripts/download_real_corpus.py

   # Parse PDFs into section-aware chunks locally
   python backend/ingestion/pdf_parser.py
   ```

3. **Re-embedding and Verification:**  
   Users can populate their own local vector database using `backend/ingestion/vector_store.py` without relying on any distributed text files.

---

## 5. Summary Table of Split Allocation & Provenance

| Split | Paper Count | Context Count | Question Families | Source Repository | Metadata Record |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Train** | 80 | 240 | 60 | arXiv.org (cs.CL, cs.CV, cs.LG) | `data/metadata/benchmark_v2_training_hash_manifest.json` |
| **Validation** | 32 | 96 | 24 | arXiv.org (cs.CL, cs.CV, cs.AI) | `data/metadata/benchmark_v2_validation_hash_manifest.json` |
| **Test** | 48 | 144 | 36 | arXiv.org (cs.CL, cs.CV, cs.IR) | `data/metadata/benchmark_v2_test_hash_manifest.json` |
| **Total** | **160** | **480** | **120** | **arXiv.org Open Preprints** | `data/metadata/full_benchmark_split_manifest.json` |
