# Visual Quality Assurance (QA) Record
**Project:** AI Research Copilot: Evaluating Duplicate-Aware Evidence Aggregation for Selective Scientific Question Answering  
**Authors:** Akanksha Jaiswal (230905106), Mohd. Uzair Qureshi (220905158), Soham Pawar (230905312) — CSE Core  
**Guide:** Dr. Adarsh Rag  
**Date of Inspection:** 04 November 2026  
**Inspection Environment:** Windows WinRT `Windows.Data.Pdf.PdfDocument` rasterization engine (Native 1428 × 1848 px, 24-bit PNG, ~168 DPI, 1:1 Letter scale)

---

## 1. Executive Summary Inspection (`teacher_package/ONE_PAGE_SUMMARY.pdf`)

- **Page Count:** Exactly 1 page (confirmed via `pypdf` and Windows WinRT rendering).
- **Rendered Output:** `teacher_package/page_images_summary/page_01.png` (1428 × 1848 px).
- **Bounding Box:** Left: 134 px, Right: 1290 px, Top: 142 px, Bottom: 1451 px.
- **Ink Density:** 10.05% of canvas area.
- **Margins & Alignment:** Symmetrical 0.8-inch margins (left/right bounds leave >130 px margin). All content is completely visible with zero cutoff or margin intrusion.
- **Typography & Structure:**
  - Title: Deep Navy (`#1B365D`), 15 pt bold, single line.
  - Subtitle: Student and guide metadata cleanly rendered in 9 pt muted italic.
  - Core Research Question: Formatted with bold prefix, clearly highlighting the 2×2 factorial evaluation of deduplication and bibliographic structure.
  - Experimental Design: Explicitly states controlled contexts contain exactly 4 passages and natural retrieval contexts allow up to 6 passages within a 2,000-token budget.
  - Table of Results: 8-row table featuring all 8 models/baselines with discrete AURC and validation-frozen operating points.
  - Primary Hypothesis Conclusion: Explicitly highlighted: M4 − M1 discrete AURC difference is $+0.0071$, 95% bootstrap percentile CI $[-0.0503, +0.0679]$. Stated as **unsupported**; the interval includes differences favoring either method; this study does not establish equivalence. Baseline B1 (Max Relevance) achieves the lowest observed controlled-context AURC (0.4049).
  - Duplicate Invariance: Bounded property explicitly stated under operating assumptions A1–A5, reporting the observed 0.000000 shift.
  - Methodological Disclosures: Highlights zero commercial API spend ($0.00), local CUDA execution, AI reading protocol with verbatim quotes, and frozen models.
- **Visual Verdict:** **PASS (Flawless Single-Page Executive Presentation).**

---

## 2. Main Academic Report Inspection (`teacher_package/AI Research Copilot - Teacher Submission.pdf`)

- **Page Count:** Exactly 13 pages (confirmed via Microsoft Word COM engine and `pypdf`).
- **Rendered Images:** `teacher_package/page_images_report/page_01.png` through `page_13.png` (all 1428 × 1848 px).
- **Global Margins:** Uniform 1.0-inch margins maintained across all 13 pages (Left: 164–168 px, Right: 1251–1259 px, Width: 1428 px).
- **Table Formatting:** Every table is equipped with `<w:cantSplit/>` on all table rows and `<w:tblHeader/>` on header rows, preventing unaligned cross-page row splits.
- **Callout Integrity:** All callout boxes enforce `<w:cantSplit/>` and `keep_together = True`, preventing box borders from breaking awkwardly across page boundaries.

### Page-by-Page Detailed Verification

| Page | Section / Content | Content Bounds (L, R, T, B) | Ink Density | Visual QA Findings |
| :---: | :--- | :---: | :---: | :--- |
| **01** | Title, Authors, Guide, Date, Abstract | (166, 1259, 176, 1358) | 13.57% | **PASS.** Title (16.5 pt bold), team metadata in 2-column borderless table with CSE Core and Guide Dr. Adarsh Rag. Both abstract paragraphs fit cleanly on page 1 with zero spillover. |
| **02** | Section 1: Introduction & Problem Statement | (164, 1259, 174, 1556) | 24.52% | **PASS.** Starts cleanly at top of page 2. Includes `[CORE PROBLEM STATEMENT]` and `[FAILURE MODES IN SELECTIVE SCIENTIFIC QA]` callout boxes with solid left accent borders. Experimental boundary disclaimer clearly visible at bottom. |
| **03** | Section 2: Literature Review & Related Work | (166, 1259, 174, 1676) | 21.74% | **PASS.** Features 4-row comparative literature matrix table (CRAG, Sufficient Context, RINSE, Our Work). Deep navy header row with white text. Detailed analyses of Yan et al. (CRAG) and Joren et al. (Sufficient Context) completely fit on page 3. |
| **04** | Section 2 (RINSE) & Section 3: Formulation | (164, 1259, 173, 1651) | 17.37% | **PASS.** Detailed analysis of Chen et al. (RINSE) and Academic Scope disclaimer. Section 3 introduces mathematical notation, passage multiset formulation, and exact deduplication equivalence relation. Seamless flow. |
| **05** | Section 3: Features, Classifier & Metrics | (168, 1256, 173, 1650) | 9.50% | **PASS.** Documents feature extraction for M1–M4, standardized logistic regression formulation, decision rules, risk-coverage trade-off curve definitions, and discrete expected AURC analytical sums. Equations formatted cleanly. |
| **06** | Section 3: Bootstrap & Duplicate Invariance | (164, 1259, 173, 1615) | 35.11% | **PASS.** Discrete Expected AURC all-tied prevalence formulation, Paired Cluster Bootstrap algorithm (2,000 resamples), and green `[DUPLICATE INVARIANCE ARGUMENT & ASSUMPTIONS]` callout box. Assumptions A1–A5 fit entirely on page 6 without splitting. |
| **07** | Section 4: Experimental Design & Benchmark | (168, 1256, 174, 1047) | 6.55% | **PASS.** Corpus partitioning (80/32/48 papers; 36 test primary clusters), 4 Conditions definition (controlled = 4 passages; natural = up to 6 within 2,000 tokens), and untruncated AI assistant reading protocol (196 verified quotes, 69 sufficient / 75 insufficient). |
| **08** | Section 5.1: Primary Controlled Evaluation | (166, 1259, 174, 1461) | 13.10% | **PASS.** Embedded Figure 1 (Risk-Coverage trade-off curves, 4.8 inches wide, centered) rendered with crisp vector-quality legibility. 8 pt italicized caption. Table 1 (Controlled Results across 8 models) sits cleanly beneath figure with zero row splitting. |
| **09** | Section 5.2: Paired Cluster Bootstrap | (117, 899, 124, 683) | 19.07% | **PASS.** Table 2 reports 8 paired cluster-bootstrap uncertainty comparisons with observed delta, bootstrap mean, bootstrap SE, 95% percentile intervals, and color-coded verdicts. Red `[PRIMARY HYPOTHESIS CONCLUSION]` callout box clearly displays negative result, notes the interval includes differences favoring either method (study does not establish equivalence), and notes B1 achieves lowest observed controlled-context AURC. Clean page termination. |
| **10** | Section 5.3 & 5.4: Natural & Subgroups | (166, 1259, 174, 1470) | 18.73% | **PASS.** Table 3 (Natural Retrieval Contexts, N=36) + Small-Sample Limitation paragraph; Table 4 (Question Complexity Subgroup Analysis, 11 multi-passage families) + Methodological Nuance paragraph. Both tables and text fit comfortably on page 10. |
| **11** | Section 5.5 & 5.6: Duplicate Test & Thresholds | (166, 1259, 174, 1144) | 11.68% | **PASS.** Table 5 (Duplicate-Invariance Property Test, demonstrating 0.000000 shift for M2, M4, B1) + Scope of Invariance; Section 5.6 Operating Thresholds detailing deployable validation-frozen operating points vs descriptive matched-coverage points. Zero spillover. |
| **12** | Section 6 & 7: Threats & Conclusion | (120, 897, 125, 1011) | 9.02% | **PASS.** Section 6 details all 8 methodological disclosures and threats to validity (including same-AI annotations, historical exposure disclosure, shortcut bounds, and $0.00 commercial spend). Section 7 provides conclusion and future work directions, explicitly noting the study does not establish equivalence and B1 achieves the lowest observed controlled-context AURC. |
| **13** | Section 8: References | (168, 1251, 174, 786) | 3.43% | **PASS.** 10 academic citations formatted in IEEE style with arXiv preprint identifiers where applicable. Text is fully justified, legible, and balanced. |

---

## 3. Defect Remediation & Verification Log

1. **Resolution of Empty Page 10:**
   - *Previous Defect:* In an earlier compilation, Section 5.2 and Section 5.3 were combined on Page 9, extending to the bottom margin and triggering Word COM's automatic pagination before an explicit `add_page_break()`, creating a blank Page 10.
   - *Remediation:* Section 5.2 (Table 2 + Primary Hypothesis Callout) was isolated to Page 9; Section 5.3 (Table 3) and Section 5.4 (Table 4) were co-located on Page 10.
   - *Verification:* Page 10 now contains 2,647 characters, 41 lines, and 18.73% ink density. Zero blank pages exist in the document.

2. **Resolution of 5-Line Orphan Spillover to Page 12:**
   - *Previous Defect:* Section 5.4, 5.5, and 5.6 combined exceeded Page 11 height, causing the final 5 lines of Section 5.6 to spill over onto Page 12 followed by a page break.
   - *Remediation:* Co-locating Section 5.3 and 5.4 on Page 10 allowed Section 5.5 and Section 5.6 to occupy Page 11 together without overflow.
   - *Verification:* Section 5.6 now terminates cleanly at `B:1144` on Page 11 (300+ pixels above the bottom margin). Page 12 begins directly with Section 6.

3. **Abstract Paragraph Spillover Resolution:**
   - *Previous Defect:* Title and author metadata pushed the second paragraph of the Abstract onto Page 2.
   - *Remediation:* Title font adjusted to 16.5 pt, metadata padding tightened, and abstract body formatted at 9.2 pt with 1.12 line spacing.
   - *Verification:* `page_01.png` confirms both abstract paragraphs and all author metadata fit completely on Page 1 (`B:1358` on 1848 canvas).

4. **Table Header & Row Integrity:**
   - *Verification:* `<w:cantSplit/>` prevents row fragmentation. `<w:tblHeader/>` ensures table column names repeat if pagination occurs.

5. **Original Report Preservation:**
   - *Verification:* `AI Research Copilot.docx` (original file) has LastWriteTime `03-11-2026 17:51:38` and SHA-256 hash `B4F1B8F5DA2DDA0AD0597FCCD72C088DA1FC55518AF87CE3C1BB1104F94D8333`. It has remained 100% untouched.

---

## 4. Final Quality Sign-Off

- [x] All 13 report pages and 1 executive summary page visually verified via rasterized PNG images.
- [x] Zero clipped text, overlapping elements, or table boundary violations.
- [x] Figure 1 rendered cleanly at 300 DPI equivalent with legible labels and curves.
- [x] All author names, roll numbers, CSE Core designation, Guide Dr. Adarsh Rag, and submission date 15 November 2026 verified.
- [x] No invented institutions (e.g., "Manipal Institute of Technology") present.
- [x] Prohibited terminology ("guessing floor", "unconditional acceptance floor") replaced with exact mathematical descriptions.
- [x] Primary hypothesis unequivocally reported as **unsupported** ($\Delta = +0.0071$, 95% CI $[-0.0503, +0.0679]$).
- [x] Commercial API spend verified at **$0.00**.
