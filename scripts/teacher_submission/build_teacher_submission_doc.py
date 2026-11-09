"""
Builds the complete, professionally formatted Teacher Submission Word Document:
'C:\\Users\\soham\\OneDrive\\Desktop\\rag\\AI Research Copilot - Teacher Submission.docx'

Strictly follows all user specifications:
- Preserves team metadata, institutional details, and department guide from original report.
- Leads with exact required problem statement.
- In-depth, verified three-paper literature comparison (CRAG, Sufficient Context, RINSE).
- Full mathematical formulation extracted from the implemented codebase.
- Rigorous duplicate-invariance argument with explicit assumptions and scope boundaries.
- Comprehensive empirical results across all 8 models/baselines, 108 controlled contexts, 36 natural retrieval contexts, and secondary subgroups.
- Unflinching reporting of negative primary result (M4 - M1 delta +0.0071, 95% CI [-0.0503, +0.0679]).
- Full disclosure of methodological caveats, same-AI review, prior Stage 4 exposure, and $0.00 API spend.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)

OUTPUT_DOCX_PATH = "C:/Users/soham/OneDrive/Desktop/rag/AI Research Copilot - Teacher Submission.docx"

# Color Palette Constants
HEX_PRIMARY = "1B365D"       # Deep Navy
HEX_SECONDARY = "2C3E50"     # Dark Slate
HEX_ACCENT_BLUE = "2980B9"   # Cerulean Blue
HEX_LIGHT_BG = "F8F9FA"      # Very Light Gray
HEX_CALLOUT_BG = "F4F6F9"    # Off-white / light blue-gray
HEX_BORDER = "D3D3D3"        # Light Gray Border
HEX_MUTED_TEXT = "555555"    # Dark Muted Gray

RGB_PRIMARY = RGBColor(0x1B, 0x36, 0x5D)
RGB_SECONDARY = RGBColor(0x2C, 0x3E, 0x50)
RGB_TEXT = RGBColor(0x22, 0x22, 0x22)
RGB_MUTED = RGBColor(0x55, 0x55, 0x55)

# ------------------------------------------------------------------------------
# XML Helper Functions for Styling
# ------------------------------------------------------------------------------

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def polish_table(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)
    
    # Prevent individual rows from splitting across pages
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
    # Repeat table header on every page if table spans across a page break
    if len(table.rows) > 0:
        header_trPr = table.rows[0]._tr.get_or_add_trPr()
        header_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

set_table_borders = polish_table

def add_callout(doc, text, title="NOTE", border_color="2980B9", bg_color="F4F6F9"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)

    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

    cell = table.cell(0, 0)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)

    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_together = True

    run_title = p.add_run(f"[{title}] ")
    run_title.bold = True
    run_title.font.name = "Arial"
    run_title.font.size = Pt(9.5)
    run_title.font.color.rgb = RGBColor.from_string(border_color)

    run_text = p.add_run(text)
    run_text.font.name = "Arial"
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = RGB_TEXT

    # Spacer paragraph after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)

def format_paragraph(p, space_before=0, space_after=6, line_spacing=1.15):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    if level == 1:
        format_paragraph(h, space_before=14, space_after=6, line_spacing=1.15)
        for r in h.runs:
            r.font.name = "Arial"
            r.font.size = Pt(14)
            r.font.bold = True
            r.font.color.rgb = RGB_PRIMARY
    elif level == 2:
        format_paragraph(h, space_before=10, space_after=4, line_spacing=1.15)
        for r in h.runs:
            r.font.name = "Arial"
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = RGB_SECONDARY
    elif level == 3:
        format_paragraph(h, space_before=8, space_after=3, line_spacing=1.15)
        for r in h.runs:
            r.font.name = "Arial"
            r.font.size = Pt(10.5)
            r.font.bold = True
            r.font.color.rgb = RGB_MUTED
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=5):
    p = doc.add_paragraph()
    format_paragraph(p, space_before=0, space_after=space_after, line_spacing=1.15)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGB_SECONDARY
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.font.color.rgb = RGB_TEXT
    return p

def add_bullet_p(doc, text, bold_prefix=None, space_after=3):
    p = doc.add_paragraph(style="List Paragraph")
    format_paragraph(p, space_before=0, space_after=space_after, line_spacing=1.15)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(9.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGB_SECONDARY
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(9.5)
    r.font.color.rgb = RGB_TEXT
    return p

# ------------------------------------------------------------------------------
# Document Generation Script
# ------------------------------------------------------------------------------

def generate_teacher_submission_report():
    print(f"Creating teacher submission report: {OUTPUT_DOCX_PATH}")
    doc = docx.Document()

    # Page Margins (Standard Academic 1.0 inch)
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.page_width = Inches(8.5)
    section.page_height = Inches(11.0)

    # --------------------------------------------------------------------------
    # Title & Metadata Header Block
    # --------------------------------------------------------------------------
    title_p = doc.add_paragraph()
    format_paragraph(title_p, space_before=0, space_after=2, line_spacing=1.05)
    run_title = title_p.add_run("AI Research Copilot: Evaluating Duplicate-Aware Evidence Aggregation for Selective Scientific Question Answering")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(16.5)
    run_title.font.bold = True
    run_title.font.color.rgb = RGB_PRIMARY

    sub_p = doc.add_paragraph()
    format_paragraph(sub_p, space_before=0, space_after=6, line_spacing=1.1)
    run_sub = sub_p.add_run("Final Academic Project Report & Empirical Evaluation Study")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(10)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGB_MUTED

    # Metadata Table
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    meta_table.columns[0].width = Inches(2.2)
    meta_table.columns[1].width = Inches(4.3)
    set_table_borders(meta_table, color="E0E0E0")

    meta_rows = [
        ("Team Members:", "Akanksha Jaiswal (230905106) – CSE Core\nMohd. Uzair Qureshi (220905158) – CSE Core\nSoham Pawar (230905312) – CSE Core"),
        ("Minor Specialization:", "Artificial Intelligence / Machine Learning (Computational Intelligence)"),
        ("Department Offering:", "Computer Science and Engineering"),
        ("Name of Guide:", "Dr. Adarsh Rag"),
        ("Date of Submission:", "15 November 2026")
    ]

    for r_idx, (label, val) in enumerate(meta_rows):
        row = meta_table.rows[r_idx]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        set_cell_background(cell_lbl, "F8F9FA")
        set_cell_background(cell_val, "FFFFFF")
        set_cell_margins(cell_lbl, top=35, bottom=35, left=80, right=80)
        set_cell_margins(cell_val, top=35, bottom=35, left=80, right=80)

        p_l = cell_lbl.paragraphs[0]
        format_paragraph(p_l, space_before=0, space_after=0, line_spacing=1.05)
        r_l = p_l.add_run(label)
        r_l.font.name = "Arial"
        r_l.font.size = Pt(8.5)
        r_l.font.bold = True
        r_l.font.color.rgb = RGB_SECONDARY

        p_v = cell_val.paragraphs[0]
        format_paragraph(p_v, space_before=0, space_after=0, line_spacing=1.05)
        r_v = p_v.add_run(val)
        r_v.font.name = "Arial"
        r_v.font.size = Pt(8.5)
        r_v.font.color.rgb = RGB_TEXT

    sp = doc.add_paragraph()
    format_paragraph(sp, space_before=2, space_after=4)

    # --------------------------------------------------------------------------
    # Abstract
    # --------------------------------------------------------------------------
    h_abs = add_styled_heading(doc, "Abstract", level=1)
    format_paragraph(h_abs, space_before=6, space_after=2)
    p_abs1 = add_body_p(
        doc,
        "Retrieval-augmented generation (RAG) systems in specialized scientific domains frequently encounter "
        "evidence insufficiency, where retrieved passages discuss relevant topics but omit exact parameters, "
        "causal definitions, or empirical mechanisms required to formulate a factual answer. Conventional systems "
        "rely on scalar cross-encoder relevance thresholds that fail to detect evidence gaps or redundant passage stuffing. "
        "This project investigates the research question: Does aggregating distinct evidence by paper and section, combined with "
        "exact within-paper passage deduplication, improve answer-or-abstain decisions over relevance-only baselines at matched coverage? "
        "We develop a controlled 120-family scientific benchmark (480 total contexts across 160 papers), rigorously partitioned into "
        "60 training families, 24 validation families, and 36 held-out test families (144 test contexts across 36 paper clusters). "
        "All models (B0 Always Accept, B1 Max Relevance, B2 Historical Heuristic Gate, M1 Relevance Raw, M2 Relevance Dedup, "
        "M3 Full Structure Raw, M4 Full Structure Dedup, and a Length Diagnostic Shortcut) were fit strictly on training data, "
        "tuned and frozen on validation data, and evaluated on the held-out test set with zero parameter fitting. "
        "Models, scalers, and validation thresholds were completely frozen before the new v2 test construction, "
        "while fully disclosing historical test-paper exposure during an earlier invalid Stage 4 run prior to the v2 benchmark protocol."
    )
    p_abs1.runs[0].font.size = Pt(9.2)
    format_paragraph(p_abs1, space_before=0, space_after=3, line_spacing=1.12)

    p_abs2 = add_body_p(
        doc,
        "On the primary evaluation partition of 108 controlled test contexts, the hypothesized superiority of M4 over M1 is NOT SUPPORTED: "
        "M4 achieved a discrete Area Under the Risk-Coverage curve (AURC) of 0.4400 compared to 0.4329 for M1, yielding an observed paired "
        "difference of +0.0071 (bootstrap 95% percentile interval [-0.0503, +0.0679] across 2,000 paired cluster resamples; "
        "negative favors M4). Because the confidence interval spans zero, the hypothesized improvement is not supported, and equivalence is not established. "
        "Baseline B1 (Max Relevance) achieved the lowest AURC (0.4049) on the controlled partition. "
        "Descriptive subgroup analysis on 11 multi-passage families revealed lower absolute AURC for deduplicated models (M4=0.3803 vs M1=0.6894), "
        "while on 36 natural retrieval contexts M4 exhibited the lowest observed AURC (0.0501), though with only three negative instances "
        "the natural bootstrap interval spans zero ([-0.1176, +0.0160]). Finally, exact deduplication confers a bounded duplicate-invariance property "
        "under explicitly stated operating assumptions, with an observed 0.000000 score shift under tested post-selection duplicate injection, "
        "whereas raw models experience over 0.31 score distortion. "
        "This report documents the software architecture, formal mathematics, verified literature comparisons, and negative empirical findings."
    )
    p_abs2.runs[0].font.size = Pt(9.2)
    format_paragraph(p_abs2, space_before=0, space_after=4, line_spacing=1.12)

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 1. Introduction & Problem Statement
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "1. Introduction & Problem Statement", level=1)
    
    # Exact lead problem statement
    add_callout(
        doc,
        "Retrieval-augmented scientific question-answering systems must decide whether retrieved passages contain enough evidence "
        "to answer a question. High relevance scores and repeated passages do not necessarily establish that all requested information is available. "
        "This project investigates whether exact deduplication and paper/section features improve evidence-sufficiency ranking over relevance-only baselines, "
        "while measuring acceptance coverage and the proportion of accepted contexts judged insufficient.",
        title="CORE PROBLEM STATEMENT",
        border_color=HEX_PRIMARY,
        bg_color=HEX_CALLOUT_BG
    )

    add_body_p(
        doc,
        "Scientific literature question answering is an intrinsically high-stakes task. Unlike general open-domain question answering, "
        "academic queries demand exact mathematical equations, chemical pathways, hyperparameters, and experimental outcomes. "
        "When presented with queries regarding state-of-the-art research, standard large language models (LLMs) frequently hallucinate "
        "convincing yet factually fabricated citations and parameters. Retrieval-Augmented Generation (RAG) mitigates parametric hallucination "
        "by conditioning generation on passages fetched dynamically from an external corpus. However, standard RAG architectures operate under "
        "an unconditional answering assumption: they attempt to synthesize a response regardless of whether the retrieved context contains "
        "the necessary evidentiary premises."
    )
    add_body_p(
        doc,
        "In specialized scientific corpora, two pervasive phenomena undermine conventional relevance-driven retrieval:",
        bold_prefix="Two Evidentiary Failure Modes: "
    )
    add_bullet_p(doc, "Cross-encoders like BAAI/bge-reranker-base assign high semantic relevance scores (e.g., >0.90) to passages that share domain concepts, entity names, or background framing with the query, even when the passage entirely lacks the critical operational formula or quantitative finding requested.", bold_prefix="1. Topical Relevance vs. Evidence Sufficiency: ")
    add_bullet_p(doc, "Academic papers frequently repeat high-level claims across the abstract, introduction, and conclusion. Standard top-k retrieval often populates the prompt window with near-identical or identical sentences from the same paper, creating an illusion of multi-passage corroboration while displacing complementary evidence from methodological or experimental sections.", bold_prefix="2. Redundant Evidence & Duplicate Stuffing: ")

    add_body_p(
        doc,
        "The software deliverable of this project—the AI Research Copilot—implements an end-to-end scientific literature assistant. "
        "Its software architecture comprises a local PDF ingestion pipeline (Docling / pypdf with section regex parsing), "
        "a persistent vector store (ChromaDB with nomic-embed-text embeddings), a sparse lexical index (BM25Okapi), "
        "a cross-encoder reranking stage (BAAI/bge-reranker-base), an in-memory citation graph (NetworkX DiGraph with PyVis visualization), "
        "and an interactive multi-tab user interface (Streamlit). Generation is handled by a hosted Gemini 3.5 Flash Lite adapter.",
        bold_prefix="System Implementation Context: "
    )

    add_callout(
        doc,
        "Clear Separation of System Features vs. Evaluated Components: The broader AI Research Copilot software application encompasses "
        "PDF ingestion, vector indexing, interactive Streamlit dashboards, and citation network visualization. However, the scientific experiment "
        "described in this report specifically isolates and evaluates the selective evidence-sufficiency gating mechanism (models M1–M4 and baselines) "
        "on frozen contexts. No claim of end-to-end hallucination reduction or answer accuracy improvement is made without a corresponding generation experiment.",
        title="SYSTEM VS. EXPERIMENT BOUNDARY",
        border_color="E67E22",
        bg_color="FEF9E7"
    )

    # --------------------------------------------------------------------------
    # 2. Literature Review: Three-Paper Comparison
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "2. Literature Review & Related Work", level=1)
    add_body_p(
        doc,
        "To establish rigorous academic context, we compare our project against three peer-reviewed and primary preprint publications "
        "that formalize retrieval evaluation, evidence gaps, and selective generation. Each primary source was verified directly from official archives."
    )

    # Table of verified literature
    lit_table = doc.add_table(rows=4, cols=4)
    lit_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    lit_table.autofit = False
    lit_table.columns[0].width = Inches(1.3)
    lit_table.columns[1].width = Inches(2.2)
    lit_table.columns[2].width = Inches(1.5)
    lit_table.columns[3].width = Inches(1.5)
    set_table_borders(lit_table, color="D3D3D3")

    lit_headers = ["Citation / Archive", "Formal Paper Title", "Authors & Venue", "Core Contribution"]
    for c_idx, h_text in enumerate(lit_headers):
        cell = lit_table.rows[0].cells[c_idx]
        set_cell_background(cell, HEX_SECONDARY)
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(9)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    lit_data = [
        ("Yan et al. [1]\narXiv:2401.15884", "Corrective Retrieval Augmented Generation (CRAG)", "S. Yan, J. Gu, Y. Zhu, Z. Ling (EMNLP 2024)", "Evaluates retrieval confidence via fine-tuned T5 evaluator to trigger document refinement or external web search."),
        ("Joren et al. [2]\narXiv:2411.06037", "Sufficient Context: A New Lens on Retrieval Augmented Generation Systems", "H. Joren, J. Zhang, C. Ferng, et al. (ICLR 2025)", "Separates context sufficiency from LLM reasoning; proposes selective generation using sufficiency classifiers."),
        ("Chen et al. [3]\narXiv:2609.37469", "Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation in RAG (RINSE)", "S. Chen, P. Hua, Y. Xiao (Preprint 2026)", "Detects pre-generation evidence gaps combining component coverage, answer evidence, and small LM sufficiency.")
    ]

    for r_idx, row_vals in enumerate(lit_data, start=1):
        row = lit_table.rows[r_idx]
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGB_TEXT

    sp = doc.add_paragraph()
    format_paragraph(sp, space_before=2, space_after=6)

    # Detailed Paper Analyses
    add_styled_heading(doc, "2.1 Detailed Primary Source Analysis", level=2)

    add_body_p(
        doc,
        "Yan et al. [1] address the issue of retrievers fetching sub-optimal or irrelevant passages that mislead downstream generators.\n"
        "Verified Method Details: CRAG trains a lightweight retrieval evaluator (fine-tuned T5-large) to score query-document relevance and establish a three-way decision rule "
        "using dual fixed scalar thresholds: (1) Correct (confidence >= upper threshold): retain internal documents and apply sentence-level decomposition and filtering; "
        "(2) Incorrect (confidence < lower threshold): discard retrieved documents entirely and trigger external web search; and (3) Ambiguous: combine decomposed internal "
        "passages with web search results. CRAG demonstrates substantial accuracy improvements on open-domain QA (PopQA, TriviaQA).\n"
        "Problem Framing & Differences: CRAG frames retrieval evaluation primarily as deciding whether to accept, rewrite, or seek external replacement for retrieved web/open-domain passages. "
        "In contrast, our project investigates a closed scientific corpus setting where external web search is absent, focusing on the empirical question of whether aggregating within-paper "
        "section diversity and deduplicating redundant passages improves selective answer-or-abstain decisions over relevance-only baselines. We do not claim general novelty over CRAG.",
        bold_prefix="A. Corrective Retrieval Augmented Generation (CRAG; Yan et al., EMNLP 2024): "
    )

    add_body_p(
        doc,
        "Joren et al. [2] address a key distinction in RAG evaluation: separating retriever failure (insufficient evidence in context) "
        "from generator failure (incorrect reasoning over sufficient evidence). They define context sufficiency as an objective property—whether a context provides enough factual premises "
        "to logically entail an answer—and demonstrate that frontier LLMs (e.g., GPT-4o, Gemini 1.5 Pro) frequently hallucinate when presented with insufficient context rather than abstaining. "
        "Crucially, Joren et al. also propose selective generation using sufficiency information, training binary classifiers to enable answer-or-abstain decisions evaluated along "
        "risk-coverage trade-off curves.\n"
        "Problem Framing & Differences: While Joren et al. formalize selective generation using sufficiency classifiers over concatenated input contexts, our project specifically investigates "
        "how passage-level evidence features behave under aggregation: evaluating whether explicitly modeling bibliographic structure (paper and section groupings) and passage deduplication "
        "improves selective gating performance compared to scalar relevance alone.",
        bold_prefix="B. Sufficient Context: A New Lens on RAG (Joren et al., ICLR 2025): "
    )

    add_body_p(
        doc,
        "Chen et al. [3] demonstrate that topical relevance and evidence sufficiency are fundamentally distinct dimensions: passages may achieve high semantic similarity scores "
        "by discussing query entities while omitting the precise quantitative parameters, definitions, or causal links required to formulate a complete answer. "
        "RINSE detects pre-generation evidence gaps by combining three components: (1) Question Component Coverage (verifying that each sub-clause/requirement of the query has "
        "corresponding passage support), (2) Passage-Level Answer Evidence (identifying direct entity/value answers within passages), and (3) a small language model's joint sufficiency "
        "judgment over the collected evidence.\n"
        "Problem Framing & Differences: RINSE addresses evidence gaps through multi-stage linguistic decomposition and local language-model validation. In contrast, our empirical project "
        "evaluates whether lightweight structural heuristics (within-paper section diversity, second-passage reinforcement) combined with exact passage deduplication can improve selective "
        "evidence gating without invoking additional generative LLM inference passes.",
        bold_prefix="C. Relevance Is Not Sufficient Evidence (RINSE; Chen et al., 2026): "
    )

    add_callout(
        doc,
        "Framing of Contribution: This project does not claim to faithfully re-implement the complete systems of CRAG, Sufficient Context, or RINSE. "
        "Rather, our contribution is framed strictly as an evaluated empirical system and controlled comparative benchmark investigating whether "
        "duplicate-aware passage aggregation and structural features improve selective question answering over relevance baselines in scientific literature. "
        "Publication novelty remains unestablished.",
        title="ACADEMIC SCOPE & NOVELTY DISCLAIMER",
        border_color="7F8C8D",
        bg_color="F2F4F4"
    )

    # --------------------------------------------------------------------------
    # 3. Mathematical Formulation
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "3. Mathematical Formulation", level=1)
    add_body_p(
        doc,
        "All data structures, scoring functions, feature transforms, and metrics described below are transcribed directly "
        "from the verified codebase in backend/selective/ and evaluation/run_benchmark_v2_test_eval.py."
    )

    add_styled_heading(doc, "3.1 Context Representation & Exact Deduplication", level=2)
    add_body_p(
        doc,
        "Let a scientific question be denoted by query q. The candidate context provided to the gating model is an ordered sequence "
        "of K scientific passages: C = (p_1, p_2, ..., p_K). Each passage p_i is an immutable dataclass record containing:",
        bold_prefix="Passage Data Structure: "
    )
    add_bullet_p(doc, "Text string text_i representing the raw passage content.", bold_prefix="• Text: ")
    add_bullet_p(doc, "Cross-encoder relevance score s_i in [0, 1], evaluated via BAAI/bge-reranker-base (sigmoid activation, fp32).", bold_prefix="• Reranker Score: ")
    add_bullet_p(doc, "Bibliographic metadata: primary source paper identifier pid_i and section heading sec_i.", bold_prefix="• Source Metadata: ")
    add_bullet_p(doc, "Recomputed token count tok_i, evaluated using the pinned cl100k_base tokenizer.", bold_prefix="• Token Count: ")
    add_bullet_p(doc, "Deterministic content hash h_i = SHA-256(normalize_whitespace(text_i)).", bold_prefix="• Hash Integrity: ")

    add_body_p(
        doc,
        "Exact within-paper deduplication is defined over the multiset of passages. Two passages p_i and p_j are identical if and only if:\n"
        "p_i equiv p_j  <=>  normalize_whitespace(p_i.text) == normalize_whitespace(p_j.text)  and  p_i.pid == p_j.pid\n"
        "Whitespace normalization collapses consecutive whitespace characters to a single space while strictly preserving case, acronyms, "
        "and mathematical/chemical notation. For each equivalence class of duplicate passages, the deduplication engine deterministically retains "
        "the single canonical representative possessing the maximum relevance score: p* = argmax_{p in [p_i]} p.s. "
        "The resulting deduplicated context is denoted C_dedup = (p'_1, ..., p'_{K'}), where K' <= K.",
        bold_prefix="Deduplication Operation (dedup): "
    )

    add_styled_heading(doc, "3.2 Feature Vector Extraction (M1–M4)", level=2)
    add_body_p(
        doc,
        "To evaluate the 2x2 factorial experimental design (Relevance vs. Combined Structure; Raw vs. Deduplicated Passages), "
        "features are extracted either from the raw context C (K passages) or the deduplicated context C_dedup (K' passages):",
        bold_prefix="Feature Representation: "
    )
    add_bullet_p(doc, "Phi_rel(C) = [s_(1), s_(2), s_bar, s_(1) - s_(2)]^T in R^4, where s_(1) is the maximum reranker score, s_(2) is the second-highest score (or s_(1) if K=1), s_bar = (1/K) sum s_i is the mean passage score, and s_(1) - s_(2) is the top-score spread.", bold_prefix="1. Relevance Features (4D): ")
    add_bullet_p(doc, "Phi_struct(C) = [s_top_paper_second, s_other, H_sec, s_bar_sec]^T in R^4, where:\n"
                     "• s_top_paper_second: Score of the second-highest passage originating from the paper that produced s_(1) (0.0 if only one passage from that paper exists).\n"
                     "• s_other: Maximum score among all passages originating from any paper OTHER than the top paper (0.0 if all passages belong to the top paper).\n"
                     "• H_sec: Normalized Shannon entropy of passage section assignments: H_sec = - sum_{j} p(sec_j) log(p(sec_j)) / log(max(2, |distinct_sections|)).\n"
                     "• s_bar_sec: Mean relevance score aggregated across distinct section groups.", bold_prefix="2. Structural Features (4D): ")

    add_body_p(
        doc,
        "The four models of the 2x2 factorial grid are defined as:\n"
        "• M1 (Relevance Raw): x = Phi_rel(C) in R^4\n"
        "• M2 (Relevance Dedup): x = Phi_rel(C_dedup) in R^4\n"
        "• M3 (Full Structure Raw): x = [Phi_rel(C); Phi_struct(C)] in R^8\n"
        "• M4 (Full Structure Dedup): x = [Phi_rel(C_dedup); Phi_struct(C_dedup)] in R^8\n"
        "• Shortcut Diagnostic: x = [K, sum tok_i] in R^2 (passage count and total context tokens).",
        bold_prefix="Model Feature Spaces: "
    )

    add_styled_heading(doc, "3.3 Standardization, Gating Score, and Decision Rule", level=2)
    add_body_p(
        doc,
        "Feature standardization is fit strictly on the training partition: mu_train = E[x_train], sigma_train = Std[x_train]. "
        "Test instances are scaled using frozen training parameters: x_scaled = (x - mu_train) / sigma_train. "
        "The gating probability is computed via an L2-regularized logistic regression model:\n"
        "P(Y_suff = 1 | C) = sigma(w^T x_scaled + b) = 1 / (1 + exp(-(w^T x_scaled + b)))\n"
        "Given an operating threshold T in [0, 1], the gating decision is binary: y_hat = 1 (Answer / Accept) if P(Y_suff = 1 | C) >= T, "
        "and y_hat = 0 (Abstain / Reject) if P(Y_suff = 1 | C) < T.",
        bold_prefix="Logistic Decision Rule: "
    )

    add_styled_heading(doc, "3.4 Evaluation Metrics & Tie-Correct Discrete AURC", level=2)
    add_body_p(
        doc,
        "Let N denote sample size, y_i in {0, 1} ground truth sufficiency (1=sufficient, 0=insufficient), and y_hat_i in {0, 1} model decisions:\n"
        "• Coverage: Cov = (1 / N) sum_{i=1}^N y_hat_i (fraction of contexts accepted).\n"
        "• Evidence Selection Risk: Risk = (sum_{i=1}^N y_hat_i (1 - y_i)) / (sum_{i=1}^N y_hat_i). "
        "Crucially, if zero contexts are accepted (sum y_hat_i = 0), risk is mathematically undefined (None / null), NOT 0.0.",
        bold_prefix="Coverage and Risk: "
    )
    add_body_p(
        doc,
        "Under uniform random tie-breaking, the discrete expected AURC is evaluated analytically across all accepted counts k in {1, ..., N}. "
        "Samples are partitioned into distinct score levels S_1 > S_2 > ... > S_M with group sizes n_m and error counts E_m = sum_{i in group_m} (1 - y_i). "
        "For any integer count k lying in boundary group m (where cum_counts_{m-1} < k <= cum_counts_m):\n"
        "Expected Errors(k) = cum_errors_{m-1} + (k - cum_counts_{m-1}) * (E_m / n_m)\n"
        "Expected Risk(k) = Expected Errors(k) / k\n"
        "Discrete Expected AURC = (1 / N) sum_{k=1}^N Expected Risk(k).\n"
        "For an all-tied scorer, Expected Risk(k) is exactly E/N for all k, and discrete AURC equals the sample insufficient prevalence E/N. "
        "The insufficient-label prevalence represents the expected all-tied ranking AURC under the implemented convention, rather than an empirical lower bound or guessing floor.",
        bold_prefix="Discrete Expected AURC: "
    )

    add_styled_heading(doc, "3.5 Paired Cluster Bootstrap", level=2)
    add_body_p(
        doc,
        "To estimate uncertainty while respecting intra-paper and intra-family correlation, we implement a paired cluster bootstrap. "
        "Clusters are defined by primary source paper identifier pid. In each of B = 2,000 bootstrap iterations (random seed 20261104):\n"
        "1. Sample M paper clusters with replacement: {pid*_1, ..., pid*_M}.\n"
        "2. Gather all contexts associated with sampled clusters, preserving exact cluster multiplicity.\n"
        "3. Evaluate all scoring models on the IDENTICAL resampled instances to compute paired differences: Delta_b = AURC_A(b) - AURC_B(b).\n"
        "4. Compute the 95% percentile confidence interval: [q_0.025(Delta), q_0.975(Delta)]. "
        "Sign convention: Delta = AURC_A - AURC_B; negative values favor Method A.",
        bold_prefix="Bootstrap Specification: "
    )

    add_styled_heading(doc, "3.6 Mathematical Duplicate-Invariance Argument", level=2)
    add_body_p(
        doc,
        "We state the exact duplicate-invariance property and its rigorous operating assumptions:",
        bold_prefix="Proposition & Scope: "
    )
    add_callout(
        doc,
        "Duplicate-Invariance Property: Let C = (p_1, ..., p_K) be a fixed candidate context record. Suppose an exact duplicate passage p_dup "
        "is appended to C, where p_dup shares identical whitespace-normalized text, identical paper ID, identical section, and identical reranker score "
        "with an existing passage p_i in C. Under the following explicit assumptions:\n"
        "(A1) Fixed Context: The input candidate pool is fixed post-retrieval; no re-retrieval occurs.\n"
        "(A2) Metadata Equivalence: p_dup has identical paper_id, section, and score s_dup = s_i.\n"
        "(A3) Deterministic Canonical Representative: Within-paper deduplication collapses {p_i, p_dup} into p* = argmax(score) = p_i.\n"
        "(A4) Downstream Feature Dependency: Gating features Phi(C) are evaluated strictly on dedup(C).\n"
        "(A5) No Token Budget Displacement: Adding p_dup does not displace other distinct evidence passages under a context token cap.\n"
        "Then: dedup(C union {p_dup}) = dedup(C), which implies Phi(dedup(C union {p_dup})) = Phi(dedup(C)), and consequently:\n"
        "| Score_M4(C union {p_dup}) - Score_M4(C) | = 0.000000 (Bounded Duplicate Invariance under A1–A5).\n\n"
        "Separately, in empirical testing on 36 fixed test contexts with post-selection duplicate injection, M2, M4, and B1 exhibit an observed 0.000000 score shift, "
        "whereas raw models exhibit shifts of 0.318703 (M1) and 0.312393 (M3).\n\n"
        "Scope Limitations: This is not an unconstrained invariance theorem. Invariance does NOT hold if: (1) duplicates displace distinct passages during budget-constrained "
        "retrieval; (2) passages are paraphrases rather than exact string matches; (3) metadata or section labels differ; or (4) retrieval queries are re-executed.",
        title="DUPLICATE INVARIANCE ARGUMENT & ASSUMPTIONS",
        border_color="27AE60",
        bg_color="EAFAF1"
    )

    # --------------------------------------------------------------------------
    # 4. Experimental Design & Benchmark Construction
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "4. Experimental Design & Benchmark Construction", level=1)
    add_body_p(
        doc,
        "The experimental evaluation is structured as a group-partitioned, 4-condition controlled benchmark across 160 eligible scientific papers "
        "(7,368 real text chunks) from arXiv. Development data (40 historical papers) was quarantined to prevent split contamination.",
        bold_prefix="Corpus Partitioning: "
    )
    add_bullet_p(doc, "80 papers, 60 question families, 240 contexts (ctx_tr_0001 to ctx_tr_0240).", bold_prefix="• Training Partition: ")
    add_bullet_p(doc, "32 papers, 24 question families, 96 contexts (ctx_val_0001 to ctx_val_0096).", bold_prefix="• Validation Partition: ")
    add_bullet_p(doc, "48 papers, 36 question families, 144 contexts (ctx_te_0001 to ctx_te_0144). Primary source paper count = 36 clusters.", bold_prefix="• Held-Out Test Partition: ")

    add_body_p(
        doc,
        "Every question family is evaluated across four parallel conditions (4 contexts per family):\n"
        "1. supported_candidate: Contains the complete designated gold evidence passage(s) + corpus distractors (exactly 4 passages per context).\n"
        "2. relevant_replacement: Essential evidence is replaced by a topically relevant passage from the same paper or topic that lacks the critical answer component (exactly 4 passages per context).\n"
        "3. duplicate_replacement: Essential evidence is replaced by an exact duplicate copy of an existing distractor passage (exactly 4 passages per context).\n"
        "4. natural_retrieval: Unconstrained end-to-end hybrid retrieval (BM25 + dense ChromaDB) reranked by BAAI/bge-reranker-base within the split corpus (allows up to 6 passages within a 2,000-token context budget).",
        bold_prefix="Four Experimental Conditions: "
    )

    add_body_p(
        doc,
        "To ensure rigorous semantic ground truth, test annotations were adjudicated through a 12-checkpoint protocol "
        "where an assistant rater displayed and read complete, untruncated numbered passages for all 144 contexts. "
        "Every positive annotation was mechanically validated to contain exact verbatim quotation substrings from displayed passages "
        "(196 verified quotes, 0 failures). Negative annotations recorded specific missing information components. "
        "Label distribution across the 144 test contexts is: 69 sufficient (47.9%) and 75 insufficient (52.1%). "
        "In the controlled partition (108 contexts), insufficient prevalence is 66.7% (36 supported vs. 72 replacements). "
        "In natural retrieval (36 contexts), insufficient prevalence is 8.3% (33 sufficient vs. 3 insufficient).",
        bold_prefix="Semantic Review Protocol: "
    )

    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 5. Empirical Results & Analysis
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "5. Empirical Results & Findings", level=1)

    # Table 1: Controlled Evaluation Table
    add_styled_heading(doc, "5.1 Primary Controlled Evaluation (N = 108 Contexts)", level=2)

    # Insert High-Resolution Figure 1
    png_fig_path = "C:/Users/soham/OneDrive/Desktop/rag/ai-research-copilot/data/metadata/risk_coverage_curve_v2_test.png"
    if os.path.exists(png_fig_path):
        p_fig = doc.add_paragraph()
        p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_fig, space_before=4, space_after=2)
        doc.add_picture(png_fig_path, width=Inches(4.8))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        format_paragraph(p_cap, space_before=2, space_after=4)
        r_cap = p_cap.add_run(
            "Figure 1: Risk vs. Coverage trade-off curves evaluated on the 108 controlled test contexts under uniform random tie ordering. "
            "M4 Full Structure Dedup (green line, discrete AURC = 0.4400) and M1 Relevance Raw (red line, discrete AURC = 0.4329) perform comparably "
            "(Delta = +0.0071, 95% bootstrap interval [-0.0503, +0.0679]). Baseline B1 Max Relevance (dark slate line) achieves the lowest discrete AURC (0.4049). "
            "B2 Historical Gate is plotted at its single discrete operating point (Coverage = 0.528, Risk = 0.404). Baseline B0 corresponds to the expected all-tied "
            "ranking AURC equal to the insufficient-label prevalence (Risk = 0.667), not a performance lower bound."
        )
        r_cap.font.name = "Arial"
        r_cap.font.size = Pt(8)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGB_MUTED

    p_t1_intro = add_body_p(doc, "Table 1 reports the primary test results on the 108 controlled contexts (prevalence of insufficient contexts = 66.7%). Operating points reflect thresholds frozen on the validation set targeting coverages 0.50, 0.75, and 0.90. The length diagnostic achieved discrete AURC = 0.6798; this establishes only that this specific fitted length-based logistic regression diagnostic performed poorly, not that the benchmark is free of all possible shortcuts.")
    format_paragraph(p_t1_intro, space_before=1, space_after=3)
    p_t1_intro.runs[0].font.size = Pt(9)

    t1 = doc.add_table(rows=9, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False
    t1.columns[0].width = Inches(2.1)
    t1.columns[1].width = Inches(1.1)
    t1.columns[2].width = Inches(1.1)
    t1.columns[3].width = Inches(1.1)
    t1.columns[4].width = Inches(1.1)
    set_table_borders(t1, color="D3D3D3")

    t1_headers = ["Model / Baseline", "Discrete AURC", "Cov@0.50 (Risk)", "Cov@0.75 (Risk)", "Cov@0.90 (Risk)"]
    for c_idx, h_text in enumerate(t1_headers):
        cell = t1.rows[0].cells[c_idx]
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, top=40, bottom=40, left=70, right=70)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(8)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    t1_rows = [
        ("B0 (Always Accept)", "0.6667", "1.000 (0.667)", "1.000 (0.667)", "1.000 (0.667)"),
        ("B1 (Max Relevance)", "0.4049", "0.315 (0.294)", "0.472 (0.372)", "0.852 (0.609)"),
        ("B2 (Historical Gate)", "N/A (Discrete)", "— (Cov=0.528)", "— (Risk=0.404)", "T1=0.319, T2=0.056"),
        ("M1 (Relevance Raw)", "0.4329", "0.250 (0.259)", "0.491 (0.377)", "0.824 (0.596)"),
        ("M2 (Relevance Dedup)", "0.4373", "0.232 (0.520)", "0.481 (0.385)", "0.824 (0.596)"),
        ("M3 (Full Structure Raw)", "0.4658", "0.352 (0.395)", "0.491 (0.377)", "0.907 (0.633)"),
        ("M4 (Full Structure Dedup)", "0.4400", "0.222 (0.500)", "0.546 (0.424)", "0.722 (0.538)"),
        ("Shortcut (Length Diag.)", "0.6798", "0.056 (0.500)", "0.296 (0.656)", "0.759 (0.683)")
    ]

    for r_idx, row_vals in enumerate(t1_rows, start=1):
        row = t1.rows[r_idx]
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=26, bottom=26, left=70, right=70)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(8)
            if c_idx == 1 and val == "0.4049":
                r.font.bold = True
            r.font.color.rgb = RGB_TEXT

    sp = doc.add_paragraph()
    format_paragraph(sp, space_before=1, space_after=2)
    doc.add_page_break()

    # Table 2: Paired Cluster Bootstrap
    add_styled_heading(doc, "5.2 Paired Cluster Bootstrap Analysis (2,000 Resamples)", level=2)
    add_body_p(doc, "Table 2 presents the paired cluster-bootstrap uncertainty analysis on controlled contexts (36 paper clusters, random seed 20261104). Delta = Method A - Method B; negative values favor Method A.")

    t2 = doc.add_table(rows=9, cols=6)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False
    t2.columns[0].width = Inches(1.8)
    t2.columns[1].width = Inches(0.8)
    t2.columns[2].width = Inches(0.8)
    t2.columns[3].width = Inches(0.8)
    t2.columns[4].width = Inches(1.3)
    t2.columns[5].width = Inches(1.0)
    set_table_borders(t2, color="D3D3D3")

    t2_headers = ["Comparison Pair", "Obs Delta", "Boot Mean", "Boot SE", "95% Percentile CI", "Verdict"]
    for c_idx, h_text in enumerate(t2_headers):
        cell = t2.rows[0].cells[c_idx]
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, top=35, bottom=35, left=50, right=50)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(8)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    t2_rows = [
        ("M4 − M1 (Primary)", "+0.0071", "+0.0092", "0.0305", "[-0.0503, +0.0679]", "Not Supported"),
        ("M2 − M1 (Ablation 1)", "+0.0044", "+0.0052", "0.0283", "[-0.0520, +0.0603]", "Contains 0"),
        ("M3 − M1 (Ablation 2)", "+0.0329", "+0.0352", "0.0228", "[-0.0058, +0.0846]", "Contains 0"),
        ("M4 − M2 (Ablation 3)", "+0.0027", "+0.0041", "0.0117", "[-0.0199, +0.0270]", "Contains 0"),
        ("M4 − M3 (Ablation 4)", "-0.0258", "-0.0260", "0.0334", "[-0.0939, +0.0363]", "Contains 0"),
        ("B1 − M1 (Baseline)", "-0.0281", "-0.0253", "0.0243", "[-0.0668, +0.0267]", "Contains 0"),
        ("M4 − B1 (Method vs B1)", "+0.0352", "+0.0345", "0.0318", "[-0.0316, +0.0949]", "Contains 0"),
        ("Length − M4 (Shortcut)", "+0.2398", "+0.2410", "0.0350", "[+0.1722, +0.3081]", "Favors M4")
    ]

    for r_idx, row_vals in enumerate(t2_rows, start=1):
        row = t2.rows[r_idx]
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=26, bottom=26, left=50, right=50)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(8)
            if c_idx == 5 and val == "Not Supported":
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
            elif c_idx == 5 and val == "Favors M4":
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x27, 0xAE, 0x60)
            else:
                r.font.color.rgb = RGB_TEXT

    sp = doc.add_paragraph()
    format_paragraph(sp, space_before=1, space_after=4)

    # Detailed Primary Result Discussion
    add_callout(
        doc,
        "Primary Hypothesis Verdict: M4 − M1 discrete AURC difference is approximately +0.0071, with a 95% bootstrap percentile "
        "interval of [-0.0503, +0.0679]. The hypothesized improvement is NOT supported by the data. The interval includes differences "
        "favoring either method; this study does not establish equivalence. Baseline B1 (Max Relevance) achieves the "
        "lowest observed controlled-context AURC (0.4049).",
        title="PRIMARY HYPOTHESIS CONCLUSION",
        border_color="C0392B",
        bg_color="FDEDEC"
    )
    doc.add_page_break()

    # Table 3: Natural Retrieval Evaluation
    add_styled_heading(doc, "5.3 Practical Generalization Check: Natural Retrieval Contexts (N = 36)", level=2)
    add_body_p(
        doc,
        "The natural retrieval partition evaluates performance in an unmanipulated end-to-end pipeline (BM25 + ChromaDB + bge-reranker-base). "
        "Only 3 of 36 contexts (8.3%) are semantically insufficient (ctx_te_0064, ctx_te_0112, ctx_te_0124)."
    )

    t3 = doc.add_table(rows=8, cols=5)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3.autofit = False
    t3.columns[0].width = Inches(2.1)
    t3.columns[1].width = Inches(1.1)
    t3.columns[2].width = Inches(1.1)
    t3.columns[3].width = Inches(1.1)
    t3.columns[4].width = Inches(1.1)
    set_table_borders(t3, color="D3D3D3")

    for c_idx, h_text in enumerate(t1_headers):
        cell = t3.rows[0].cells[c_idx]
        set_cell_background(cell, HEX_SECONDARY)
        set_cell_margins(cell, top=35, bottom=35, left=70, right=70)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(8)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    t3_rows = [
        ("B0 (Always Accept)", "0.0833", "1.000 (0.083)", "1.000 (0.083)", "1.000 (0.083)"),
        ("B1 (Max Relevance)", "0.0707", "0.750 (0.074)", "0.944 (0.059)", "1.000 (0.083)"),
        ("M1 (Relevance Raw)", "0.0861", "0.889 (0.062)", "0.889 (0.062)", "1.000 (0.083)"),
        ("M2 (Relevance Dedup)", "0.1037", "0.972 (0.057)", "0.972 (0.057)", "1.000 (0.083)"),
        ("M3 (Full Structure Raw)", "0.0962", "0.889 (0.062)", "0.917 (0.061)", "1.000 (0.083)"),
        ("M4 (Full Structure Dedup)", "0.0501", "0.972 (0.057)", "1.000 (0.083)", "1.000 (0.083)"),
        ("Shortcut (Length Diag.)", "0.0644", "1.000 (0.083)", "1.000 (0.083)", "1.000 (0.083)")
    ]

    for r_idx, row_vals in enumerate(t3_rows, start=1):
        row = t3.rows[r_idx]
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=26, bottom=26, left=70, right=70)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(8)
            if c_idx == 1 and val == "0.0501":
                r.font.bold = True
            r.font.color.rgb = RGB_TEXT

    sp = doc.add_paragraph()
    format_paragraph(sp, space_before=1, space_after=3)

    add_body_p(
        doc,
        "Caution Regarding Natural Retrieval Findings: Although M4 achieves the lowest observed AURC (0.0501) on the natural retrieval split, "
        "the partition contains only three insufficient contexts out of 36. Under paired cluster bootstrapping, the difference M4 − M1 has a 95% "
        "confidence interval of [-0.1176, +0.0160], which spans zero. Due to the very small negative sample count, this result cannot support "
        "claims of general superiority in practical retrieval settings.",
        bold_prefix="Small-Sample Limitation: "
    )

    # Table 4: Question Complexity Disaggregation
    add_styled_heading(doc, "5.4 Descriptive Subgroup Analysis: Question Complexity", level=2)
    add_body_p(
        doc,
        "Table 4 examines discrete AURC across question complexity typologies. Single-passage questions require extracting evidence from one passage; "
        "multi-passage questions require synthesizing evidence across distinct sections. Note that multi-passage questions comprise only 11 families "
        "(33 controlled contexts), making this an exploratory descriptive finding."
    )

    t4 = doc.add_table(rows=5, cols=7)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    t4.autofit = False
    t4.columns[0].width = Inches(1.8)
    t4.columns[1].width = Inches(0.6)
    t4.columns[2].width = Inches(0.8)
    t4.columns[3].width = Inches(0.8)
    t4.columns[4].width = Inches(0.8)
    t4.columns[5].width = Inches(0.8)
    t4.columns[6].width = Inches(0.9)
    set_table_borders(t4, color="D3D3D3")

    t4_headers = ["Subgroup", "N", "Insuff %", "B1 MaxRel", "M1 Raw", "M2 Dedup", "M4 Dedup"]
    for c_idx, h_text in enumerate(t4_headers):
        cell = t4.rows[0].cells[c_idx]
        set_cell_background(cell, HEX_SECONDARY)
        set_cell_margins(cell, top=35, bottom=35, left=50, right=50)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(8)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    t4_rows = [
        ("Controlled Single-Passage", "75", "66.7%", "0.3166", "0.3743", "0.3472", "0.3388"),
        ("Controlled Multi-Passage", "33", "66.7%", "0.5316", "0.6894", "0.3508", "0.3803"),
        ("All Single-Passage", "100", "50.0%", "0.1617", "0.1739", "0.1662", "0.1671"),
        ("All Multi-Passage", "44", "56.8%", "0.4214", "0.4559", "0.3343", "0.2988")
    ]

    for r_idx, row_vals in enumerate(t4_rows, start=1):
        row = t4.rows[r_idx]
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=26, bottom=26, left=50, right=50)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(8)
            r.font.color.rgb = RGB_TEXT

    sp = doc.add_paragraph()
    format_paragraph(sp, space_before=1, space_after=3)

    add_body_p(
        doc,
        "Descriptive Multi-Passage Observation: On the 11 multi-passage families (33 controlled contexts), deduplicated models exhibit lower absolute "
        "discrete AURC values (M2 = 0.3508, M4 = 0.3803) compared to raw models (M1 = 0.6894, M3 = 0.6900). "
        "However, this is an 11-family descriptive observation and represents a difference in discrete AURC, NOT a '44.8% relative risk reduction' "
        "at a specified operational coverage. It does not establish causal claims about multi-passage reasoning or synthesis capabilities.",
        bold_prefix="Methodological Nuance: "
    )
    doc.add_page_break()

    # Table 5: Duplicate Invariance Property Test
    add_styled_heading(doc, "5.5 Duplicate-Invariance Property Test", level=2)
    add_body_p(
        doc,
        "Table 5 reports the observed absolute score perturbation |Score(C_dup) - Score(C)| when an exact duplicate of the highest-scoring passage "
        "is injected into all 36 supported_candidate test contexts post-selection."
    )

    t5 = doc.add_table(rows=7, cols=5)
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    t5.autofit = False
    t5.columns[0].width = Inches(2.1)
    t5.columns[1].width = Inches(1.3)
    t5.columns[2].width = Inches(1.1)
    t5.columns[3].width = Inches(1.1)
    t5.columns[4].width = Inches(1.1)
    set_table_borders(t5, color="D3D3D3")

    t5_headers = ["Model / Baseline", "Strictly Invariant?", "Mean Abs Shift", "Max Abs Shift", "Std Shift"]
    for c_idx, h_text in enumerate(t5_headers):
        cell = t5.rows[0].cells[c_idx]
        set_cell_background(cell, HEX_SECONDARY)
        set_cell_margins(cell, top=35, bottom=35, left=60, right=60)
        p = cell.paragraphs[0]
        format_paragraph(p, space_before=0, space_after=0)
        r = p.add_run(h_text)
        r.font.name = "Arial"
        r.font.size = Pt(8)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    t5_rows = [
        ("B1 (Max Relevance)", "YES (Identical 0.0)", "0.000000", "0.000000", "0.000000"),
        ("M1 (Relevance Raw)", "NO (Vulnerable)", "0.318703", "0.382319", "0.038141"),
        ("M2 (Relevance Dedup)", "YES (Identical 0.0)", "0.000000", "0.000000", "0.000000"),
        ("M3 (Full Structure Raw)", "NO (Vulnerable)", "0.312393", "0.387396", "0.037412"),
        ("M4 (Full Structure Dedup)", "YES (Identical 0.0)", "0.000000", "0.000000", "0.000000"),
        ("Shortcut (Length Diag.)", "NO (Vulnerable)", "0.473078", "0.478142", "0.001642")
    ]

    for r_idx, row_vals in enumerate(t5_rows, start=1):
        row = t5.rows[r_idx]
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=26, bottom=26, left=60, right=60)
            p = cell.paragraphs[0]
            format_paragraph(p, space_before=0, space_after=0)
            r = p.add_run(val)
            r.font.name = "Arial"
            r.font.size = Pt(8)
            if "YES" in val:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x27, 0xAE, 0x60)
            elif "NO" in val:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
            else:
                r.font.color.rgb = RGB_TEXT

    sp = doc.add_paragraph()
    format_paragraph(sp, space_before=1, space_after=3)

    add_body_p(
        doc,
        "Duplicate Property Bounds: M2 and M4 exhibit exact zero score change specifically for the tested post-selection duplication operation "
        "on fixed contexts. This does NOT establish robustness to paraphrases, different metadata tags, re-retrieval pipeline changes, "
        "or evidence displaced by a strict token budget.",
        bold_prefix="Scope of Invariance: "
    )

    add_styled_heading(doc, "5.6 Operating Thresholds: Achieved vs. Descriptive Matched Coverage", level=2)
    p_th = add_body_p(
        doc,
        "Deployable validation-frozen operating points must be strictly distinguished from descriptive matched-coverage points:\n"
        "1. Deployable Frozen Operating Points: Operating thresholds T = {0.5518, 0.4923, 0.0414} for M1 and T = {0.5381, 0.2657, 0.0621} for M4 "
        "were frozen on the validation set targeting coverages 0.50, 0.75, and 0.90. On the held-out controlled test partition (N=108), M1 realized coverages "
        "of 0.250 (risk 0.259), 0.491 (risk 0.377), and 0.824 (risk 0.596). M4 realized coverages of 0.222 (risk 0.500), 0.546 (risk 0.424), and 0.722 (risk 0.538). "
        "The substantial deviations from target coverage reflect cross-split score distribution shifts.\n"
        "2. Descriptive Matched-Coverage Points: When evaluating curves at exact integer ranks k = round(target * N) under uniform tie-breaking, "
        "M1 and M4 exhibit near-identical expected risks: at matched cov 0.50 (k=54), M1 risk=0.3889 vs M4 risk=0.3951; at matched cov 0.75 (k=81), both exhibit 0.5556 risk.",
        bold_prefix="Threshold Distinction: "
    )
    p_th.runs[0].font.size = Pt(9)
    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 6. Threats to Validity & Disclosures
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "6. Threats to Validity & Methodological Disclosures", level=1)
    
    add_bullet_p(doc, "Test labels were generated through a 12-checkpoint untruncated AI assistant reading protocol with verbatim quotation verification. While every quotation was asserted against context text, labels are AI semantic annotations with prior domain exposure, not independent human judgments.", bold_prefix="1. Same-AI Semantic Annotations: ")
    add_bullet_p(doc, "All model parameters, scalers, and validation thresholds were completely frozen before the new v2 test contexts, questions, and annotations were constructed; however, we fully disclose historical test-paper exposure during an earlier invalid Stage 4 run prior to the v2 benchmark protocol.", bold_prefix="2. Model Freezing & Historical Exposure: ")
    add_bullet_p(doc, "The controlled benchmark uses single-passage replacements (relevant and duplicate distractors), yielding a 66.7% synthetic insufficient prevalence, compared to 8.3% in natural retrieval.", bold_prefix="3. Constructed Benchmark Nature: ")
    add_bullet_p(doc, "The natural retrieval test split contains only 3 insufficient contexts (ctx_te_0064, ctx_te_0112, ctx_te_0124), yielding wide confidence intervals that span zero.", bold_prefix="4. Small Natural Negative Sample: ")
    add_bullet_p(doc, "Shared distractor passages across question families can create residual inter-cluster dependence during bootstrap resampling.", bold_prefix="5. Shared Distractor Dependence: ")
    add_bullet_p(doc, "A documented file re-copy occurred during checkpoint consolidation; the authoritative SHA-256 hash (fcd815fc...) was independently re-verified against raw passages.", bold_prefix="6. Provenance Gap Documentation: ")
    add_bullet_p(doc, "The shortcut diagnostic achieves a discrete AURC of 0.6798; this poor performance establishes only that this specific fitted length-based diagnostic performed poorly, and does not prove the absence of all possible shortcuts or spurious correlations in the benchmark.", bold_prefix="7. Shortcut Bounds: ")
    add_bullet_p(doc, "All experiments were executed locally on CUDA hardware using open-source models; commercial API spend is strictly $0.00.", bold_prefix="8. Zero Commercial API Spend: ")

    # --------------------------------------------------------------------------
    # 7. Conclusion & Future Work
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "7. Conclusion & Future Work", level=1)
    add_body_p(
        doc,
        "This project presents an empirical investigation of evidence-sufficiency gating in scientific question answering. "
        "Our primary hypothesis—that full deduplicated structural gating (M4) achieves lower discrete AURC than raw relevance gating (M1) "
        "on controlled scientific contexts—is NOT supported by the data (observed delta +0.0071, 95% bootstrap interval [-0.0503, +0.0679]). "
        "The interval includes differences favoring either method; this study does not establish equivalence. Simple maximum relevance (B1) "
        "achieves the lowest observed controlled-context AURC (0.4049) on the controlled benchmark.\n\n"
        "Nonetheless, the empirical study yields two valuable architectural findings: (1) exact deduplication confers bounded duplicate invariance "
        "under specified operating assumptions (observed 0.000000 score change under post-selection duplication on fixed contexts), preventing the severe "
        "score perturbations observed in raw relevance models (>0.31 shift); and (2) on multi-passage questions, deduplication avoids the severe "
        "degradation observed in raw models (0.3803 vs 0.6894 discrete AURC). "
        "Future work should expand multi-passage family sizes, incorporate semantic paraphrase deduplication, and evaluate selective abstention directly "
        "within downstream generative synthesis."
    )
    doc.add_page_break()

    # --------------------------------------------------------------------------
    # 8. References
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "8. References", level=1)
    
    refs = [
        "[1] S. Yan, J. Gu, Y. Zhu, and Z. Ling, \"Corrective Retrieval Augmented Generation,\" in Findings of the Association for Computational Linguistics: EMNLP 2024, arXiv preprint arXiv:2401.15884, 2024.",
        "[2] H. Joren, J. Zhang, C. Ferng, et al., \"Sufficient Context: A New Lens on Retrieval Augmented Generation Systems,\" in International Conference on Learning Representations (ICLR 2025), arXiv preprint arXiv:2411.06037, 2024.",
        "[3] S. Chen, P. Hua, and Y. Xiao, \"Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation in RAG,\" arXiv preprint arXiv:2609.37469, 2026.",
        "[4] P. Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,\" in Advances in Neural Information Processing Systems (NeurIPS 2020), vol. 33, pp. 9459–9474, 2020.",
        "[5] Y. Geifman and R. El-Yaniv, \"Selective Classification for Deep Neural Networks,\" in Advances in Neural Information Processing Systems (NeurIPS 2017), vol. 30, 2017.",
        "[6] A. Kamath, R. Jia, and P. Liang, \"Selective Question Answering under Domain Shift,\" in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), pp. 5684–5696, 2020.",
        "[7] N. Varshney, S. Mishra, and C. Baral, \"Can Open-Domain QA Models Abstain When They Cannot Answer?,\" in Findings of the Association for Computational Linguistics: EACL 2023, pp. 883–894, 2023.",
        "[8] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, \"RAGAS: Automated Evaluation of Retrieval Augmented Generation,\" arXiv preprint arXiv:2309.15217, 2023.",
        "[9] D. Edge et al., \"From Local to Global: A Graph RAG Approach to Query-Focused Summarization,\" arXiv preprint arXiv:2404.16130, 2024.",
        "[10] A. Vaswani et al., \"Attention Is All You Need,\" in Advances in Neural Information Processing Systems (NeurIPS 2017), vol. 30, 2017."
    ]

    for ref in refs:
        p_ref = doc.add_paragraph()
        format_paragraph(p_ref, space_before=1, space_after=3, line_spacing=1.1)
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = "Arial"
        r_ref.font.size = Pt(8.5)
        r_ref.font.color.rgb = RGB_TEXT

    doc.save(OUTPUT_DOCX_PATH)
    print(f"Successfully generated teacher submission report at: {OUTPUT_DOCX_PATH}")

if __name__ == "__main__":
    generate_teacher_submission_report()
