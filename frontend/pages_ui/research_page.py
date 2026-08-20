"""
Research QA & Synthesis Engine Page Layout.
Directs technical query execution across hybrid retrieval, cross-reranking, and grounded answer synthesis.
Academic Scientific Instrument Styling (Zero Emojis, Clean Inline SVGs, Native Type-Ahead Autocomplete).
"""

import streamlit as st
from backend import research_engine
from frontend.components.answer_card import render_answer_card
from frontend.components.evidence_viewer import render_evidence_viewer
from frontend.components.icons import svg_icon
from frontend.styles.theme import get_theme_colors


PRESET_RESEARCH_QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG) and why was it introduced?",
    "How does Reciprocal Rank Fusion (RRF) combine dense and sparse BM25 scores?",
    "What are the key computational limitations of transformer self-attention mechanisms?",
]


def render_research_page():
    """Render Research QA Engine UI."""
    colors = get_theme_colors()

    # Unboxed Page Header
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div class="academic-title" style="font-size: 1.45rem; margin-bottom: 2px; color: {colors['text_primary']};">
                Scientific Query &amp; Answer Engine
            </div>
            <div style="color: {colors['text_secondary']}; font-size: 0.88rem;">
                Ask technical research questions across the indexed literature with source-grounded evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    health = research_engine.get_system_health()

    st.markdown(
        f"""
        <div style="font-size: 0.82rem; color: {colors['text_secondary']}; margin-bottom: 16px;">
            Corpus Index: <b>{health['total_papers']} papers</b> ({health['domain_breakdown'].get('NLP', 0)} NLP / {health['domain_breakdown'].get('CV', 0)} CV) &nbsp;·&nbsp;
            Docling Parsed: <b>{health['successfully_parsed']}/{health['total_papers']}</b> ({health['parse_success_rate']}%) &nbsp;·&nbsp;
            Total Chunks: <b>{health['total_estimated_chunks']:,}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Query Input (Native Autocomplete with Type-Ahead & Free-Form Typing) & Mode Controls
    col_input, col_mode = st.columns([3, 1])
    
    with col_input:
        selected_query = st.selectbox(
            "Enter scientific research question:",
            options=PRESET_RESEARCH_QUESTIONS,
            index=None,
            accept_new_options=True,
            filter_mode="fuzzy",
            placeholder="Type to search presets or enter custom question...",
            key="query_input_selectbox"
        )
        query_text = str(selected_query).strip() if selected_query else ""

    with col_mode:
        search_mode = st.selectbox(
            "Research Mode:",
            ["Question Answering (QA)", "Multi-Paper Summary", "Literature Review"]
        )

    submit_clicked = st.button("Run Research Pipeline", type="primary")

    # State preservation for query results
    if "active_result" not in st.session_state:
        st.session_state["active_result"] = None
        st.session_state["active_query"] = None

    # Process Query Execution on Submit
    if submit_clicked:
        if not query_text:
            st.warning("Please enter or select a research question first.")
        else:
            mode_key = "qa"
            if search_mode == "Multi-Paper Summary":
                mode_key = "summary"
            elif search_mode == "Literature Review":
                mode_key = "literature_review"

            with st.spinner("Executing hybrid retrieval (ChromaDB + BM25s) ➜ bge-reranker-base ➜ Gemini 3.5 Flash Lite..."):
                result = research_engine.query(query_text, mode=mode_key)
                st.session_state["active_result"] = result
                st.session_state["active_query"] = query_text

    # Render Results
    if st.session_state.get("active_result"):
        result = st.session_state["active_result"]
        active_q = st.session_state.get("active_query", query_text)

        st.markdown(f"<div style='border-top: 1px solid {colors['border']}; margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)

        # Action Bar: Download Report
        report_md = f"# Research Report: {active_q}\n\n{result.answer}\n\n## Sources\n"
        for idx, s in enumerate(result.sources, 1):
            report_md += f"{idx}. {s.title} ({s.year}) - {', '.join(s.authors[:3])}\n"

        col_dl, col_blank = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="Download Report (.md)",
                data=report_md,
                file_name=f"Research_Report_{active_q[:20].replace(' ', '_')}.md",
                mime="text/markdown"
            )

        # Tabbed View for Answer vs Evidence (Zero Emojis)
        tab1, tab2 = st.tabs(["Grounded Synthesis & Sources", "Retrieval Pipeline Transparency"])

        with tab1:
            render_answer_card(result)

        with tab2:
            render_evidence_viewer(result)
