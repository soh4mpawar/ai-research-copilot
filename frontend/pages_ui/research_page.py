"""
Research QA Page Layout.
Main interface for submitting scientific queries, selecting search modes,
viewing grounded answers, and downloading Markdown reports.
Academic Scientific Instrument Styling.
"""

import streamlit as st
from backend import research_engine
from backend.corpus import CorpusManager
from frontend.components.answer_card import render_answer_card
from frontend.components.evidence_viewer import render_evidence_viewer

# Initialize Corpus Manager for Corpus Stats
corpus_mgr = CorpusManager()


def render_research_page():
    """Render main Research QA interface."""
    st.markdown("<div class='academic-title' style='font-size: 1.5rem; margin-bottom: 4px;'>⌂ Scientific Query & Answer Engine</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #525252; font-size: 0.92rem; margin-bottom: 14px;'>Ask technical research questions across 184 scientific papers with grounded citation provenance.</div>", unsafe_allow_html=True)

    # Corpus Health Summary Bar
    health = corpus_mgr.get_corpus_health_summary()
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E5E5E3; border-radius: 6px; padding: 8px 14px; margin-bottom: 16px; font-size: 0.82rem; color: #4B5563; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <span>📚 <b>Corpus Index</b>: {health['total_papers']} papers ({health['nlp_papers_count']} NLP / {health['cv_papers_count']} CV)</span>
            <span>📑 <b>Docling Parsed</b>: {health['successfully_parsed']}/{health['total_papers']} ({health['parse_success_rate']}%)</span>
            <span>🧩 <b>Total Chunks</b>: {health['total_estimated_chunks']:,}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize Session State Query
    if "query_input" not in st.session_state:
        st.session_state["query_input"] = "What is Retrieval-Augmented Generation (RAG) and why was it introduced?"

    # Preset Example Query Chips
    st.markdown("<span style='font-size: 0.82rem; color: #6B7280; font-weight: 500;'>Quick Preset Queries:</span>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What is RAG and why was it introduced?"):
            st.session_state["query_input"] = "What is Retrieval-Augmented Generation (RAG) and why was it introduced?"
    with col2:
        if st.button("How does RRF fuse BM25 and Vector Search?"):
            st.session_state["query_input"] = "How does Reciprocal Rank Fusion (RRF) combine dense and sparse BM25 scores?"
    with col3:
        if st.button("What are transformer self-attention limits?"):
            st.session_state["query_input"] = "What are the key computational limitations of transformer self-attention mechanisms?"

    # Query Input & Mode Controls
    col_input, col_mode = st.columns([3, 1])
    
    with col_input:
        query_text = st.text_input(
            "Enter your scientific research question:",
            key="query_input",
            placeholder="e.g. Compare dense passage retrieval with sparse BM25 search..."
        )

    with col_mode:
        search_mode = st.selectbox(
            "Research Mode:",
            ["Question Answering (QA)", "Multi-Paper Summary", "Literature Review"]
        )

    submit_clicked = st.button("Run Research Pipeline", type="primary")

    # Process Query Execution
    if submit_clicked or query_text.strip():
        mode_key = "qa"
        if search_mode == "Multi-Paper Summary":
            mode_key = "summary"
        elif search_mode == "Literature Review":
            mode_key = "literature_review"

        with st.spinner("Executing hybrid retrieval (ChromaDB + BM25s) ➜ bge-reranker-base ➜ Gemini 3.5 Flash Lite..."):
            result = research_engine.query(query_text, mode=mode_key)

        st.markdown("---")

        # Action Bar: Download Report
        report_md = f"# Research Report: {query_text}\n\n{result.answer}\n\n## Sources\n"
        for idx, s in enumerate(result.sources, 1):
            report_md += f"{idx}. {s.title} ({s.year}) - {', '.join(s.authors[:3])}\n"

        col_dl, col_blank = st.columns([1, 3])
        with col_dl:
            st.download_button(
                label="📥 Download Answer Report (.md)",
                data=report_md,
                file_name=f"Research_Report_{query_text[:20].replace(' ', '_')}.md",
                mime="text/markdown"
            )

        # Tabbed View for Answer vs Evidence
        tab1, tab2 = st.tabs(["📝 Grounded Answer & Citations", "🔎 Retrieval Evidence Transparency"])

        with tab1:
            render_answer_card(result)

        with tab2:
            render_evidence_viewer(result)
