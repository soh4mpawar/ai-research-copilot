"""
AI Research Copilot — Master Streamlit Application.
Main router, sidebar navigation, custom theme injection, singleton model caching (FR-19), and page state management.
Owned by S (Soham Pawar).
"""

import sys
import os
import streamlit as st

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.components.header import render_header
from frontend.pages_ui.research_page import render_research_page
from frontend.pages_ui.lit_review_page import render_lit_review_page
from frontend.pages_ui.paper_explorer_page import render_paper_explorer_page
from frontend.pages_ui.citation_graph_page import render_citation_graph_page
from frontend.pages_ui.eval_dashboard_page import render_eval_dashboard_page

# Page Configuration
st.set_page_config(
    page_title="AI Research Copilot | Scientific Literature RAG Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom Glassmorphism Styling (v1.0.8 - Day 6 Phase 4 Integration)
css_file = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Application-Scoped Singleton Model & Client Caching (FR-19)
@st.cache_resource
def get_cached_orchestrator():
    """Cache heavyweight models & DB connections as application singletons (FR-19)."""
    try:
        from backend.pipeline import get_orchestrator
        return get_orchestrator()
    except Exception:
        return None


# Query Result Caching Keyed by (query, corpus_version) per FR-19
@st.cache_data(show_spinner=False)
def execute_cached_query(query_text: str, mode: str, corpus_version: str):
    """Cache query results keyed by (query, corpus_version) to prevent redundant API calls on page refresh (FR-19)."""
    from backend import research_engine
    return research_engine.query(query_text, mode=mode)


def main():
    """Main application navigation router."""
    # Top Header Component
    render_header()

    # Pre-warm heavyweight singletons (FR-19)
    orchestrator = get_cached_orchestrator()
    corpus_v = orchestrator.corpus_version if orchestrator else "corpus_v1.0_default"

    # Sidebar Navigation Menu
    st.sidebar.markdown("## 🧭 Navigation")
    
    page_choice = st.sidebar.radio(
        "Select System View:",
        [
            "⌂ Research QA Engine",
            "📚 Literature Review Studio",
            "📄 Scientific Paper Explorer",
            "🕸 Citation Network Graph",
            "📊 Evaluation & RAGAS Dashboard"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Configuration")
    
    use_mock = st.sidebar.checkbox(
        "Use Mock Backend Engine",
        value=False,
        help="Uncheck to run real backend pipeline (ChromaDB + BM25 + RRF + bge-reranker-base + Gemini 3.5 Flash Lite)."
    )
    os.environ["USE_MOCK_ENGINE"] = "true" if use_mock else "false"

    st.sidebar.markdown(f"<div style='font-size:0.75rem; color:#94a3b8; margin-top:4px;'>Corpus Version: <code>{corpus_v}</code></div>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.5;">
            <b>AI Research Copilot v1.0</b><br/>
            Created by <b>Soham Pawar</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Page Router Execution
    if page_choice == "⌂ Research QA Engine":
        render_research_page()
    elif page_choice == "📚 Literature Review Studio":
        render_lit_review_page()
    elif page_choice == "📄 Scientific Paper Explorer":
        render_paper_explorer_page()
    elif page_choice == "🕸 Citation Network Graph":
        render_citation_graph_page()
    elif page_choice == "📊 Evaluation & RAGAS Dashboard":
        render_eval_dashboard_page()


if __name__ == "__main__":
    main()
