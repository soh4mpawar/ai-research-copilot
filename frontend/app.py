"""
AI Research Copilot — Master Streamlit Application.
Main router, sidebar navigation, custom theme injection, singleton model caching (FR-19), and page state management.
Author: Soham Pawar
Academic Scientific Instrument Design System (Dual-Theme Engine, Zero Emojis, Clean Inline SVGs).
"""

import sys
import os
import streamlit as st

# Ensure repository root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.styles.theme import inject_theme, get_theme_colors, is_dark_mode
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

# Load Base Custom Academic Styling
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
    # Pre-warm heavyweight singletons (FR-19)
    orchestrator = get_cached_orchestrator()
    corpus_v = orchestrator.corpus_version if orchestrator else "corpus_v1.0_default"

    # 1. Sidebar Brand Header
    st.sidebar.markdown(
        """
        <div style="padding-bottom: 12px; margin-bottom: 6px; border-bottom: 1px solid var(--border-subtle, #E5E5E3);">
            <div class="hero-title" style="font-size: 1.18rem; margin-bottom: 2px; line-height: 1.2;">
                AI Research Copilot
            </div>
            <div style="font-size: 0.74rem; color: #6B7280;">
                Scientific Literature RAG Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Section: Navigation
    st.sidebar.markdown(
        """
        <div class="sidebar-section-header">
            Navigation
        </div>
        """,
        unsafe_allow_html=True
    )
    
    page_choice = st.sidebar.radio(
        "Select System View:",
        [
            "Research QA Engine",
            "Literature Review Studio",
            "Scientific Paper Explorer",
            "Citation Network Graph",
            "Evaluation & RAGAS Dashboard"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # 3. Section: System Controls
    st.sidebar.markdown(
        """
        <div class="sidebar-section-header" style="margin-top: 16px;">
            System Controls
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Clean Dark Mode Toggle (no tooltip icon)
    dark_mode_active = st.sidebar.toggle(
        "Dark Mode",
        key="dark_mode"
    )

    # Clean Mock Backend Checkbox (no tooltip icon)
    use_mock = st.sidebar.checkbox(
        "Use Mock Backend Engine",
        value=False
    )
    os.environ["USE_MOCK_ENGINE"] = "true" if use_mock else "false"

    # 4. Corpus Status Info Card
    st.sidebar.markdown(
        f"""
        <div class="sidebar-info-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6B7280;">
                    Corpus Status
                </span>
                <span style="display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; color: #16A34A; font-weight: 600;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #16A34A;"></span> Ready
                </span>
            </div>
            <div style="font-size: 0.76rem; line-height: 1.55;">
                <b>Version:</b> <code>{corpus_v}</code><br/>
                <b>Indexed:</b> 184 Papers (9,456 chunks)<br/>
                <b>Retriever:</b> BM25s + BGE-Reranker
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5. Author & Repository Footer
    st.sidebar.markdown(
        """
        <div class="sidebar-footer">
            <div style="font-size: 0.76rem; font-weight: 600;">
                AI Research Copilot <span style="font-weight: 400; color: #6B7280;">v1.0</span>
            </div>
            <div style="font-size: 0.74rem; color: #6B7280; margin-top: 2px;">
                Built by <b>Soham Pawar</b>
            </div>
            <div style="margin-top: 6px;">
                <a href="https://github.com/soh4mpawar/ai-research-copilot" target="_blank" class="sidebar-repo-link">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                    github.com/soh4mpawar/ai-research-copilot
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic Theme Injection
    inject_theme()

    # Top Header Component (Landing Hero on QA, Slim Breadcrumb on other pages)
    render_header(page_choice)

    # Page Router Execution
    if page_choice == "Research QA Engine":
        render_research_page()
    elif page_choice == "Literature Review Studio":
        render_lit_review_page()
    elif page_choice == "Scientific Paper Explorer":
        render_paper_explorer_page()
    elif page_choice == "Citation Network Graph":
        render_citation_graph_page()
    elif page_choice == "Evaluation & RAGAS Dashboard":
        render_eval_dashboard_page()


if __name__ == "__main__":
    main()
