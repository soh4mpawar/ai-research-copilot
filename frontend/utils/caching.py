"""
Frontend Caching Utilities (FR-19).
Application-scoped singleton caching for heavyweight models/clients and query result caching.
"""

import streamlit as st


@st.cache_resource
def get_cached_orchestrator():
    """Cache heavyweight models & DB connections as application singletons (FR-19)."""
    try:
        from backend.pipeline import get_orchestrator
        return get_orchestrator()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def execute_cached_query(query_text: str, mode: str, corpus_version: str):
    """Cache query results keyed by (query, corpus_version) to prevent redundant API calls on page refresh (FR-19)."""
    from backend import research_engine
    return research_engine.query(query_text, mode=mode)
