"""
Citation Network Graph Page Layout.
Interactive paper citation graph network visualizer and relational explorer.
Academic Scientific Instrument Styling.
"""

import streamlit as st
from backend import research_engine
from frontend.components.citation_graph_ui import render_citation_graph


def render_citation_graph_page():
    """Render Citation Graph UI."""
    st.markdown("<div class='academic-title' style='font-size: 1.5rem; margin-bottom: 4px;'>🕸 Interactive Scientific Citation Network Graph</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color: #525252; font-size: 0.92rem; margin-bottom: 14px;'>Explore citation relationships, foundational paper seeds, and cross-document lineage in the indexed corpus.</div>",
        unsafe_allow_html=True
    )

    graph_data = research_engine.get_citation_graph()

    # Graph Network Stats Banner
    n_nodes = len(graph_data.nodes)
    n_edges = len(graph_data.edges)
    density = (n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
    avg_deg = (n_edges / n_nodes) if n_nodes > 0 else 0.0

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Graph Nodes (Papers)", n_nodes)
    with s2:
        st.metric("Citation Edges (Links)", n_edges)
    with s3:
        st.metric("Graph Density", f"{density:.4f}")
    with s4:
        st.metric("Avg In-Degree", f"{avg_deg:.2f} links")

    st.markdown("---")

    # Render PyVis Graph
    render_citation_graph(graph_data, height=580)

    st.info(
        "💡 **Network Navigation**: Click and drag nodes to inspect paper clusters. "
        "Foundational seed papers (e.g. *Attention Is All You Need*, *BERT*, *RAG*) appear as darker primary hubs."
    )
