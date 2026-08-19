"""
Citation Network Graph Page Layout.
Interactive paper citation graph network visualizer and relational explorer.
Academic Scientific Instrument Styling (Zero Emojis, Clean Inline SVGs, Dual-Theme Aware).
"""

import streamlit as st
from backend import research_engine
from frontend.components.citation_graph_ui import render_citation_graph
from frontend.components.icons import svg_icon
from frontend.styles.theme import get_theme_colors


def render_citation_graph_page():
    """Render Citation Graph UI."""
    colors = get_theme_colors()

    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div class="academic-title" style="font-size: 1.45rem; margin-bottom: 2px; color: {colors['text_primary']};">
                Interactive Scientific Citation Network Graph
            </div>
            <div style="color: {colors['text_secondary']}; font-size: 0.88rem;">
                Explore citation relationships, foundational paper seeds, and cross-document lineage in the indexed corpus.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    graph_data = research_engine.get_citation_graph()

    # Graph Network Stats as Compact Horizontal Key-Value Strip
    n_nodes = len(graph_data.nodes)
    n_edges = len(graph_data.edges)
    density = (n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
    avg_deg = (n_edges / n_nodes) if n_nodes > 0 else 0.0

    st.markdown(
        f"""
        <div style="display: flex; gap: 14px; align-items: center; flex-wrap: wrap; font-size: 0.78rem; color: {colors['text_secondary']}; padding-bottom: 8px; border-bottom: 1px solid {colors['border']}; margin-bottom: 16px;">
            <span style="display: flex; align-items: center; gap: 4px;">
                {svg_icon('network', size=13, color=colors['text_secondary'])}
                Nodes: <code style="font-size:0.82rem;">{n_nodes} papers</code>
            </span>
            <span>·</span>
            <span>Edges: <code style="font-size:0.82rem;">{n_edges} links</code></span>
            <span>·</span>
            <span>Density: <code style="font-size:0.82rem;">{density:.4f}</code></span>
            <span>·</span>
            <span>Avg Degree: <code style="font-size:0.82rem;">{avg_deg:.2f} links/paper</code></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render PyVis Graph
    render_citation_graph(graph_data, height=580)

    st.markdown(
        f"""
        <div style="font-size: 0.80rem; color: {colors['text_secondary']}; margin-top: 10px; line-height: 1.5;">
            <b>Network Navigation</b>: Click and drag nodes to inspect paper clusters. 
            Foundational seed papers (e.g. <i>Attention Is All You Need</i>, <i>BERT</i>, <i>RAG</i>) appear as primary hubs.
        </div>
        """,
        unsafe_allow_html=True
    )
