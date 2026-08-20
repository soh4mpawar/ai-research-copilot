"""
Citation Network Graph Page Layout.
Interactive paper citation graph network visualizer and relational explorer.
Academic Scientific Instrument Styling (Dual-Theme Aware, Subgraph Filtering, Hub Emphasis, Legend).
"""

import streamlit as st
import pandas as pd
from backend import research_engine
from backend.contract import CitationGraphData
from frontend.components.citation_graph_ui import render_citation_graph
from frontend.components.icons import svg_icon
from frontend.components.data_table import render_academic_table
from frontend.styles.theme import get_theme_colors


def render_citation_graph_page():
    """Render Citation Graph UI."""
    colors = get_theme_colors()

    # Unboxed Page Header
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

    raw_graph_data = research_engine.get_citation_graph()

    # Controls Strip (Filter connected vs isolated, domain, degree)
    col_filter1, col_filter2, col_filter3 = st.columns([2, 1, 1])
    
    with col_filter1:
        only_connected = st.checkbox(
            "Filter to Connected Citation Network (hide isolated papers)",
            value=True
        )
    with col_filter2:
        domain_choice = st.selectbox(
            "Filter Category:",
            ["All Domains", "cs.CL (NLP)", "cs.CV (Vision)"]
        )
    with col_filter3:
        min_connections = st.selectbox(
            "Min Connections:",
            ["All Connected (1+)", "Major Hubs Only (4+)", "All Papers (0+)"],
            index=0 if only_connected else 2
        )

    # Determine connection threshold
    if min_connections == "Major Hubs Only (4+)":
        min_deg = 4
    elif min_connections == "All Connected (1+)" or only_connected:
        min_deg = 1
    else:
        min_deg = 0

    # Filter Nodes
    filtered_nodes = []
    filtered_node_ids = set()

    for node in raw_graph_data.nodes:
        deg = node.get("degree", node.get("in_degree", 0) + node.get("out_degree", 0))
        cat = node.get("category", node.get("group", "cs.CL"))
        
        if deg < min_deg:
            continue
            
        if domain_choice == "cs.CL (NLP)" and "CL" not in cat:
            continue
        elif domain_choice == "cs.CV (Vision)" and "CV" not in cat:
            continue

        filtered_nodes.append(node)
        filtered_node_ids.add(node["id"])

    # Filter Edges (both endpoints must be in filtered nodes)
    filtered_edges = [
        e for e in raw_graph_data.edges
        if e["from"] in filtered_node_ids and e["to"] in filtered_node_ids
    ]

    filtered_graph_data = CitationGraphData(nodes=filtered_nodes, edges=filtered_edges)

    # Graph Network Stats as Compact Horizontal Key-Value Strip
    n_nodes = len(filtered_nodes)
    n_edges = len(filtered_edges)
    density = (n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
    avg_deg = (n_edges / n_nodes) if n_nodes > 0 else 0.0

    st.markdown(
        f"""
        <div style="display: flex; gap: 14px; align-items: center; flex-wrap: wrap; font-size: 0.78rem; color: {colors['text_secondary']}; padding-bottom: 8px; border-bottom: 1px solid {colors['border']}; margin-bottom: 10px;">
            <span style="display: flex; align-items: center; gap: 4px;">
                {svg_icon('network', size=13, color=colors['text_secondary'])}
                Showing: <code style="font-size:0.82rem;">{n_nodes} nodes</code>
            </span>
            <span>·</span>
            <span>Edges: <code style="font-size:0.82rem;">{n_edges} links</code></span>
            <span>·</span>
            <span>Graph Density: <code style="font-size:0.82rem;">{density:.4f}</code></span>
            <span>·</span>
            <span>Avg Degree: <code style="font-size:0.82rem;">{avg_deg:.2f} links/paper</code></span>
            <span>·</span>
            <span>Corpus Baseline: <code style="font-size:0.82rem;">54 connected / 200 total</code></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Visual Legend Bar
    st.markdown(
        f"""
        <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap; font-size: 0.76rem; color: {colors['text_secondary']}; margin-bottom: 12px; padding: 6px 12px; background: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 6px;">
            <span style="font-weight: 600; color: {colors['text_primary']};">Legend:</span>
            <span style="display: inline-flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background-color: #3B82F6;"></span> cs.CL (NLP / Language)
            </span>
            <span style="display: inline-flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background-color: #F59E0B;"></span> cs.CV (Computer Vision)
            </span>
            <span style="display: inline-flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background-color: #10B981;"></span> cs.AI / Other
            </span>
            <span style="display: inline-flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 14px; height: 2px; background-color: #5B7FB5;"></span> Intra-Domain Link
            </span>
            <span style="display: inline-flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 14px; height: 2px; background-color: #EF4444; border-top: 2px dashed #EF4444;"></span> Cross-Domain Citation
            </span>
            <span style="display: inline-flex; align-items: center; gap: 5px;">
                <span style="display: inline-block; width: 13px; height: 13px; border-radius: 50%; background-color: #3B82F6; border: 2px solid #93C5FD;"></span> Major Hub Paper
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render PyVis Graph
    render_citation_graph(filtered_graph_data, height=560)

    # Primary Hub Papers Table & Analysis
    with st.expander("Inspect Primary Citation Hubs & Cross-Domain Lineage", expanded=False):
        hub_data = []
        for n in raw_graph_data.nodes:
            deg = n.get("degree", n.get("in_degree", 0) + n.get("out_degree", 0))
            if deg >= 4:
                hub_data.append({
                    "ArXiv ID": n["id"],
                    "Paper Title": n.get("full_title", n.get("label", "")),
                    "Category": n.get("category", "cs.CL"),
                    "Total Connections": deg,
                    "Cited By (In)": n.get("in_degree", 0),
                    "Cites (Out)": n.get("out_degree", 0),
                    "Corpus Year": n.get("year", 2026)
                })
        
        if hub_data:
            hub_df = pd.DataFrame(hub_data).sort_values("Total Connections", ascending=False)
            render_academic_table(hub_df)

    st.markdown(
        f"""
        <div style="font-size: 0.78rem; color: {colors['text_secondary']}; margin-top: 8px; line-height: 1.5;">
            <b>Corpus Scope Note</b>: In a closed 200-paper collection, 54 papers form 87 mutual internal citations connecting to foundational hubs (<i>Attention</i>, <i>RAG</i>, <i>ResNet</i>, <i>BERT</i>, <i>Swin</i>). The remaining 146 recent papers cite literature outside this corpus.
        </div>
        """,
        unsafe_allow_html=True
    )
