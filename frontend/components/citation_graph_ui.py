"""
Citation Graph Visualizer Component.
Generates an interactive PyVis / NetworkX citation network embedded directly into Streamlit.
Academic Scientific Instrument Styling.
"""

import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from backend.contract import CitationGraphData


def render_citation_graph(graph_data: CitationGraphData, height: int = 580):
    """Render interactive paper citation graph network with light academic styling."""
    try:
        net = Network(
            height=f"{height}px",
            width="100%",
            bgcolor="#FAFAF9",
            font_color="#111827",
            directed=True,
            notebook=False
        )

        # Add Nodes with academic color scheme
        for node in graph_data.nodes:
            is_seed = node.get("group") == "Seed" or node.get("val", 15) > 20
            color = "#2B4C7E" if is_seed else "#64748B"
            net.add_node(
                n_id=node["id"],
                label=node["label"],
                title=node["title"],
                group=node.get("group", "Paper"),
                color=color,
                value=node.get("val", 15)
            )

        # Add Edges
        for edge in graph_data.edges:
            net.add_edge(
                source=edge["from"],
                to=edge["to"],
                title=edge.get("label", "cites"),
                value=edge.get("weight", 1),
                color="#CBD5E1"
            )

        net.set_options("""
        var options = {
          "nodes": {
            "font": {"size": 13, "face": "Inter", "color": "#111827"},
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "shadow": false
          },
          "edges": {
            "color": {"color": "#CBD5E1", "highlight": "#2B4C7E"},
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3500,
              "centralGravity": 0.25,
              "springLength": 100
            },
            "minVelocity": 0.75
          }
        }
        """)

        # Save to temp file and render via Streamlit HTML component
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            with open(tmp_file.name, "r", encoding="utf-8") as f:
                html_content = f.read()

        components.html(html_content, height=height + 20, scrolling=False)

    except Exception as e:
        st.warning(f"Citation Graph Interactive Renderer Note: {e}")
        st.info("Displaying fallback citation relationship summary table.")
        import pandas as pd
        nodes_df = pd.DataFrame(graph_data.nodes)
        st.dataframe(nodes_df)
