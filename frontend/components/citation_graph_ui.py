"""
Citation Graph Visualizer Component.
Generates an interactive PyVis / NetworkX citation network embedded directly into Streamlit.
"""

import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from backend.contract import CitationGraphData


def render_citation_graph(graph_data: CitationGraphData, height: int = 580):
    """Render interactive paper citation graph network."""
    try:
        net = Network(
            height=f"{height}px",
            width="100%",
            bgcolor="#0b0f19",
            font_color="#e2e8f0",
            directed=True,
            notebook=False
        )

        # Add Nodes
        for node in graph_data.nodes:
            net.add_node(
                n_id=node["id"],
                label=node["label"],
                title=node["title"],
                group=node.get("group", "Paper"),
                value=node.get("val", 15)
            )

        # Add Edges
        for edge in graph_data.edges:
            net.add_edge(
                source=edge["from"],
                to=edge["to"],
                title=edge.get("label", "cites"),
                value=edge.get("weight", 1)
            )

        net.set_options("""
        var options = {
          "nodes": {
            "font": {"size": 14, "face": "Inter"},
            "shadow": true
          },
          "edges": {
            "color": {"inherit": "from"},
            "smooth": {"type": "continuous"}
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -4000,
              "centralGravity": 0.3,
              "springLength": 110
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
