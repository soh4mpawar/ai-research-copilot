"""
Citation Graph Visualizer Component.
Generates an interactive PyVis / NetworkX citation network embedded directly into Streamlit.
Academic Scientific Instrument Styling (Dual-Theme Aware, Zero Emojis).
"""

import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from backend.contract import CitationGraphData
from frontend.styles.theme import get_theme_colors, is_dark_mode


def render_citation_graph(graph_data: CitationGraphData, height: int = 580):
    """Render interactive paper citation graph network with theme-aware academic styling."""
    colors = get_theme_colors()
    dark = is_dark_mode()

    try:
        net = Network(
            height=f"{height}px",
            width="100%",
            bgcolor=colors["bg"],
            font_color=colors["text_primary"],
            directed=True,
            notebook=False
        )

        # Add Nodes with theme-aware color scheme
        for node in graph_data.nodes:
            is_seed = node.get("group") == "Seed" or node.get("val", 15) > 20
            color = colors["pyvis_node_primary"] if is_seed else colors["pyvis_node_secondary"]
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
                color=colors["pyvis_edge"]
            )

        font_color = "#F3F4F6" if dark else "#111827"
        highlight_color = "#7AA2DC" if dark else "#2B4C7E"

        net.set_options(f"""
        var options = {{
          "nodes": {{
            "font": {{"size": 13, "face": "Inter", "color": "{font_color}"}},
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "shadow": false
          }},
          "edges": {{
            "color": {{"color": "{colors['pyvis_edge']}", "highlight": "{highlight_color}"}},
            "smooth": {{"type": "continuous"}},
            "arrows": {{"to": {{"enabled": true, "scaleFactor": 0.5}}}}
          }},
          "physics": {{
            "barnesHut": {{
              "gravitationalConstant": -3500,
              "centralGravity": 0.25,
              "springLength": 100
            }},
            "minVelocity": 0.75
          }}
        }}
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
