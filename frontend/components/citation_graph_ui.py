"""
Citation Graph Visualizer Component.
Generates an interactive PyVis / NetworkX citation network embedded directly into Streamlit.
Academic Scientific Instrument Styling (Dual-Theme Aware, Zero Emojis, Hub Sizing, Tooltips).
"""

import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from backend.contract import CitationGraphData
from frontend.styles.theme import get_theme_colors, is_dark_mode


def render_citation_graph(graph_data: CitationGraphData, height: int = 600):
    """Render interactive paper citation graph network with theme-aware academic styling."""
    colors = get_theme_colors()
    dark = is_dark_mode()

    try:
        net = Network(
            height=f"{height}px",
            width="100%",
            bgcolor=colors["card_bg"],
            font_color=colors["text_primary"],
            directed=True,
            notebook=False
        )

        font_color = "#F3F4F6" if dark else "#111827"
        edge_default = "#5B7FB5" if dark else "#2B4C7E"
        edge_cross = "#F87171" if dark else "#DC2626"
        edge_highlight = "#93C5FD" if dark else "#1E40AF"

        # Add Nodes with category colors and degree-weighted sizing
        for node in graph_data.nodes:
            deg = node.get("degree", node.get("in_degree", 0) + node.get("out_degree", 0))
            is_hub = deg >= 6
            cat = node.get("category", node.get("group", "cs.CL"))
            
            # Palette: cs.CL -> Blue, cs.CV -> Amber, cs.AI/other -> Emerald
            if "CV" in cat:
                base_color = "#FBBF24" if dark else "#D97706"
                border_color = "#FDE68A" if dark else "#B45309"
            elif "AI" in cat or "LG" in cat:
                base_color = "#34D399" if dark else "#059669"
                border_color = "#A7F3D0" if dark else "#047857"
            else:
                base_color = "#60A5FA" if dark else "#2563EB"
                border_color = "#BFDBFE" if dark else "#1D4ED8"

            net.add_node(
                n_id=node["id"],
                label=node["label"],
                title=node["title"],
                group=cat,
                color={
                    "background": base_color,
                    "border": border_color,
                    "highlight": {"background": "#FFFFFF" if dark else "#111827", "border": edge_highlight},
                    "hover": {"background": "#E2E8F0" if dark else "#334155", "border": edge_highlight}
                },
                shape="dot",
                size=node.get("val", 15) if not is_hub else max(24, node.get("val", 28)),
                borderWidth=2.5 if is_hub else 1.2,
                borderWidthSelected=3.5
            )

        # Add Edges
        for edge in graph_data.edges:
            is_cross = edge.get("is_cross", False)
            edge_color = edge_cross if is_cross else edge_default
            
            net.add_edge(
                source=edge["from"],
                to=edge["to"],
                title=edge.get("label", "CITES"),
                value=edge.get("weight", 1.0),
                color={
                    "color": edge_color,
                    "highlight": edge_highlight,
                    "hover": edge_highlight,
                    "opacity": 0.85 if dark else 0.78
                },
                width=2.0 if is_cross else 1.5,
                dashes=True if is_cross else False
            )

        net.set_options(f"""
        var options = {{
          "nodes": {{
            "font": {{
              "size": 12,
              "face": "Inter, -apple-system, sans-serif",
              "color": "{font_color}",
              "strokeWidth": 2,
              "strokeColor": "{colors['bg']}"
            }},
            "shadow": false
          }},
          "edges": {{
            "smooth": {{"type": "curvedCW", "roundness": 0.15}},
            "arrows": {{"to": {{"enabled": true, "scaleFactor": 0.55}}}},
            "selectionWidth": 2.5
          }},
          "physics": {{
            "barnesHut": {{
              "gravitationalConstant": -3800,
              "centralGravity": 0.22,
              "springLength": 140,
              "springConstant": 0.04,
              "damping": 0.2,
              "avoidOverlap": 0.65
            }},
            "minVelocity": 0.75,
            "solver": "barnesHut"
          }},
          "interaction": {{
            "hover": true,
            "tooltipDelay": 150,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": false
          }}
        }}
        """)

        # Save to temporary file and customize iframe container
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            with open(tmp_file.name, "r", encoding="utf-8") as f:
                html_content = f.read()

        # Clean default pyvis grey border and apply card border
        html_content = html_content.replace(
            "border: 1px solid lightgray;",
            f"border: 1px solid {colors['border']}; border-radius: 8px; background: {colors['card_bg']};"
        )

        components.html(html_content, height=height + 10, scrolling=False)

    except Exception as e:
        st.warning(f"Citation Graph Interactive Renderer Note: {e}")
        st.info("Displaying fallback citation relationship summary table.")
        import pandas as pd
        nodes_df = pd.DataFrame(graph_data.nodes)
        st.dataframe(nodes_df)
