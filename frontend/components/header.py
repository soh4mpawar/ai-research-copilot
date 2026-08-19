"""
Top Header and System Metadata Banner Component.
Academic Scientific Instrument Styling with Single-Line Breadcrumb for Inner Pages.
Dual-Theme (Light/Dark) Aware, Zero Emojis, Clean Lucide SVG Icons.
"""

import streamlit as st
from frontend.components.icons import svg_icon
from frontend.styles.theme import get_theme_colors


def render_header(page_name: str = "Research QA Engine"):
    """
    Render main landing hero header on QA Engine,
    or a slim, single-line breadcrumb navigation bar on all other modules.
    """
    colors = get_theme_colors()

    if page_name in ["Research QA Engine", "⌂ Research QA Engine"]:
        # Landing Hero Masthead (Clean, unboxed masthead)
        st.markdown(
            f"""
            <div style="border-bottom: 1px solid {colors['border']}; padding-bottom: 14px; margin-bottom: 18px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div class="hero-title" style="font-size: 1.85rem; margin-bottom: 2px; color: {colors['text_primary']};">
                            AI Research Copilot
                        </div>
                        <div class="hero-subtitle" style="margin-bottom: 0; font-size: 0.88rem; color: {colors['text_secondary']};">
                            Retrieval-Augmented Scientific Literature Analysis Engine &amp; Citation Network Explorer
                        </div>
                    </div>
                    <div style="display: flex; gap: 6px; flex-wrap: wrap; align-items: center;">
                        <span class="badge-pill badge-strong" style="font-size: 0.72rem;">
                            {svg_icon("cpu", size=12, color=colors['badge_strong_text'])} Core Engine Active
                        </span>
                        <span class="badge-pill badge-slate" style="font-size: 0.72rem;">
                            {svg_icon("layers", size=12, color=colors['text_secondary'])} Hybrid RRF + Reranker
                        </span>
                        <span class="badge-pill badge-outline" style="font-size: 0.72rem;">
                            {svg_icon("database", size=12, color=colors['text_primary'])} 184 Papers Indexed
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Slim Single-Line Breadcrumb for Inner Pages
        clean_page_name = page_name.replace("⌂ ", "").replace("📚 ", "").replace("📄 ", "").replace("🕸 ", "").replace("📊 ", "")
        st.markdown(
            f"""
            <div style="border-bottom: 1px solid {colors['border']}; padding-bottom: 8px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: {colors['text_secondary']};">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="font-weight: 600; color: {colors['text_primary']}; font-family: 'Lora', Georgia, serif;">AI Research Copilot</span>
                    <span>/</span>
                    <span style="font-weight: 500; color: {colors['accent']};">{clean_page_name}</span>
                </div>
                <div style="display: flex; gap: 8px; align-items: center; font-size: 0.75rem;">
                    <span>Corpus: <code>184 papers</code></span>
                    <span>·</span>
                    <span style="color: {colors['badge_strong_text']};">● Engine Ready</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
