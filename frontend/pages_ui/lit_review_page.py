"""
Literature Review Studio Page Layout.
Generates multi-paper synthesis, structural markdown tables, and research gap reports.
Academic Scientific Instrument Styling (Zero Emojis, Clean Inline SVGs, Dual-Theme Aware).
"""

import streamlit as st
import pandas as pd
from backend import research_engine
from frontend.components.icons import svg_icon
from frontend.components.copy_button import render_copy_button
from frontend.components.data_table import render_academic_table
from frontend.styles.theme import get_theme_colors, is_dark_mode


def render_lit_review_page():
    """Render Literature Review Studio UI."""
    colors = get_theme_colors()
    dark = is_dark_mode()

    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div class="academic-title" style="font-size: 1.45rem; margin-bottom: 2px; color: {colors['text_primary']};">
                Literature Review &amp; Multi-Paper Synthesis Studio
            </div>
            <div style="color: {colors['text_secondary']}; font-size: 0.88rem;">
                Synthesize overarching paradigms, architectural evolutions, and open research gaps across multiple papers simultaneously.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Preset Topics
    preset_topics = [
        "Evolution of Transformer Architectures and Dense Neural Retrieval",
        "Hybrid Search: Reciprocal Rank Fusion of BM25 and Vector Embeddings",
        "Evaluation Protocols for Large Language Model Hallucinations in RAG"
    ]

    selected_topic = st.selectbox("Select literature review topic focus:", preset_topics)
    custom_topic = st.text_input("Or specify custom topic focus:", placeholder="e.g. Multi-modal document parsing and layout extraction...")

    topic_to_run = custom_topic.strip() if custom_topic.strip() else selected_topic

    if st.button("Generate Multi-Paper Literature Review", type="primary"):
        with st.spinner("Synthesizing multi-paper evidence matrix & generating comparison tables..."):
            lit_res = research_engine.generate_lit_review(topic_to_run)

        st.markdown(f"<div style='border-top: 1px solid {colors['border']}; margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)

        # Introduction
        st.markdown(lit_res.introduction)

        # Multi-Paper Comparison Matrix Table
        st.markdown(f"<div class='academic-title' style='font-size: 1.15rem; margin-top: 14px; margin-bottom: 8px; color: {colors['text_primary']};'>Multi-Paper Comparative Summary Matrix</div>", unsafe_allow_html=True)
        df_comp = pd.DataFrame(lit_res.comparison_table)
        render_academic_table(df_comp, wrap_cells=True)

        st.markdown(lit_res.architectural_evolution)
        st.markdown(lit_res.methodology_synthesis)

        # Research Gaps Section
        st.markdown(f"<div class='academic-title' style='font-size: 1.15rem; margin-top: 16px; margin-bottom: 8px; color: {colors['text_primary']};'>Identified Research Gaps &amp; Open Challenges</div>", unsafe_allow_html=True)
        gap_bg = "#3A230B" if dark else "#FFFBEB"
        gap_border = "#854D0E" if dark else "#FDE68A"
        gap_text = "#FBBF24" if dark else "#92400E"

        for idx, gap in enumerate(lit_res.identified_research_gaps, 1):
            st.markdown(
                f"""
                <div style="background: {gap_bg}; border: 1px solid {gap_border}; border-left: 3px solid #D97706; border-radius: 4px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.88rem; color: {gap_text};">
                    <b>Gap #{idx}</b>: {gap}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(lit_res.conclusion)

        # Export & Copy Actions Strip
        st.markdown(f"<div style='border-top: 1px solid {colors['border']}; margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)
        export_md = f"# Literature Review: {topic_to_run}\n\n{lit_res.introduction}\n\n{lit_res.architectural_evolution}\n\n{lit_res.methodology_synthesis}\n\n{lit_res.conclusion}"
        
        col_exp1, col_exp2 = st.columns([1, 2])
        with col_exp1:
            st.download_button(
                label="Export Literature Review (.md)",
                data=export_md,
                file_name=f"Literature_Review_{topic_to_run.replace(' ', '_')[:30]}.md",
                mime="text/markdown"
            )
        with col_exp2:
            render_copy_button(
                text_to_copy=export_md,
                label="Copy Full Synthesis Markdown",
                tooltip="Copy full literature review synthesis to clipboard",
                height=36
            )
