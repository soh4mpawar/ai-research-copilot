"""
RAGAS Framework & Benchmark Evaluation Dashboard Page Layout.
Renders quantitative metrics, baseline comparison plots, and evaluation logs.
Academic Scientific Instrument Styling (Zero Emojis, Clean Inline SVGs, Dual-Theme Aware).
"""

import streamlit as st
import pandas as pd
from backend import research_engine
from frontend.components.metrics_card import (
    render_ragas_scorecards,
    render_rag_vs_baseline_chart,
    render_retrieval_stage_chart,
)
from frontend.components.icons import svg_icon
from frontend.components.data_table import render_academic_table
from frontend.styles.theme import get_theme_colors


def render_eval_dashboard_page():
    """Render Evaluation Dashboard UI."""
    colors = get_theme_colors()

    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div class="academic-title" style="font-size: 1.45rem; margin-bottom: 2px; color: {colors['text_primary']};">
                RAGAS Evaluation &amp; Baseline Benchmark Dashboard
            </div>
            <div style="color: {colors['text_secondary']}; font-size: 0.88rem;">
                Quantitative reference-free evaluation metrics measuring Faithfulness, Context Precision, Recall, and Answer Relevance against held-out benchmark data.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    metrics = research_engine.get_eval_metrics()

    # Executive Scorecards (Theme aware)
    render_ragas_scorecards(metrics)

    st.markdown(f"<div style='border-top: 1px solid {colors['border']}; margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)

    # Plotly Charts Grid
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        render_rag_vs_baseline_chart(metrics)
        
    with col_chart2:
        render_retrieval_stage_chart(metrics)

    st.markdown(f"<div style='border-top: 1px solid {colors['border']}; margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='academic-title' style='font-size: 1.15rem; margin-bottom: 8px; color: {colors['text_primary']};'>Held-out QA Evaluation Benchmark Dataset (40 Test Pairs)</div>", unsafe_allow_html=True)

    # Test Samples DataFrame
    df_samples = pd.DataFrame(metrics.eval_samples)
    render_academic_table(df_samples, wrap_cells=True, max_height_px=400)
