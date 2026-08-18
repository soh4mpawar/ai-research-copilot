"""
RAGAS Framework & Benchmark Evaluation Dashboard Page Layout.
Renders quantitative metrics, baseline comparison plots, and evaluation logs.
Academic Scientific Instrument Styling.
"""

import streamlit as st
import pandas as pd
from backend import research_engine
from frontend.components.metrics_card import (
    render_ragas_scorecards,
    render_rag_vs_baseline_chart,
    render_retrieval_stage_chart,
)


def render_eval_dashboard_page():
    """Render Evaluation Dashboard UI."""
    st.markdown("<div class='academic-title' style='font-size: 1.5rem; margin-bottom: 4px;'>📊 RAGAS Evaluation & Baseline Benchmark Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color: #525252; font-size: 0.92rem; margin-bottom: 14px;'>Quantitative reference-free evaluation metrics measuring Faithfulness, Context Precision, Recall, and Answer Relevance against held-out benchmark data.</div>",
        unsafe_allow_html=True
    )

    metrics = research_engine.get_eval_metrics()

    # Executive Scorecards
    render_ragas_scorecards(metrics)

    st.markdown("---")

    # Plotly Charts Grid
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        render_rag_vs_baseline_chart(metrics)
        
    with col_chart2:
        render_retrieval_stage_chart(metrics)

    st.markdown("---")
    st.markdown("<div class='academic-title' style='font-size: 1.15rem; margin-bottom: 8px;'>📋 Held-out QA Evaluation Benchmark Dataset (40 Test Pairs)</div>", unsafe_allow_html=True)

    # Test Samples DataFrame
    df_samples = pd.DataFrame(metrics.eval_samples)
    st.dataframe(
        df_samples,
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("QA ID", width="small"),
            "question": st.column_config.TextColumn("Question", width="large"),
            "ground_truth": st.column_config.TextColumn("Ground Truth Target", width="large"),
            "faithfulness": st.column_config.NumberColumn("Faithfulness", format="%.4f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "status": st.column_config.TextColumn("Status", width="small"),
        }
    )
