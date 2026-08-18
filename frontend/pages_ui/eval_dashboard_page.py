"""
RAGAS Framework & Benchmark Evaluation Dashboard Page Layout.
Renders quantitative metrics, baseline comparison plots, and evaluation logs.
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
    st.markdown("## 📊 RAGAS Evaluation & Baseline Benchmark Dashboard")
    st.markdown(
        "Quantitative reference-free evaluation metrics measuring Faithfulness, Context Precision, Recall, and Answer Relevance."
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
    st.markdown("### 📋 Held-out QA Evaluation Benchmark Dataset (30-50 Test Pairs)")

    # Test Samples DataFrame
    df_samples = pd.DataFrame(metrics.eval_samples)
    st.dataframe(
        df_samples,
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("QA ID", width="small"),
            "question": st.column_config.TextColumn("Question", width="large"),
            "ground_truth": st.column_config.TextColumn("Ground Truth Target", width="large"),
            "faithfulness": st.column_config.NumberColumn("Faithfulness", format="%.2f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.2f"),
            "status": st.column_config.TextColumn("Status", width="small"),
        }
    )
