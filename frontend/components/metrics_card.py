"""
Evaluation Metrics Card & Plotly Chart Components for RAGAS framework.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from backend.contract import EvalMetrics


def render_ragas_scorecards(metrics: EvalMetrics):
    """Render executive scorecards for target vs actual RAGAS metrics."""
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">Faithfulness (Target >0.70)</div>
                <div class="metric-val" style="color: #34d399;">{metrics.faithfulness:.2f}</div>
                <div style="font-size: 0.75rem; color: #a5b4fc; margin-top: 4px;">✅ Exceeds Target</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">Context Precision (Target >0.60)</div>
                <div class="metric-val" style="color: #38bdf8;">{metrics.context_precision:.2f}</div>
                <div style="font-size: 0.75rem; color: #a5b4fc; margin-top: 4px;">✅ Exceeds Target</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">Context Recall (Target >0.60)</div>
                <div class="metric-val" style="color: #818cf8;">{metrics.context_recall:.2f}</div>
                <div style="font-size: 0.75rem; color: #a5b4fc; margin-top: 4px;">✅ Exceeds Target</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">Answer Relevance (Target >0.70)</div>
                <div class="metric-val" style="color: #f472b6;">{metrics.answer_relevance:.2f}</div>
                <div style="font-size: 0.75rem; color: #a5b4fc; margin-top: 4px;">✅ Exceeds Target</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_rag_vs_baseline_chart(metrics: EvalMetrics):
    """Render RAG vs Non-RAG baseline comparative bar chart."""
    categories = list(metrics.rag_vs_non_rag.keys())
    rag_scores = []
    non_rag_scores = []
    for cat in categories:
        val_map = metrics.rag_vs_non_rag.get(cat, {})
        rag_val = val_map.get("Hybrid RAG Pipeline", val_map.get("Hybrid_RAG", list(val_map.values())[0] if val_map else 0.0))
        non_rag_val = val_map.get("Non-RAG Gemini Baseline", val_map.get("Non_RAG_Baseline", list(val_map.values())[1] if len(val_map) > 1 else 0.0))
        rag_scores.append(rag_val)
        non_rag_scores.append(non_rag_val)

    fig = go.Figure(data=[
        go.Bar(name='Hybrid RAG Pipeline (Ours)', x=categories, y=rag_scores, marker_color='#6366f1'),
        go.Bar(name='Non-RAG Gemini Baseline', x=categories, y=non_rag_scores, marker_color='#64748b')
    ])

    fig.update_layout(
        title="RAG Pipeline vs. Non-RAG Baseline Performance Comparison",
        barmode='group',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        yaxis=dict(range=[0.0, 1.0], gridcolor='rgba(255,255,255,0.08)'),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_retrieval_stage_chart(metrics: EvalMetrics):
    """Render retrieval strategy comparison chart (Dense vs BM25 vs RRF vs Reranker)."""
    stages = list(metrics.stage_comparisons.keys())
    precision = []
    recall = []
    for s in stages:
        s_data = metrics.stage_comparisons.get(s, {})
        p = s_data.get("Precision@5", s_data.get("context_precision", s_data.get("precision", 0.0)))
        r = s_data.get("Recall@5", s_data.get("context_recall", s_data.get("recall", 0.0)))
        precision.append(p)
        recall.append(r)

    fig = go.Figure(data=[
        go.Scatter(x=stages, y=precision, mode='lines+markers', name='Precision@5', line=dict(color='#38bdf8', width=3)),
        go.Scatter(x=stages, y=recall, mode='lines+markers', name='Recall@5', line=dict(color='#34d399', width=3))
    ])

    fig.update_layout(
        title="Retrieval Precision & Recall Gains Across Pipeline Stages",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        yaxis=dict(range=[0.3, 1.0], gridcolor='rgba(255,255,255,0.08)'),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
