"""
Evidence Panel and Retrieval Transparency Component.
Exposes multi-stage candidate funnels, comparative rank matrices, cross-encoder scores,
and passage text inspectors.
Academic Scientific Instrument Styling (Zero Emojis, Clean Inline SVGs, Dual-Theme Aware).
"""

import streamlit as st
import pandas as pd
from backend.contract import QueryResult
from frontend.components.icons import svg_icon
from frontend.components.copy_button import render_copy_button
from frontend.components.data_table import render_academic_table
from frontend.styles.theme import get_theme_colors


def render_evidence_viewer(result: QueryResult):
    """Render interactive retrieval transparency panel and chunk inspector."""
    colors = get_theme_colors()

    st.markdown(
        f"""
        <div class="academic-title" style="font-size: 1.2rem; margin-bottom: 4px; color: {colors['text_primary']};">
            Retrieval Pipeline Transparency &amp; Multi-Stage Inspector
        </div>
        <div style="font-size: 0.84rem; color: {colors['text_secondary']}; margin-bottom: 14px;">
            Candidate progression across dense vector search, lexical BM25, Reciprocal Rank Fusion, and cross-encoder scoring.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Connected Multi-Stage Funnel Horizontal Strip
    st.markdown(
        f"""
        <div style="background: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 6px; padding: 14px 18px; margin-bottom: 18px;">
            <div style="font-size: 0.75rem; font-weight: 600; color: {colors['text_secondary']}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">
                Multi-Stage Retrieval Funnel
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px;">
                <div style="border-right: 1px solid {colors['border']}; padding-right: 8px;">
                    <div style="font-size: 0.72rem; color: {colors['text_secondary']};">1. Dense Vector</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: {colors['text_primary']};">{result.metrics.dense_candidates_count}</div>
                    <div style="font-size: 0.70rem; color: {colors['text_secondary']};">Top ChromaDB</div>
                </div>
                <div style="border-right: 1px solid {colors['border']}; padding-right: 8px;">
                    <div style="font-size: 0.72rem; color: {colors['text_secondary']};">2. Sparse BM25</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: {colors['text_primary']};">{result.metrics.bm25_candidates_count}</div>
                    <div style="font-size: 0.70rem; color: {colors['text_secondary']};">Top Lexical</div>
                </div>
                <div style="border-right: 1px solid {colors['border']}; padding-right: 8px;">
                    <div style="font-size: 0.72rem; color: {colors['text_secondary']};">3. RRF Fusion</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: {colors['text_primary']};">{result.metrics.rrf_candidates_count}</div>
                    <div style="font-size: 0.70rem; color: {colors['text_secondary']};">Merged Candidates</div>
                </div>
                <div style="border-right: 1px solid {colors['border']}; padding-right: 8px;">
                    <div style="font-size: 0.72rem; color: {colors['text_secondary']};">4. Cross-Reranker</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: {colors['text_primary']};">{result.metrics.reranked_candidates_count}</div>
                    <div style="font-size: 0.70rem; color: {colors['text_secondary']};">bge-reranker-base</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; color: {colors['badge_strong_text']}; font-weight: 600;">5. Injected Context</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 600; color: {colors['badge_strong_text']};">{result.metrics.final_context_chunks_count}</div>
                    <div style="font-size: 0.70rem; color: {colors['badge_strong_text']};">Prompt Passages</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"<div class='academic-title' style='font-size: 1.05rem; margin-bottom: 8px; color: {colors['text_primary']};'>Comparative Rank Matrix Across Retrieval Strategies</div>", unsafe_allow_html=True)

    # Build Comparative Matrix DataFrame
    matrix_data = []
    for idx, c in enumerate(result.retrieved_chunks, 1):
        matrix_data.append({
            "Context #": f"#{idx}",
            "Chunk ID": c.chunk_id,
            "Paper Title": c.paper_title,
            "Docling Section": c.section,
            "Dense Rank": f"#{c.dense_rank}",
            "Sparse BM25 Rank": f"#{c.bm25_rank}",
            "RRF Merged Rank": f"#{c.rrf_rank}",
            "Reranker Score": f"{c.rerank_score:.3f}",
            "Status": "Injected in Context"
        })

    df_matrix = pd.DataFrame(matrix_data)
    render_academic_table(df_matrix)

    st.markdown(f"<div style='border-top: 1px solid {colors['border']}; margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='academic-title' style='font-size: 1.05rem; margin-bottom: 8px; color: {colors['text_primary']};'>Injected Context Passages &amp; Text Inspector</div>", unsafe_allow_html=True)

    # Render Chunks Passage Cards
    for idx, chunk in enumerate(result.retrieved_chunks, 1):
        rel_pct = int(chunk.score * 100)
        with st.expander(f"Passage #{idx}: {chunk.paper_title} — [{chunk.section}] (Score: {chunk.rerank_score:.3f})", expanded=(idx == 1)):
            col_info, col_ranks = st.columns([2, 1])
            
            with col_info:
                st.markdown(f"**Source Paper**: {chunk.paper_title}")
                st.markdown(f"**Section**: `{chunk.section}` · **Page**: {chunk.page} · **Chunk ID**: `{chunk.chunk_id}`")
                st.markdown(f"**Authors**: {chunk.authors}")
            
            with col_ranks:
                st.markdown(
                    f"""
                    <div style="background: {colors['card_bg']}; padding: 8px 12px; border-radius: 6px; border: 1px solid {colors['border']}; font-size: 0.80rem; line-height: 1.6;">
                        <div><b>Dense Rank</b>: #{chunk.dense_rank}</div>
                        <div><b>BM25 Rank</b>: #{chunk.bm25_rank}</div>
                        <div><b>RRF Rank</b>: #{chunk.rrf_rank}</div>
                        <div style="color:{colors['badge_strong_text']}; margin-top: 2px;"><b>Cross-Rerank Score</b>: {chunk.rerank_score:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-header">
                        <span>EXTRACTED PASSAGE TEXT</span>
                        <span class="badge-pill badge-strong" style="font-size: 0.72rem;">Relevance: {rel_pct}%</span>
                    </div>
                    <div class="evidence-text">
                        "{chunk.text}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            render_copy_button(
                text_to_copy=chunk.text,
                label="Copy Passage",
                tooltip="Copy extracted passage text to clipboard",
                height=30
            )
