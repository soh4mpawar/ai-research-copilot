"""
Evidence Panel and Retrieval Transparency Component.
Exposes multi-stage candidate funnels, comparative rank matrices, cross-encoder scores,
and passage text inspectors.
Academic Scientific Instrument Styling.
"""

import streamlit as st
import pandas as pd
from backend.contract import QueryResult


def render_evidence_viewer(result: QueryResult):
    """Render interactive retrieval transparency panel and chunk inspector."""
    st.markdown("<div class='academic-title' style='font-size: 1.2rem; margin-bottom: 8px;'>🔎 Retrieval Transparency & Multi-Stage Evidence Inspector</div>", unsafe_allow_html=True)
    
    st.info(
        "💡 **Retrieval Pipeline**: Multi-stage funnel passing candidates through "
        "Dense ChromaDB Search + Sparse BM25 Search ➜ Reciprocal Rank Fusion (RRF) ➜ Cross-Encoder Reranking (`bge-reranker-base`) ➜ Gemini Prompt Context."
    )

    # Multi-Stage Funnel Progress Metric Cards
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">1. Dense ChromaDB</div>
                <div class="metric-val">{result.metrics.dense_candidates_count}</div>
                <div style="font-size:0.72rem; color:#6B7280; margin-top:2px;">Top Vector Matches</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">2. Sparse BM25</div>
                <div class="metric-val">{result.metrics.bm25_candidates_count}</div>
                <div style="font-size:0.72rem; color:#6B7280; margin-top:2px;">Top Lexical Matches</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">3. RRF Fusion</div>
                <div class="metric-val">{result.metrics.rrf_candidates_count}</div>
                <div style="font-size:0.72rem; color:#6B7280; margin-top:2px;">Rank Fusion Pool</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">4. Cross-Reranker</div>
                <div class="metric-val">{result.metrics.reranked_candidates_count}</div>
                <div style="font-size:0.72rem; color:#6B7280; margin-top:2px;">bge-reranker-base</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with f5:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">5. LLM Context</div>
                <div class="metric-val">{result.metrics.final_context_chunks_count}</div>
                <div style="font-size:0.72rem; color:#1E5631; margin-top:2px; font-weight: 500;">Injected Passages</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("<div class='academic-title' style='font-size: 1.05rem; margin-bottom: 8px;'>📊 Comparative Rank Matrix Across Retrieval Strategies</div>", unsafe_allow_html=True)

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
            "Status": "✅ Injected in Context"
        })

    df_matrix = pd.DataFrame(matrix_data)
    st.dataframe(
        df_matrix,
        use_container_width=True,
        column_config={
            "Context #": st.column_config.TextColumn("Context #", width="small"),
            "Chunk ID": st.column_config.TextColumn("Chunk ID", width="small"),
            "Paper Title": st.column_config.TextColumn("Paper Title", width="large"),
            "Docling Section": st.column_config.TextColumn("Docling Section", width="medium"),
            "Reranker Score": st.column_config.TextColumn("Reranker Score", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium"),
        }
    )

    st.markdown("---")
    st.markdown("<div class='academic-title' style='font-size: 1.05rem; margin-bottom: 8px;'>📄 Injected Context Passages & Text Inspector</div>", unsafe_allow_html=True)

    # Render Chunks Passage Cards
    for idx, chunk in enumerate(result.retrieved_chunks, 1):
        rel_pct = int(chunk.score * 100)
        with st.expander(f"Passage #{idx}: {chunk.paper_title} — [{chunk.section}] (Relevance: {rel_pct}%)", expanded=(idx == 1)):
            col_info, col_ranks = st.columns([2, 1])
            
            with col_info:
                st.markdown(f"**Source Paper**: {chunk.paper_title}")
                st.markdown(f"**Section**: `{chunk.section}` | **Page**: {chunk.page} | **Chunk ID**: `{chunk.chunk_id}`")
                st.markdown(f"**Authors**: {chunk.authors}")
            
            with col_ranks:
                st.markdown(
                    f"""
                    <div style="background: #F9FAFB; padding: 10px; border-radius: 6px; border: 1px solid #E5E7EB; font-size: 0.80rem; line-height: 1.6;">
                        <div><b>Dense Rank</b>: #{chunk.dense_rank}</div>
                        <div><b>BM25 Rank</b>: #{chunk.bm25_rank}</div>
                        <div><b>RRF Rank</b>: #{chunk.rrf_rank}</div>
                        <div style="color:#1E5631; margin-top: 2px;"><b>Cross-Rerank Score</b>: {chunk.rerank_score:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-header">
                        <span>EXTRACTED PASSAGE TEXT (Docling Parsed ~310 Tokens)</span>
                        <span class="badge-pill badge-strong" style="font-size: 0.72rem;">Relevance: {rel_pct}%</span>
                    </div>
                    <div class="evidence-text">
                        "{chunk.text}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("📋 Copy Passage Text"):
                st.code(chunk.text, language="text")
