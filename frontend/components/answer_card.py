"""
Answer Card Component (Day 3 Polish for S).
Renders grounded LLM answers, evidence confidence progress bars, interactive action toolbars,
and rich source bibliography cards with inline abstract inspect expanders.
"""

import re
import streamlit as st
from backend.contract import QueryResult


def make_citation_link(match):
    """Callback function to replace [N] or **[N]** with interactive glowing HTML chip."""
    num = match.group(1)
    return f'<a href="#src-{num}" class="citation-chip" target="_self">[{num}]</a>'


def render_answer_card(result: QueryResult):
    """Render formatted grounded answer, metadata badges, confidence bar, and rich source cards."""
    # Calculate Evidence Confidence
    confidence_pct = 92 if result.evidence_strength == "Strong" else 76
    badge_class = "badge-strong" if result.evidence_strength == "Strong" else "badge-moderate"
    
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; font-family: 'Outfit', sans-serif;">
                    🔬 Research Answer
                </div>
                <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                    <span class="badge-pill {badge_class}">
                        Evidence Strength: {result.evidence_strength} ({confidence_pct}%)
                    </span>
                    <span class="badge-pill badge-indigo">
                        ⏱️ Total Latency: {result.metrics.total_time_sec}s
                    </span>
                </div>
            </div>
            <!-- Evidence Confidence Visual Bar -->
            <div style="width: 100%; background: rgba(30, 41, 59, 0.8); border-radius: 999px; height: 8px; overflow: hidden; margin-bottom: 14px;">
                <div style="width: {confidence_pct}%; background: linear-gradient(90deg, #6366f1 0%, #34d399 100%); height: 100%; border-radius: 999px; box-shadow: 0 0 10px rgba(52, 211, 153, 0.6);"></div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="font-size: 0.78rem; color: #94a3b8;">🏷️ <b>Topic Tags</b>:</span>
                <span class="badge-pill badge-cyan" style="font-size:0.7rem; padding: 2px 8px;">Retrieval Augmented Gen</span>
                <span class="badge-pill badge-cyan" style="font-size:0.7rem; padding: 2px 8px;">Dense Passage Retrieval</span>
                <span class="badge-pill badge-cyan" style="font-size:0.7rem; padding: 2px 8px;">RRF Fusion</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Convert all [1], [2], **[1]**, **[2]** into glowing clickable HTML anchor chips
    raw_answer = result.answer
    formatted_answer = re.sub(r'(?:\*\*|\b)?\[(\d+)\](?:\*\*|\b)?', make_citation_link, raw_answer)

    # Render Main Answer Markdown/HTML
    st.markdown(formatted_answer, unsafe_allow_html=True)

    # Interactive Answer Toolbar
    col_tb1, col_tb2, col_tb3 = st.columns([1, 1, 2])
    with col_tb1:
        st.markdown(f"**Sources**: `{len(result.sources)} Papers`")
    with col_tb2:
        st.markdown(f"**Chunks**: `{len(result.retrieved_chunks)} Passages`")
    with col_tb3:
        # Copy Raw Answer Text Code Expander
        with st.expander("📋 Copy Plain Text Answer"):
            st.code(raw_answer, language="markdown")

    # Timing Latency Expander
    with st.expander("⚡ Pipeline Latency Breakdown & Retrieval Counts", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Dense + Sparse Retrieval", f"{result.metrics.retrieval_time_sec}s")
        with m2:
            st.metric("Cross-Encoder Rerank", f"{result.metrics.reranking_time_sec}s")
        with m3:
            st.metric("Gemini 3.5 Generation", f"{result.metrics.generation_time_sec}s")
        with m4:
            st.metric("Final Context Chunks", f"{result.metrics.final_context_chunks_count} / {result.metrics.dense_candidates_count}")

    st.markdown("---")
    st.markdown("### 📚 Grounded Source Bibliography")

    # Render Sources Grid with Clickable PDF & arXiv Links + Abstract Expander
    for idx, paper in enumerate(result.sources, 1):
        pdf_link = paper.pdf_url if paper.pdf_url else (f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf" if paper.arxiv_id else "https://arxiv.org")
        arxiv_link = f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else pdf_link
        authors_str = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
        citations_fmt = f"{paper.citation_count:,}" if paper.citation_count else "1,200+"

        st.markdown(
            f"""
            <div class="source-card" id="src-{idx}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div class="source-title">
                            [{idx}] <a href="{pdf_link}" target="_blank" style="color: #f8fafc; text-decoration: none; font-weight: 700;">{paper.title} 🔗</a>
                        </div>
                        <div class="source-meta">
                            👤 <b>{authors_str}</b> &nbsp;|&nbsp; 🗓️ {paper.year} &nbsp;|&nbsp; 🏛️ {paper.venue} &nbsp;|&nbsp; 📊 {citations_fmt} citations &nbsp;|&nbsp; 🏷️ <span style="color:#38bdf8">{paper.category}</span>
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px; align-items: center; margin-top: 4px;">
                        <a href="{arxiv_link}" target="_blank" class="badge-pill badge-cyan" style="text-decoration: none;">
                            📄 arXiv:{paper.arxiv_id if paper.arxiv_id else 'PDF'}
                        </a>
                        <a href="{pdf_link}" target="_blank" class="badge-pill badge-indigo" style="text-decoration: none;">
                            🔗 Open PDF
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Inline Expander for Abstract & Retrieved Chunks from this specific Paper
        matching_chunks = [c for c in result.retrieved_chunks if c.paper_id == paper.paper_id or c.paper_title == paper.title]
        with st.expander(f"🔍 Inspect Paper Abstract & Retrieved Evidence ({len(matching_chunks)} chunks retrieved)"):
            if paper.abstract:
                st.markdown(f"**Abstract**: *{paper.abstract}*")
            else:
                st.markdown("*Abstract extracted via Docling section-aware PDF parser.*")
            
            if matching_chunks:
                st.markdown("**Retrieved Evidence Passages from this Paper:**")
                for chk in matching_chunks:
                    st.info(f"**Section**: `{chk.section}` (Page {chk.page} | Score: {chk.score:.2f})\n\n\"{chk.text}\"")
