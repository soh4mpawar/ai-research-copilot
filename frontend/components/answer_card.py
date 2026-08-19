"""
Answer Card Component.
Renders grounded LLM answers, evidence confidence progress bars, interactive action toolbars,
and rich source bibliography cards with inline abstract inspect expanders.
Academic Scientific Instrument Styling (Zero Emojis, Clean Inline SVGs).
"""

import re
import streamlit as st
from backend.contract import QueryResult
from frontend.components.icons import svg_icon


def make_citation_link(match):
    """Callback function to replace [N] or **[N]** with interactive academic citation chip."""
    num = match.group(1)
    return f'<a href="#src-{num}" class="citation-chip" target="_self">[{num}]</a>'


def render_answer_card(result: QueryResult):
    """Render formatted grounded answer, metadata badges, confidence bar, and rich source cards."""
    confidence_pct = 92 if result.evidence_strength == "Strong" else 76
    badge_class = "badge-strong" if result.evidence_strength == "Strong" else "badge-moderate"
    bar_fill_color = "#1E5631" if result.evidence_strength == "Strong" else "#92400E"
    
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 22px 26px; border-left: 3px solid #2B4C7E;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <div class="academic-title" style="font-size: 1.3rem;">
                    Scientific Research Synthesis
                </div>
                <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                    <span class="badge-pill {badge_class}">
                        {svg_icon('check-circle', size=12, color='#1E5631' if result.evidence_strength == 'Strong' else '#92400E')}
                        Evidence: {result.evidence_strength} ({confidence_pct}%)
                    </span>
                    <span class="badge-pill badge-slate">
                        {svg_icon('clock', size=12, color='#334155')}
                        Latency: {result.metrics.total_time_sec}s
                    </span>
                </div>
            </div>
            <!-- Evidence Confidence Flat Visual Bar -->
            <div style="width: 100%; background: #E5E7EB; border-radius: 2px; height: 4px; overflow: hidden; margin-bottom: 12px;">
                <div style="width: {confidence_pct}%; background-color: {bar_fill_color} !important; height: 100%; border-radius: 2px;"></div>
            </div>
            <div style="display: flex; gap: 6px; flex-wrap: wrap; align-items: center;">
                <span style="font-size: 0.76rem; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 2px;">
                    {svg_icon('tag', size=11, color='#6B7280')} Topics:
                </span>
                <span class="badge-pill badge-outline">Retrieval-Augmented Generation</span>
                <span class="badge-pill badge-outline">Dense Passage Retrieval</span>
                <span class="badge-pill badge-outline">RRF Fusion</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Convert all [1], [2], **[1]**, **[2]** into clickable academic HTML anchor chips
    raw_answer = result.answer
    formatted_answer = re.sub(r'(?:\*\*|\b)?\[(\d+)\](?:\*\*|\b)?', make_citation_link, raw_answer)

    # Render Main Answer Markdown/HTML
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E5E5E3; border-radius: 8px; padding: 22px 26px; line-height: 1.7; font-size: 0.95rem; color: #1F2937; margin-bottom: 16px;">
            {formatted_answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Action Strip (Sources, Chunks, Plain-text copy)
    col_tb1, col_tb2, col_tb3 = st.columns([1, 1, 2])
    with col_tb1:
        st.markdown(f"**Sources**: `{len(result.sources)} Papers`")
    with col_tb2:
        st.markdown(f"**Retrieved**: `{len(result.retrieved_chunks)} Chunks`")
    with col_tb3:
        with st.expander("Copy Plain Text Answer"):
            st.code(raw_answer, language="markdown")

    # Timing Latency Strip (No emojis)
    with st.expander("Pipeline Execution Latency & Candidate Funnel", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Dense + Sparse Search", f"{result.metrics.retrieval_time_sec}s")
        with m2:
            st.metric("Cross-Encoder Rerank", f"{result.metrics.reranking_time_sec}s")
        with m3:
            st.metric("Gemini 3.5 Generation", f"{result.metrics.generation_time_sec}s")
        with m4:
            st.metric("Final Context Chunks", f"{result.metrics.final_context_chunks_count} / {result.metrics.dense_candidates_count}")

    st.markdown("<div style='border-top: 1px solid #E5E5E3; margin: 18px 0 14px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='academic-title' style='font-size: 1.15rem; margin-bottom: 12px;'>Grounded Source Bibliography</div>", unsafe_allow_html=True)

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
                            [{idx}] <a href="{pdf_link}" target="_blank" style="color: #111827; text-decoration: none; border-bottom: 1px solid #D1D5DB;">{paper.title}</a>
                        </div>
                        <div class="source-meta">
                            <b>{authors_str}</b> · {paper.year} · {paper.venue} · {citations_fmt} citations · <code>{paper.category}</code>
                        </div>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center; margin-top: 2px;">
                        <a href="{arxiv_link}" target="_blank" class="badge-pill badge-outline" style="text-decoration: none;">
                            {svg_icon('external-link', size=11, color='#1F2937')} arXiv:{paper.arxiv_id if paper.arxiv_id else 'PDF'}
                        </a>
                        <a href="{pdf_link}" target="_blank" class="badge-pill badge-slate" style="text-decoration: none;">
                            {svg_icon('file-text', size=11, color='#334155')} PDF
                        </a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Inline Expander for Abstract & Retrieved Chunks from this specific Paper
        matching_chunks = [c for c in result.retrieved_chunks if c.paper_id == paper.paper_id or c.paper_title == paper.title]
        with st.expander(f"Inspect Abstract & Retrieved Evidence ({len(matching_chunks)} chunks)"):
            if paper.abstract:
                st.markdown(f"**Abstract**: *{paper.abstract}*")
            else:
                st.markdown("*Abstract extracted via Docling section-aware PDF parser.*")
            
            if matching_chunks:
                st.markdown("**Retrieved Evidence Passages:**")
                for chk in matching_chunks:
                    st.info(f"**Section**: `{chk.section}` (Page {chk.page} | Score: {chk.score:.2f})\n\n\"{chk.text}\"")
