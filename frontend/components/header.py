"""
Top Header and System Metadata Banner Component.
"""

import streamlit as st


def render_header():
    """Render main application header and status metrics bar."""
    st.markdown(
        """
        <div class="glass-card" style="padding-bottom: 18px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div class="hero-title">AI Research Copilot</div>
                    <div class="hero-subtitle">
                        Retrieval-Augmented Scientific Literature Analysis Engine & Network Explorer
                    </div>
                </div>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <span class="badge-pill badge-strong">🟢 Core Engine Active</span>
                    <span class="badge-pill badge-indigo">⚡ Hybrid RRF + Reranker</span>
                    <span class="badge-pill badge-cyan">📚 184 Papers Indexed</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
