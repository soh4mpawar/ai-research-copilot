"""
Top Header and System Metadata Banner Component.
Academic Scientific Instrument Styling.
"""

import streamlit as st


def render_header():
    """Render main application header and status metrics bar."""
    st.markdown(
        """
        <div class="glass-card" style="padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div class="hero-title">AI Research Copilot</div>
                    <div class="hero-subtitle" style="margin-bottom: 0;">
                        Retrieval-Augmented Scientific Literature Analysis Engine & Network Explorer
                    </div>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="badge-pill badge-strong">● Core Engine Active</span>
                    <span class="badge-pill badge-slate">⚡ Hybrid RRF + Reranker</span>
                    <span class="badge-pill badge-outline">📚 184 Papers Indexed</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
