"""
Literature Review Studio Page Layout.
Generates multi-paper synthesis, structural markdown tables, and research gap reports.
"""

import streamlit as st
import pandas as pd
from backend import research_engine


def render_lit_review_page():
    """Render Literature Review Studio UI."""
    st.markdown("## 📚 Literature Review & Multi-Paper Synthesis Studio")
    st.markdown(
        "Synthesize overarching paradigms, architectural evolutions, and open research gaps across multiple papers simultaneously."
    )

    # Preset Topics
    preset_topics = [
        "Evolution of Transformer Architectures and Dense Neural Retrieval",
        "Hybrid Search: Reciprocal Rank Fusion of BM25 and Vector Embeddings",
        "Evaluation Protocols for Large Language Model Hallucinations in RAG"
    ]

    selected_topic = st.selectbox("Select or enter a literature review topic:", preset_topics)
    custom_topic = st.text_input("Or specify custom topic focus:", placeholder="e.g. Multi-modal document parsing and layout extraction...")

    topic_to_run = custom_topic.strip() if custom_topic.strip() else selected_topic

    if st.button("🚀 Generate Multi-Paper Literature Review", type="primary"):
        with st.spinner("Synthesizing multi-paper evidence matrix & generating comparison tables..."):
            lit_res = research_engine.generate_lit_review(topic_to_run)

        st.markdown("---")

        # Introduction
        st.markdown(lit_res.introduction)

        # Multi-Paper Comparison Matrix Table
        st.markdown("### 📊 Multi-Paper Comparative Summary Matrix")
        df_comp = pd.DataFrame(lit_res.comparison_table)
        st.dataframe(
            df_comp,
            use_container_width=True,
            column_config={
                "Paper Title": st.column_config.TextColumn("Paper Title", width="medium"),
                "Year": st.column_config.NumberColumn("Year", format="%d"),
                "Core Approach": st.column_config.TextColumn("Core Approach", width="medium"),
                "Key Contribution": st.column_config.TextColumn("Key Contribution", width="large"),
                "Limitations": st.column_config.TextColumn("Limitations", width="large"),
            }
        )

        st.markdown(lit_res.architectural_evolution)
        st.markdown(lit_res.methodology_synthesis)

        # Research Gaps Section
        st.markdown("### 🔍 Identified Research Gaps & Open Challenges")
        for idx, gap in enumerate(lit_res.identified_research_gaps, 1):
            st.error(f"**Gap #{idx}**: {gap}")

        st.markdown(lit_res.conclusion)

        # Export Button
        st.markdown("---")
        export_md = f"# Literature Review: {topic_to_run}\n\n{lit_res.introduction}\n\n{lit_res.architectural_evolution}\n\n{lit_res.methodology_synthesis}\n\n{lit_res.conclusion}"
        st.download_button(
            label="📥 Export Literature Review (.md)",
            data=export_md,
            file_name=f"Literature_Review_{topic_to_run.replace(' ', '_')[:30]}.md",
            mime="text/markdown"
        )
