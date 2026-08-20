import streamlit as st

st.title("Test Native Type-Ahead Autocomplete")

PRESET_QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG) and why was it introduced?",
    "How does Reciprocal Rank Fusion (RRF) combine dense and sparse BM25 scores?",
    "What are the key computational limitations of transformer self-attention mechanisms?",
    "How does bge-reranker-base calculate cross-encoder relevance scores?",
    "What are the primary differences between dense vector embeddings and sparse BM25 retrieval?"
]

selected_query = st.selectbox(
    "Enter or select scientific research question:",
    options=PRESET_QUESTIONS,
    index=None,
    accept_new_options=True,
    placeholder="Type to search presets or enter custom question..."
)

st.write(f"Selected/Entered Query: {selected_query}")
