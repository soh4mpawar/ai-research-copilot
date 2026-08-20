import time
import streamlit as st

st.set_page_config(page_title="Test Selectbox Enter Key", layout="wide")

PRESETS = [
    "What is Retrieval-Augmented Generation (RAG) and why was it introduced?",
    "How does Reciprocal Rank Fusion (RRF) combine dense and sparse BM25 scores?",
    "What are the key computational limitations of transformer self-attention mechanisms?"
]

st.title("Enter Key & Autocomplete Test")

query = st.selectbox(
    "Enter scientific research question:",
    options=PRESETS,
    index=None,
    accept_new_options=True,
    filter_mode="fuzzy",
    placeholder="Type to search presets or enter custom question...",
    key="test_query_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    btn = st.button("Run Research Pipeline", type="primary")

if query:
    st.success(f"Selected/Committed Query Value: '{query}'")
else:
    st.info("No query committed yet.")
