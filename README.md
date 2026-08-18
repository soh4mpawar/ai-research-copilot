# AI Research Copilot

### *A Retrieval-Augmented Generation System for Automated Scientific Literature Analysis*

**Author**: Soham Pawar  
**System Type**: End-to-End Scientific Paper RAG & Literature Review Assistant  
**Stack**: Python 3.10+, Streamlit, ChromaDB, BM25s, BGE-Reranker-Base, Google GenAI SDK (Gemini 3.5 Flash Lite), NetworkX

---

## 🌟 Overview

The **AI Research Copilot** is a high-precision, locally-deployed Retrieval-Augmented Generation (RAG) platform designed to synthesize research papers, conduct multi-paper literature reviews, and explore academic citation graphs.

### Key Architecture Highlights
* **Hybrid Retrieval Engine**: Dense embeddings (`nomic-ai/nomic-embed-text-v1.5` on CUDA) + Sparse BM25 (`bm25s`) fused via Reciprocal Rank Fusion (RRF).
* **Cross-Encoder Reranker**: `BAAI/bge-reranker-base` for deep sequence classification.
* **Source-Grounded Coherence Gate (FR-11)**: Multi-signal relevance gate preventing hallucinations and blocking out-of-domain queries (100% precision on 23-query adversarial suite).
* **GraphRAG 1-Hop Traversal (FR-14/15)**: Bidirectional academic citation graph over verified citation networks with pre-computed candidate injection.
* **Independent Evaluation (FR-20)**: Rigorous RAGAS evaluation using OpenAI `gpt-4o-mini` as independent judge (**0.9482 Faithfulness**, **0.9390 Context Precision**, **0.8785 Context Recall**, **0.7231 Answer Relevance** across 40 technical benchmark samples).

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
Set `GEMINI_API_KEY` (and optionally `OPENAI_API_KEY` for evaluation).

### 3. Launch Streamlit Application
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎨 Core Application Features

1. **⌂ Research QA Engine (`frontend/pages_ui/research_page.py`)**
   * Instant search across scientific papers with preset and free-form queries.
   * Grounded answers with inline clickable citation chips `[arXiv:YYMM.NNNNN]`.
   * Complete source bibliography with direct PDF and arXiv links.

2. **🔎 Retrieval Transparency Panel (`frontend/components/evidence_viewer.py`)**
   * Visualizes candidate filtering across pipeline stages (Dense Top-50 + Sparse BM25 Top-50 ➜ RRF Top-50 ➜ Cross-Encoder Top-15 ➜ Final Context Chunks).
   * Displays exact retrieved chunk text, section metadata, and similarity scores.

3. **📚 Literature Review Studio (`frontend/pages_ui/lit_review_page.py`)**
   * Synthesizes multi-paper comparative matrices across architectural families.
   * Highlights identified research gaps, open challenges, and future directions.
   * One-click export to Markdown (`.md`).

4. **📄 Scientific Paper Explorer (`frontend/pages_ui/paper_explorer_page.py`)**
   * Filter indexed papers by category (`cs.CL`, `cs.CV`), year, or keyword.
   * Instant paper summaries detailing core objectives, methodology, and limitations.

5. **🕸 Citation Network Graph (`frontend/pages_ui/citation_graph_page.py`)**
   * Interactive PyVis / NetworkX citation graph visualization with draggable nodes and degree statistics.

6. **📊 Evaluation & RAGAS Dashboard (`frontend/pages_ui/eval_dashboard_page.py`)**
   * Interactive Plotly charts showing RAG vs Non-RAG baseline comparisons and multi-metric stability scorecards.

---

## 🧪 Testing & Verification

Run the automated regression test suite:
```bash
python scripts/run_regression_suite.py
```
Asserts 16/16 unit and end-to-end tests covering threshold gating, adversarial rejection, ligature segmentation, rate-limit backoff, and dataset integrity.
