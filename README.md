# AI Research Copilot

### *A Retrieval-Augmented System for Automated Scientific Literature Analysis*

---

## 👥 Team & Subsystem Ownership

| Member | Role | Primary Responsibility | Owned Directory |
| :--- | :--- | :--- | :--- |
| **A — Akanksha Jaiswal** | RAG Engine + Backend Architect | PDF Ingestion (Docling), Chunking, Embeddings, ChromaDB, Sparse BM25, RRF Fusion, Cross-Encoder Reranker, Gemini LLM | `backend/` |
| **U — Mohd. Uzair Qureshi** | Data + Evaluation + Research Engineering | Corpus creation (150-200 papers), QA Benchmark dataset, RAGAS framework, Non-RAG baseline, Plot generation | `data/`, `evaluation/` |
| **S — Soham Pawar** | Frontend + Product + Graph Explorer | Streamlit Application, Glassmorphism UI, Evidence Transparency Drawer, Literature Review Studio, Citation Network | `frontend/` |

---

## 🚀 Quick Start Guide (For S Frontend Development & Demonstration)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run frontend/app.py
```

---

## 🔌 Frontend-Backend Contract Interface (`backend/contract.py`)

To allow **S** to build the entire product UI without waiting for **A** to finish backend ingestion or **U** to complete corpus collection, the frontend connects to a proxy interface `backend/research_engine.py`.

* **Mock Mode (Default)**: Set `USE_MOCK_ENGINE=true` (or toggle checkbox in sidebar) to run with realistic simulated data.
* **Production Real Mode**: When **A** completes `backend/pipeline.py`, set `USE_MOCK_ENGINE=false` to route queries directly into the real ChromaDB vector store and `bge-reranker-base` cross-encoder!

---

## 🎨 Core Application Modules (Owned by S)

1. **⌂ Research QA Engine (`frontend/pages_ui/research_page.py`)**
   * Instant search over 180+ scientific papers with preset research questions.
   * Grounded Markdown answers with inline clickable citation chips `[1]`, `[2]`.
   * Sources bibliography grid with arXiv links and metadata.

2. **🔎 Retrieval Transparency Panel (`frontend/components/evidence_viewer.py`)**
   * Visualizes candidate filtering across pipeline stages (Dense Top-20 + Sparse BM25 Top-20 ➜ RRF Top-25 ➜ Cross-Encoder Top-10 ➜ Gemini Context Top-4).
   * Displays exact retrieved chunk passages, section names, and similarity scores.

3. **📚 Literature Review Studio (`frontend/pages_ui/lit_review_page.py`)**
   * Synthesizes multi-paper comparative matrices across models (BERT, RAG, DPR, GraphRAG).
   * Highlights identified research gaps and open challenges.
   * One-click export to Markdown (`.md`).

4. **📄 Scientific Paper Explorer (`frontend/pages_ui/paper_explorer_page.py`)**
   * Filter 180+ indexed papers by category (`cs.CL`, `cs.CV`), year (2017-2026), or keyword.
   * Instant AI paper summary detailing core objectives, key findings, and limitations.

5. **🕸 Citation Network Graph (`frontend/pages_ui/citation_graph_page.py`)**
   * Interactive PyVis / NetworkX citation graph visualization.
   * Draggable nodes, foundational paper seed highlights, and degree statistics.

6. **📊 Evaluation & RAGAS Dashboard (`frontend/pages_ui/eval_dashboard_page.py`)**
   * Executive scorecards: **Faithfulness (0.86)**, **Context Precision (0.74)**, **Context Recall (0.78)**, **Answer Relevance (0.82)**.
   * Interactive Plotly charts comparing **RAG vs Non-RAG Baseline** and **Dense vs Hybrid vs Reranked Stages**.
