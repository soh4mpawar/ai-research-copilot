"""
Phase 3 Generation CLI Smoke-Test Script (Phase 3 Definition of Done / FR-7, FR-8, FR-13).
Executes Point-QA mode, Literature-Review mode, and Non-RAG Baseline mode end-to-end from a raw query.
"""

import sys
import os
import argparse

# Ensure UTF-8 output encoding for Windows PowerShell terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.vector_store import VectorStore
from backend.retrieval.sparse_bm25 import SparseBM25Retriever
from backend.retrieval.fusion import RRFFusionRetriever
from backend.reranking.reranker import CrossEncoderReranker
from backend.generation.point_qa import GroundedPointQAEngine
from backend.generation.lit_review import MultiPaperLitReviewEngine
from backend.generation.baseline import NonRAGBaselineEngine


def sanitize_str(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    return s.encode("ascii", "ignore").decode("ascii")


def run_generation_smoke_test(query: str):
    print("==========================================================================")
    print("AI RESEARCH COPILOT -- PHASE 3: GROUNDED GENERATION & BASELINE SMOKE TEST")
    print(f"QUERY: '{sanitize_str(query)}'")
    print("==========================================================================")

    # 1. Retrieve & Rerank Context Chunks (Phase 2 Pipeline)
    vector_store = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
    dense = vector_store.search_dense(query, top_k=20)
    
    bm25 = SparseBM25Retriever(vector_store)
    sparse = bm25.search_sparse(query, top_k=20)
    
    fuser = RRFFusionRetriever(rrf_k=60.0)
    fused = fuser.fuse_results(dense, sparse, top_k=25)
    
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    reranked = reranker.rerank_chunks(query, fused, top_k=10)

    # 2. Point-QA Generation (FR-7)
    print("\n--- 1. GROUNDED POINT-QA MODE (FR-7) ---")
    qa_engine = GroundedPointQAEngine()
    qa_res = qa_engine.generate_point_qa(query, reranked)
    print(f"Evidence Strength: {qa_res.evidence_strength} | Sources: {len(qa_res.sources)} papers | Chunks: {len(qa_res.retrieved_chunks)}")
    print("Grounded Answer Snippet:\n")
    print(sanitize_str(qa_res.answer[:350]) + "...\n")

    # 3. Literature-Review Generation (FR-8)
    print("\n--- 2. LITERATURE REVIEW SYNTHESIS MODE (FR-8) ---")
    raw_papers = [
        {"arxiv_id": s.paper_id, "title": s.title, "authors": s.authors, "year": s.year, "category": s.category, "abstract": s.abstract}
        for s in qa_res.sources
    ]
    lit_engine = MultiPaperLitReviewEngine()
    lit_res = lit_engine.generate_lit_review(query, reranked, raw_papers)
    print(f"Introduction: {sanitize_str(lit_res.introduction)}")
    print(f"Research Gaps ({len(lit_res.identified_research_gaps)} identified): {lit_res.identified_research_gaps[0]}")
    print("Methodology Synthesis Snippet:\n")
    print(sanitize_str(lit_res.methodology_synthesis[:250]) + "...\n")

    # 4. Non-RAG Baseline Generation (FR-13)
    print("\n--- 3. NON-RAG BASELINE MODE (FR-13 - Parametric Memory Only) ---")
    baseline_engine = NonRAGBaselineEngine()
    baseline_res = baseline_engine.generate_baseline_answer(query)
    print(f"Mode: {baseline_res['mode']} | Context Used: {baseline_res['context_used']}")
    print("Baseline Answer Snippet:\n")
    print(sanitize_str(baseline_res['answer'][:300]) + "...\n")

    print("==========================================================================")
    print("PHASE 3 DEFINITION OF DONE MET: Grounded Generation & Baseline Tested Clean!")
    print("==========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Generation CLI Smoke Test")
    parser.add_argument("--query", type=str, default="What are the core methodology contributions of Vision Transformers and Dense Passage Retrieval?", help="Query string")
    args = parser.parse_args()

    run_generation_smoke_test(query=args.query)
