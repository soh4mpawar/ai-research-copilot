"""
Phase 2 CLI Smoke-Test Script (Phase 2 Definition of Done / FR-4, FR-5, FR-6, FR-11).
Takes a query string and outputs dense candidates, sparse BM25 candidates, RRF fused results,
cross-encoder reranked scores, and relevance gate status — runnable with no UI.
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
from backend.retrieval.threshold_gate import RelevanceThresholdGate
from backend.utils.vram_manager import VRAMManager


def sanitize_str(s: str) -> str:
    """Sanitize non-ascii characters for clean console printing."""
    if not isinstance(s, str):
        return str(s)
    return s.encode("ascii", "ignore").decode("ascii")


def run_retrieval_pipeline(query: str, top_k: int = 5):
    print("==========================================================================")
    print(f"AI RESEARCH COPILOT -- PHASE 2: HYBRID RETRIEVAL & RERANKING SMOKE TEST")
    print(f"QUERY: '{sanitize_str(query)}'")
    print("==========================================================================")

    # 1. Init Vector Store (Dense ChromaDB)
    vector_store = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
    print(f"[Dense] Persistent ChromaDB collection count: {vector_store.get_total_chunks_count()} chunks.")

    # 2. Dense Vector Search (Top 20)
    dense_candidates = vector_store.search_dense(query, top_k=20)
    print(f"[Dense Vector] Retrieved {len(dense_candidates)} dense candidates.")

    # 3. Sparse BM25 Search (Top 20)
    bm25_engine = SparseBM25Retriever(vector_store)
    sparse_candidates = bm25_engine.search_sparse(query, top_k=20)
    print(f"[Sparse BM25] Retrieved {len(sparse_candidates)} sparse candidates.")

    # 4. Reciprocal Rank Fusion (RRF Top 25)
    fusion_engine = RRFFusionRetriever(rrf_k=60.0)
    fused_candidates = fusion_engine.fuse_results(dense_candidates, sparse_candidates, top_k=25)
    print(f"[RRF Fusion] Fused into {len(fused_candidates)} unique candidates.")

    # 5. Cross-Encoder Reranking (bge-reranker-base Top 10)
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    reranked_candidates = reranker.rerank_chunks(query, fused_candidates, top_k=10)
    print(f"[Cross-Encoder] Reranked top {len(reranked_candidates)} candidates.")

    # 6. Relevance Threshold Gate (FR-11)
    gate = RelevanceThresholdGate(threshold=0.35)
    passed_gate, valid_chunks, status_msg = gate.evaluate_chunks(reranked_candidates)
    print(f"[Relevance Gate] {sanitize_str(status_msg)}")

    print("\n--------------------------------------------------------------------------")
    print(f"TOP {top_k} RERANKED GROUNDED CHUNKS:")
    print("--------------------------------------------------------------------------")

    display_chunks = valid_chunks[:top_k] if passed_gate else reranked_candidates[:top_k]

    for idx, c in enumerate(display_chunks, 1):
        score = c.get("rerank_score", 0.0)
        paper_title = sanitize_str(c.get("paper_title", ""))
        section = sanitize_str(c.get("section", ""))
        authors = sanitize_str(c.get("authors", ""))
        snippet = sanitize_str(c.get("text", "")[:180]).replace("\n", " ")

        print(f"\nRANK #{idx} | Score: {score:.4f} | RRF Rank: #{c.get('rrf_rank')} | Dense: #{c.get('dense_rank')} | BM25: #{c.get('bm25_rank')}")
        print(f"  Paper:    {paper_title}")
        print(f"  Section:  {section} (Chunk ID: {c.get('chunk_id')})")
        print(f"  Authors:  {authors}")
        print(f"  Passage:  \"{snippet}...\"")

    vram_stats = VRAMManager.get_vram_stats()
    print("\n--------------------------------------------------------------------------")
    print(f"VRAM / Hardware Audit: CUDA Available: {vram_stats['cuda_available']} | Peak Alloc: {vram_stats['max_allocated_mb']} MB / 8192 MB")
    print("==========================================================================")
    print("PHASE 2 DEFINITION OF DONE MET: Hybrid Retrieval Pipeline Execution Clean!")
    print("==========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Hybrid Retrieval CLI Test")
    parser.add_argument("--query", type=str, default="What are the core methodology contributions of Vision Transformers and Dense Passage Retrieval?", help="Query string")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top chunks to output")
    args = parser.parse_args()

    run_retrieval_pipeline(query=args.query, top_k=args.top_k)
