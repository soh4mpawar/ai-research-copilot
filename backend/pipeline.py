"""
Real Production Pipeline Orchestrator (Phases 4-6 / FR-9, FR-14, FR-15, FR-16, FR-18, FR-19).
Connects:
- ChromaDB Vector Store (nomic-embed-text-v1.5 CUDA embeddings)
- BM25 Sparse Retriever (bm25s)
- Reciprocal Rank Fusion (RRF k=60.0)
- GraphRAG 1-Hop Traversal Candidate Source (FR-15)
- Cross-Encoder Reranker (BAAI/bge-reranker-base CUDA)
- Grounded Gemini Point-QA Engine (FR-7)
- Multi-Paper Literature Review Engine (FR-8)
- Citation Graph Engine & Subgraph Visualizer (FR-14, FR-16)
- Non-RAG Baseline Engine (FR-13)
"""

import time
import os
from dotenv import load_dotenv

load_dotenv()
from typing import List, Dict, Any, Tuple, Optional

from backend.contract import (
    QueryResult,
    SourcePaper,
    LitReviewResult,
    CitationGraphData,
    EvalMetrics
)
from backend.ingestion.vector_store import VectorStore
from backend.retrieval.sparse_bm25 import SparseBM25Retriever
from backend.retrieval.fusion import RRFFusionRetriever
from backend.reranking.reranker import CrossEncoderReranker
from backend.generation.point_qa import GroundedPointQAEngine
from backend.generation.lit_review import MultiPaperLitReviewEngine
from backend.generation.baseline import NonRAGBaselineEngine
from backend.graph.neo4j_builder import CitationGraphEngine
from backend.graph.graph_traversal import GraphRAGTraversalRetriever


class ResearchPipelineOrchestrator:
    """Master backend pipeline orchestrator with singleton component caching (FR-19)."""

    def __init__(self):
        self.vector_store = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
        self.bm25_retriever = SparseBM25Retriever(self.vector_store)
        self.fusion_retriever = RRFFusionRetriever(rrf_k=60.0)
        self.reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
        self.graph_engine = CitationGraphEngine()
        self.graph_retriever = GraphRAGTraversalRetriever(self.graph_engine, self.vector_store)
        self.qa_engine = GroundedPointQAEngine()
        self.lit_review_engine = MultiPaperLitReviewEngine()
        self.baseline_engine = NonRAGBaselineEngine()
        self.corpus_version = f"corpus_v1.0_{self.vector_store.get_total_chunks_count()}chunks"

    def execute_query(
        self,
        query_text: str,
        mode: str = "qa",
        enable_graph_rag: bool = True
    ) -> QueryResult:
        """
        Execute full hybrid retrieval, GraphRAG 1-hop candidate expansion (FR-15),
        cross-encoder reranking, and grounded answer generation.
        """
        t0 = time.time()

        # 1. Hybrid Retrieval (Dense Vector + Sparse BM25)
        dense_candidates = self.vector_store.search_dense(query_text, top_k=25)
        sparse_candidates = self.bm25_retriever.search_sparse(query_text, top_k=25)

        # 2. Reciprocal Rank Fusion (RRF)
        fused_candidates = self.fusion_retriever.fuse_results(dense_candidates, sparse_candidates, top_k=30)

        # 3. GraphRAG 1-Hop Traversal Candidate Source (FR-15)
        candidate_pool = list(fused_candidates)
        graph_candidate_count = 0
        if enable_graph_rag:
            graph_candidates = self.graph_retriever.traverse_and_fetch_chunks(fused_candidates, max_graph_candidates=10)
            if graph_candidates:
                candidate_pool.extend(graph_candidates)
                graph_candidate_count = len(graph_candidates)

        # 4. Cross-Encoder Reranking
        reranked_candidates = self.reranker.rerank_chunks(query_text, candidate_pool, top_k=12)

        # 5. Grounded Answer Generation via Gemini
        result = self.qa_engine.generate_point_qa(
            query=query_text,
            reranked_chunks=reranked_candidates,
            dense_count=len(dense_candidates),
            sparse_count=len(sparse_candidates),
            rrf_count=len(fused_candidates)
        )

        return result

    def execute_lit_review(self, topic: str, enable_graph_rag: bool = True) -> LitReviewResult:
        """Execute multi-paper literature review synthesis with GraphRAG lineage discovery."""
        dense_candidates = self.vector_store.search_dense(topic, top_k=20)
        sparse_candidates = self.bm25_retriever.search_sparse(topic, top_k=20)
        fused_candidates = self.fusion_retriever.fuse_results(dense_candidates, sparse_candidates, top_k=25)

        candidate_pool = list(fused_candidates)
        if enable_graph_rag:
            graph_candidates = self.graph_retriever.traverse_and_fetch_chunks(fused_candidates, max_graph_candidates=15)
            if graph_candidates:
                candidate_pool.extend(graph_candidates)

        reranked_candidates = self.reranker.rerank_chunks(topic, candidate_pool, top_k=12)
        raw_papers = self.bm25_retriever.corpus_chunks[:10]
        return self.lit_review_engine.generate_lit_review(topic, reranked_candidates, raw_papers)

    def execute_baseline(self, query_text: str) -> Dict[str, Any]:
        """Execute Non-RAG baseline query (FR-13)."""
        return self.baseline_engine.generate_baseline_answer(query_text)

    def get_citation_subgraph(self, seed_paper_id: str) -> Dict[str, Any]:
        """Fetch 1-hop citation subgraph around seed paper for UI visualization (FR-16)."""
        return self.graph_engine.get_subgraph_data(seed_paper_id)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Fetch citation graph network topology statistics."""
        return self.graph_engine.get_citation_stats()


_ORCHESTRATOR_SINGLETON = None


def get_orchestrator() -> ResearchPipelineOrchestrator:
    """Return application-scoped singleton orchestrator instance."""
    global _ORCHESTRATOR_SINGLETON
    if _ORCHESTRATOR_SINGLETON is None:
        _ORCHESTRATOR_SINGLETON = ResearchPipelineOrchestrator()
    return _ORCHESTRATOR_SINGLETON


def execute_real_query(query_text: str, mode: str = "qa", enable_graph_rag: bool = True) -> QueryResult:
    """Public wrapper for real backend query execution."""
    orchestrator = get_orchestrator()
    return orchestrator.execute_query(query_text, mode=mode, enable_graph_rag=enable_graph_rag)
