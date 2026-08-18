"""
Contract Interface Definitions between Backend (A) and Frontend (S).
Defines strict data schemas for queries, retrieval context, citations,
literature reviews, evaluation metrics, and citation graph data.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SourcePaper(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: int
    arxiv_id: Optional[str] = None
    category: str = "cs.CL"
    venue: Optional[str] = "arXiv"
    citation_count: int = 0
    pdf_url: Optional[str] = None
    abstract: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    paper_id: str
    paper_title: str
    authors: str
    section: str
    text: str
    score: float  # Final normalized relevance score [0.0, 1.0]
    dense_rank: int
    bm25_rank: int
    rrf_rank: int
    rerank_score: float
    page: Optional[int] = 1


class PipelineMetrics(BaseModel):
    retrieval_time_sec: float
    reranking_time_sec: float
    generation_time_sec: float
    total_time_sec: float
    dense_candidates_count: int = 20
    bm25_candidates_count: int = 20
    rrf_candidates_count: int = 25
    reranked_candidates_count: int = 10
    final_context_chunks_count: int = 6


class QueryResult(BaseModel):
    query: str
    mode: str = "qa"  # qa | summary | literature_review
    answer: str
    evidence_strength: str = "Strong"  # Strong | Moderate | Weak
    sources: List[SourcePaper]
    retrieved_chunks: List[RetrievedChunk]
    metrics: PipelineMetrics
    generator_model: str = "Google/gemini-3.7-flash"


class LitReviewResult(BaseModel):
    topic: str
    introduction: str
    comparison_table: List[Dict[str, Any]]
    architectural_evolution: str
    methodology_synthesis: str
    identified_research_gaps: List[str]
    conclusion: str
    sources: List[SourcePaper]


class CitationGraphData(BaseModel):
    nodes: List[Dict[str, Any]]  # id, label, group, title, val
    edges: List[Dict[str, Any]]  # from, to, label, weight


class EvalMetrics(BaseModel):
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevance: float
    rag_vs_non_rag: Dict[str, Dict[str, float]]
    stage_comparisons: Dict[str, Dict[str, float]]
    eval_samples: List[Dict[str, Any]]
