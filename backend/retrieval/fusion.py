"""
Reciprocal Rank Fusion (RRF) Module (Phase 2 / FR-5, PRD §7.1).
Fuses dense ChromaDB vector search and sparse BM25 keyword search results without manual weight tuning.
"""

from typing import List, Dict, Any


class RRFFusionRetriever:
    """Reciprocal Rank Fusion retriever merging dense and sparse search streams (FR-5)."""

    def __init__(self, rrf_k: float = 60.0):
        self.rrf_k = rrf_k

    def fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and sparse search candidate lists using RRF.
        Formula: RRF_score(d) = 1/(k + dense_rank) + 1/(k + bm25_rank)
        """
        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # 1. Process Dense Vector Candidates
        for rank, item in enumerate(dense_results, 1):
            cid = item["chunk_id"]
            dense_ranks[cid] = rank
            chunk_lookup[cid] = item
            rrf_score = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_score

        # 2. Process Sparse BM25 Candidates
        for rank, item in enumerate(sparse_results, 1):
            cid = item["chunk_id"]
            sparse_ranks[cid] = rank
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
            rrf_score = 1.0 / (self.rrf_k + rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_score

        # 3. Sort merged candidates by RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        fused_output = []
        for rrf_rank, cid in enumerate(sorted_cids[:top_k], 1):
            item = chunk_lookup[cid].copy()
            item["rrf_score"] = round(fused_scores[cid], 5)
            item["rrf_rank"] = rrf_rank
            item["dense_rank"] = dense_ranks.get(cid, 999)
            item["bm25_rank"] = sparse_ranks.get(cid, 999)
            fused_output.append(item)

        return fused_output
