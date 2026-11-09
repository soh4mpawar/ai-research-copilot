"""
Natural Retrieval Adapter for Controlled Selective Experiments (Protocol Amendment v1.1).
Executes genuine hybrid retrieval (Dense Vector + Sparse BM25 + RRF Fusion + Cross-Encoder Reranking)
strictly restricted to an assigned paper pool (e.g. training split) prior to candidate ranking.
"""

import time
import hashlib
from typing import List, Dict, Any, Set, Optional

import numpy as np
import torch
import bm25s
from sentence_transformers import CrossEncoder

from backend.ingestion.vector_store import VectorStore
from backend.retrieval.fusion import RRFFusionRetriever
from backend.selective.context import (
    Passage,
    FinalContextRecord,
    select_final_context,
    compute_text_sha256,
    count_tokens
)


class SplitRestrictedHybridRetriever:
    """
    Retrieval adapter implementing the production pipeline strictly restricted to a paper pool.
    Pipeline:
      1. Dense search (nomic-embed-text-v1.5) over pool chunks.
      2. Sparse keyword search (bm25s) indexed exclusively over pool chunks.
      3. Reciprocal Rank Fusion (RRF, k=60).
      4. Cross-encoder reranking (BAAI/bge-reranker-base) on CUDA.
      5. Final context selection (6 passages, <= 2,000 tokens).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: CrossEncoder,
        allowed_paper_ids: List[str],
        dense_top_k: int = 25,
        sparse_top_k: int = 25,
        fusion_top_k: int = 30,
        final_max_passages: int = 6,
        final_max_tokens: int = 2000
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.allowed_paper_ids: Set[str] = set(allowed_paper_ids)
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.fusion_top_k = fusion_top_k
        self.final_max_passages = final_max_passages
        self.final_max_tokens = final_max_tokens
        
        self.fusion = RRFFusionRetriever(rrf_k=60.0)
        self.pool_chunks: List[Dict[str, Any]] = []
        self.pool_chunk_ids: Set[str] = set()
        self.chunk_by_id: Dict[str, Dict[str, Any]] = {}
        self.bm25_retriever: Optional[bm25s.BM25] = None
        
        # Enforce and record actual loaded model identities
        if getattr(self.vector_store, "embed_model", None) is None:
            raise RuntimeError(
                "Experiment mode violation: vector_store.embed_model is None. "
                "A verified embedding model is strictly required for research experiments."
            )
        embed_obj = self.vector_store.embed_model
        if hasattr(embed_obj, "tokenizer") and hasattr(embed_obj.tokenizer, "name_or_path") and embed_obj.tokenizer.name_or_path:
            self.embedding_model_name = str(embed_obj.tokenizer.name_or_path)
        elif hasattr(embed_obj, "name_or_path") and embed_obj.name_or_path:
            self.embedding_model_name = str(embed_obj.name_or_path)
        else:
            self.embedding_model_name = type(embed_obj).__name__

        if self.reranker is None:
            raise RuntimeError(
                "Experiment mode violation: reranker is None. "
                "A verified CrossEncoder reranker is strictly required for research experiments."
            )
        if hasattr(self.reranker, "model") and hasattr(self.reranker.model, "name_or_path") and self.reranker.model.name_or_path:
            self.reranker_model_name = str(self.reranker.model.name_or_path)
        elif hasattr(self.reranker, "tokenizer") and hasattr(self.reranker.tokenizer, "name_or_path") and self.reranker.tokenizer.name_or_path:
            self.reranker_model_name = str(self.reranker.tokenizer.name_or_path)
        else:
            self.reranker_model_name = type(self.reranker).__name__

        self.fusion_method = "Reciprocal Rank Fusion (k=60)"
        self.sparse_method = "bm25s (stemmer=None)"

        self._build_restricted_pool()

    def _build_restricted_pool(self):
        """Load all chunks for allowed papers and construct the restricted BM25 index."""
        t0 = time.time()
        print(f"[RestrictedRetriever] Loading chunks for {len(self.allowed_paper_ids)} papers...")
        
        # Load paper by paper to prevent ChromaDB where-clause size limitations
        self.pool_chunks = []
        self.pool_chunk_ids = set()
        self.chunk_by_id = {}
        self.total_allowed_chunks = 0
        for pid in sorted(list(self.allowed_paper_ids)):
            res = self.vector_store.collection.get(
                where={"paper_id": pid},
                include=["metadatas", "documents"]
            )
            if not res or not res.get("ids"):
                continue
            self.total_allowed_chunks += len(res["ids"])
            for cid, meta, doc in zip(res["ids"], res["metadatas"], res["documents"]):
                sec = meta.get("section", "General") if meta else "General"
                # Exclude reference sections and empty text from candidate pool
                if "ref" in sec.lower() or not doc.strip():
                    continue
                chunk_record = {
                    "chunk_id": cid,
                    "paper_id": pid,
                    "section": sec,
                    "text": doc,
                    "token_count": count_tokens(doc)
                }
                self.pool_chunks.append(chunk_record)
                self.pool_chunk_ids.add(cid)
                self.chunk_by_id[cid] = chunk_record

        print(f"[RestrictedRetriever] Loaded {len(self.pool_chunks)} substantive chunks (out of {self.total_allowed_chunks} total paper chunks) in {time.time() - t0:.2f}s.")
        
        # Build BM25 index exclusively on these pool chunks
        if self.pool_chunks:
            corpus_texts = [c["text"] for c in self.pool_chunks]
            corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=None)
            self.bm25_retriever = bm25s.BM25()
            self.bm25_retriever.index(corpus_tokens)
            print(f"[RestrictedRetriever] Built pool BM25 index with {len(corpus_texts)} documents.")
        else:
            raise RuntimeError(f"No chunks found for allowed papers: {self.allowed_paper_ids}")

    def search_dense(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run dense retrieval strictly restricted to the allowed paper pool and eligible pool chunks.
        
        Dense candidate policy:
        Because ChromaDB does not support complex metadata regex filtering (e.g. section title checks)
        within its query where-clause, this method queries ChromaDB with paper-level split filtering
        (`paper_id IN allowed_papers`) and filters out ineligible chunks (such as reference sections or
        boilerplate).
        To ensure valid candidates are not lost when many high-ranking ineligible chunks precede eligible
        ones, it employs progressive query expansion: starting at max(top_k * 4, 30), it progressively
        doubles n_results within the allowed paper pool until at least top_k eligible candidates are found
        or the paper pool is exhausted. It never falls back to an unconstrained global query.
        """
        query_vec = self.vector_store._compute_dense_vector(query, is_query=True, strict=True)
        pids_list = sorted(list(self.allowed_paper_ids))
        max_possible = getattr(self, "total_allowed_chunks", len(self.pool_chunks) * 2)
        if max_possible == 0:
            return []

        batch_size = min(max(top_k * 4, 30), max_possible)
        results = []

        while True:
            try:
                # Query ChromaDB with split filter
                res = self.vector_store.collection.query(
                    query_embeddings=[query_vec],
                    n_results=batch_size,
                    where={"paper_id": {"$in": pids_list}}
                )
            except Exception as e:
                # Unauthorized fallback to unconstrained global query is strictly prohibited.
                raise RuntimeError(f"Split-filtered dense query failed: {e}") from e

            results = []
            returned_count = 0
            if res and res.get("ids") and len(res["ids"]) > 0:
                returned_count = len(res["ids"][0])
                rank = 1
                for i in range(returned_count):
                    cid = res["ids"][0][i]
                    # Enforce identical chunk eligibility between dense and sparse BM25:
                    # Any chunk not in pool_chunk_ids (e.g. references or boilerplate) is strictly excluded
                    if cid not in self.pool_chunk_ids:
                        continue
                    
                    chunk_info = self.chunk_by_id[cid]
                    dist = res["distances"][0][i] if res.get("distances") else 0.5
                    score = round(max(0.0, 1.0 - dist), 4)

                    results.append({
                        "chunk_id": cid,
                        "paper_id": chunk_info["paper_id"],
                        "section": chunk_info["section"],
                        "text": chunk_info["text"],
                        "dense_score": score,
                        "dense_rank": rank
                    })
                    rank += 1
                    if len(results) >= top_k:
                        break

            # If we collected enough eligible results, or ChromaDB exhausted the pool
            if len(results) >= top_k or returned_count < batch_size or batch_size >= max_possible:
                break

            # Progressively expand batch size within the allowed paper pool
            next_batch = min(batch_size * 2, max_possible)
            if next_batch <= batch_size:
                break
            batch_size = next_batch

        return results[:top_k]

    def search_sparse(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run BM25 search over the pool-specific index."""
        if not self.bm25_retriever or not self.pool_chunks:
            return []
        
        query_tokens = bm25s.tokenize([query], stemmer=None)
        k_val = min(top_k, len(self.pool_chunks))
        results, scores = self.bm25_retriever.retrieve(query_tokens, k=k_val)

        out = []
        for rank_idx, doc_idx in enumerate(results[0]):
            score = float(scores[0][rank_idx])
            c = self.pool_chunks[doc_idx].copy()
            c["bm25_score"] = round(score, 4)
            c["bm25_rank"] = rank_idx + 1
            out.append(c)
        return out

    def retrieve_hybrid_context(
        self,
        query: str,
        max_passages: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute full pipeline: Dense + Sparse BM25 -> RRF -> Cross-Encoder -> select_final_context.
        """
        target_passages = max_passages or self.final_max_passages
        target_tokens = max_tokens or self.final_max_tokens
        
        t0 = time.time()
        
        # 1. Dense retrieval
        t_dense_0 = time.time()
        dense_results = self.search_dense(query, top_k=self.dense_top_k)
        t_dense = time.time() - t_dense_0
        
        # 2. Sparse BM25 retrieval
        t_sparse_0 = time.time()
        sparse_results = self.search_sparse(query, top_k=self.sparse_top_k)
        t_sparse = time.time() - t_sparse_0
        
        # 3. Reciprocal Rank Fusion
        t_fuse_0 = time.time()
        fused_candidates = self.fusion.fuse_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=self.fusion_top_k
        )
        t_fuse = time.time() - t_fuse_0
        
        # 4. Cross-Encoder Reranking
        t_rerank_0 = time.time()
        pairs = [[query, c["text"]] for c in fused_candidates]
        if pairs:
            raw_scores = self.reranker.predict(pairs)
            for idx, c in enumerate(fused_candidates):
                c["rerank_score"] = round(float(raw_scores[idx]), 4)
        
        # Sort strictly by rerank_score descending
        reranked_candidates = sorted(fused_candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        for rank_idx, c in enumerate(reranked_candidates, 1):
            c["rerank_rank"] = rank_idx
        t_rerank = time.time() - t_rerank_0
        
        # 5. Deterministic Final Context Selection
        candidate_passages = [
            Passage.from_dict({
                "passage_id": c["chunk_id"],
                "paper_id": c["paper_id"],
                "section": c["section"],
                "text": c["text"],
                "score": c.get("rerank_score", 0.0),
                "token_count": c.get("token_count") or count_tokens(c["text"])
            })
            for c in reranked_candidates
        ]
        
        final_record: FinalContextRecord = select_final_context(
            query=query,
            candidates=candidate_passages,
            max_passages=target_passages,
            token_budget=target_tokens
        )
        
        t_total = time.time() - t0
        
        return {
            "final_record": final_record,
            "candidate_trace": reranked_candidates,
            "timings": {
                "dense_sec": round(t_dense, 4),
                "sparse_sec": round(t_sparse, 4),
                "fusion_sec": round(t_fuse, 4),
                "rerank_sec": round(t_rerank, 4),
                "total_sec": round(t_total, 4)
            },
            "model_metadata": {
                "embedding_model": self.embedding_model_name,
                "sparse_retriever": self.sparse_method,
                "fusion_algorithm": self.fusion_method,
                "cross_encoder": self.reranker_model_name,
                "target_passages": target_passages,
                "target_tokens": target_tokens,
                "allowed_papers_count": len(self.allowed_paper_ids)
            }
        }
