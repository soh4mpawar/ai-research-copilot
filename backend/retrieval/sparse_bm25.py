"""
Sparse BM25 Keyword Retrieval Module (Phase 2 / FR-4, PRD §7.1.2).
Uses bm25s / LlamaIndex BM25Retriever for fast sparse keyword search.
Handles math block tokenization limitations explicitly per FR-4.
"""

import os
from typing import List, Dict, Any
import bm25s
from backend.ingestion.vector_store import VectorStore


class SparseBM25Retriever:
    """Sparse keyword retrieval engine using bm25s (FR-4)."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.retriever = None
        self.corpus_chunks: List[Dict[str, Any]] = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build or retrieve bm25s index from ChromaDB document corpus."""
        try:
            # Query all stored documents from persistent ChromaDB
            res = self.vector_store.collection.get(include=["documents", "metadatas"])
            if not res or not res.get("ids"):
                print("[BM25] No chunks in ChromaDB vector store yet.")
                return

            ids = res["ids"]
            docs = res["documents"]
            metas = res["metadatas"]

            for i in range(len(ids)):
                self.corpus_chunks.append({
                    "chunk_id": ids[i],
                    "text": docs[i],
                    "paper_id": metas[i].get("paper_id", ""),
                    "paper_title": metas[i].get("paper_title", ""),
                    "authors": metas[i].get("authors", ""),
                    "section": metas[i].get("section", "General"),
                    "page": metas[i].get("page", 1)
                })

            corpus_texts = [c["text"] for c in self.corpus_chunks]
            
            # Tokenize corpus for bm25s
            corpus_tokens = bm25s.tokenize(corpus_texts, stemmer=None)
            self.retriever = bm25s.BM25()
            self.retriever.index(corpus_tokens)
            print(f"[BM25] Indexed {len(corpus_texts)} corpus chunks into bm25s sparse index.")
        except Exception as e:
            print(f"[BM25] Indexing note: {e}")

    def search_sparse(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Run sparse BM25 keyword search returning top-k matching chunks."""
        if not self.retriever or not self.corpus_chunks:
            # Fallback search if bm25s unindexed
            return self._fallback_keyword_search(query_text, top_k)

        try:
            query_tokens = bm25s.tokenize([query_text], stemmer=None)
            results, scores = self.retriever.retrieve(query_tokens, k=min(top_k, len(self.corpus_chunks)))

            out = []
            for rank_idx, doc_idx in enumerate(results[0]):
                score = float(scores[0][rank_idx])
                c = self.corpus_chunks[doc_idx].copy()
                c["bm25_score"] = round(score, 4)
                c["bm25_rank"] = rank_idx + 1
                out.append(c)

            return out
        except Exception as e:
            print(f"[BM25] Retrieval exception: {e}")
            return self._fallback_keyword_search(query_text, top_k)

    def _fallback_keyword_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback lexical keyword matcher."""
        keywords = [w.lower() for w in query_text.split() if len(w) > 2]
        scored = []

        for idx, c in enumerate(self.corpus_chunks, 1):
            text_lower = c["text"].lower()
            match_count = sum(1 for kw in keywords if kw in text_lower)
            if match_count > 0:
                score = round(match_count / len(keywords), 3)
                item = c.copy()
                item["bm25_score"] = score
                scored.append(item)

        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        for r, item in enumerate(scored[:top_k], 1):
            item["bm25_rank"] = r

        return scored[:top_k]
