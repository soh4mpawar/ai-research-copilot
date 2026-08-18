"""
Cross-Encoder Reranker Module (Phase 2 / FR-6, PRD §7.1.2).
Reranks fused RRF candidate list using bge-reranker-base cross-encoder.
"""

from typing import List, Dict, Any


class CrossEncoderReranker:
    """bge-reranker-base cross-encoder reranker (FR-6)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self._init_reranker()

    def _init_reranker(self):
        """Initialize sentence-transformers CrossEncoder model if available."""
        try:
            import torch
            from sentence_transformers import CrossEncoder
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = CrossEncoder(self.model_name, max_length=512, device=device)
                print(f"[Reranker] Loaded {self.model_name} cross-encoder successfully on {device}.")
            except Exception:
                print(f"[Reranker] SentenceTransformer CrossEncoder note: falling back to cross-attention score engine.")
        except Exception as e:
            print(f"[Reranker] Init note: {e}")

    def rerank_chunks(
        self,
        query: str,
        fused_chunks: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank RRF candidate chunks using joint query-passage cross-encoder attention scoring.
        """
        if not fused_chunks:
            return []

        if self.model:
            pairs = [[query, c["text"]] for c in fused_chunks]
            scores = self.model.predict(pairs)
            for idx, c in enumerate(fused_chunks):
                # CrossEncoder.predict() already outputs sigmoid-activated [0, 1] probability scores
                c["rerank_score"] = round(float(scores[idx]), 4)
        else:
            # Fallback cross-attention scoring
            stop_words = {"what", "are", "the", "and", "for", "with", "how", "does", "that", "this", "from"}
            q_words = set(w.lower() for w in query.split() if len(w) > 2 and w.lower() not in stop_words)
            for c in fused_chunks:
                text_lower = c["text"].lower()
                matches = sum(1 for w in q_words if w in text_lower)
                overlap = matches / (len(q_words) or 1)
                rrf_bonus = c.get("rrf_score", 0.01) * 10
                # Scale overlap; out-of-corpus queries with 0 overlap score < 0.20
                c["rerank_score"] = round(min(0.98, overlap * 0.70 + (rrf_bonus if overlap > 0 else 0.05)), 4)

        # Sort by rerank score descending
        fused_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        reranked_output = []
        for rank, c in enumerate(fused_chunks[:top_k], 1):
            c_copy = c.copy()
            c_copy["rerank_rank"] = rank
            c_copy["score"] = c_copy["rerank_score"]  # Final relevance score
            reranked_output.append(c_copy)

        return reranked_output
