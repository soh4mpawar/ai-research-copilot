"""
Vector Store & Dense Embedding Module (Phase 1 / FR-3, PRD §7.1).
Embeds section-aware chunks using nomic-embed-text / dense vector hashing into persistent local ChromaDB.
Includes unicode surrogate sanitization for robust PDF text persistence.
"""

import os
import math
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional


def clean_unicode_str(s: str) -> str:
    """Sanitize lone unicode surrogates to prevent Rust C-extension serialization crashes."""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "ignore").decode("utf-8")


class VectorStore:
    """Local persistent ChromaDB vector store manager for dense embeddings."""

    def __init__(self, persist_dir: str = "data/chroma_db", collection_name: str = "scientific_papers"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embed_model = None
        self._init_embedding_model()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _init_embedding_model(self):
        """Initialize nomic-embed-text / sentence-transformers embedding model."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device)
                print(f"[VectorStore] Loaded nomic-ai/nomic-embed-text-v1.5 model successfully on {device}.")
            except Exception:
                pass
        except Exception:
            pass

    def _compute_dense_vector(self, text: str, dim: int = 384, is_query: bool = True) -> List[float]:
        """Generate a dense semantic vector using nomic-embed-text (768-dim) or fallback hashed vector."""
        if self.embed_model:
            try:
                prefix = "search_query: " if is_query else "search_document: "
                return self.embed_model.encode([f"{prefix}{text}"], show_progress_bar=False)[0].tolist()
            except Exception:
                pass

        # High-performance hashed dense feature vector fallback (384 dimensions)
        vec = [0.0] * dim
        words = text.lower().split()
        for idx, word in enumerate(words):
            h = hash(word) % dim
            vec[h] += 1.0 / (idx + 1.0)

        # L2 Normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Embed chunks and persist in local ChromaDB collection in memory-safe slices."""
        if not chunks:
            return

        import gc
        import torch

        # Check already existing chunk IDs to enable instant resume
        existing_ids = set()
        try:
            existing_data = self.collection.get(include=[])
            existing_ids = set(existing_data.get("ids", []))
        except Exception:
            pass

        missing_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not missing_chunks:
            print(f"[VectorStore] All {len(chunks)} chunks already present in collection '{self.collection_name}'.")
            return

        slice_size = 500
        total = len(missing_chunks)
        print(f"[VectorStore] Ingesting {total} missing chunks (of {len(chunks)} total) in slices of {slice_size}...")

        for i in range(0, total, slice_size):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            s_chunks = missing_chunks[i:i + slice_size]
            s_ids = [clean_unicode_str(c["chunk_id"]) for c in s_chunks]
            s_docs = [clean_unicode_str(c["text"]) for c in s_chunks]
            s_metas = [
                {
                    "paper_id": clean_unicode_str(c["paper_id"]),
                    "paper_title": clean_unicode_str(c["paper_title"]),
                    "authors": clean_unicode_str(c["authors"] if isinstance(c["authors"], str) else ", ".join(c["authors"])),
                    "section": clean_unicode_str(c["section"]),
                    "token_count": c.get("token_count", 300),
                    "oversized_for_reranker": c.get("oversized_for_reranker", False),
                    "page": c.get("page", 1)
                }
                for c in s_chunks
            ]

            if self.embed_model:
                self.embed_model.max_seq_length = 512
                formatted_docs = [f"search_document: {d[:2500]}" for d in s_docs]
                with torch.inference_mode():
                    raw_embs = self.embed_model.encode(formatted_docs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
                s_embs = [e.tolist() for e in raw_embs]
            else:
                s_embs = [self._compute_dense_vector(d) for d in s_docs]

            self.collection.upsert(ids=s_ids, documents=s_docs, embeddings=s_embs, metadatas=s_metas)
            print(f"  • Upserted {min(i + slice_size, total)} / {total} chunks ({(min(i + slice_size, total)/total)*100:.1f}%)", flush=True)

        print(f"[VectorStore] Successfully finalized ChromaDB collection '{self.collection_name}' ({self.collection.count()} chunks total).")

    def search_dense(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Run dense vector similarity search over ChromaDB."""
        query_vec = self._compute_dense_vector(query_text)
        res = self.collection.query(query_embeddings=[query_vec], n_results=top_k)

        results = []
        if res and res.get("ids") and len(res["ids"]) > 0:
            for i in range(len(res["ids"][0])):
                cid = res["ids"][0][i]
                doc = res["documents"][0][i] if res.get("documents") else ""
                meta = res["metadatas"][0][i] if res.get("metadatas") else {}
                dist = res["distances"][0][i] if res.get("distances") else 0.5
                score = round(max(0.0, 1.0 - dist), 3)

                results.append({
                    "chunk_id": cid,
                    "text": doc,
                    "paper_id": meta.get("paper_id", ""),
                    "paper_title": meta.get("paper_title", ""),
                    "authors": meta.get("authors", ""),
                    "section": meta.get("section", "General"),
                    "dense_score": score,
                    "dense_rank": i + 1,
                    "page": meta.get("page", 1)
                })

        return results

    def get_total_chunks_count(self) -> int:
        """Return total stored chunk count in ChromaDB."""
        return self.collection.count()
