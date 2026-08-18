"""
GraphRAG Traversal Candidate Source (Phase 6 Extension / FR-15, PRD §7.1).
Seeds graph traversal from top-N RRF candidates, traverses 1-hop outward in the citation network,
and surfaces pre-computed ChromaDB vector chunks with silent fallback on empty graph streams.
"""

from typing import List, Dict, Any
from backend.graph.neo4j_builder import CitationGraphEngine
from backend.ingestion.vector_store import VectorStore


class GraphRAGTraversalRetriever:
    """1-Hop Citation Graph Traversal Candidate Source (FR-15)."""

    def __init__(self, graph_engine: CitationGraphEngine, vector_store: VectorStore):
        self.graph_engine = graph_engine
        self.vector_store = vector_store

    def traverse_and_fetch_chunks(
        self,
        rrf_seed_chunks: List[Dict[str, Any]],
        max_graph_candidates: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Seed 1-hop graph traversal from top-N RRF candidates.
        Fetch pre-computed ChromaDB chunks for surfaced citation-connected papers.
        """
        if not rrf_seed_chunks or self.graph_engine.graph.number_of_nodes() == 0:
            # Silent fallback to RRF alone per FR-15
            return []

        # 1. Extract Seed Paper ArXiv IDs
        seed_paper_ids = set()
        for c in rrf_seed_chunks[:5]:
            pid = c.get("paper_id", "")
            if pid:
                seed_paper_ids.add(pid)

        # 2. 1-Hop Outward Graph Traversal per FR-15
        traversed_paper_ids = set()
        for seed_id in seed_paper_ids:
            if seed_id in self.graph_engine.graph:
                for successor in self.graph_engine.graph.successors(seed_id):
                    traversed_paper_ids.add(successor)
                for predecessor in self.graph_engine.graph.predecessors(seed_id):
                    traversed_paper_ids.add(predecessor)

        # Remove original seeds from graph candidates pool
        graph_target_ids = list(traversed_paper_ids - seed_paper_ids)

        if not graph_target_ids:
            # Silent fallback per FR-15
            return []

        # 3. Fetch pre-computed ChromaDB chunks for graph target papers (No new embedding calls per FR-15)
        graph_chunks = []
        try:
            for target_pid in graph_target_ids[:max_graph_candidates]:
                res = self.vector_store.collection.get(
                    where={"paper_id": target_pid},
                    limit=2
                )
                if res and res.get("ids"):
                    for i in range(len(res["ids"])):
                        doc = res["documents"][i]
                        meta = res["metadatas"][i]
                        graph_chunks.append({
                            "chunk_id": res["ids"][i],
                            "text": doc,
                            "paper_id": meta.get("paper_id", target_pid),
                            "paper_title": meta.get("paper_title", "Graph Citation Paper"),
                            "authors": meta.get("authors", "Graph Author"),
                            "section": meta.get("section", "Citation Context"),
                            "graph_traversed": True,
                            "dense_rank": 999,
                            "bm25_rank": 999,
                            "score": 0.65
                        })
        except Exception as e:
            print(f"[GraphRAG] Chunk retrieval note: {e}")

        print(f"[GraphRAG FR-15] Traversed 1-hop graph from {len(seed_paper_ids)} seed papers -> surfaced {len(graph_chunks)} pre-computed ChromaDB chunks.")
        return graph_chunks
