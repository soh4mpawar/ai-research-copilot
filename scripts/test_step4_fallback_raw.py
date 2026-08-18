import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.graph.neo4j_builder import CitationGraphEngine
from backend.graph.graph_traversal import GraphRAGTraversalRetriever
from backend.ingestion.vector_store import VectorStore

print("--- EXECUTING STEP 4: EMPTY GRAPH FALLBACK TEST ---")
engine = CitationGraphEngine()
vs = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
retriever = GraphRAGTraversalRetriever(engine, vs)

# Case A: Seed papers are completely absent from graph
print("\nCase A: Query produces seeds not present in citation graph:")
empty_seeds = [
    {"paper_id": "9999.99999", "paper_title": "Non-existent Paper 1", "text": "Unconnected snippet", "score": 0.1},
    {"paper_id": "8888.88888", "paper_title": "Non-existent Paper 2", "text": "Another snippet", "score": 0.05}
]
print(f"Input seeds passed to retriever: {[s['paper_id'] for s in empty_seeds]}")
fallback_chunks_a = retriever.traverse_and_fetch_chunks(empty_seeds)
print(f"Output chunks returned: {fallback_chunks_a}")
print(f"Number of graph chunks returned: {len(fallback_chunks_a)}")

# Case B: Empty seeds list passed (e.g. dense/BM25 returned 0 chunks)
print("\nCase B: Empty seeds list passed to retriever:")
fallback_chunks_b = retriever.traverse_and_fetch_chunks([])
print(f"Output chunks returned: {fallback_chunks_b}")
print(f"Number of graph chunks returned: {len(fallback_chunks_b)}")
