"""
Phase 6 GraphRAG & Citation Network Verification Test Suite.
Tests:
1. Citation Graph Initialization (FR-14)
2. Network Topology & Community Metrics
3. 1-Hop GraphRAG Traversal Candidate Expansion (FR-15)
4. Subgraph Extraction for Interactive Visualization (FR-16)
5. End-to-End Orchestrator Query with GraphRAG
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from backend.graph.neo4j_builder import CitationGraphEngine
from backend.graph.graph_traversal import GraphRAGTraversalRetriever
from backend.ingestion.vector_store import VectorStore
from backend.pipeline import ResearchPipelineOrchestrator


def run_phase6_tests():
    print("=" * 80)
    print("PHASE 6: GRAPHRAG CITATION NETWORK VERIFICATION SUITE")
    print("=" * 80)

    # 1. Test Citation Graph Engine (FR-14)
    print("\n[Step 1/5] Testing CitationGraphEngine...")
    engine = CitationGraphEngine()
    stats = engine.get_citation_stats()
    print(f"  • Total Paper Nodes: {stats['total_nodes']}")
    print(f"  • Total Citation Edges: {stats['total_edges']}")
    print(f"  • Within-Category Edges: {stats['within_category_edges']}")
    print(f"  • Cross-Category Edges: {stats['cross_category_edges']} ({stats['cross_category_pct']}%)")
    print("  • Top Citation Hubs:")
    for h in stats["top_citation_hubs"]:
        print(f"    - [{h['paper_id']}] {h['title'][:45]}... (In-Degree Citations: {h['in_citations']})")

    assert stats["total_edges"] >= 80, f"Expected >= 80 edges, found {stats['total_edges']}"
    assert stats["cross_category_edges"] > 0, "Expected non-zero cross-category edges"
    print("  => Step 1 PASSED!")

    # 2. Test 1-Hop Subgraph Extraction (FR-16)
    print("\n[Step 2/5] Testing Subgraph Extraction for Interactive UI Visualization (FR-16)...")
    subgraph_attention = engine.get_subgraph_data("1706.03762")
    print(f"  • Subgraph for 'Attention Is All You Need' (1706.03762):")
    print(f"    - Rendered Nodes: {len(subgraph_attention['nodes'])}")
    print(f"    - Rendered Edges: {len(subgraph_attention['edges'])}")
    
    subgraph_rag = engine.get_subgraph_data("2005.11401")
    print(f"  • Subgraph for 'RAG' (2005.11401):")
    print(f"    - Rendered Nodes: {len(subgraph_rag['nodes'])}")
    print(f"    - Rendered Edges: {len(subgraph_rag['edges'])}")

    assert len(subgraph_attention["nodes"]) > 1, "Failed to build 1-hop neighborhood for Attention"
    print("  => Step 2 PASSED!")

    # 3. Test GraphRAG Traversal Candidate Retriever (FR-15)
    print("\n[Step 3/5] Testing GraphRAG Traversal Candidate Retriever (FR-15)...")
    vs = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
    retriever = GraphRAGTraversalRetriever(engine, vs)

    # Seed with real RRF candidates
    synthetic_rrf_seeds = [
        {"paper_id": "2005.11401", "paper_title": "Retrieval-Augmented Generation", "text": "RAG text", "score": 0.9},
        {"paper_id": "2004.04906", "paper_title": "Dense Passage Retrieval", "text": "DPR text", "score": 0.85}
    ]

    traversed_chunks = retriever.traverse_and_fetch_chunks(synthetic_rrf_seeds, max_graph_candidates=10)
    print(f"  • Surfaced Pre-Computed ChromaDB Chunks: {len(traversed_chunks)}")
    for tc in traversed_chunks[:3]:
        print(f"    - Chunk [{tc['chunk_id']}] from Connected Paper [{tc['paper_id']}] '{tc['paper_title'][:35]}...' (Graph Traversed: {tc['graph_traversed']})")

    assert len(traversed_chunks) > 0, "Expected traversed chunks from RAG/DPR citation neighborhood"
    print("  => Step 3 PASSED!")

    # 4. Test Silent Fallback on Disconnected / Empty Seed (FR-15)
    print("\n[Step 4/5] Testing Silent Fallback on Disconnected / Empty Graph Stream (FR-15)...")
    empty_seeds = [{"paper_id": "9999.99999", "paper_title": "Non-existent Paper", "text": "...", "score": 0.1}]
    fallback_chunks = retriever.traverse_and_fetch_chunks(empty_seeds)
    print(f"  • Fallback Candidate Count: {len(fallback_chunks)} (Graceful silent degradation with 0 exceptions)")
    assert len(fallback_chunks) == 0, "Expected empty fallback list"
    print("  => Step 4 PASSED!")

    # 5. End-to-End Pipeline Execution with GraphRAG (FR-15, FR-18)
    print("\n[Step 5/5] Testing End-to-End Orchestrator Query with GraphRAG Active...")
    orchestrator = ResearchPipelineOrchestrator()
    query = "How did Dense Passage Retrieval and BERT contribute to Retrieval-Augmented Generation models?"
    res = orchestrator.execute_query(query, enable_graph_rag=True)
    
    print(f"  • Generated Answer Length: {len(res.answer)} chars")
    print(f"  • Sources Referenced: {len(res.sources)}")
    print(f"  • Retrieved Chunks Count: {len(res.retrieved_chunks)}")
    print(f"  • Pipeline Metrics: {res.metrics.model_dump() if hasattr(res.metrics, 'model_dump') else res.metrics}")
    print("  => Step 5 PASSED!")

    print("\n" + "=" * 80)
    print("ALL PHASE 6 GRAPHRAG & CITATION NETWORK TESTS COMPLETED SUCCESSFULLY (100% PASS)")
    print("=" * 80)


if __name__ == "__main__":
    run_phase6_tests()
