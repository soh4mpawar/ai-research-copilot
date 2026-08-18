import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator
from backend.retrieval.threshold_gate import RelevanceThresholdGate

orch = ResearchPipelineOrchestrator()
gate = RelevanceThresholdGate()

def diagnose_query(q):
    dense = orch.vector_store.search_dense(q, top_k=25)
    sparse = orch.bm25_retriever.search_sparse(q, top_k=25)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=30)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
    reranked = orch.reranker.rerank_chunks(q, pool, top_k=12)
    passed, valid, msg = gate.evaluate_chunks(reranked)
    print("=" * 80)
    print(f"QUERY: '{q}'")
    print(f"RESULT: passed={passed} | msg={msg}")
    print(f"TOP 5 CHUNKS:")
    for idx, c in enumerate(reranked[:5]):
        print(f"  [{idx+1}] Score: {c.get('rerank_score'):.4f} | Paper: {c.get('paper_id')} | Section: {c.get('section')} | Text: {c.get('text')[:100].replace(chr(10), ' ')}")

diagnose_query("What are the primary symptoms and diagnostic tests for acute appendicitis?")
diagnose_query("Why does VGG use stacks of 3x3 convolutions instead of larger receptive fields?")
