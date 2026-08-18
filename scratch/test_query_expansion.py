import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

def expand_query_terms(query: str) -> str:
    """
    Heuristic / Domain Synonyms Expansion for Convolutional & Architectural Queries:
    Maps colloquial terms to scientific vocabulary (kernels -> convolution filters, 
    receptive fields, small/big -> 3x3, 7x7, parameter savings).
    """
    expanded = query
    q_lower = query.lower()
    
    synonyms = []
    if "kernel" in q_lower or "filter" in q_lower:
        synonyms.append("receptive field convolution 3x3 filter conv layers")
    if "vgg" in q_lower:
        synonyms.append("VGGNet 1409.1556 ConvNet configurations")
    if "small" in q_lower or "big" in q_lower or "large" in q_lower:
        synonyms.append("parameters non-linear rectification layers 7x7 5x5")
        
    if synonyms:
        expanded = f"{query} {' '.join(synonyms)}"
    return expanded

queries = [
    "Why does VGG use small filters instead of big ones?",
    "why not just use bigger kernels in VGG",
    "what's the benefit of small kernel sizes in CNNs",
]

print("=" * 115)
print("TESTING DOMAIN QUERY EXPANSION ON COLLOQUIAL VGG QUERIES")
print("=" * 115)

for q in queries:
    expanded_q = expand_query_terms(q)
    print(f"\nOriginal: '{q}'")
    print(f"Expanded: '{expanded_q}'")
    
    dense = orch.vector_store.search_dense(expanded_q, top_k=50)
    sparse = orch.bm25_retriever.search_sparse(expanded_q, top_k=50)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=50)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
        
    reranked = orch.reranker.rerank_chunks(q, pool, top_k=15)
    
    vgg_chunks = [c for c in reranked if c.get("paper_id") == "1409.1556"]
    print(f"  • VGG Chunks in reranked top-15: {len(vgg_chunks)}")
    for idx, c in enumerate(vgg_chunks[:3]):
        print(f"    [{idx+1}] Score: {c.get('rerank_score'):.4f} | Section: {c.get('section')} | Text: {c.get('text')[:100].replace(chr(10), ' ')}...")
    if reranked:
        print(f"  • Top-1 Overall: Paper {reranked[0].get('paper_id')} | Score: {reranked[0].get('rerank_score'):.4f}")
