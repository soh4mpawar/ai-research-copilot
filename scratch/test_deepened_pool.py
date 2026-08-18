import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

test_queries = [
    "Why does VGG use small filters instead of big ones?",
    "why not just use bigger kernels in VGG",
    "what's the benefit of small kernel sizes in CNNs",
    "why stack small convolutions instead of one large one",
    "What is the theoretical benefit of stacking two 3x3 convolution layers instead of using a single 5x5 convolution layer in VGGNet?"
]

print("=" * 115)
print("TESTING RETRIEVAL DEPTH (K=50) AND EXPANDED RERANKING (K=20)")
print("=" * 115)

for q in test_queries:
    print(f"\n>>> Query: '{q}'")
    dense = orch.vector_store.search_dense(q, top_k=50)
    sparse = orch.bm25_retriever.search_sparse(q, top_k=50)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=50)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
    reranked = orch.reranker.rerank_chunks(q, pool, top_k=15)
    
    # Check paper 1409.1556 chunks in reranked
    vgg_reranked = [c for c in reranked if c.get("paper_id") == "1409.1556"]
    print(f"  • Total VGG chunks in reranked top-15: {len(vgg_reranked)}")
    for idx, c in enumerate(vgg_reranked[:3]):
        print(f"    [{idx+1}] Score: {c.get('rerank_score'):.4f} | Section: {c.get('section')} | Text: {c.get('text')[:100].replace(chr(10), ' ')}...")
    
    top_overall = reranked[0] if reranked else None
    if top_overall:
        print(f"  • Top-1 Overall: Paper {top_overall.get('paper_id')} | Score: {top_overall.get('rerank_score'):.4f}")
