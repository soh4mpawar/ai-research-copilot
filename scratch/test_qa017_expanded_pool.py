import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()
q = "In paper 2608.11947, why do model accuracy and prompt order sensitivity diverge when evaluating language models under permutation?"

# Test top_k=35 in dense/sparse
dense = orch.vector_store.search_dense(q, top_k=35)
sparse = orch.bm25_retriever.search_sparse(q, top_k=35)
fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=35)
graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
pool = list(fused)
if graph:
    pool.extend(graph)

reranked = orch.reranker.rerank_chunks(q, pool, top_k=15)

paper_scores = {}
for c in reranked:
    pid = c.get("paper_id", "")
    sc = c.get("rerank_score", 0.0)
    if pid not in paper_scores:
        paper_scores[pid] = []
    paper_scores[pid].append(sc)

print("=" * 85)
print("PAPER SCORES FOR qa_017 WITH TOP_K=35 POOL:")
print("=" * 85)
for pid, scs in paper_scores.items():
    print(f"Paper: {pid} | Scores: {scs}")
