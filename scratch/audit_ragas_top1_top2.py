import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    benchmark_samples = json.load(f)

print("=" * 115)
print("EVALUATING TOP-1 AND TOP-2 DISTRIBUTIONS ACROSS ALL 40 BENCHMARK SAMPLES")
print("=" * 115)

ragas_scores = []
for idx, s in enumerate(benchmark_samples, 1):
    q = s["question"]
    sid = s["id"]
    pid = s["source_paper_id"]
    
    dense_candidates = orch.vector_store.search_dense(q, top_k=20)
    sparse_candidates = orch.bm25_retriever.search_sparse(q, top_k=20)
    fused_candidates = orch.fusion_retriever.fuse_results(dense_candidates, sparse_candidates, top_k=25)
    graph_candidates = orch.graph_retriever.traverse_and_fetch_chunks(fused_candidates, max_graph_candidates=10)
    candidate_pool = list(fused_candidates)
    if graph_candidates:
        candidate_pool.extend(graph_candidates)
    reranked = orch.reranker.rerank_chunks(q, candidate_pool, top_k=10)
    scores = [c.get("rerank_score", 0.0) for c in reranked]
    
    top1 = scores[0] if len(scores) > 0 else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    top3 = scores[2] if len(scores) > 2 else 0.0
    valid_035 = sum(1 for sc in scores if sc >= 0.35)
    
    ragas_scores.append({
        "id": sid,
        "paper_id": pid,
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "valid_035": valid_035,
        "question": q
    })
    
    print(f"[{sid}] Top-1:{top1:.4f} | Top-2:{top2:.4f} | Top-3:{top3:.4f} | Chunks>=0.35:{valid_035} | Paper:{pid}")

with open("scratch/ragas_top1_top2_scores.json", "w", encoding="utf-8") as f:
    json.dump(ragas_scores, f, indent=2)

print("\n" + "=" * 115)
print("BENCHMARK SAMPLE AUDIT SAVED TO scratch/ragas_top1_top2_scores.json")
print("=" * 115)
