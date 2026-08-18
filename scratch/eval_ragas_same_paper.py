import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    benchmark_samples = json.load(f)

def evaluate_same_paper_clustering(reranked_chunks):
    if not reranked_chunks:
        return False, "No chunks retrieved"
    top1 = reranked_chunks[0].get("rerank_score", 0.0)
    if top1 < 0.35:
        return False, f"Top score ({top1:.4f}) < 0.35"
    
    paper_scores = {}
    for c in reranked_chunks:
        pid = c.get("paper_id", "unknown")
        sc = c.get("rerank_score", 0.0)
        if pid not in paper_scores:
            paper_scores[pid] = []
        paper_scores[pid].append(sc)

    # Condition A: Same-paper dual support (Top1_P >= 0.35 AND Top2_P >= 0.15)
    for pid, sc_list in paper_scores.items():
        if len(sc_list) >= 2:
            sc_sorted = sorted(sc_list, reverse=True)
            if sc_sorted[0] >= 0.35 and sc_sorted[1] >= 0.15:
                return True, f"Same-Paper Support ({pid}: [{sc_sorted[0]:.3f}, {sc_sorted[1]:.3f}])"

    # Condition B: Multi-paper consensus (>= 2 distinct papers with score >= 0.35)
    qualifying_papers = [pid for pid, sc_list in paper_scores.items() if max(sc_list) >= 0.35]
    if len(qualifying_papers) >= 2:
        return True, f"Multi-Paper Consensus ({qualifying_papers[:2]})"

    return False, f"Isolated single chunk without same-paper or multi-paper support (Top-1: {top1:.3f})"

print("=" * 115)
print("TESTING SAME-PAPER DUAL SUPPORT ON ALL 40 RAGAS BENCHMARK SAMPLES")
print("=" * 115)

blocked_samples = []
passed_samples = []

for s in benchmark_samples:
    q = s["question"]
    sid = s["id"]
    pid = s["source_paper_id"]
    
    dense = orch.vector_store.search_dense(q, top_k=20)
    sparse = orch.bm25_retriever.search_sparse(q, top_k=20)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=25)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
    reranked = orch.reranker.rerank_chunks(q, pool, top_k=10)
    
    passed, reason = evaluate_same_paper_clustering(reranked)
    if passed:
        passed_samples.append(sid)
        print(f"[{sid}] PASS ({pid}) -> {reason}")
    else:
        blocked_samples.append((sid, pid, reason))
        print(f"[{sid}] BLOCKED ({pid}) -> {reason}")

print("\n" + "=" * 115)
print(f"SUMMARY: {len(passed_samples)}/40 PASS ({len(passed_samples)/40*100:.1f}%)")
print(f"Blocked samples: {blocked_samples}")
print("=" * 115)
