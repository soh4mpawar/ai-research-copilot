import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

# Let's test the Same-Paper Clustered Support rule on:
# 1. All 40 RAGAS benchmark samples
# 2. All 5 Narrow single-fact queries
# 3. All 3 Broad in-domain queries
# 4. All 23 Adversarial OOD queries

with open("scratch/stress_test_scores.json", "r", encoding="utf-8") as f:
    stress_data = json.load(f)

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    draft_qa = json.load(f)

# Helper function to evaluate same-paper clustering
def evaluate_same_paper_clustering(reranked_chunks):
    """
    Evaluates candidate chunks for source-grounded reinforcement.
    Rule:
      A. Same-Paper Dual Support:
         At least ONE paper has >= 2 retrieved chunks where:
         - Top chunk from paper >= 0.35
         - 2nd chunk from same paper >= 0.20
      OR
      B. Multi-Paper Cross-Source Consensus:
         At least TWO distinct papers each have a chunk with score >= 0.35.
    """
    if not reranked_chunks:
        return False, "No chunks retrieved"

    top1 = reranked_chunks[0].get("rerank_score", 0.0)
    if top1 < 0.35:
        return False, f"Top score ({top1:.4f}) below minimum threshold (0.35)"

    # Group scores by paper_id
    paper_scores = {}
    for c in reranked_chunks:
        pid = c.get("paper_id", "unknown")
        sc = c.get("rerank_score", 0.0)
        if pid not in paper_scores:
            paper_scores[pid] = []
        paper_scores[pid].append(sc)

    # Check Condition A: Same-paper dual support
    for pid, sc_list in paper_scores.items():
        if len(sc_list) >= 2:
            sc_list_sorted = sorted(sc_list, reverse=True)
            if sc_list_sorted[0] >= 0.35 and sc_list_sorted[1] >= 0.20:
                return True, f"Passed via Same-Paper Support ({pid}: [{sc_list_sorted[0]:.3f}, {sc_list_sorted[1]:.3f}])"

    # Check Condition B: Multi-paper cross-source consensus
    qualifying_papers = [pid for pid, sc_list in paper_scores.items() if max(sc_list) >= 0.35]
    if len(qualifying_papers) >= 2:
        return True, f"Passed via Multi-Paper Consensus (Papers: {qualifying_papers[:2]})"

    return False, f"Rejected: Isolated single chunk without same-paper reinforcement or multi-paper consensus (Top-1: {top1:.3f})"

print("=" * 115)
print("AUDITING SAME-PAPER / MULTI-PAPER CLUSTERED COHERENCE GATE ACROSS ALL QUERIES")
print("=" * 115)

def run_query_and_eval(q):
    dense = orch.vector_store.search_dense(q, top_k=20)
    sparse = orch.bm25_retriever.search_sparse(q, top_k=20)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=25)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
    reranked = orch.reranker.rerank_chunks(q, pool, top_k=10)
    passed, reason = evaluate_same_paper_clustering(reranked)
    return passed, reason, reranked

# 1. Test Narrow In-Domain
print("\n--- 1. NARROW IN-DOMAIN QUERIES (5) ---")
for item in stress_data[:5]:
    passed, reason, _ = run_query_and_eval(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

# 2. Test Broad In-Domain
print("\n--- 2. BROAD IN-DOMAIN CONTROLS (3) ---")
for item in stress_data[5:8]:
    passed, reason, _ = run_query_and_eval(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

# 3. Test Adversarial OOD (Original 10)
print("\n--- 3. ORIGINAL ADVERSARIAL OOD (10) ---")
for item in stress_data[8:18]:
    passed, reason, _ = run_query_and_eval(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

# 4. Test Adversarial OOD (Expanded 15)
print("\n--- 4. EXPANDED ADVERSARIAL OOD (15) ---")
for item in stress_data[18:]:
    passed, reason, _ = run_query_and_eval(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")
