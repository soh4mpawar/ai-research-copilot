import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

with open("scratch/stress_test_scores.json", "r", encoding="utf-8") as f:
    stress_data = json.load(f)

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    draft_qa = json.load(f)

def evaluate_principled_source_gate(reranked_chunks):
    """
    Principled Source-Grounded Coherence Gate:
    Passes IF:
      1. Same-Paper Dual Reinforcement:
         At least one paper P has:
           - Top chunk >= 0.35
           - AND 2nd chunk from same paper >= 0.15
      OR
      2. Same-Paper Deep Topic Focus:
         At least one paper P has:
           - Top chunk >= 0.85
           - AND >= 3 chunks from paper P are present in the candidate pool
      OR
      3. Multi-Paper Cross-Source Consensus:
         At least two distinct papers P1, P2 each have a chunk >= 0.35.
    """
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

    # 1. Same-Paper Dual Reinforcement
    for pid, sc_list in paper_scores.items():
        if len(sc_list) >= 2:
            sc_sorted = sorted(sc_list, reverse=True)
            if sc_sorted[0] >= 0.35 and sc_sorted[1] >= 0.15:
                return True, f"Passed: Same-Paper Dual Reinforcement ({pid}: [{sc_sorted[0]:.3f}, {sc_sorted[1]:.3f}])"

    # 2. Same-Paper Deep Topic Focus
    for pid, sc_list in paper_scores.items():
        if len(sc_list) >= 3 and max(sc_list) >= 0.85:
            return True, f"Passed: Same-Paper Deep Topic Focus ({pid}: Top={max(sc_list):.3f}, {len(sc_list)} candidate chunks)"

    # 3. Multi-Paper Cross-Source Consensus
    qualifying_papers = [pid for pid, sc_list in paper_scores.items() if max(sc_list) >= 0.35]
    if len(qualifying_papers) >= 2:
        return True, f"Passed: Multi-Paper Cross-Source Consensus ({qualifying_papers[:2]})"

    return False, f"Blocked: Isolated single-chunk hook without intra-paper or cross-paper support (Top-1: {top1:.3f})"

print("=" * 115)
print("TESTING PRINCIPLED SOURCE-GROUNDED GATE ACROSS ALL 73 QUERIES")
print("=" * 115)

def run_query(q):
    dense = orch.vector_store.search_dense(q, top_k=25)
    sparse = orch.bm25_retriever.search_sparse(q, top_k=25)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=30)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
    reranked = orch.reranker.rerank_chunks(q, pool, top_k=12)
    return evaluate_principled_source_gate(reranked)

# 1. 40 Benchmark samples
print("\n--- 1. 40 RAGAS BENCHMARK SAMPLES ---")
ragas_pass = 0
for s in draft_qa:
    passed, reason = run_query(s["question"])
    if passed:
        ragas_pass += 1
    else:
        print(f"  • [{s['id']}] -> BLOCKED: {reason}")
print(f"RAGAS Benchmark Pass Rate: {ragas_pass}/40 ({ragas_pass/40*100:.1f}%)")

# 2. Narrow In-Domain (5)
print("\n--- 2. NARROW SINGLE-FACT IN-DOMAIN QUERIES (5) ---")
for item in stress_data[:5]:
    passed, reason = run_query(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

# 3. Broad In-Domain (3)
print("\n--- 3. BROAD IN-DOMAIN CONTROLS (3) ---")
for item in stress_data[5:8]:
    passed, reason = run_query(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

# 4. Adversarial OOD (Original 10)
print("\n--- 4. ORIGINAL ADVERSARIAL OOD QUERIES (10) ---")
for item in stress_data[8:18]:
    passed, reason = run_query(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")

# 5. Adversarial OOD (Expanded 15)
print("\n--- 5. EXPANDED ADVERSARIAL OOD QUERIES (15) ---")
for item in stress_data[18:]:
    passed, reason = run_query(item["query"])
    print(f"[{item['label']}] -> {'PASS' if passed else 'BLOCKED'}: {reason}")
