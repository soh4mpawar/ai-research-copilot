import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

import torch
from backend.pipeline import ResearchPipelineOrchestrator
from backend.retrieval.threshold_gate import RelevanceThresholdGate

orch = ResearchPipelineOrchestrator()
gate = RelevanceThresholdGate(threshold=0.35)

test_queries = [
    # The 2 failed queries reported
    ("Weather", "What's the weather going to be like tomorrow?"),
    ("Bicycle Tire", "How do I fix a flat bicycle tire?"),
    ("Champions League", "Who won the Champions League last season?"),
    
    # The other 7 adversarial queries
    ("Lasagna", "How to bake a classic meat lasagna with ricotta cheese and tomato sauce?"),
    ("Taxes", "How do I file my personal income taxes online?"),
    ("Capital Australia", "What is the capital city of Australia?"),
    ("Stock Price", "What is the current stock price of Apple Inc (AAPL)?"),
    ("Poem", "Write a rhyming poem about a cat sitting on a windowsill."),
    ("Workout Plan", "What is a 4-day gym split workout plan for building muscle?"),
    ("Sci-Fi Movies", "What are the top 10 best sci-fi movies of all time?"),

    # In-domain controls
    ("Transformer (In-Domain)", "How does multi-head self-attention work in the Transformer architecture?"),
    ("ResNet (In-Domain)", "Why do residual shortcut connections in ResNet prevent vanishing gradients?"),
    ("VGG (In-Domain)", "What is the theoretical benefit of using a stack of three 3x3 convolution layers instead of a single 7x7 layer in VGGNet?")
]

print("=" * 115)
print("AUDITING OUT-OF-DOMAIN & ADVERSARIAL QUERY SCORES IN LIVE PIPELINE")
print("=" * 115)

for label, q in test_queries:
    print(f"\n[{label}] Query: '{q}'")
    
    # 1. Search dense & sparse
    dense_candidates = orch.vector_store.search_dense(q, top_k=20)
    sparse_candidates = orch.bm25_retriever.search_sparse(q, top_k=20)
    
    # 2. Fuse
    fused_candidates = orch.fusion_retriever.fuse_results(dense_candidates, sparse_candidates, top_k=25)
    
    # 3. Graph traversal
    graph_candidates = orch.graph_retriever.traverse_and_fetch_chunks(fused_candidates, max_graph_candidates=10)
    candidate_pool = list(fused_candidates)
    if graph_candidates:
        candidate_pool.extend(graph_candidates)
    
    # 4. Rerank
    reranked = orch.reranker.rerank_chunks(q, candidate_pool, top_k=10)
    
    # 5. Evaluate Gate
    passed, valid_chunks, status_msg = gate.evaluate_chunks(reranked)
    
    top_chunk = reranked[0] if reranked else {}
    top_score = top_chunk.get("rerank_score", 0.0)
    top_paper = top_chunk.get("paper_id", "None")
    top_sec = top_chunk.get("section", "None")
    top_text = top_chunk.get("text", "")[:200].replace("\n", " ")
    
    print(f"  • Candidate Pool Size: {len(candidate_pool)}")
    print(f"  • Top Rerank Score: {top_score:.4f} | Paper: {top_paper} ({top_sec})")
    print(f"  • Gate Result: {'PASSED (Generation Triggered)' if passed else 'BLOCKED (FR-11 Short-Circuit)'} | Valid Chunks (>=0.25): {len(valid_chunks)}")
    print(f"  • Top Chunk Text: {top_text}...")
    
    # If passed, print all chunks that passed the gate
    if passed:
        print("  • ALL PASSING CHUNKS:")
        for idx, vc in enumerate(valid_chunks):
            print(f"    [{idx+1}] Score: {vc.get('rerank_score'):.4f} | Paper: {vc.get('paper_id')} | Sec: {vc.get('section')} | Text: {vc.get('text')[:120]}...")
