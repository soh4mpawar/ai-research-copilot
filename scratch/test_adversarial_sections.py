import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

# Test queries designed to target Acknowledgments, Appendix, Background, and Related Work
adversarial_section_queries = [
    ("OOD Grant Funding", "In the acknowledgments, who funded the grant for the high performance computing cluster?"),
    ("OOD Pandas Library", "In the appendix, what Python library version is used for pandas and numpy dataframes?"),
    ("OOD European History", "In the related work section, what historical background is provided on the Renaissance in Europe?"),
    ("OOD Author Affiliation", "Which university department did the second author graduate from in computer engineering?"),
    ("OOD LaTeX Style", "In the appendix, what LaTeX style file and font package was used for typesetting?")
]

print("=" * 115)
print("TESTING ADVERSARIAL SECTION-TARGETING QUERIES (NON-REFERENCES)")
print("=" * 115)

for label, q in adversarial_section_queries:
    dense = orch.vector_store.search_dense(q, top_k=20)
    sparse = orch.bm25_retriever.search_sparse(q, top_k=20)
    fused = orch.fusion_retriever.fuse_results(dense, sparse, top_k=25)
    graph = orch.graph_retriever.traverse_and_fetch_chunks(fused, max_graph_candidates=10)
    pool = list(fused)
    if graph:
        pool.extend(graph)
    reranked = orch.reranker.rerank_chunks(q, pool, top_k=10)
    
    top1 = reranked[0].get("rerank_score", 0.0) if reranked else 0.0
    top2 = reranked[1].get("rerank_score", 0.0) if len(reranked) > 1 else 0.0
    top_sec = reranked[0].get("section", "None") if reranked else "None"
    top_pid = reranked[0].get("paper_id", "None") if reranked else "None"
    top_text = reranked[0].get("text", "")[:120].replace("\n", " ") if reranked else ""
    
    print(f"[{label}]")
    print(f"  • Top-1 Score: {top1:.4f} | Top-2 Score: {top2:.4f} | Paper: {top_pid} ({top_sec})")
    print(f"  • Top Text: {top_text}...")
