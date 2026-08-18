import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

import json
from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

# 1. Inspect all chunks of paper 1409.1556
res = orch.vector_store.collection.get(where={'paper_id': '1409.1556'}, include=['documents', 'metadatas'])
docs = res['documents']
metas = res['metadatas']

print("=" * 115)
print(f"PAPER 1409.1556 (VGGNet) HAS {len(docs)} TOTAL CHUNKS IN CHROMADB")
print("=" * 115)

target_chunks = []
for i, (d, m) in enumerate(zip(docs, metas)):
    sec = m.get("section", "")
    # Check if this chunk discusses convolution size / filters / 3x3 / receptive field / configurations
    if any(k in d.lower() for k in ["3 × 3", "3x3", "receptive field", "7 × 7", "7x7", "convolution", "filter", "stride"]):
        target_chunks.append((i, sec, d))
        print(f"\n--- Chunk #{i+1} [Section: '{sec}'] ---")
        print(d[:350].replace('\n', ' ') + "...\n")

# 2. Test natural query variants against hybrid retrieval & reranker
test_queries = [
    ("Query 1 (User Natural)", "Why does VGG use small filters instead of big ones?"),
    ("Query 2 (Natural Paraphrase A)", "why not just use bigger kernels in VGG"),
    ("Query 3 (Natural Paraphrase B)", "what's the benefit of small kernel sizes in CNNs"),
    ("Query 4 (Natural Paraphrase C)", "why stack small convolutions instead of one large one"),
    ("Query 5 (qa_008 Textbook)", "What is the theoretical benefit of stacking two 3x3 convolution layers instead of using a single 5x5 convolution layer in VGGNet?"),
    ("Query 6 (Literal Paper Terms)", "How does the architecture of VGGNet use very small 3x3 convolution filters and stacks of conv layers?")
]

print("=" * 115)
print("AUDITING REACHABILITY AND RANKINGS ACROSS PHRASINGS")
print("=" * 115)

for label, q in test_queries:
    print(f"\n>>> {label}: '{q}'")
    dense = orch.vector_store.search_dense(q, top_k=50)
    sparse = orch.bm25_retriever.search_sparse(q, top_k=50)
    
    # Check where paper 1409.1556 appears in Dense vs Sparse
    dense_vgg = [(idx+1, c.get("paper_id"), c.get("section"), c.get("score")) for idx, c in enumerate(dense) if c.get("paper_id") == "1409.1556"]
    sparse_vgg = [(idx+1, c.get("paper_id"), c.get("section"), c.get("score")) for idx, c in enumerate(sparse) if c.get("paper_id") == "1409.1556"]
    
    fused_30 = orch.fusion_retriever.fuse_results(dense[:25], sparse[:25], top_k=30)
    fused_50 = orch.fusion_retriever.fuse_results(dense[:50], sparse[:50], top_k=50)
    
    fused_30_vgg = [(idx+1, c.get("paper_id"), c.get("section")) for idx, c in enumerate(fused_30) if c.get("paper_id") == "1409.1556"]
    fused_50_vgg = [(idx+1, c.get("paper_id"), c.get("section")) for idx, c in enumerate(fused_50) if c.get("paper_id") == "1409.1556"]
    
    # Rerank with top-12
    reranked = orch.reranker.rerank_chunks(q, fused_30, top_k=12)
    top_score = reranked[0].get("rerank_score", 0.0) if reranked else 0.0
    top_pid = reranked[0].get("paper_id", "") if reranked else ""
    
    print(f"  • Dense Rank(s) in Top 50:  {dense_vgg[:3]}")
    print(f"  • BM25 Rank(s) in Top 50:   {sparse_vgg[:3]}")
    print(f"  • Fused Rank(s) @ K=30:     {fused_30_vgg}")
    print(f"  • Fused Rank(s) @ K=50:     {fused_50_vgg}")
    print(f"  • Top Rerank Score:         {top_score:.4f} (Paper: {top_pid})")
