import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

# Fetch chunks 7, 8, 9 of 1409.1556
res = orch.vector_store.collection.get(where={'paper_id': '1409.1556'}, include=['documents', 'metadatas'])
docs = res['documents']
chunk_8 = docs[7]
chunk_9 = docs[8]

pairs = [
    ("Colloquial Query Raw", "Why does VGG use small filters instead of big ones?"),
    ("Colloquial Query with Concept Expansion", "Why does VGG use small filters instead of big ones? (VGGNet 3x3 vs 7x7 convolution filters receptive field parameter reduction)"),
    ("Colloquial Kernels Query Raw", "why not just use bigger kernels in VGG"),
    ("Colloquial Kernels with Concept Expansion", "why not just use bigger kernels in VGG (VGGNet 3x3 vs 7x7 convolution filters receptive fields)"),
    ("qa_008 Textbook Raw", "What is the theoretical benefit of stacking two 3x3 convolution layers instead of using a single 5x5 convolution layer in VGGNet?"),
    ("Literal Paper Query", "So what have we gained by using a stack of three 3x3 conv layers instead of a single 7x7 layer in VGG?")
]

print("=" * 115)
print("BGE CROSS-ENCODER SCORE COMPARISON: RAW VS CONCEPT-EXPANDED QUERY")
print("=" * 115)

for label, q in pairs:
    sc8 = orch.reranker.rerank_chunks(q, [{"text": chunk_8, "score": 0.0}], top_k=1)[0]["rerank_score"]
    sc9 = orch.reranker.rerank_chunks(q, [{"text": chunk_9, "score": 0.0}], top_k=1)[0]["rerank_score"]
    print(f"\n[{label}]")
    print(f"  Query: '{q}'")
    print(f"  • Chunk #8 Score (Non-linear rectification & receptive field): {sc8:.4f}")
    print(f"  • Chunk #9 Score (Parameter reduction 27C^2 vs 49C^2):          {sc9:.4f}")
