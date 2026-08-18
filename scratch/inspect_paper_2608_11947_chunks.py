import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

import json
from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

# Fetch all chunks of 2608.11947 from ChromaDB
res = orch.vector_store.collection.get(where={'paper_id': '2608.11947'}, include=['documents', 'metadatas'])
docs = res['documents']
metas = res['metadatas']

q = "In paper 2608.11947, why do model accuracy and prompt order sensitivity diverge when evaluating language models under permutation?"

print("=" * 85)
print(f"PAPER 2608.11947 HAS {len(docs)} TOTAL CHUNKS IN CHROMADB")
print("=" * 85)

for i, (d, m) in enumerate(zip(docs, metas)):
    score = orch.reranker.rerank_chunks(q, [{"text": d, "score": 0.0, "paper_id": "2608.11947", "section": m.get("section", "")}], top_k=1)[0]["rerank_score"]
    print(f"Chunk #{i+1} (Section: '{m.get('section')}') | Score: {score:.4f} | Text: {d[:120].replace(chr(10), ' ')}...")
