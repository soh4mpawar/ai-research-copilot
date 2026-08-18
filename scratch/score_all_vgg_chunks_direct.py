import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()

# Fetch all chunks of 1409.1556
res = orch.vector_store.collection.get(where={'paper_id': '1409.1556'}, include=['documents', 'metadatas'])
docs = res['documents']
metas = res['metadatas']

q1 = "Why does VGG use small filters instead of big ones?"
q2 = "why not just use bigger kernels in VGG"
q3 = "why stack small convolutions instead of one large one in VGG"

print("=" * 115)
print(f"DIRECT CROSS-ENCODER SCORING OF ALL {len(docs)} VGG (1409.1556) CHUNKS")
print("=" * 115)

for q_label, q in [("Query 1", q1), ("Query 2", q2), ("Query 3", q3)]:
    print(f"\n>>> {q_label}: '{q}'")
    scored_chunks = []
    for idx, (d, m) in enumerate(zip(docs, metas)):
        # score chunk
        sc = orch.reranker.rerank_chunks(q, [{"text": d, "score": 0.0}], top_k=1)[0]["rerank_score"]
        scored_chunks.append((sc, idx+1, m.get("section"), d))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    for sc, chunk_num, sec, txt in scored_chunks[:5]:
        print(f"  • Score: {sc:.4f} | Chunk #{chunk_num} [{sec}] | Text: {txt[:120].replace(chr(10), ' ')}...")
