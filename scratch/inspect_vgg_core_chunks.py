import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, ".")

from backend.pipeline import ResearchPipelineOrchestrator

orch = ResearchPipelineOrchestrator()
res = orch.vector_store.collection.get(where={'paper_id': '1409.1556'}, include=['documents', 'metadatas'])
docs = res['documents']
metas = res['metadatas']

print("=" * 115)
print("INSPECTING CHUNKS 7, 8, 9, 10, 11 OF 1409.1556")
print("=" * 115)

for idx in [6, 7, 8, 9, 10]:
    if idx < len(docs):
        print(f"\n--- CHUNK #{idx+1} [Section: '{metas[idx].get('section')}'] ---")
        print(docs[idx])
