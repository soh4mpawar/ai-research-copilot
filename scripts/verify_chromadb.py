import sys
import os
import random

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8")

from backend.ingestion.vector_store import VectorStore

vs = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")
total_count = vs.get_total_chunks_count()
print(f"Final ChromaDB Total Chunks Count: {total_count}")

# Retrieve records directly from ChromaDB
data = vs.collection.get(include=["metadatas", "documents"])
all_ids = data["ids"]
all_metas = data["metadatas"]
all_docs = data["documents"]

print(f"Direct ChromaDB Record Verification: {len(all_ids)} records retrieved.")

# True random sample of 8 records using random.sample()
sample_indices = random.sample(range(len(all_ids)), 8)

print("\n=================== 8 RANDOMLY SAMPLED CHROMADB RECORDS ===================")
for i, idx in enumerate(sample_indices, 1):
    meta = all_metas[idx]
    chunk_id = all_ids[idx]
    doc_snip = all_docs[idx][:160].replace("\n", " ")
    print(f"\nSample {i}:")
    print(f"  Chunk ID:     {chunk_id}")
    print(f"  Paper ID:     {meta.get('paper_id')}")
    print(f"  Paper Title:  {meta.get('paper_title')}")
    print(f"  Section:      {meta.get('section')}")
    print(f"  Token Count:  {meta.get('token_count')}")
    print(f"  Text Preview: \"{doc_snip}...\"")
