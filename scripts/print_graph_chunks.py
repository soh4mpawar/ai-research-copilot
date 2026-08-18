import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingestion.vector_store import VectorStore

vs = VectorStore(persist_dir="data/chroma_db", collection_name="scientific_papers")

for cid in ["2608.12138_sec_001", "2608.13546_sec_001"]:
    res = vs.collection.get(ids=[cid])
    print("=" * 80)
    print(f"CHUNK ID: {cid}")
    print("=" * 80)
    if res and res.get("ids"):
        print(f"Paper Title: {res['metadatas'][0].get('paper_title')}")
        print(f"Section: {res['metadatas'][0].get('section')}")
        print(f"Authors: {res['metadatas'][0].get('authors')}")
        print("-" * 80)
        print("RAW TEXT CONTENT:")
        text = res['documents'][0]
        # Print first 800 characters of the raw chunk document
        print(text[:800])
        if len(text) > 800:
            print("... [truncated for display] ...")
    else:
        print("Chunk not found!")
