import glob
import json
import random
import sys

sys.stdout.reconfigure(encoding='utf-8')

files = sorted(glob.glob("data/parsed/*.json"))
print(f"Total parsed papers: {len(files)}")

all_chunks = []
for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        d = json.load(fp)
        for c in d.get("chunks", []):
            c["paper_title"] = d.get("title", c.get("paper_title", ""))
            all_chunks.append(c)

total_chunks = len(all_chunks)
token_counts = [c["token_count"] for c in all_chunks]
min_tokens = min(token_counts)
max_tokens = max(token_counts)
avg_tokens = sum(token_counts) / len(token_counts)

# Token distribution buckets
buckets = {"<100": 0, "100-200": 0, "200-350": 0, "350-512": 0, ">512 (oversized)": 0}
for t in token_counts:
    if t < 100:
        buckets["<100"] += 1
    elif t <= 200:
        buckets["100-200"] += 1
    elif t <= 350:
        buckets["200-350"] += 1
    elif t <= 512:
        buckets["350-512"] += 1
    else:
        buckets[">512 (oversized)"] += 1

oversized_chunks = [c for c in all_chunks if c.get("oversized_for_reranker")]

print("=================== STATS ===================")
print(f"Total Papers: {len(files)}")
print(f"Total Chunks: {total_chunks}")
print(f"Min Tokens per Chunk: {min_tokens}")
print(f"Max Tokens per Chunk: {max_tokens}")
print(f"Avg Tokens per Chunk: {avg_tokens:.1f}")
print(f"Token Buckets: {buckets}")
print(f"Flagged Oversized Chunks (token_count > 512): {len(oversized_chunks)}")

random.seed(42)
sample_indices = [150, 2500, 7200]
print("\n=================== 3 SAMPLE CHUNKS ===================")
for idx in sample_indices:
    c = all_chunks[idx]
    cid = c["chunk_id"]
    pt = c["paper_title"]
    sec = c["section"]
    tc = c["token_count"]
    flag = c.get("oversized_for_reranker", False)
    print(f"\n--- [Sample Chunk ID: {cid}] ---")
    print(f"Paper Title: {pt}")
    print(f"Section: {sec}")
    print(f"Token Count: {tc} | Oversized Flag: {flag}")
    print("Text Content:\n" + c["text"])
    print("-" * 80)
