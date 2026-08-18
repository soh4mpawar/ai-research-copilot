import csv

with open("data/metadata/citation_edges.csv", "r", encoding="utf-8") as f:
    edges = list(csv.DictReader(f))

# Let's find 2 strong candidate edges to recommend for user verification
candidates = []
for e in edges:
    src = e["source_paper_id"]
    tgt = e["target_paper_id"]
    if src not in ["2608.13045", "2608.13482", "2608.11879", "2608.12737", "1512.03385"]:
        candidates.append(e)

print(f"Total candidate edges: {len(candidates)}")
for i, c in enumerate(candidates[:10], 1):
    print(f"{i}. [{c['source_paper_id']}] '{c['source_paper_title']}' -> [{c['target_paper_id']}] '{c['target_paper_title']}'")
