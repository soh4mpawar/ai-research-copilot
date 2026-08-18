import csv

with open("data/metadata/citation_edges.csv", "r", encoding="utf-8") as f:
    reader = list(csv.DictReader(f))

print(f"Total edges in CSV: {len(reader)}")

# Find specific samples
print("\nFirst 5 edges in CSV:")
for i, r in enumerate(reader[:5], 1):
    print(f"{i}. [{r['source_paper_id']}] '{r['source_paper_title'][:40]}...' -> [{r['target_paper_id']}] '{r['target_paper_title'][:40]}...' (Cross: {r['is_cross_category']})")

print("\nEdges from 2026 preprints (which used PDF fallback / S2 references):")
for r in reader:
    if "2608" in r["source_paper_id"]:
        print(f"  • [{r['source_paper_id']}] -> [{r['target_paper_id']}] ({r['target_paper_title'][:35]}...)")
