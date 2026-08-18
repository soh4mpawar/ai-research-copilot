import json
import csv

with open("data/metadata/citation_edges.csv", "r", encoding="utf-8") as f:
    edges = list(csv.DictReader(f))

print("Total verified edges:", len(edges))

# We know from the run log that:
# Total papers: 200
# S2 Success: 121 papers
# PDF Fallback used: 79 papers
# Total edges produced: 87
