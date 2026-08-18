import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

recent_pairs = [d for d in data if d.get("source_paper_id", "").startswith("2608")]
print(f"Total QA pairs from 2026 papers: {len(recent_pairs)} (out of {len(data)} total)")

for i, d in enumerate(recent_pairs[:4]):
    print(f"\n=================== 2026 SAMPLE QA PAIR {i+1} ===================")
    print(f"Paper ID:             {d.get('source_paper_id')}")
    print(f"Paper Title:          {d.get('source_paper_title')}")
    print(f"Category:             {d.get('source_category')}")
    print(f"Question:\n  {d.get('question')}")
    print(f"Ground Truth Answer:\n  {d.get('ground_truth_answer')}")
    print(f"Ground Truth Passage:\n  {d.get('ground_truth_passage')}")
