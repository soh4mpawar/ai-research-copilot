import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total QA pairs: {len(data)}")
for i, d in enumerate(data[:10]):
    print(f"\n{i+1}. [{d['source_paper_id']}] {d['source_paper_title']}")
    print(f"   Q: {d['question']}")
    print(f"   A: {d['ground_truth_answer']}")
    print(f"   Passage: {d.get('ground_truth_passage', '')[:120]}...")
