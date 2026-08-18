import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total QA pairs: {len(data)}\n")

# Display 5 varied samples from different 2026 papers across different categories
sample_indices = [11, 14, 18, 23, 29]

for i, idx in enumerate(sample_indices, 1):
    d = data[idx]
    print(f"=================== SAMPLE {i} (Index {idx+1} | Type: {d.get('type')}) ===================")
    print(f"Paper ID:             {d.get('source_paper_id')}")
    print(f"Paper Title:          {d.get('source_paper_title')}")
    print(f"Category:             {d.get('source_category')}")
    print(f"Question Type:        {d.get('type')}")
    print(f"Question:\n  {d.get('question')}")
    print(f"Ground Truth Answer:\n  {d.get('ground_truth_answer')[:250]}...")
    print(f"Passage:\n  {d.get('ground_truth_passage')[:180]}...\n")
