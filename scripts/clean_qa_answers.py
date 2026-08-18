import json
import re

with open("data/metadata/draft_qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    gt = item["ground_truth_answer"]
    
    # Remove author/affiliation noise if present
    gt = re.sub(r'[\w\.-]+@[\w\.-]+', '', gt)
    gt = re.sub(r'^\s*(?:[A-Z0-9\s\.\-]{4,}\s+)?(?:Preprint|\d+\s+)?(?:[A-Z\s]{4,}:)?\s*', '', gt)
    gt = re.sub(r'^(?:[A-Z\s]{5,}\s+)+', '', gt)
    gt = re.sub(r'^(?:[A-Z][a-z]+\s+){1,4}(?:University|Institute|School|Laboratories|Thailand|Beijing|London|Isfahan|Shanghai)\s*', '', gt, flags=re.IGNORECASE)
    
    # Ensure starts with clean capital letter
    gt = gt.strip()
    match = re.search(r'[A-Z]', gt)
    if match:
        gt = gt[match.start():]
    
    item["ground_truth_answer"] = gt
    item["ground_truth_passage"] = gt[:350]

with open("data/metadata/draft_qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Cleaned all ground truth statements in draft_qa_dataset.json.")
