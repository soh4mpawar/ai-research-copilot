import json
import os
import sys

# Load both datasets
with open("scratch/stress_test_scores.json", "r", encoding="utf-8") as f:
    stress_data = json.load(f)

with open("scratch/ragas_top1_top2_scores.json", "r", encoding="utf-8") as f:
    ragas_data = json.load(f)

# Combine into in-domain vs out-of-domain
in_domain_items = []
out_of_domain_items = []

for item in stress_data:
    if item["is_in_domain"]:
        in_domain_items.append(item)
    else:
        out_of_domain_items.append(item)

for item in ragas_data:
    in_domain_items.append({
        "label": item["id"],
        "query": item["question"],
        "top1": item["top1"],
        "top2": item["top2"],
        "top3": item["top3"],
        "valid_035": item["valid_035"],
        "is_in_domain": True
    })

print(f"Total In-Domain Queries: {len(in_domain_items)} (40 RAGAS + 5 Narrow + 3 Broad)")
print(f"Total Out-Of-Domain Queries: {len(out_of_domain_items)} (10 Original + 15 Expanded)")

# Candidate Gate Rules:
def gate_current(top1, top2, scores):
    """Rule 0: Current Baseline (Top-1 >= 0.35)"""
    return top1 >= 0.35

def gate_candidate_1(top1, top2, scores):
    """
    Rule 1: Primary Threshold + Secondary Coherence Reinforcement.
    Pass if:
      (Top-1 >= 0.35 AND Top-2 >= 0.25)
    """
    return (top1 >= 0.35 and top2 >= 0.25)

def gate_candidate_2(top1, top2, scores):
    """
    Rule 2: Primary Threshold with Score Cliff / Isolation Penalty.
    Pass if:
      Top-1 >= 0.35 AND NOT (Top-2 < 0.22 AND (Top-1 - Top-2) > 0.30)
    """
    if top1 < 0.35:
        return False
    is_isolated_spike = (top2 < 0.22 and (top1 - top2) > 0.30)
    return not is_isolated_spike

def gate_candidate_3(top1, top2, scores):
    """
    Rule 3: Dual Harmonic Energy Gate.
    Pass if:
      Top-1 >= 0.35 AND (Top-1 + Top-2 >= 0.65) AND Top-2 >= 0.20
    """
    return (top1 >= 0.35 and (top1 + top2 >= 0.65) and top2 >= 0.20)

rules = [
    ("Current Baseline (Top-1 >= 0.35)", gate_current),
    ("Candidate 1: Dual Reinforcement (Top-1 >= 0.35 & Top-2 >= 0.25)", gate_candidate_1),
    ("Candidate 2: Isolation Cliff Rejection (Top-1 >= 0.35 & !(Top-2 < 0.22 & Gap > 0.30))", gate_candidate_2),
    ("Candidate 3: Dual Energy Gate (Top-1 >= 0.35 & Sum >= 0.65 & Top-2 >= 0.20)", gate_candidate_3),
]

print("\n" + "=" * 115)
print("SIMULATION RESULTS ACROSS 73 TOTAL QUERIES")
print("=" * 115)

for r_name, r_fn in rules:
    id_passed = 0
    id_failed = 0
    id_failed_list = []
    
    ood_passed = 0
    ood_blocked = 0
    ood_passed_list = []
    
    for item in in_domain_items:
        # Note: qa_008 has top1=0.280 (known lexical boundary)
        passed = r_fn(item["top1"], item["top2"], [item["top1"], item["top2"], item.get("top3", 0.0)])
        if passed:
            id_passed += 1
        else:
            id_failed += 1
            id_failed_list.append((item["label"], item["top1"], item["top2"]))
            
    for item in out_of_domain_items:
        passed = r_fn(item["top1"], item["top2"], [item["top1"], item["top2"], item.get("top3", 0.0)])
        if passed:
            ood_passed += 1
            ood_passed_list.append((item["label"], item["top1"], item["top2"]))
        else:
            ood_blocked += 1

    print(f"\n--- {r_name} ---")
    print(f"  • In-Domain Pass Rate:     {id_passed}/{len(in_domain_items)} ({id_passed/len(in_domain_items)*100:.1f}%)")
    if id_failed_list:
        print(f"    In-Domain Blocked: {id_failed_list}")
    print(f"  • Out-Of-Domain Block Rate: {ood_blocked}/{len(out_of_domain_items)} ({ood_blocked/len(out_of_domain_items)*100:.1f}%) [False Positives: {ood_passed}]")
    if ood_passed_list:
        print(f"    OOD False Positives: {ood_passed_list}")
