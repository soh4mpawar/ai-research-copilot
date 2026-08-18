import json

with open("scratch/stress_test_scores.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 125)
print(f"{'Group':<35} | {'Label':<25} | {'Top-1':<7} | {'Top-2':<7} | {'Top-3':<7} | {'Gap':<7} | {'>=0.35':<6} | {'Top Paper ID'}")
print("=" * 125)

for d in data:
    print(f"{d['group']:<35} | {d['label']:<25} | {d['top1']:<7.4f} | {d['top2']:<7.4f} | {d['top3']:<7.4f} | {d['gap']:<7.4f} | {d['valid_035']:<6} | {d['top_paper']}")
