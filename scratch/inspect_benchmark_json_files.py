import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

for fname in ["ragas_benchmark_run1.json", "ragas_benchmark_run2_final.json", "ragas_benchmark_run3_final.json"]:
    p = f"evaluation/{fname}"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n==================================================")
    print(f"FILE: {p}")
    print(f"==================================================")
    if "aggregate_metrics" in data:
        print("aggregate_metrics:", json.dumps(data["aggregate_metrics"], indent=2))
    elif "metrics" in data:
        print("metrics:", json.dumps(data["metrics"], indent=2))
    else:
        # Compute mean from samples if needed
        samples = data.get("samples", [])
        print(f"Total samples: {len(samples)}")
        for k in ["faithfulness", "context_precision", "context_recall", "answer_relevance"]:
            vals = [s.get("metrics", {}).get(k, 0) for s in samples if k in s.get("metrics", {})]
            if vals:
                print(f"  {k}: {sum(vals)/len(vals):.4f}")
