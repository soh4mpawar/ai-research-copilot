import json

with open("evaluation/ragas_benchmark_results_40samples.json", "r", encoding="utf-8") as f:
    data = json.load(f)

agg = data["aggregate_ragas_scores"]
samples = data["per_sample_breakdown"]

print("### Aggregate Scores:")
print(f"Faithfulness: {agg['faithfulness']:.4f}")
print(f"Answer Relevancy: {agg['answer_relevancy']:.4f}")
print(f"Context Precision: {agg['context_precision']:.4f}")
print(f"Context Recall: {agg['context_recall']:.4f}")

print("\n### 40-Sample Breakdown Table:")
print("| # | Sample ID | Paper ID | Question / Title | Faithfulness | Relevancy | Context Precision | Context Recall | Judge Model |")
print("|---|---|---|---|---|---|---|---|---|")

for idx, s in enumerate(samples, 1):
    q_title = f"**{s['source_paper_title'][:30]}**"
    sc = s["scores"]
    j_model = s.get("judge_model", "OpenAI/gpt-4o-mini")
    print(f"| {idx} | `{s['sample_id']}` | `{s['source_paper_id']}` | {q_title} | `{sc.get('faithfulness', 0):.4f}` | `{sc.get('answer_relevancy', 0):.4f}` | `{sc.get('context_precision', 0):.4f}` | `{sc.get('context_recall', 0):.4f}` | `{j_model}` |")
