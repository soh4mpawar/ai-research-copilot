import json
import os
import sys

sys.path.insert(0, ".")

from backend.reranking.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")

dataset_path = "data/metadata/draft_qa_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    qa_dataset = json.load(f)

pairs = [[s["question"], s.get("ground_truth_passage", s.get("ground_truth_answer", ""))] for s in qa_dataset]
scores = reranker.model.predict(pairs)

results = []
for s, score in zip(qa_dataset, scores):
    results.append({
        "id": s["id"],
        "paper_id": s["source_paper_id"],
        "score": float(score),
        "question": s["question"],
        "ground_truth": s.get("ground_truth_passage", s.get("ground_truth_answer", ""))
    })

# Sort by score ascending
results.sort(key=lambda x: x["score"])

print("=" * 105)
print(f"{'ID':<8} | {'Paper ID':<12} | {'Alignment Score':<16} | {'Status (<0.35 is Flagged)':<25} | {'Question Preview'}")
print("=" * 105)

for r in results:
    status = "FLAGGED (Low Direct Alignment)" if r["score"] < 0.35 else "STRONG ALIGNMENT (>=0.35)"
    q_preview = r["question"][:45] + "..." if len(r["question"]) > 45 else r["question"]
    print(f"{r['id']:<8} | {r['paper_id']:<12} | {r['score']:<16.4f} | {status:<25} | {q_preview}")

print("\n" + "=" * 105)
print("DEEP DIVE ON qa_008 (VGGNet):")
qa008 = next(r for r in results if r["id"] == "qa_008")
print(f"ID: {qa008['id']} | Paper: {qa008['paper_id']} | Alignment Score: {qa008['score']:.4f}")
print(f"Question: {qa008['question']}")
print(f"Ground Truth Passage: {qa008['ground_truth']}")
print("=" * 105)
