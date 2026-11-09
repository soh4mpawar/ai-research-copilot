"""
Ground-Truth Factual Accuracy Evaluation Runner (Phase 5/6, PRD §8.2).
Evaluates candidate answers across all 3 pipeline configurations against curated ground-truth answers.
Uses automated LLM-as-judge (OpenAI gpt-4o-mini per PRIMARY_JUDGE_MODEL) as a standardized proxy
for PRD §8.2 manual factual accuracy verification.
Saves complete per-sample breakdown and aggregate scores to data/metadata/factual_accuracy_eval_results.json.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from openai import OpenAI

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import PRIMARY_JUDGE_MODEL
from backend.pipeline import ResearchPipelineOrchestrator


def grade_factuality_with_judge(
    client: OpenAI,
    judge_model: str,
    question: str,
    ground_truth: str,
    candidate_answer: str
) -> tuple[float, str]:
    """
    Grade factual accuracy on a 0.0 to 1.0 continuous scale against ground truth using the judge model.
    """
    if not candidate_answer or "[WARN] No sufficiently relevant context found" in candidate_answer or "do not have information about" in candidate_answer.lower():
        return 0.0, "Abstention, safe fallback warning, or knowledge cutoff refusal"

    sys_prompt = (
        "You are an expert scientific evaluator. Evaluate whether the Candidate Answer is factually "
        "accurate and consistent with the Reference Ground Truth Answer for the given scientific Question."
    )
    user_content = (
        f"Question: {question}\n\n"
        f"Reference Ground Truth Answer: {ground_truth}\n\n"
        f"Candidate Model Answer: {candidate_answer}\n\n"
        "Rate the factual accuracy on a continuous scale from 0.0 to 1.0:\n"
        "- 1.0: Completely factually accurate; correctly captures the specific technical mechanisms, numbers, or claims in the ground truth.\n"
        "- 0.5: Partially correct, high-level, or generic, but misses critical specific numbers/findings.\n"
        "- 0.0: Factually contradictory, completely hallucinated, or ungrounded.\n\n"
        "Output ONLY valid JSON with keys 'score' (float between 0.0 and 1.0) and 'reason' (short 1-sentence rationale)."
    )

    for attempt in range(5):
        try:
            res = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            data = json.loads(res.choices[0].message.content)
            return float(data.get("score", 0.0)), data.get("reason", "")
        except Exception as e:
            time.sleep(1.5)

    return 0.0, "Judge API evaluation failure"


def run_factual_accuracy_evaluation(num_samples: int = 40) -> Dict[str, Any]:
    """
    Run full factual accuracy benchmark across all 3 ablation configurations (FR-12, FR-13, FR-17).
    """
    print("==========================================================================")
    print("AI RESEARCH COPILOT -- GROUND-TRUTH FACTUAL ACCURACY BENCHMARK")
    print(f"Judge Model: OpenAI/{PRIMARY_JUDGE_MODEL} | Samples: {num_samples}")
    print("==========================================================================")

    qa_path = "data/metadata/draft_qa_dataset.json"
    if not os.path.exists(qa_path):
        print(f"[Error] QA benchmark dataset not found at {qa_path}")
        return {}

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)[:num_samples]

    pipeline = ResearchPipelineOrchestrator()
    client = OpenAI()

    records = []
    print(f"\n[1/1] Generating and grading all 3 configs across {len(qa_pairs)} benchmark samples...")

    for idx, item in enumerate(qa_pairs, 1):
        q = item["question"]
        gt = item["ground_truth_answer"]
        sid = item["id"]

        # Config 1: Full Hybrid RAG + Cross-Encoder + GraphRAG
        res1 = pipeline.execute_query(q)
        ans1 = res1.answer
        s1, r1 = grade_factuality_with_judge(client, PRIMARY_JUDGE_MODEL, q, gt, ans1)

        # Config 2: Dense Vector Search Only (No BM25, No Reranker)
        dense_chunks = pipeline.vector_store.search_dense(q, top_k=6)
        for c in dense_chunks:
            c["score"] = c.get("dense_score", 0.8)
            c["rerank_score"] = c.get("dense_score", 0.8)
        res2 = pipeline.qa_engine.generate_point_qa(q, dense_chunks)
        ans2 = res2.answer
        s2, r2 = grade_factuality_with_judge(client, PRIMARY_JUDGE_MODEL, q, gt, ans2)

        # Config 3: Non-RAG Baseline (Direct Parametric Generation)
        res3 = pipeline.execute_baseline(q)
        ans3 = res3.get("answer", "")
        s3, r3 = grade_factuality_with_judge(client, PRIMARY_JUDGE_MODEL, q, gt, ans3)

        records.append({
            "sample_id": sid,
            "question": q,
            "ground_truth": gt,
            "config1_hybrid_factual_acc": s1,
            "config1_reason": r1,
            "config2_dense_factual_acc": s2,
            "config2_reason": r2,
            "config3_baseline_factual_acc": s3,
            "config3_reason": r3
        })
        print(f"  • Sample {idx:02d}/{len(qa_pairs)} [{sid}] -> C1 (Hybrid): {s1:.2f} | C2 (Dense): {s2:.2f} | C3 (Baseline): {s3:.2f}")

    avg_s1 = sum(r["config1_hybrid_factual_acc"] for r in records) / (len(records) or 1)
    avg_s2 = sum(r["config2_dense_factual_acc"] for r in records) / (len(records) or 1)
    avg_s3 = sum(r["config3_baseline_factual_acc"] for r in records) / (len(records) or 1)

    out_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(records),
        "judge_model": f"OpenAI/{PRIMARY_JUDGE_MODEL}",
        "evaluation_methodology": "LLM-as-judge automated ground-truth factual accuracy scoring (0.0 to 1.0)",
        "aggregate_factual_accuracy": {
            "Config 1: Full Hybrid RAG + Reranker + GraphRAG": round(avg_s1, 4),
            "Config 2: Dense Vector Only (No BM25, No Reranker)": round(avg_s2, 4),
            "Config 3: Non-RAG Baseline (Direct Parametric Generation)": round(avg_s3, 4)
        },
        "per_sample_breakdown": records
    }

    os.makedirs("data/metadata", exist_ok=True)
    out_file = "data/metadata/factual_accuracy_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print("\n" + "=" * 90)
    print(f"FACTUAL ACCURACY BENCHMARK SUMMARY ({len(records)} SAMPLES):")
    print("=" * 90)
    print(f"  • Config 1 (Full Hybrid RAG):     {avg_s1 * 100:.1f}% ({avg_s1:.4f})")
    print(f"  • Config 2 (Dense Vector Only):    {avg_s2 * 100:.1f}% ({avg_s2:.4f})")
    print(f"  • Config 3 (Non-RAG Baseline):     {avg_s3 * 100:.1f}% ({avg_s3:.4f})")
    print(f"Saved evaluation report to {out_file}")
    print("==========================================================================")

    return out_data


if __name__ == "__main__":
    run_factual_accuracy_evaluation(40)
