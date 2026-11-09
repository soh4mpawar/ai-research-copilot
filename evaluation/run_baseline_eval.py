"""
Non-RAG Baseline Evaluation Runner (Phase 5 / FR-13, PRD §8.2).
Runs direct Gemini baseline over held-out QA test dataset to compare RAG vs Non-RAG performance.
Computes genuine RAGAS answer_relevancy using the centralized judge LLM (OpenAIRagasJudge).
"""

import sys
import os
import json
import time
from typing import Dict, Any

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Module monkeypatch for RAGAS vertexai import compatibility
import langchain_google_vertexai
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy

from backend.config import PRIMARY_JUDGE_MODEL
from backend.generation.baseline import NonRAGBaselineEngine
from evaluation.run_ragas_eval import OpenAIRagasJudge, LocalRagasEmbeddings


def run_baseline_evaluation(num_samples: int = 35) -> Dict[str, Any]:
    print("==========================================================================")
    print("AI RESEARCH COPILOT -- PHASE 5: NON-RAG BASELINE EVALUATION (FR-13)")
    print(f"Judge LLM: {PRIMARY_JUDGE_MODEL} | Samples: {num_samples}")
    print("==========================================================================")

    qa_file = "data/metadata/draft_qa_dataset.json"
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)[:num_samples]
    else:
        from evaluation.draft_qa_dataset import draft_qa_pairs_from_200_corpus
        qa_pairs = draft_qa_pairs_from_200_corpus(num_samples)
    baseline_engine = NonRAGBaselineEngine()

    questions = []
    answers = []
    contexts = []
    baseline_records = []

    print(f"\n[1/2] Generating non-retrieval baseline answers for {len(qa_pairs)} QA pairs...")
    for idx, item in enumerate(qa_pairs, 1):
        q = item["question"]
        res = baseline_engine.generate_baseline_answer(q)
        ans_text = res.get("answer", "")

        questions.append(q)
        answers.append(ans_text)
        # Baseline has no retrieval context; answer_relevancy evaluates question-answer alignment.
        # Passing an empty string context list to satisfy Dataset schema without providing external context.
        contexts.append([""])

        baseline_records.append({
            "sample_id": item["id"],
            "question": q,
            "baseline_answer": ans_text[:180] + "..." if len(ans_text) > 180 else ans_text,
            "raw_answer": ans_text
        })
        print(f"  • Generated baseline answer {idx}/{len(qa_pairs)} for {item['id']}")

    print(f"\n[2/2] Evaluating genuine RAGAS answer_relevancy with {PRIMARY_JUDGE_MODEL} judge...")
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts
    })

    judge_llm = OpenAIRagasJudge(model_name=PRIMARY_JUDGE_MODEL)
    embedder = LocalRagasEmbeddings()

    eval_result = evaluate(
        dataset=ragas_dataset,
        metrics=[answer_relevancy],
        llm=judge_llm,
        embeddings=embedder
    )

    df = eval_result.to_pandas()
    relevance_scores = []
    baseline_samples = []

    for idx, record in enumerate(baseline_records):
        a_score = 0.0
        if "answer_relevancy" in df.columns:
            val = df.iloc[idx]["answer_relevancy"]
            a_score = round(float(val), 4) if not (val is None or str(val) == "nan") else 0.0

        relevance_scores.append(a_score)
        baseline_samples.append({
            "sample_id": record["sample_id"],
            "question": record["question"],
            "baseline_answer": record["baseline_answer"],
            "answer_relevance": a_score,
            # Manual factual accuracy requires human expert spot-check grading per PRD §8.2 / FR-12 / FR-13
            "manual_factual_accuracy": None
        })

    avg_relevance = round(sum(relevance_scores) / (len(relevance_scores) or 1), 3)

    print(f"\n--- NON-RAG BASELINE BENCHMARK RESULTS ({len(qa_pairs)} Samples) ---")
    print(f"  • Baseline Answer Relevance:      {avg_relevance:.3f}")
    print(f"  • Manual Factual Accuracy Check:  Requires human review (PRD §8.2)")
    print(f"  • Faithfulness / Context Recall:  N/A (Undefined without retrieval)")

    report = {
        "total_samples": len(qa_pairs),
        "baseline_metrics": {
            "answer_relevance": avg_relevance,
            "manual_factual_accuracy": None,
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None
        },
        "samples": baseline_samples
    }

    os.makedirs("data/metadata", exist_ok=True)
    with open("data/metadata/baseline_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved baseline evaluation report to data/metadata/baseline_eval_results.json")
    print("==========================================================================")
    print("PHASE 5 DEFINITION OF DONE MET: Baseline Evaluation Complete!")
    print("==========================================================================")

    return report


if __name__ == "__main__":
    run_baseline_evaluation(35)

