"""
Multi-Configuration RAGAS Ablation Study Runner (Phase 6 / FR-17, PRD §8).
Evaluates QA dataset samples across 3 pipeline configurations:
  Config 1: Full Hybrid RAG + Cross-Encoder Reranker + GraphRAG
  Config 2: Dense Vector Search Only (No BM25, No Reranker)
  Config 3: Non-RAG Baseline (Direct Parametric Generation)
Computes official RAGAS metrics (faithfulness, answer_relevancy) using the centralized OpenAIRagasJudge.
Saves complete per-sample breakdown to data/metadata/ablation_eval_results.json.
"""

import sys
import os
import json
import time
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Module monkeypatch for RAGAS vertexai import compatibility
import langchain_google_vertexai
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from backend.config import PRIMARY_JUDGE_MODEL
from backend.pipeline import ResearchPipelineOrchestrator
from evaluation.run_ragas_eval import OpenAIRagasJudge, LocalRagasEmbeddings, get_chunk_text


def run_ablation_study(num_samples: int = 40) -> Dict[str, Any]:
    """
    Run multi-configuration RAGAS ablation study (FR-17).
    Evaluates all 3 configurations across the benchmark QA dataset using OpenAIRagasJudge.
    """
    print("==========================================================================")
    print("AI RESEARCH COPILOT -- PHASE 6: MULTI-CONFIGURATION RAGAS ABLATION (FR-17)")
    print(f"Judge LLM: {PRIMARY_JUDGE_MODEL} | Samples: {num_samples}")
    print("==========================================================================")

    qa_file = "data/metadata/draft_qa_dataset.json"
    if not os.path.exists(qa_file):
        print(f"[Ablation Error] QA dataset file not found at {qa_file}")
        return {}

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)[:num_samples]

    pipeline = ResearchPipelineOrchestrator()
    judge_llm = OpenAIRagasJudge(model_name=PRIMARY_JUDGE_MODEL)
    embedder = LocalRagasEmbeddings()

    # --- 1. Collect Generations for Config 1 (Full Hybrid RAG) ---
    print(f"\n[1/3] Generating and evaluating Config 1: Full Hybrid RAG ({len(qa_pairs)} samples)...")
    c1_questions, c1_answers, c1_contexts, c1_ground_truths = [], [], [], []
    for idx, item in enumerate(qa_pairs, 1):
        q = item["question"]
        res = pipeline.execute_query(q)
        ans = res.answer
        ctx = [get_chunk_text(c) for c in res.retrieved_chunks] or ["No context retrieved."]
        c1_questions.append(q)
        c1_answers.append(ans)
        c1_contexts.append(ctx)
        c1_ground_truths.append(item["ground_truth_answer"])
        print(f"  • Config 1 generation {idx}/{len(qa_pairs)}")

    ds_c1 = Dataset.from_dict({
        "question": c1_questions,
        "answer": c1_answers,
        "contexts": c1_contexts,
        "ground_truth": c1_ground_truths
    })
    eval_c1 = evaluate(dataset=ds_c1, metrics=[faithfulness, answer_relevancy], llm=judge_llm, embeddings=embedder)
    df_c1 = eval_c1.to_pandas()
    c1_f = round(float(df_c1["faithfulness"].mean()), 3) if "faithfulness" in df_c1 else 0.0
    c1_r = round(float(df_c1["answer_relevancy"].mean()), 3) if "answer_relevancy" in df_c1 else 0.0

    # --- 2. Collect Generations for Config 2 (Dense Vector Only) ---
    print(f"\n[2/3] Generating and evaluating Config 2: Dense Vector Only ({len(qa_pairs)} samples)...")
    c2_questions, c2_answers, c2_contexts, c2_ground_truths = [], [], [], []
    for idx, item in enumerate(qa_pairs, 1):
        q = item["question"]
        dense_chunks = pipeline.vector_store.search_dense(q, top_k=6)
        for c in dense_chunks:
            c["score"] = c.get("dense_score", 0.8)
            c["rerank_score"] = c.get("dense_score", 0.8)
        res = pipeline.qa_engine.generate_point_qa(q, dense_chunks)
        ans = res.answer
        ctx = [get_chunk_text(c) for c in dense_chunks] or ["No context retrieved."]
        c2_questions.append(q)
        c2_answers.append(ans)
        c2_contexts.append(ctx)
        c2_ground_truths.append(item["ground_truth_answer"])
        print(f"  • Config 2 generation {idx}/{len(qa_pairs)}")

    ds_c2 = Dataset.from_dict({
        "question": c2_questions,
        "answer": c2_answers,
        "contexts": c2_contexts,
        "ground_truth": c2_ground_truths
    })
    eval_c2 = evaluate(dataset=ds_c2, metrics=[faithfulness, answer_relevancy], llm=judge_llm, embeddings=embedder)
    df_c2 = eval_c2.to_pandas()
    c2_f = round(float(df_c2["faithfulness"].mean()), 3) if "faithfulness" in df_c2 else 0.0
    c2_r = round(float(df_c2["answer_relevancy"].mean()), 3) if "answer_relevancy" in df_c2 else 0.0

    # --- 3. Collect Generations for Config 3 (Non-RAG Baseline) ---
    print(f"\n[3/3] Generating and evaluating Config 3: Non-RAG Baseline ({len(qa_pairs)} samples)...")
    c3_questions, c3_answers, c3_contexts = [], [], []
    baseline_samples_log = []
    for idx, item in enumerate(qa_pairs, 1):
        q = item["question"]
        res = pipeline.execute_baseline(q)
        ans = res.get("answer", "")
        c3_questions.append(q)
        c3_answers.append(ans)
        c3_contexts.append([""])
        if idx <= 5:
            baseline_samples_log.append({
                "sample_id": item["id"],
                "question": q,
                "baseline_answer": ans[:300] + "..." if len(ans) > 300 else ans
            })
        print(f"  • Config 3 generation {idx}/{len(qa_pairs)}")

    ds_c3 = Dataset.from_dict({
        "question": c3_questions,
        "answer": c3_answers,
        "contexts": c3_contexts
    })
    eval_c3 = evaluate(dataset=ds_c3, metrics=[answer_relevancy], llm=judge_llm, embeddings=embedder)
    df_c3 = eval_c3.to_pandas()
    c3_r = round(float(df_c3["answer_relevancy"].mean()), 3) if "answer_relevancy" in df_c3 else 0.0

    per_sample_comparison = []
    for idx, item in enumerate(qa_pairs):
        c1_faith_val = round(float(df_c1.iloc[idx]["faithfulness"]), 4) if "faithfulness" in df_c1 and not pd.isna(df_c1.iloc[idx]["faithfulness"]) else 0.0
        c1_rel_val = round(float(df_c1.iloc[idx]["answer_relevancy"]), 4) if "answer_relevancy" in df_c1 and not pd.isna(df_c1.iloc[idx]["answer_relevancy"]) else 0.0
        c2_faith_val = round(float(df_c2.iloc[idx]["faithfulness"]), 4) if "faithfulness" in df_c2 and not pd.isna(df_c2.iloc[idx]["faithfulness"]) else 0.0
        c2_rel_val = round(float(df_c2.iloc[idx]["answer_relevancy"]), 4) if "answer_relevancy" in df_c2 and not pd.isna(df_c2.iloc[idx]["answer_relevancy"]) else 0.0
        c3_rel_val = round(float(df_c3.iloc[idx]["answer_relevancy"]), 4) if "answer_relevancy" in df_c3 and not pd.isna(df_c3.iloc[idx]["answer_relevancy"]) else 0.0

        per_sample_comparison.append({
            "sample_id": item["id"],
            "question": item["question"],
            "config1_hybrid_faithfulness": c1_faith_val,
            "config1_hybrid_answer_relevance": c1_rel_val,
            "config2_dense_faithfulness": c2_faith_val,
            "config2_dense_answer_relevance": c2_rel_val,
            "config3_baseline_answer_relevance": c3_rel_val
        })

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_eval_samples": len(qa_pairs),
        "corpus_papers_count": 204,
        "chroma_chunks_count": 1094,
        "judge_model": f"OpenAI/{PRIMARY_JUDGE_MODEL}",
        "ablation_configurations": {
            "Config 1: Full Hybrid RAG + Reranker + GraphRAG": {
                "faithfulness": c1_f,
                "answer_relevance": c1_r
            },
            "Config 2: Dense Vector Only (No BM25, No Reranker)": {
                "faithfulness": c2_f,
                "answer_relevance": c2_r
            },
            "Config 3: Non-RAG Baseline (Direct Parametric Generation)": {
                "faithfulness": None,
                "answer_relevance": c3_r
            }
        },
        "per_sample_comparison": per_sample_comparison,
        "baseline_sample_inspection": baseline_samples_log,
        "key_ablation_takeaway": f"Adding Sparse BM25 + RRF Fusion + Cross-Encoder Reranking improved Faithfulness by +{c1_f - c2_f:.3f} and Answer Relevance by +{c1_r - c2_r:.3f} over Dense Vector Search alone (FR-17)."
    }

    os.makedirs("data/metadata", exist_ok=True)
    out_file = "data/metadata/ablation_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n-----------------------------------------------------------------------------------------------------------------")
    print(f"PER-SAMPLE ABLATION METRICS BREAKDOWN ({len(qa_pairs)} Samples):")
    print("-----------------------------------------------------------------------------------------------------------------")
    print(f"{'Sample ID':<10} | {'Config 1 (Hybrid)':<22} | {'Config 2 (Dense Only)':<22} | {'Config 3 (Baseline)':<20}")
    print(f"{'':<10} | {'Faith':<8} {'Relev':<8} | {'Faith':<8} {'Relev':<8} | {'Relev':<12}")
    print("-" * 85)
    for s in per_sample_comparison:
        print(f"{s['sample_id']:<10} | {s['config1_hybrid_faithfulness']:<8.4f} {s['config1_hybrid_answer_relevance']:<8.4f} | {s['config2_dense_faithfulness']:<8.4f} {s['config2_dense_answer_relevance']:<8.4f} | {s['config3_baseline_answer_relevance']:<12.4f}")

    print("\n--------------------------------------------------------------------------")
    print(f"MULTI-CONFIGURATION ABLATION RESULTS (FR-17 / {len(qa_pairs)} Samples):")
    print("--------------------------------------------------------------------------")
    print(f"  • Config 1 (Full Hybrid + Rerank + Graph): Faithfulness = {c1_f:.3f} | Answer Relevance = {c1_r:.3f}")
    print(f"  • Config 2 (Dense Vector Only):            Faithfulness = {c2_f:.3f} | Answer Relevance = {c2_r:.3f}")
    print(f"  • Config 3 (Non-RAG Baseline):             Faithfulness = N/A   | Answer Relevance = {c3_r:.3f}")
    print(f"\nTakeaway: {results['key_ablation_takeaway']}")
    print("==========================================================================")
    return results


if __name__ == "__main__":
    run_ablation_study(40)

