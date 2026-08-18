"""
Official RAGAS Benchmark Evaluation Runner (Phase 5 / FR-12, FR-20, PRD §8).
STRICT SPECIFICATION:
- Pipeline Generator: Google Gemini 3.7 Flash (centralized in backend/config.py)
- Judge LLM: 100% OpenAI GPT-4o-mini (PRD §8 Gold Standard Spec, strictly independent provider)
- Embeddings: nomic-embed-text-v1.5 (CUDA)
- Full per-sample logging of judge model, latency, prompt tokens, and output tokens.
"""

import sys
import os
import json
import time
import argparse
import re
import asyncio

# Ensure unbuffered stdout for live progress visibility in terminal
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

# Module monkeypatch for RAGAS vertexai import compatibility
import langchain_google_vertexai
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai

import torch
from openai import OpenAI, AsyncOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.outputs import LLMResult, Generation

from backend.config import PRIMARY_GENERATOR_MODEL, PRIMARY_JUDGE_MODEL, JUDGE_PROVIDER
from backend.pipeline import ResearchPipelineOrchestrator


class OpenAIRagasJudge(BaseRagasLLM):
    """
    Custom RAGAS Judge LLM strictly wrapping OpenAI GPT-4o-mini (PRD §8 Gold Standard).
    Logs latency, prompt tokens, and output tokens for every single evaluation API call.
    """

    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.call_count = 0
        print(f"[OpenAIJudge] Initialized OpenAI RAGAS Judge ({self.model_name}) successfully.", flush=True)

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        txt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        generations = []

        for _ in range(n):
            text_out = ""
            for attempt in range(8):
                t0 = time.time()
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": txt}],
                        model=self.model_name,
                        temperature=temperature,
                    )
                    raw_text = chat_completion.choices[0].message.content or ""
                    
                    cleaned_text = raw_text.strip()
                    if "```json" in cleaned_text:
                        match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                        if match:
                            cleaned_text = match.group(1).strip()
                    
                    text_out = cleaned_text
                    usage = chat_completion.usage
                    p_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    o_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                    dur = time.time() - t0
                    self.call_count += 1
                    print(f"[OpenAI Judge API Call #{self.call_count}] Model: {self.model_name} | Latency: {dur:.2f}s | Prompt Tokens: {p_tokens} | Output Tokens: {o_tokens}", flush=True)
                    break
                except Exception as e:
                    print(f"[OpenAI Judge Error] Attempt {attempt+1}: {e}", flush=True)
                    time.sleep(2)

            generations.append(Generation(text=text_out))

        return LLMResult(generations=[generations])

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        txt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
        generations = []

        for _ in range(n):
            text_out = ""
            for attempt in range(8):
                t0 = time.time()
                try:
                    chat_completion = await self.async_client.chat.completions.create(
                        messages=[{"role": "user", "content": txt}],
                        model=self.model_name,
                        temperature=temperature,
                    )
                    raw_text = chat_completion.choices[0].message.content or ""
                    
                    cleaned_text = raw_text.strip()
                    if "```json" in cleaned_text:
                        match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                        if match:
                            cleaned_text = match.group(1).strip()
                    
                    text_out = cleaned_text
                    usage = chat_completion.usage
                    p_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    o_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                    dur = time.time() - t0
                    self.call_count += 1
                    print(f"[OpenAI Judge API Call #{self.call_count}] Model: {self.model_name} | Latency: {dur:.2f}s | Prompt Tokens: {p_tokens} | Output Tokens: {o_tokens}", flush=True)
                    break
                except Exception as e:
                    print(f"[OpenAI Judge Error] Attempt {attempt+1}: {e}", flush=True)
                    await asyncio.sleep(2)

            generations.append(Generation(text=text_out))

        return LLMResult(generations=[generations])

    def is_finished(self, response: LLMResult) -> bool:
        return True


class LocalRagasEmbeddings(BaseRagasEmbeddings):
    """Custom RAGAS Embeddings wrapping nomic-embed-text-v1.5 model."""

    def __init__(self):
        super().__init__()
        self.model_name = "nomic-ai/nomic-embed-text-v1.5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.st_model = SentenceTransformer(self.model_name, trust_remote_code=True, device=device)
        print(f"[LocalRagasEmbeddings] Loaded {self.model_name} on {device} successfully.")

    def embed_query(self, text: str):
        if isinstance(text, str):
            text = f"search_query: {text}"
        return self.st_model.encode(text).tolist()

    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        formatted = [f"search_document: {t}" for t in texts]
        return self.st_model.encode(formatted, batch_size=16).tolist()

    async def aembed_query(self, text: str):
        return self.embed_query(text)

    async def aembed_documents(self, texts):
        return self.embed_documents(texts)


def get_chunk_text(c) -> str:
    if hasattr(c, "text"):
        return getattr(c, "text")
    elif isinstance(c, dict):
        return c.get("text", "")
    return str(c)


def run_ragas_evaluation_pass(num_samples: int = 40):
    print("==========================================================================")
    print(f"RUNNING OFFICIAL RAGAS EVALUATION PASS ({num_samples} SAMPLES)")
    print(f"Generator Model: {PRIMARY_GENERATOR_MODEL} (Centralized in backend/config.py)")
    print(f"Judge LLM: {PRIMARY_JUDGE_MODEL} ({JUDGE_PROVIDER} - PRD §8 Gold Standard)")
    print("Embeddings: nomic-ai/nomic-embed-text-v1.5 (CUDA)")
    print("==========================================================================")

    qa_file = "data/metadata/draft_qa_dataset.json"
    if not os.path.exists(qa_file):
        print(f"[RAGAS Error] QA dataset file not found at {qa_file}")
        return

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)[:num_samples]

    pipeline = ResearchPipelineOrchestrator()

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    metadata_records = []

    print(f"\n[1/3] Generating RAG pipeline answers for {len(qa_pairs)} QA pairs...")
    for idx, item in enumerate(qa_pairs, 1):
        q = item["question"]
        gt_ans = item["ground_truth_answer"]
        src_id = item["source_paper_id"]

        res = pipeline.execute_query(q)
        ans_text = res.answer
        retrieved_chunks = res.retrieved_chunks
        actual_generator = res.generator_model
        ctx_list = [get_chunk_text(c) for c in retrieved_chunks]
        if not ctx_list:
            ctx_list = ["No context retrieved."]

        questions.append(q)
        answers.append(ans_text)
        contexts.append(ctx_list)
        ground_truths.append(gt_ans)

        metadata_records.append({
            "sample_id": item["id"],
            "question_type": item.get("type", "General"),
            "source_paper_id": src_id,
            "source_paper_title": item["source_paper_title"],
            "question": q,
            "ground_truth_answer": gt_ans,
            "pipeline_answer": ans_text,
            "retrieved_chunks_count": len(retrieved_chunks),
            "generator_model": actual_generator,
            "judge_model": f"OpenAI/{PRIMARY_JUDGE_MODEL}",
            "judge_provider": JUDGE_PROVIDER
        })

        print(f"  • Generated pipeline answer {idx}/{len(qa_pairs)} for [{src_id}] '{item['source_paper_title'][:40]}' using {actual_generator}")
        time.sleep(4.2)

    print(f"\n[2/3] Constructing RAGAS Dataset and initializing Judge LLM & Embedder...")
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    judge_llm = OpenAIRagasJudge(model_name=PRIMARY_JUDGE_MODEL)
    embedder = LocalRagasEmbeddings()

    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall]

    print(f"\n[3/3] Executing official ragas.evaluate()...")
    eval_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics_list,
        llm=judge_llm,
        embeddings=embedder
    )

    print("\n=== OFFICIAL RAGAS EVALUATION COMPLETE ===")
    print(eval_result)

    # Convert evaluation result to dict / pandas dataframe breakdown
    df = eval_result.to_pandas()

    per_sample_breakdown = []
    for idx, record in enumerate(metadata_records):
        sample_scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if metric_name in df.columns:
                val = df.iloc[idx][metric_name]
                sample_scores[metric_name] = round(float(val), 4) if not (val is None or str(val) == "nan") else 0.0
            else:
                sample_scores[metric_name] = 0.0

        record["scores"] = sample_scores
        per_sample_breakdown.append(record)

    agg_scores = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric_name in df.columns:
            mean_val = float(df[metric_name].mean())
            agg_scores[metric_name] = round(mean_val, 4) if not (mean_val is None or str(mean_val) == "nan") else 0.0
        else:
            agg_scores[metric_name] = 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "COMPLETED",
        "eval_engine": "official_ragas_library",
        "generator_model": f"Google/{PRIMARY_GENERATOR_MODEL}",
        "judge_model": f"OpenAI/{PRIMARY_JUDGE_MODEL}",
        "judge_provider": JUDGE_PROVIDER,
        "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
        "total_eval_samples": len(per_sample_breakdown),
        "aggregate_ragas_scores": agg_scores,
        "prd_targets": {
            "faithfulness": ">0.70",
            "context_precision": ">0.60",
            "context_recall": ">0.60",
            "answer_relevancy": ">0.70"
        },
        "target_pass": all([
            agg_scores.get("faithfulness", 0) >= 0.70,
            agg_scores.get("context_precision", 0) >= 0.60,
            agg_scores.get("context_recall", 0) >= 0.60,
            agg_scores.get("answer_relevancy", 0) >= 0.70
        ]),
        "per_sample_breakdown": per_sample_breakdown
    }

    out_file = f"evaluation/ragas_benchmark_results_{num_samples}samples.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Official RAGAS Benchmark Evaluation")
    parser.add_argument("--samples", type=int, default=40, help="Number of QA pairs to evaluate (default: 40)")
    args = parser.parse_args()

    run_ragas_evaluation_pass(num_samples=args.samples)
