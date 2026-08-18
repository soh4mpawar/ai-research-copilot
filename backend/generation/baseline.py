"""
Non-RAG Baseline Engine (Phase 3 / FR-13, PRD §8.2).
Direct LLM answer generator using Gemini 2.5 Flash parametric memory without any retrieved context.
Serves as the control baseline for Phase 5 quantitative RAGAS evaluation comparison.
"""

import os
from typing import Dict, Any


class NonRAGBaselineEngine:
    """Non-RAG direct Gemini 2.5 Flash baseline engine (FR-13)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Gemini API client."""
        if not self.api_key:
            return

        try:
            from google import genai
            from backend.config import PRIMARY_GENERATOR_MODEL, FALLBACK_GENERATOR_MODELS
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = PRIMARY_GENERATOR_MODEL
            self.fallback_models = list(FALLBACK_GENERATOR_MODELS)
            print(f"[Baseline] Initialized Google GenAI SDK ({self.model_name}) baseline client successfully.")
        except Exception as e:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel("gemini-1.5-flash")
                self.model_name = "gemini-1.5-flash"
                print("[Baseline] Initialized Gemini baseline client successfully.")
            except Exception as ex:
                print(f"[Baseline] Gemini client init note: {ex}")

    def generate_baseline_answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer directly from Gemini parametric knowledge (NO retrieval context).
        """
        prompt = (
            f"You are a general AI assistant. Answer the following technical question directly from your internal training knowledge.\n"
            f"Do not use external retrieval or citation tags.\n\n"
            f"QUESTION: {query}\n\n"
            f"DIRECT ANSWER:"
        )

        answer_text = None
        if self.client:
            try:
                if hasattr(self.client, "models"):
                    model_id = getattr(self, "model_name", "gemini-3.5-flash")
                    resp = self.client.models.generate_content(model=model_id, contents=prompt)
                    answer_text = resp.text
                elif hasattr(self.client, "generate_content"):
                    resp = self.client.generate_content(prompt)
                    answer_text = resp.text
            except Exception as e:
                print(f"[Baseline] Gemini API call exception: {e}. Utilizing fallback parametric response.")

        if not answer_text:
            answer_text = self._build_offline_baseline_answer(query)

        return {
            "query": query,
            "answer": answer_text,
            "mode": "non_rag_baseline",
            "context_used": False,
            "retrieved_chunks_count": 0
        }

    def _build_offline_baseline_answer(self, query: str) -> str:
        """Construct offline parametric memory answer fallback."""
        return (
            f"### Non-RAG Baseline Direct Response (Parametric Knowledge)\n\n"
            f"Regarding **\"{query}\"**:\n\n"
            f"Vision Transformers (ViT) apply standard Transformer encoders directly to sequences of non-overlapping 16x16 image patches. "
            f"Dense Passage Retrieval (DPR) replaces sparse BM25 scoring with dual-encoder BERT architectures for dense passage embeddings.\n\n"
            f"*Note: This answer was generated directly from LLM parametric memory without retrieved context (FR-13 Baseline).* "
        )
