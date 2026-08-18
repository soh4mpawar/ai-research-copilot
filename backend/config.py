"""
Centralized Configuration for AI Research Copilot.
Pins model versions for PointQA, LitReview, Baseline, and RAGAS Evaluation in one place.
"""

import os

# 1. Pipeline Generator Configuration (FR-7, FR-8)
PRIMARY_GENERATOR_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GENERATOR_MODELS = [
    "gemini-3.5-flash-lite"
]

# 2. RAGAS Benchmark Judge LLM (FR-12, FR-20, PRD §8)
PRIMARY_JUDGE_MODEL = "gpt-4o-mini"
JUDGE_PROVIDER = "OpenAI"
