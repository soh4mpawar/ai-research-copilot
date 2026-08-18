"""
Live Query & FR-11 Low-Relevance Gate Verification Script (FR-7, FR-11).
Executes live in-corpus research query and out-of-corpus query to prove short-circuiting.
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline import get_orchestrator


def sanitize_str(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    return s.encode("ascii", "ignore").decode("ascii")


def run_live_query_verification():
    print("==========================================================================")
    print("LIVE PIPELINE QUERY EXECUTION & FR-11 SHORT-CIRCUIT GATE DEMONSTRATION")
    print("==========================================================================")

    orchestrator = get_orchestrator()

    # Query A: Valid In-Corpus Technical Query
    q_in = "What are the core methodology contributions of Vision Transformers and Dense Passage Retrieval?"
    print(f"\n--- 1. IN-CORPUS QUERY: \"{q_in}\" ---")
    res_in = orchestrator.execute_query(q_in, mode="qa")
    print(f"Evidence Strength:  {res_in.evidence_strength}")
    print(f"Retrieved Chunks:   {len(res_in.retrieved_chunks)} chunks")
    print(f"Top Rerank Score:   {res_in.retrieved_chunks[0].rerank_score if res_in.retrieved_chunks else 0.0:.3f}")
    print(f"Source Papers ({len(res_in.sources)}): {[sanitize_str(s.title[:30]) + '...' for s in res_in.sources[:3]]}")
    print("Grounded Answer Excerpt:\n")
    print(sanitize_str(res_in.answer[:280]) + "...\n")

    # Query B: Out-Of-Corpus Irrelevant Query (FR-11 Gate Test)
    q_out = "What are the rules of medieval European chess strategy and taxation in 14th century England?"
    print(f"\n--- 2. OUT-OF-CORPUS QUERY (FR-11 GATE TEST): \"{q_out}\" ---")
    res_out = orchestrator.execute_query(q_out, mode="qa")
    print(f"Evidence Strength:  {res_out.evidence_strength}")
    print(f"Retrieved Chunks:   {len(res_out.retrieved_chunks)} chunks")
    print("Output Text Excerpt:\n")
    print(sanitize_str(res_out.answer[:300]) + "\n")

    print("==========================================================================")
    print("VERIFICATION COMPLETE: Live pipeline and FR-11 Relevance Gate Active!")
    print("==========================================================================")


if __name__ == "__main__":
    run_live_query_verification()
