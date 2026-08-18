"""
Interim Spot-Check & Relevance Threshold Calibrator (Phase 5 / FR-11, PRD §8.3).
Runs lightweight 10-15 QA pair spot-check pass to empirically calibrate FR-11's relevance threshold constant.
"""

import sys
import os
import json

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.pipeline import get_orchestrator
from evaluation.draft_qa_dataset import draft_qa_pairs_from_corpus


def run_interim_spot_check(num_pairs: int = 15) -> float:
    print("==========================================================================")
    print("AI RESEARCH COPILOT -- PHASE 5: INTERIM SPOT-CHECK & THRESHOLD CALIBRATION")
    print("==========================================================================")

    qa_pairs = draft_qa_pairs_from_corpus(num_pairs)
    orchestrator = get_orchestrator()

    top_scores = []
    passed_count = 0

    for idx, item in enumerate(qa_pairs[:num_pairs], 1):
        q = item["question"]
        res = orchestrator.execute_query(q, mode="qa")
        
        passed = len(res.retrieved_chunks) > 0
        top_score = res.retrieved_chunks[0].rerank_score if passed else 0.15
        top_scores.append(top_score)

        if passed:
            passed_count += 1

        print(f"QA #{idx:02d} | Top Score: {top_score:.3f} | Gate Passed: {passed} | Q: \"{q[:55]}...\"")

    avg_score = sum(top_scores) / (len(top_scores) or 1)
    calibrated_threshold = round(max(0.30, avg_score * 0.55), 2)

    print("\n--------------------------------------------------------------------------")
    print(f"SPOT-CHECK SUMMARY ({len(qa_pairs[:num_pairs])} QA Pairs Evaluated):")
    print(f"  • Average Top Rerank Score:   {avg_score:.3f}")
    print(f"  • Gate Passed Count:          {passed_count} / {len(qa_pairs[:num_pairs])} ({passed_count/(len(qa_pairs[:num_pairs]) or 1)*100:.1f}%)")
    print(f"  • Calibrated FR-11 Threshold: {calibrated_threshold}")
    print("==========================================================================")

    return calibrated_threshold


if __name__ == "__main__":
    threshold = run_interim_spot_check(15)
