"""
Source-Grounded Coherence Gate (Phase 2 / FR-11, PRD §7.4).
Checks reranked candidates against an empirically calibrated primary relevance threshold (0.35)
and enforces source-grounded clustered coherence (same-paper dual support, deep topic focus,
or multi-paper consensus) to prevent isolated cross-encoder false-positive spikes on out-of-domain
queries while preserving narrow single-fact and deep-focus in-domain retrieval.
"""

from typing import List, Dict, Any, Tuple

# Empirically calibrated relevance threshold constants (PRD §7.4 / FR-11)
DEFAULT_RELEVANCE_THRESHOLD = 0.35
DEFAULT_SECONDARY_THRESHOLD = 0.15
DEFAULT_DEEP_FOCUS_THRESHOLD = 0.85
DEFAULT_MIN_DEEP_FOCUS_CANDIDATES = 3


class RelevanceThresholdGate:
    """Source-grounded relevance threshold gate preventing noisy retrieval from reaching generator (FR-11)."""

    def __init__(
        self,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        secondary_threshold: float = DEFAULT_SECONDARY_THRESHOLD,
        deep_focus_threshold: float = DEFAULT_DEEP_FOCUS_THRESHOLD,
        min_deep_focus_candidates: int = DEFAULT_MIN_DEEP_FOCUS_CANDIDATES,
    ):
        self.threshold = threshold
        self.secondary_threshold = secondary_threshold
        self.deep_focus_threshold = deep_focus_threshold
        self.min_deep_focus_candidates = min_deep_focus_candidates

    def evaluate_chunks(self, reranked_chunks: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Evaluate candidate rerank scores and source distribution against coherence gate rules.
        Returns: (passed_gate, valid_chunks, status_message)
        """
        if not reranked_chunks:
            msg = "[WARN] No sufficiently relevant context found in the ingested paper corpus for your question."
            return False, [], msg

        top_score = reranked_chunks[0].get("rerank_score", reranked_chunks[0].get("score", 0.0))

        # Absolute baseline floor check
        if top_score < self.threshold:
            msg = (
                f"[WARN] No sufficiently relevant context found in the ingested paper corpus for your question "
                f"(Top relevance score: {top_score:.3f} < threshold {self.threshold:.2f})."
            )
            return False, [], msg

        # Group candidate scores by source paper_id
        paper_scores: Dict[str, List[float]] = {}
        for c in reranked_chunks:
            pid = c.get("paper_id", "unknown")
            sc = c.get("rerank_score", c.get("score", 0.0))
            if pid not in paper_scores:
                paper_scores[pid] = []
            paper_scores[pid].append(sc)

        # 1. Condition A: Same-Paper Dual Reinforcement
        # At least one source paper has a primary chunk >= threshold and secondary chunk >= secondary_threshold
        same_paper_passed = False
        for pid, sc_list in paper_scores.items():
            if len(sc_list) >= 2:
                sorted_sc = sorted(sc_list, reverse=True)
                if sorted_sc[0] >= self.threshold and sorted_sc[1] >= self.secondary_threshold:
                    same_paper_passed = True
                    break

        # 2. Condition B: Same-Paper Deep Topic Focus
        # A paper with very high top relevance (>=0.85) heavily represented in the retrieval pool (>= 3 candidates)
        deep_focus_passed = False
        if not same_paper_passed:
            for pid, sc_list in paper_scores.items():
                if len(sc_list) >= self.min_deep_focus_candidates and max(sc_list) >= self.deep_focus_threshold:
                    deep_focus_passed = True
                    break

        # 3. Condition C: Multi-Paper Cross-Source Consensus
        # At least two distinct papers each have a chunk >= threshold
        multi_paper_passed = False
        if not same_paper_passed and not deep_focus_passed:
            qualifying_papers = [pid for pid, sc_list in paper_scores.items() if max(sc_list) >= self.threshold]
            if len(qualifying_papers) >= 2:
                multi_paper_passed = True

        if not (same_paper_passed or deep_focus_passed or multi_paper_passed):
            msg = (
                f"[WARN] Retrieval rejected: Isolated single-chunk hook without intra-paper or cross-paper support "
                f"(Top score: {top_score:.3f}, lacking secondary source reinforcement)."
            )
            return False, [], msg

        # Filter valid chunks that meet or exceed minimum threshold
        valid_chunks = [c for c in reranked_chunks if c.get("rerank_score", c.get("score", 0.0)) >= (self.threshold - 0.1)]
        msg = f"[OK] Context passed source-grounded relevance gate (Top score: {top_score:.3f} >= {self.threshold:.2f})."
        return True, valid_chunks, msg
