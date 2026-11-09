"""
Cross-encoder scoring and caching helpers for Benchmark v2 protocol.
Ensures deterministic execution, exact query-text-config caching,
and identical scores for duplicate passage occurrences.
"""

from typing import Dict, List, Tuple, Any, Optional


def score_chunk_with_cache(
    query: str,
    text: str,
    scorer: Any,
    cache: Dict[Tuple[str, str, str], float],
    config_str: str = "model=BAAI/bge-reranker-base;max_length=512;activation=sigmoid;device=cuda:0"
) -> float:
    """
    Production shared scoring function with exact (query, text, config) caching.
    Reuses existing scores for identical query-text pairs, ensuring duplicate
    passages receive identical cross-encoder scores without redundant computation.
    """
    cache_key = (query, text, config_str)
    if cache_key in cache:
        return cache[cache_key]
    raw_scores = scorer.predict([[query, text]])
    score = round(float(raw_scores[0]), 4)
    cache[cache_key] = score
    return score


def score_and_package_passages(
    query: str,
    passages: List[Dict[str, Any]],
    scorer: Any,
    cache: Dict[Tuple[str, str, str], float],
    config_str: str = "model=BAAI/bge-reranker-base;max_length=512;activation=sigmoid;device=cuda:0"
) -> List[Dict[str, Any]]:
    """
    Scores each passage dictionary in a context record in-place using score_chunk_with_cache,
    strictly preserving passage order and assigning identical scores to duplicate chunks.
    """
    for p in passages:
        p["rerank_score"] = score_chunk_with_cache(
            query=query,
            text=p["text"],
            scorer=scorer,
            cache=cache,
            config_str=config_str
        )
    return passages


class CrossEncoderScoringManager:
    """
    Object-oriented wrapper around the shared scoring helper for persistent pipelines.
    """
    def __init__(
        self,
        scorer: Any,
        config_str: str = "model=BAAI/bge-reranker-base;max_length=512;activation=sigmoid;device=cuda:0",
        cache: Optional[Dict[Tuple[str, str, str], float]] = None
    ):
        self.scorer = scorer
        self.config_str = config_str
        self.cache: Dict[Tuple[str, str, str], float] = cache if cache is not None else {}

    def score(self, query: str, text: str, config_str: Optional[str] = None) -> float:
        cfg = config_str or self.config_str
        return score_chunk_with_cache(
            query=query,
            text=text,
            scorer=self.scorer,
            cache=self.cache,
            config_str=cfg
        )

    def score_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        config_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cfg = config_str or self.config_str
        return score_and_package_passages(
            query=query,
            passages=passages,
            scorer=self.scorer,
            cache=self.cache,
            config_str=cfg
        )
