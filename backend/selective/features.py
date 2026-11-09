"""
Feature Extraction for Selective Answering Classifiers.
Extracts relevance-only and source/section structural features strictly from FinalContextRecord.

Feature Nomenclature:
- Relevance Features (4D):
  [s_(1), s_(2), mean_score, score_spread]
- Source/Section Features (4D):
  - In raw configuration: [s_top_paper_second_passage, s_other, section_diversity, section_mean_score]
  - In deduplicated configuration: [s_top_paper_second_distinct_passage, s_other, section_diversity, section_mean_score]
"""

from typing import Sequence, Dict, List, Optional, Tuple
import numpy as np
from backend.selective.context import FinalContextRecord, Passage, deduplicate_passages_exact


GENERIC_SECTIONS = {"general", "none", "", "unknown", "section"}


def _get_source_id(p: Passage) -> str:
    """Return deterministic source identity: paper_id if present and non-empty, else per-record unknown ID."""
    if p.paper_id and p.paper_id.strip():
        return p.paper_id.strip()
    return f"__unknown_source_{p.passage_id}__"


def extract_relevance_features(passages: Sequence[Passage]) -> np.ndarray:
    """
    Relevance-Only Feature Vector (4 dimensions):
    [s_(1), s_(2), mean_score, score_spread]
    Deterministic tie-breaking on score descending, passage_id ascending.
    """
    if not passages:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Deterministic sort: score descending, passage_id ascending
    sorted_passages = sorted(passages, key=lambda p: (-p.score, p.passage_id))
    s1 = sorted_passages[0].score
    s2 = sorted_passages[1].score if len(sorted_passages) > 1 else 0.0
    scores = [p.score for p in passages]
    s_mean = float(np.mean(scores))
    s_spread = float(s1 - sorted_passages[-1].score) if len(sorted_passages) > 1 else 0.0

    return np.array([s1, s2, s_mean, s_spread], dtype=np.float64)


def extract_source_features(passages: Sequence[Passage]) -> np.ndarray:
    """
    Source and Section Structural Feature Vector (4 dimensions):
    [s_p*_2, s_other, section_diversity, section_mean_score]

    - Deterministic tie-breaking: top passage chosen by highest score, then passage_id ascending.
    - Deterministic source identity: paper_id if present, else '__unknown_source_{passage_id}__'.
    - If top passage has unknown source, it maps to its own source group; s_other includes all other sources.
    - Section grouping: strictly by (source_identity, normalized_known_section), preventing cross-paper section merging.
    - Generic/missing sections ({None, '', 'general', 'unknown', 'section'}) excluded from known sections.
    """
    if not passages:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # 1. Deterministic top passage selection
    top_passage = min(passages, key=lambda p: (-p.score, p.passage_id))
    top_source_id = _get_source_id(top_passage)

    # 2. Group passages by deterministic source identity
    source_groups: Dict[str, List[Passage]] = {}
    for p in passages:
        sid = _get_source_id(p)
        source_groups.setdefault(sid, []).append(p)

    # Top paper's second passage score (deterministic tie-breaking)
    top_group = source_groups.get(top_source_id, [])
    sorted_top_group = sorted(top_group, key=lambda p: (-p.score, p.passage_id))
    s_p_star_2 = sorted_top_group[1].score if len(sorted_top_group) > 1 else 0.0

    # Highest score from any other source (including other unknown sources)
    other_scores: List[float] = []
    for sid, group in source_groups.items():
        if sid != top_source_id:
            other_scores.extend([p.score for p in group])
    s_other = max(other_scores) if other_scores else 0.0

    # 3. Known-section grouping strictly by (source_identity, normalized_known_section)
    section_scores: Dict[Tuple[str, str], List[float]] = {}
    for p in passages:
        sec = (p.section or "").strip().lower()
        if sec and sec not in GENERIC_SECTIONS:
            sid = _get_source_id(p)
            sec_key = (sid, sec)
            section_scores.setdefault(sec_key, []).append(p.score)

    total_passages = len(passages)
    section_diversity = len(section_scores) / float(total_passages) if total_passages > 0 else 0.0
    sec_means = [float(np.mean(scs)) for scs in section_scores.values()] if section_scores else [0.0]
    section_mean_score = float(np.mean(sec_means))

    return np.array([s_p_star_2, s_other, section_diversity, section_mean_score], dtype=np.float64)


def extract_features_from_record(
    record: FinalContextRecord,
    feature_mode: str = "relevance_only",
    deduplicate: bool = False
) -> np.ndarray:
    """
    Extract feature vector strictly from final context record.
    feature_mode: 'relevance_only' (4 features), 'source_section' (4 features), or 'combined' (8 features).
    deduplicate: if True, runs exact deduplication within paper partitions before feature extraction.
    """
    passages = record.passages
    if deduplicate:
        passages = deduplicate_passages_exact(passages)

    rel_feats = extract_relevance_features(passages)
    if feature_mode == "relevance_only":
        return rel_feats

    src_feats = extract_source_features(passages)
    if feature_mode == "source_section":
        return src_feats
    elif feature_mode == "combined":
        return np.concatenate([rel_feats, src_feats])
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
