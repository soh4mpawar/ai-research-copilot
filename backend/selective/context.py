"""
Immutable Context Data Structures, Deterministic Selection Engine, and Exact Deduplication.
Phase 2 / Stage 2 Experimental Harness for Duplicate-Aware Source Aggregation.

Design Principles:
1. Strict case preservation: Exact string equality after whitespace normalization only.
2. Hash integrity: Content hashes recomputed from text; mismatched supplied hashes rejected.
3. Deterministic deduplication: Retains maximum-scoring duplicate even on unsorted input.
4. Dual context budget: Enforces 6-passage cap AND 2,000-token budget via pinned tiktoken encoder.
5. Invariance separation: Post-selection duplicate tests are kept separate from fixed-budget displacement tests.
"""

from dataclasses import dataclass, field
import hashlib
import logging
from typing import Sequence, Tuple, Optional, List, Dict, Any
import tiktoken

logger = logging.getLogger("selective_context")

PINNED_TOKENIZER_NAME = "cl100k_base"
_ENCODER = tiktoken.get_encoding(PINNED_TOKENIZER_NAME)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace characters (collapse runs of whitespace to a single space, strip margins).
    PRESERVES EXACT CASE: Scientific notation, casing, acronyms, and genomic/mathematical
    symbols (e.g., 'A' vs 'a', 'p-value' vs 'P-value') must not be conflated.
    """
    if not text:
        return ""
    return " ".join(text.strip().split())


def compute_text_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hash of whitespace-normalized text (preserving case)."""
    normalized = normalize_whitespace(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    """Count tokens using the pinned cl100k_base tokenizer."""
    if not text:
        return 0
    return len(_ENCODER.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class Passage:
    """Immutable representation of a retrieved scientific passage."""
    passage_id: str
    paper_id: Optional[str]
    section: Optional[str]
    text: str
    score: float
    content_hash: str
    token_count: int

    def __post_init__(self):
        # Validate hash integrity
        computed = compute_text_sha256(self.text)
        if self.content_hash != computed:
            raise ValueError(
                f"Inconsistent content_hash for passage '{self.passage_id}': "
                f"supplied='{self.content_hash}' != computed='{computed}'"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Passage":
        text = str(data.get("text", ""))
        computed_hash = compute_text_sha256(text)
        supplied_hash = data.get("content_hash")
        if supplied_hash is not None and supplied_hash != computed_hash:
            raise ValueError(
                f"Supplied content_hash '{supplied_hash}' does not match computed hash '{computed_hash}'."
            )

        t_count = data.get("token_count")
        if t_count is None:
            t_count = count_tokens(text)
        else:
            t_count = int(t_count)

        raw_score = data.get("rerank_score", data.get("score", 0.0))
        try:
            score_val = float(raw_score)
        except (ValueError, TypeError):
            score_val = 0.0

        return cls(
            passage_id=str(data.get("chunk_id", data.get("passage_id", "unknown"))),
            paper_id=data.get("paper_id"),
            section=data.get("section"),
            text=text,
            score=score_val,
            content_hash=computed_hash,
            token_count=t_count
        )


@dataclass(frozen=True)
class FinalContextRecord:
    """
    Immutable final context record evaluated by gating models and provided to the generation adapter.
    Guarantees that gate decision features and generator prompt context evaluate identical evidence.
    """
    query: str
    passages: Tuple[Passage, ...]
    max_passages: int
    token_budget: int
    tokenizer_identity: str
    total_tokens: int
    skipped_passages: Tuple[Tuple[str, int], ...]  # (passage_id, token_count)
    truncation_policy: str
    serialized_payload: str
    payload_hash: str


def format_context_payload(passages: Sequence[Passage]) -> str:
    """Serialize ordered passages into the standard generative prompt payload."""
    blocks = []
    for idx, p in enumerate(passages, 1):
        pid = p.paper_id or "unknown_paper"
        sec = p.section or "General"
        blocks.append(f"[{idx}] Paper: '{pid}' - Section: [{sec}]\nPassage: \"{p.text}\"")
    return "\n\n".join(blocks)


def select_final_context(
    query: str,
    candidates: Sequence[Passage],
    max_passages: int = 6,
    token_budget: int = 2000,
    truncation_policy: str = "top_k_score_desc_id_asc_token_budget_2000"
) -> FinalContextRecord:
    """
    Deterministically select top-k candidate passages within a dual budget:
    1. At most `max_passages` passages (default 6).
    2. At most `token_budget` tokens (default 2000) using pinned cl100k_base tokenizer.

    Candidates are sorted deterministically: score descending, then passage_id ascending.
    Whole-passage selection is enforced: oversized passages that exceed remaining token budget
    are skipped and logged.
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda p: (-p.score, p.passage_id)
    )

    selected: List[Passage] = []
    skipped: List[Tuple[str, int]] = []
    current_tokens = 0

    for p in sorted_candidates:
        if len(selected) >= max_passages:
            break

        cand_payload = format_context_payload(selected + [p])
        cand_tokens = count_tokens(cand_payload)
        if cand_tokens <= token_budget:
            selected.append(p)
            current_tokens = cand_tokens
        else:
            p_tokens = p.token_count if p.token_count > 0 else count_tokens(p.text)
            skipped.append((p.passage_id, p_tokens))
            logger.info(
                f"[Budget Truncation] Skipped oversized passage {p.passage_id} "
                f"({p_tokens} tokens, payload would be {cand_tokens} > {token_budget})"
            )

    selected_tuple = tuple(selected)
    payload = format_context_payload(selected_tuple)
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    actual_payload_tokens = count_tokens(payload)

    return FinalContextRecord(
        query=query,
        passages=selected_tuple,
        max_passages=max_passages,
        token_budget=token_budget,
        tokenizer_identity=PINNED_TOKENIZER_NAME,
        total_tokens=actual_payload_tokens,
        skipped_passages=tuple(skipped),
        truncation_policy=truncation_policy,
        serialized_payload=payload,
        payload_hash=payload_hash
    )


def deduplicate_passages_exact(passages: Sequence[Passage]) -> Tuple[Passage, ...]:
    """
    Exact-text deduplication within each paper partition.
    Case-sensitive equality after whitespace normalization.

    Rules:
    1. Missing paper IDs are treated as distinct unknown sources to avoid false conflation.
    2. Retains the MAXIMUM-SCORING duplicate deterministically, even for unsorted input.
       Ties in score are broken deterministically by passage_id ascending.
    3. Section metadata conflict handling:
       If the winning passage has a missing, generic ('General', 'Unknown', None) section while an
       identical duplicate has a specific section name, populate the specific section.
    4. Input ordering: Preserves the position of the first occurrence of the unique content in `passages`.
    """
    if not passages:
        return ()

    # Group passages by (paper_key, content_hash)
    groups: Dict[Tuple[str, str], List[Passage]] = {}
    first_seen_order: List[Tuple[str, str]] = []

    for idx, p in enumerate(passages):
        paper_key = p.paper_id if p.paper_id else f"__unknown_paper_{idx}__"
        key = (paper_key, p.content_hash)
        if key not in groups:
            groups[key] = []
            first_seen_order.append(key)
        groups[key].append(p)

    deduped: List[Passage] = []
    for key in first_seen_order:
        candidates = groups[key]
        if len(candidates) == 1:
            deduped.append(candidates[0])
            continue

        # Deterministically select max-scoring candidate
        # Sort by: score descending, then passage_id ascending
        best_p = max(candidates, key=lambda p: (p.score, -1 if p.passage_id is None else 0, p.passage_id))

        # Check section metadata resolution
        generic_sections = {None, "", "General", "Unknown", "unknown"}
        final_section = best_p.section
        if final_section in generic_sections:
            for other in candidates:
                if other.section not in generic_sections:
                    final_section = other.section
                    break

        if final_section != best_p.section:
            # Recreate passage with resolved section
            resolved_p = Passage(
                passage_id=best_p.passage_id,
                paper_id=best_p.paper_id,
                section=final_section,
                text=best_p.text,
                score=best_p.score,
                content_hash=best_p.content_hash,
                token_count=best_p.token_count
            )
            deduped.append(resolved_p)
        else:
            deduped.append(best_p)

    return tuple(deduped)


def inject_duplicates_post_selection(
    record: FinalContextRecord,
    source_idx: int = 0
) -> FinalContextRecord:
    """
    Aggregation-Invariance Test Context:
    Appends an exact duplicate of passage at source_idx AFTER context selection with no subsequent truncation.
    Used exclusively to evaluate whether duplicate-aware aggregation is invariant to post-retrieval redundant passages.
    """
    if not record.passages:
        return record

    target = record.passages[min(source_idx, len(record.passages) - 1)]
    dup_passage = Passage(
        passage_id=f"{target.passage_id}_dup_post",
        paper_id=target.paper_id,
        section=target.section,
        text=target.text,
        score=target.score,
        content_hash=target.content_hash,
        token_count=target.token_count
    )
    new_passages = record.passages + (dup_passage,)
    new_payload = format_context_payload(new_passages)
    new_hash = hashlib.sha256(new_payload.encode("utf-8")).hexdigest()
    new_tokens = count_tokens(new_payload)

    return FinalContextRecord(
        query=record.query,
        passages=new_passages,
        max_passages=record.max_passages + 1,
        token_budget=record.token_budget + dup_passage.token_count,
        tokenizer_identity=record.tokenizer_identity,
        total_tokens=new_tokens,
        skipped_passages=record.skipped_passages,
        truncation_policy="post_selection_duplicate_injection_no_truncation",
        serialized_payload=new_payload,
        payload_hash=new_hash
    )


def inject_duplicates_with_displacement(
    query: str,
    candidates: Sequence[Passage],
    max_passages: int = 6,
    token_budget: int = 2000,
    duplicate_source_idx: int = 0
) -> FinalContextRecord:
    """
    Fixed-Budget Replacement Test:
    Injects high-scoring duplicate passages into the candidate pool PRIOR to selection.
    Duplicates displace useful lower-ranked distinct evidence under the fixed budget constraint.
    """
    if not candidates:
        return select_final_context(query, candidates, max_passages=max_passages, token_budget=token_budget)

    target = candidates[min(duplicate_source_idx, len(candidates) - 1)]
    dup_passage = Passage(
        passage_id=f"{target.passage_id}_dup_displace",
        paper_id=target.paper_id,
        section=target.section,
        text=target.text,
        score=target.score,
        content_hash=target.content_hash,
        token_count=target.token_count
    )
    expanded = list(candidates) + [dup_passage]
    return select_final_context(
        query=query,
        candidates=expanded,
        max_passages=max_passages,
        token_budget=token_budget,
        truncation_policy=f"budget_{max_passages}_passages_{token_budget}_tokens_displacement_selection"
    )
