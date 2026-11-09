"""
Autorater Adapter Inspired by Sufficient Context (Joren et al., ICLR 2025).

Provenance & Disclosed Adaptations:
- Reference Publication: Joren, H., Zhang, J., Ferng, C. S., et al. (2025).
  'Sufficient Context: A New Lens on Retrieval Augmented Generation Systems.' ICLR 2025. arXiv:2411.06037.
- Official Code Reference: https://github.com/hljoren/sufficientcontext
- Adaptation Notice: This adapter implements a structured evidence-sufficiency evaluation rubric
  inspired by Sufficient Context. It is NOT a verbatim reproduction of the 1-shot Gemini 1.5 Pro prompt
  from the published benchmark. All adaptations are disclosed:
  1. Requests concise evidence-based justification and supporting passage IDs rather than raw private chain-of-thought.
  2. Parses strict terminal verdicts [[1]] (sufficient) and [[0]] (insufficient) only.
  3. Offline cache misses return status 'pending' (verdict=None); synthetic labels never enter research cache.
  4. Cache keys enforce full provenance: query, context hash, prompt hash, model identity, inference settings, and adapter version.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import logging
import os
import re
from typing import Optional, Dict, Any, Tuple, List

from backend.selective.context import FinalContextRecord

logger = logging.getLogger("selective_autorater")

ADAPTER_VERSION = "2.0.0_inspired_by_sufficientcontext"
DEFAULT_CACHE_PATH = "data/metadata/autorater_cache.json"

PROMPT_TEMPLATE = """You are an evidence sufficiency auditor for a scientific retrieval-augmented question answering system.

Task: Determine whether the retrieved passages contain sufficient factual information to answer the question completely and factually, without requiring outside assumptions, speculation, or domain extrapolation.

Question:
{query}

Retrieved Passages:
{context_payload}

Instructions:
1. Evidence Justification: In 1-3 sentences, state specifically what required facts or numerical values are present or missing in the retrieved passages.
2. Supporting Passage IDs: List the passage IDs (e.g., [1], [2]) that directly substantiate the answer. If context is insufficient, state "None".
3. Final Verdict: On the final line, output exactly [[1]] if the retrieved context is sufficient to answer the question, or [[0]] if the context is insufficient.

Response Format:
Justification: <concise factual explanation>
Supporting Passage IDs: <bracketed IDs or None>
Verdict: [[1]] or [[0]]"""


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AutoraterResult:
    """Immutable result of an evidence sufficiency autorater evaluation."""
    query: str
    context_hash: str
    prompt_hash: str
    model_identity: str
    inference_settings: Dict[str, Any]
    adapter_version: str
    raw_response: Optional[str]
    parse_status: str  # 'parsed_valid', 'parsed_malformed', 'pending_execution'
    verdict: Optional[int]  # 1 (sufficient), 0 (insufficient), None (pending or malformed)
    justification: Optional[str]
    supporting_passage_ids: Tuple[str, ...]
    is_cached: bool


class SufficientContextInspiredAutorater:
    """
    Autorater adapter inspired by Sufficient Context (Joren et al., 2025).
    Guarantees that cache misses return 'pending' rather than fabricating a verdict.
    """

    def __init__(
        self,
        model_identity: str = "offline_pending_adapter",
        inference_settings: Optional[Dict[str, Any]] = None,
        cache_path: str = DEFAULT_CACHE_PATH,
        allow_live_api: bool = False
    ):
        self.model_identity = model_identity
        self.inference_settings = inference_settings or {"temperature": 0.0, "max_tokens": 512}
        self.cache_path = cache_path
        self.allow_live_api = allow_live_api
        self.adapter_version = ADAPTER_VERSION
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Quarantine entries lacking provenance
                valid_entries = {}
                for k, v in data.items():
                    if isinstance(v, dict) and "prompt_hash" in v and "model_identity" in v:
                        valid_entries[k] = v
                    else:
                        logger.warning(f"Quarantining unverified cache entry '{k}' lacking provenance.")
                self.cache = valid_entries
            except Exception as e:
                logger.error(f"Error loading autorater cache from {self.cache_path}: {e}")
                self.cache = {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def format_prompt(self, record: FinalContextRecord) -> Tuple[str, str]:
        """Construct the autorater prompt and return (prompt_text, prompt_sha256)."""
        prompt = PROMPT_TEMPLATE.format(
            query=record.query,
            context_payload=record.serialized_payload
        )
        return prompt, compute_sha256(prompt)

    def compute_cache_key(self, record: FinalContextRecord, prompt_hash: str) -> str:
        """Construct deterministic cache key covering all provenance dimensions."""
        key_dict = {
            "query": record.query,
            "context_hash": record.payload_hash,
            "prompt_hash": prompt_hash,
            "model_identity": self.model_identity,
            "inference_settings": self.inference_settings,
            "adapter_version": self.adapter_version
        }
        return compute_sha256(json.dumps(key_dict, sort_keys=True))

    def parse_response(self, raw_text: str) -> Tuple[str, Optional[int], Optional[str], Tuple[str, ...]]:
        """
        Parse raw model response into (parse_status, verdict, justification, supporting_passage_ids).
        Strict rules:
        - Only exact terminal [[1]] or [[0]] is accepted as a valid verdict.
        - If both [[1]] and [[0]] are present, or neither is present, marked as 'parsed_malformed' with verdict=None.
        - Never infer a verdict from the word 'sufficient'.
        """
        if not raw_text or not raw_text.strip():
            return "parsed_malformed", None, None, ()

        # Check verdict tokens
        matches = re.findall(r'\[\[([01])\]\]', raw_text)
        if len(matches) != 1:
            # Contradictory, ambiguous, or missing verdict
            return "parsed_malformed", None, None, ()

        verdict = int(matches[0])

        # Extract justification
        just_match = re.search(r'Justification:\s*(.*?)(?=\nSupporting Passage IDs:|\nVerdict:|$)', raw_text, re.DOTALL | re.IGNORECASE)
        justification = just_match.group(1).strip() if just_match else None

        # Extract passage IDs
        passages_match = re.search(r'Supporting Passage IDs:\s*(.*?)(?=\nVerdict:|$)', raw_text, re.DOTALL | re.IGNORECASE)
        passage_ids: List[str] = []
        if passages_match:
            raw_pids = passages_match.group(1).strip()
            if raw_pids.lower() != "none":
                found = re.findall(r'\[(\d+)\]', raw_pids)
                passage_ids = [f"[{x}]" for x in found]

        return "parsed_valid", verdict, justification, tuple(passage_ids)

    def evaluate(self, record: FinalContextRecord) -> AutoraterResult:
        """
        Evaluate context sufficiency.
        If entry exists in cache, returns cached result.
        If cache miss and allow_live_api is False, returns 'pending_execution' with verdict=None.
        NEVER fabricates a verdict.
        """
        prompt, prompt_hash = self.format_prompt(record)
        cache_key = self.compute_cache_key(record, prompt_hash)

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            return AutoraterResult(
                query=record.query,
                context_hash=record.payload_hash,
                prompt_hash=prompt_hash,
                model_identity=entry["model_identity"],
                inference_settings=entry["inference_settings"],
                adapter_version=entry["adapter_version"],
                raw_response=entry.get("raw_response"),
                parse_status=entry["parse_status"],
                verdict=entry["verdict"],
                justification=entry.get("justification"),
                supporting_passage_ids=tuple(entry.get("supporting_passage_ids", ())),
                is_cached=True
            )

        if not self.allow_live_api:
            # Return pending execution with NO fabricated verdict
            return AutoraterResult(
                query=record.query,
                context_hash=record.payload_hash,
                prompt_hash=prompt_hash,
                model_identity=self.model_identity,
                inference_settings=self.inference_settings,
                adapter_version=self.adapter_version,
                raw_response=None,
                parse_status="pending_execution",
                verdict=None,  # Strictly None, never 0 or 1
                justification=None,
                supporting_passage_ids=(),
                is_cached=False
            )

        raise RuntimeError(
            "Live API evaluation is disabled in this bounded stage. "
            "Autorater evaluation must be populated via offline cache or explicit batch annotation."
        )

    def record_genuine_response(
        self,
        record: FinalContextRecord,
        raw_response: str
    ) -> AutoraterResult:
        """
        Record a verified genuine model response into the cache.
        Synthetic fixture responses must NEVER be passed to this method.
        """
        prompt, prompt_hash = self.format_prompt(record)
        cache_key = self.compute_cache_key(record, prompt_hash)

        parse_status, verdict, justification, passage_ids = self.parse_response(raw_response)

        entry = {
            "query": record.query,
            "context_hash": record.payload_hash,
            "prompt_hash": prompt_hash,
            "model_identity": self.model_identity,
            "inference_settings": self.inference_settings,
            "adapter_version": self.adapter_version,
            "raw_response": raw_response,
            "parse_status": parse_status,
            "verdict": verdict,
            "justification": justification,
            "supporting_passage_ids": list(passage_ids)
        }
        self.cache[cache_key] = entry
        self._save_cache()

        return AutoraterResult(
            query=record.query,
            context_hash=record.payload_hash,
            prompt_hash=prompt_hash,
            model_identity=self.model_identity,
            inference_settings=self.inference_settings,
            adapter_version=self.adapter_version,
            raw_response=raw_response,
            parse_status=parse_status,
            verdict=verdict,
            justification=justification,
            supporting_passage_ids=passage_ids,
            is_cached=True
        )
