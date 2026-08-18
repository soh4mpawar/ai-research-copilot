"""
Section-Aware Chunking Module (Phase 1 / FR-2, FR-3, PRD §7.1).
Splits section Markdown into ~250-350 token chunks using boundary-aware splitting.
Treats fenced and LaTeX math blocks as atomic units and flags oversized blocks.
"""

import re
from typing import List, Dict, Any, Tuple


class SectionChunker:
    """Boundary-aware chunker respecting section boundaries and atomic math equations."""

    def __init__(self, target_chunk_tokens: int = 300, max_reranker_tokens: int = 512):
        self.target_tokens = target_chunk_tokens
        self.max_reranker_tokens = max_reranker_tokens

    def chunk_paper_sections(
        self,
        paper_id: str,
        title: str,
        authors: List[str],
        sections: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Chunk paper sections into size-bounded sub-chunks (~250-350 tokens)."""
        all_chunks = []
        chunk_idx = 0

        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue

            sub_chunks = self._chunk_section_text(section_text)
            for sub_text in sub_chunks:
                chunk_idx += 1
                token_count = self._estimate_tokens(sub_text)
                
                # Flag oversized atomic blocks per FR-3 & PRD §7.4
                oversized_reranker = token_count > self.max_reranker_tokens

                chunk_obj = {
                    "chunk_id": f"{paper_id}_sec_{chunk_idx:03d}",
                    "paper_id": paper_id,
                    "paper_title": title,
                    "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
                    "section": section_name,
                    "text": sub_text,
                    "token_count": token_count,
                    "oversized_for_reranker": oversized_reranker,
                    "page": 1
                }
                all_chunks.append(chunk_obj)

        return all_chunks

    def _chunk_section_text(self, text: str) -> List[str]:
        """Split text into ~250-350 token chunks while keeping math blocks atomic."""
        # Detect LaTeX math blocks: $$...$$, \[...\], \begin{equation}...\end{equation}
        math_pattern = r'(\$\$.*?\$\$|\\\[.*?\\\]|\\begin\{equation\}.*?\\end\{equation\})'
        raw_blocks = re.split(math_pattern, text, flags=re.DOTALL)

        units = []
        for blk in raw_blocks:
            if not blk.strip():
                continue
            is_math = bool(re.match(r'^(\$\$|\\\[|\\begin\{equation\})', blk.strip()))
            if is_math:
                units.append((blk.strip(), True))
            else:
                # Split regular text by double-newline or sentences
                paras = [p.strip() for p in blk.split("\n\n") if p.strip()]
                for p in paras:
                    p_tokens = self._estimate_tokens(p)
                    if p_tokens > self.target_tokens:
                        sentences = re.split(r'(?<=[.!?])\s+', p)
                        cur_sent_group = []
                        cur_sent_tokens = 0
                        for s in sentences:
                            s_tokens = self._estimate_tokens(s)
                            if cur_sent_tokens + s_tokens > self.target_tokens and cur_sent_group:
                                units.append((" ".join(cur_sent_group), False))
                                cur_sent_group = [s]
                                cur_sent_tokens = s_tokens
                            else:
                                cur_sent_group.append(s)
                                cur_sent_tokens += s_tokens
                        if cur_sent_group:
                            units.append((" ".join(cur_sent_group), False))
                    else:
                        units.append((p, False))

        chunks = []
        current_chunk = []
        current_tokens = 0

        for unit_text, is_math in units:
            unit_tokens = self._estimate_tokens(unit_text)

            if is_math and unit_tokens >= self.target_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk).strip())
                    current_chunk = []
                    current_tokens = 0
                chunks.append(unit_text)
                continue

            if current_tokens + unit_tokens > self.target_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk).strip())
                current_chunk = [unit_text]
                current_tokens = unit_tokens
            else:
                current_chunk.append(unit_text)
                current_tokens += unit_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())

        return [c for c in chunks if c.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx 1.3 tokens per word)."""
        words = len(text.split())
        return int(words * 1.3)
