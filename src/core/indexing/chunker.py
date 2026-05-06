from __future__ import annotations

import re
from typing import Final, Iterator

from src.config.settings import settings
from src.core.schema import Chunk, ChunkType

# ---------------------------------------------------------------------------
# Pre-compiled regular expressions
# ---------------------------------------------------------------------------

RE_H1: Final = re.compile(r"^# (.+)$", re.MULTILINE)
RE_H2: Final = re.compile(r"^## (.+)$", re.MULTILINE)
RE_MARK: Final = re.compile(r"<mark>(.*?)</mark>")
RE_TABLE_ROW: Final = re.compile(r"^\|", re.MULTILINE)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
  
class MarkdownChunker:
    """
    Splits Markdown documents into token-bounded :class:`Chunk` objects at H2
    section boundaries.

    The chunker performs three passes:
    1. ``_split_into_sections`` — yields ``(heading, parent_heading, body)``
       triples by walking H1/H2 markers in document order.
    2. ``_split_if_oversized`` — further divides bodies that exceed
       ``max_tokens`` on paragraph boundaries (or hard-cuts single oversized
       paragraphs).
    3. ``_detect_type`` — classifies each part as ``"text"``, ``"table"``, or
       ``"mixed"`` based on the fraction of Markdown table rows.

    Args:
        max_tokens: Upper token budget per chunk.  Tokens are approximated as
            ``len(text) // settings.chars_per_token``.

    Example::

        chunker = MarkdownChunker()
        chunks = chunker.chunk(text, source_file="om_syntax_core.md")
    """

    def __init__(self, max_tokens: int = settings.default_max_tokens) -> None:
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, source_file: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0
        for heading, parent_heading, body in self._split_into_sections(text):
            for part in self._split_if_oversized(body):
                chunks.append(Chunk(
                    text=f"{heading}\n\n{part}".strip(),
                    heading=heading,
                    parent_heading=parent_heading,
                    chunk_type=self._detect_type(part),
                    source_file=source_file,
                    chunk_index=idx,
                ))
                idx += 1
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> Iterator[tuple[str, str, str]]:
        """
        Walk all H1/H2 headings in document order and yield
        ``(heading, parent_heading, body)`` triples.

        H1 headings update ``current_h1`` context but do not emit a chunk
        themselves.  Text that precedes the first heading is yielded as an
        ``"Introduction"`` section.
        """
        matches = sorted(
            list(RE_H1.finditer(text)) + list(RE_H2.finditer(text)),
            key=lambda x: x.start()
        )

        if not matches:
            yield("Document", "", text)
            return
        
        current_h1 = "Document"
        
        if matches[0].start() > 0:
            orphan = text[:matches[0].start()].strip()
            if orphan:
                yield ("Introduction", "Document", orphan)

        for i, match in enumerate(matches):
            if match.re == RE_H1:
                current_h1 = match.group(1).strip()
                continue

            heading = RE_MARK.sub(r'\1', match.group(1)).strip()
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(text)
            body = text[start_pos:end_pos].strip()

            if body:
                yield (heading, current_h1, body)
    
    def _split_if_oversized(self, text: str) -> list[str]:
        """
        Split *text* into parts that each fit within ``max_tokens``.

        Splitting prefers paragraph boundaries (``\\n\\n``).  Single
        paragraphs that exceed the budget are hard-cut on character width.
        """
        if len(text) // settings.chars_per_token <= self.max_tokens:
            return [text]
        
        parts: list[str] = []
        current_batch: list[str] = []
        current_len = 0

        for para in text.split("\n\n"):
            para = para.strip()
            if not para: continue
            para_len = len(para) // settings.chars_per_token

            if para_len > self.max_tokens:
                if current_batch:
                    parts.append("\n\n".join(current_batch))
                    current_batch, current_len = [], 0
                step = self.max_tokens * settings.chars_per_token
                for i in range(0, len(para), step):
                    parts.append(para[i : i + step])
                continue

            if current_len + para_len > self.max_tokens and current_batch:
                parts.append("\n\n".join(current_batch))
                current_batch, current_len = [], 0
            current_batch.append(para)
            current_len += para_len

        if current_batch:
            parts.append("\n\n".join(current_batch))

        return parts
    
    @staticmethod
    def _detect_type(body: str) -> ChunkType:
        """
        Classify *body* by the ratio of Markdown table rows to total lines.

        Thresholds (from ``settings``):
            - ``> table_threshold``  → ``"table"``
            - ``> mixed_threshold``  → ``"mixed"``
            - otherwise              → ``"text"``
        """
        lines = body.splitlines()
        table_lines = sum(1 for ln in lines if RE_TABLE_ROW.match(ln))
        ratio = table_lines / max(len(lines), 1)
        if ratio > settings.table_threshold: return "table"
        if ratio > settings.mixed_threshold: return "mixed"
        return "text"