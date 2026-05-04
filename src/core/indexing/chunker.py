"""
Markdown chunker: splits document at H2 boundaries.

Each "## Функция X" section becomes one atomic chunk -
preserves syntax + arguments + example together.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterator, Literal

RE_H2: Final = re.compile(r'^## (.+)$', re.MULTILINE)
RE_H1: Final = re.compile(r'^# (.+)$', re.MULTILINE)
RE_MARK: Final = re.compile(r'<mark>(.*?)</mark>')
RE_TABLE_ROW: Final = re.compile(r'^\|', re.MULTILINE)

CHARS_PER_TOKEN: Final = 2
TABLE_THRESHOLD: Final = 0.8 
MIXED_THRESHOLD: Final = 0.2

ChunkType = Literal["text", "table", "mixed"]


@dataclass
class Chunk:
    text:str
    heading: str
    parent_heading: str
    chunk_type: ChunkType
    source_file: str
    chunk_index: int
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        fingerprint = f"{self.source_file}::{self.chunk_index}::{self.text}"
        self.chunk_id = str(uuid.UUID(hashlib.md5(fingerprint.encode()).hexdigest()))

    
class MarkdownChunker:
    """
    Splits Markdown into chunks at H2 boundaries.

    Usage:
        chunker = MarkdownChunker()
        chunks = chunker.chunk(text, source_file="om_syntax_core.md")
    """

    def __init__(self, max_tokens: int = 1800) -> None:
        self.max_tokens = max_tokens

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
    # Private Logic
    # ------------------------------------------------------------------

    def _split_into_sections(self, text: str) -> Iterator[tuple[str, str, str]]:
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
        if len(text) // CHARS_PER_TOKEN <= self.max_tokens:
            return [text]
        
        parts: list[str] = []
        current_batch: list[str] = []
        current_len = 0

        for para in text.split("\n\n"):
            para = para.strip()
            if not para: continue
            para_len = len(para) // CHARS_PER_TOKEN

            if para_len > self.max_tokens:
                if current_batch:
                    parts.append("\n\n".join(current_batch))
                    current_batch, current_len = [], 0
                step = self.max_tokens * CHARS_PER_TOKEN
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
        lines = body.splitlines()
        table_lines = sum(1 for ln in lines if RE_TABLE_ROW.match(ln))
        ratio = table_lines / max(len(lines), 1)
        if ratio > TABLE_THRESHOLD:
            return "table"
        if ratio > MIXED_THRESHOLD:
            return "mixed"
        return "text"