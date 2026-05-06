from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ChunkType = Literal["text", "table", "mixed"]
ExpandType = Literal["text", "markdown", "items"]
PathLike = Union[str, Path]
Tier = Literal["fast", "cost_effective", "agentic", "agentic_plus"]


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    A single text unit produced by the chunker.

    ``chunk_id`` is derived deterministically from ``source_file``,
    ``chunk_index``, and ``text`` so that re-indexing the same content
    always produces the same UUID (idempotent upserts in Qdrant).
    """

    text: str
    heading: str
    parent_heading: str
    chunk_type: ChunkType
    source_file: str
    chunk_index: int
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        fingerprint = f"{self.source_file}::{self.chunk_index}::{self.text}"
        self.chunk_id = str(uuid.UUID(hashlib.md5(fingerprint.encode()).hexdigest()))


@dataclass(frozen=True)
class RetrievedChunk:
    """
    An immutable view of a chunk returned by the retriever, enriched with
    the relevance `score` assigned by Qdrant.
    """

    text: str
    heading: str
    parent_heading: str
    chunk_type: str
    source_file: str
    chunk_idx: int
    score: float