from __future__ import annotations

from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions
from qdrant_client.models import (
    Distance,
    OptimizersConfigDiff,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config.settings import settings
from src.core.indexing.chunker import Chunk
from src.core.indexing.embedder import EmbeddingResult
from src.utils.logger import setup_logger

logger = setup_logger("indexer")


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class QdrantIndexer:
    """
    Manages the collection lifecycle and bulk ingestion for Qdrant.

    Supports hybrid search via parallel dense (cosine ANN) and sparse
    (BM25-style) vector spaces fused at query time with RRF.

    Args:
        client: Authenticated :class:`QdrantClient` instance.

    Notes:
        All vectors and payloads are stored on disk (``on_disk=True``) to
        avoid exhausting memory when the collection grows large.
    """

    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection_name = settings.qdrant_collection_name

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        """
        Create the collection if it does not already exist.

        Args:
            recreate: When ``True``, delete the existing collection first.
                      **Destructive** — use only during re-indexing.
        """
        if recreate:
            self.delete_collection()
        
        try:
            if self.client.collection_exists(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists.")
                return
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=settings.dense_dim,
                        distance=Distance.COSINE,
                        on_disk=True
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=True,
                            full_scan_threshold=1000
                        )
                    )
                },
                optimizers_config=OptimizersConfigDiff(memmap_threshold=2000)
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the collection if it exists.  This action is irreversible."""
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                logger.warning(f"Collection '{self.collection_name}' has been deleted.")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert_batch(self, chunks: List[Chunk], embeddings: List[EmbeddingResult]) -> None:
        """
        Upsert a batch of *chunks* together with their pre-computed *embeddings*.

        Chunk IDs are deterministic (see :class:`Chunk`), so repeated calls
        with the same data are idempotent.

        Args:
            chunks:     Source chunks produced by :class:`MarkdownChunker`.
            embeddings: Parallel list of dual-vector results from
                        :class:`BGEEmbedder`.

        Raises:
            ValueError: When ``len(chunks) != len(embeddings)``.
        """
        if len(chunks) != len(embeddings):
            error_msg = f"Inconsistent data: {len(chunks)} chunks != {len(embeddings)} embeddings"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector={
                    "dense": emb["dense"],
                    "sparse": SparseVector(
                        indices=list(emb["sparse"].keys()),
                        values=list(emb["sparse"].values())
                    )
                },
                payload={
                    "text": chunk.text,
                    "metadata": {
                        "heading": chunk.heading,
                        "parent_heading": chunk.parent_heading,
                        "chunk_type": chunk.chunk_type,
                        "source": chunk.source_file,
                        "chunk_idx": chunk.chunk_index,
                    }
                }
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        except exceptions.UnexpectedResponse as e:
            logger.error(f"Qdrant upsert failed: {e}")
            raise