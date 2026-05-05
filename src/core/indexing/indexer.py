from __future__ import annotations
from typing import List, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    OptimizersConfigDiff
)

from src.config import settings
from src.core.indexing.chunker import Chunk
from src.core.indexing.embedder import EmbeddingResult
from src.utils.logger import setup_logger

logger = setup_logger("indexer")


class QdrantIndexer:
    """
    A class for managing the lifecycle of a collection and indexing data in Quadrant.
    Implements support for hybrid search (Dense + Sparse).
    """

    def __init__(self, client: QdrantClient):
        self.client = client
        self.collection_name = settings.qdrant_collection_name

    def ensure_collection(self, recreate: bool = False) -> None:
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

    def upsert_batch(self, chunks: List[Chunk], embeddings: List[EmbeddingResult]) -> None:
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

    def delete_collection(self) -> None:
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                logger.warning(f"Collection '{self.collection_name}' has been deleted.")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise