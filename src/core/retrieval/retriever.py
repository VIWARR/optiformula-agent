from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    Prefetch,
    ScoredPoint,
    SparseVector,
)

from src.config import settings
from src.core.indexing.embedder import BGEEmbedder
from src.core.schema import ChunkType, RetrievedChunk
from src.utils.logger import setup_logger

logger = setup_logger("retriever")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid dense+sparse retriever backed by Qdrant with RRF fusion.

    Two prefetch queries are issued in parallel — one against the ``dense``
    (cosine ANN) vector space and one against the ``sparse`` (lexical-weight)
    space — and the results are re-ranked using Reciprocal Rank Fusion.

    Args:
        client:     Authenticated :class:`QdrantClient` instance.
        embedder:   :class:`BGEEmbedder` used to vectorise incoming queries.
        prefetch_k: Number of candidates fetched per vector space before
                    fusion.  Higher values improve recall at the cost of
                    latency.

    Example::

        retriever = HybridRetriever(client, embedder)
        chunks = retriever.search("как работает функция SUMIF", top_k=5)
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: BGEEmbedder,
        prefetch_k: int = settings.prefetch_k,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection_name = settings.qdrant_collection_name
        self.prefetch_k = prefetch_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type_filter: Optional[ChunkType] = None,
    ) -> List[RetrievedChunk]:
        """
        Run a hybrid search and return the top *top_k* chunks.

        Args:
            query:             Natural-language query string.
            top_k:             Maximum number of results to return.
            chunk_type_filter: When set, restrict results to chunks whose
                               ``metadata.chunk_type`` matches this value.

        Returns:
            List of :class:`RetrievedChunk`, sorted by descending RRF score.
            Returns an empty list on :class:`~qdrant_client.http.exceptions.UnexpectedResponse`.
        """
        try:
            emb = self.embedder.embed_query(query)
            query_filter = self._build_filter(chunk_type_filter)

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=NamedVector(name="dense", vector=emb["dense"]),
                        limit=self.prefetch_k,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=NamedSparseVector(
                            name="sparse",
                            vector=SparseVector(
                                indices=list(emb["sparse"].keys()),
                                values=list(emb["sparse"].values()),
                            ),
                        ),
                        limit=self.prefetch_k,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return [self._map_to_chunk(p) for p in results.points]

        except exceptions.UnexpectedResponse as exc:
            logger.error("Qdrant search error: %s", exc)
            return []
        except Exception as exc:
            logger.error("Unexpected retrieval error: %s", exc)
            raise

    def search_dense_only(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Dense-only ANN search.

        Useful for ablation experiments and latency-sensitive scenarios where
        lexical matching is not required.
        """
        emb = self.embedder.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=NamedVector(name="dense", vector=emb["dense"]),
            limit=top_k,
            with_payload=True,
        )
        return [self._map_to_chunk(p) for p in results.points]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(chunk_type: Optional[ChunkType]) -> Optional[Filter]:
        """Construct a Qdrant payload filter for *chunk_type*, or ``None``."""
        if not chunk_type:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="metadata.chunk_type",
                    match=MatchValue(value=chunk_type),
                )
            ]
        )

    @staticmethod
    def _map_to_chunk(point: ScoredPoint) -> RetrievedChunk:
        """Map a raw :class:`ScoredPoint` to a domain :class:`RetrievedChunk`."""
        payload = point.payload or {}
        meta = payload.get("metadata", {})
        return RetrievedChunk(
            text=payload.get("text", ""),
            heading=meta.get("heading", "N/A"),
            parent_heading=meta.get("parent_heading", "N/A"),
            chunk_type=meta.get("chunk_type", "text"),
            source_file=meta.get("source", "unknown"),
            chunk_idx=meta.get("chunk_idx", 0),
            score=point.score,
        )