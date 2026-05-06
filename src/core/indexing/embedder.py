from __future__ import annotations

from typing import TypedDict

from FlagEmbedding import BGEM3FlagModel

from src.config.settings import settings


# ---------------------------------------------------------------------------
# Typed result container
# ---------------------------------------------------------------------------

class EmbeddingResult(TypedDict):
    """
    Dual-vector representation produced by BGE-M3.

    Attributes:
        dense:  1 024-dimensional float vector for ANN search.
        sparse: Token-id → weight mapping for exact lexical matching.
    """

    dense: list[float]
    sparse: dict[int, float]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class BGEEmbedder:
    """
    Thin wrapper around ``BAAI/bge-m3`` that returns both dense and sparse
    (lexical-weight) vectors required for hybrid search.

    Args:
        model_name: HuggingFace model identifier.
        use_fp16:   Run inference in FP16 to halve VRAM usage with negligible
                    accuracy loss.
        batch_size: Number of texts encoded per forward pass.

    Example::

        embedder = BGEEmbedder()
        results = embedder.embed_batch(["text one", "text two"])
        results[0]["dense"]   # list[float], dim=1024
        results[0]["sparse"]  # dict[int, float]
    """

    DENSE_DIM: int = settings.dense_dim

    def __init__(
            self, 
            model_name: str = settings.bge_model_name,
            use_fp16: bool = True,
            batch_size: int = settings.embed_batch_size
    ) -> None:
        self.batch_size = batch_size
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_batch(self, text: list[str]) -> list[EmbeddingResult]:
        """Encode a batch of *texts* and return dual-vector results."""
        output = self._model.encode(
            text,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=settings.max_seq_length
        )
        return [
            EmbeddingResult(
                dense=dense.tolist(),
                sparse={int(k): float(v) for k, v in sparse.items()}
            )
            for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"])
        ]
    
    def embed_query(self, text:str) -> EmbeddingResult:
        """Convenience wrapper — encode a single query string."""
        return self.embed_batch([text])[0]