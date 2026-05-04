from __future__ import annotations
from typing import TypedDict
from FlagEmbedding import BGEM3FlagModel


class EmbeddingResult(TypedDict):
    dense: list[float]
    sparse: dict[int, float]


class BGEmbedder:
    """
    Wraps BAAI/bge-m3 for dual-vector output.

    Usage:
        embedder = BGEEmbedder()
        results = embedder.embed_batch(["текст 1", "текст 2"])
        results[0]["dense"]   # list[float], dim=1024
        results[0]["sparse"]  # dict[int, float]
    """

    DENSE_DIM = 1024

    def __init__(
            self, 
            model_name: str = "BAAI/bge-m3",
            use_fp16: bool = True,
            batch_size: int = 16
    ) -> None:
        self.batch_size = batch_size
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed_batch(self, text: list[str]) -> list[EmbeddingResult]:
        output = self._model.encode(
            text,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=8192
        )
        return [
            EmbeddingResult(
                dense=dense.tolist(),
                sparse={int(k): float(v) for k, v in sparse.items()}
            )
            for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"])
        ]
    
    def embed_query(self, text:str) -> EmbeddingResult:
        """Single query — convenience wrapper around embed_batch."""
        return self.embed_batch([text])[0]