from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration loaded from environment variables or .env file.

    Sections:
        - Chunker: controls token-budget and table-detection thresholds.
        - Embedder: BAAI/bge-m3 model parameters.
        - Retrieval: prefetch and vector-dimension settings.
        - Qdrant: connection parameters for the vector store.
    """

    # ------------------------------------------------------------------
    # Chunker
    # ------------------------------------------------------------------
    chars_per_token: int = 2
    table_threshold: float = 0.8
    mixed_threshold: float = 0.2
    default_max_tokens: int = 1800

    # ------------------------------------------------------------------
    # Embedder
    # ------------------------------------------------------------------
    bge_model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 8192
    embed_batch_size: int = 16

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    prefetch_k: int = 40
    dense_dim: int = 1024

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "optimacros_docs"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()