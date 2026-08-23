"""
Centralized configuration for the Personal Knowledge Assistant.

All tunable parameters are exposed as environment variables so the system
can be reconfigured without touching code (see .env.example).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Paths -----------------------------------------------------------------
    data_dir: Path = Field(default=Path("./data"))
    upload_dir: Path = Field(default=Path("./data/uploads"))
    vector_db_path: Path = Field(default=Path("./data/chroma"))
    sqlite_path: Path = Field(default=Path("./data/sqlite/pka.db"))

    # --- Embedding model ---------------------------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")  # "cpu" | "cuda" | "mps"

    # --- Semantic chunking ---------------------------------------------------
    chunk_similarity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # threshold is interpreted as: split when cosine similarity between
    # consecutive sentences drops BELOW this value (i.e. a topic change).
    chunk_min_size_tokens: int = Field(default=80)
    chunk_max_size_tokens: int = Field(default=400)
    chunk_overlap_tokens: int = Field(default=30)
    prefer_paragraph_boundaries: bool = Field(default=True)

    # baseline chunkers, used for comparison / evaluation
    fixed_chunk_size_tokens: int = Field(default=250)
    fixed_chunk_overlap_tokens: int = Field(default=30)
    recursive_chunk_size_tokens: int = Field(default=250)
    recursive_chunk_overlap_tokens: int = Field(default=30)

    # --- Retrieval -----------------------------------------------------------
    top_k: int = Field(default=5)
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = Field(default="hybrid")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # weight on vector score

    # --- Reranking -------------------------------------------------------------
    use_reranker: bool = Field(default=False)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_top_n: int = Field(default=20)

    # --- Summarization ---------------------------------------------------------
    enable_hierarchical_summarization: bool = Field(default=True)
    summary_max_tokens: int = Field(default=200)

    # --- Ollama / LLM ------------------------------------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:1b")
    ollama_temperature: float = Field(default=0.1)
    llm_context_max_tokens: int = Field(default=4000)

    # --- App -------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(
        default=(
            "http://127.0.0.1:5500,"
            "http://localhost:5500,"
            "http://localhost:5173,"
            "http://127.0.0.1:5173,"
            "http://localhost:3000,"
            "http://127.0.0.1:3000"
        )
    )
    def ensure_dirs(self) -> None:
        for p in [self.data_dir, self.upload_dir, self.vector_db_path, self.sqlite_path.parent]:
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
