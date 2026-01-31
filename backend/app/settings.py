"""
Configuration management for the application using Pydantic.
Handles loading environment variables and defining default settings.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration for the RAG service."""

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    pinecone_api_key: str = Field(..., alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(..., alias="PINECONE_INDEX_NAME")

    data_dir: Path = Field(Path("Word"), alias="DATA_DIR")

    # Pinecone Models (Serverless Inference)
    pinecone_embedding_model: str = Field("multilingual-e5-large", alias="PINECONE_EMBEDDING_MODEL")
    pinecone_rerank_model: str = Field("bge-reranker-v2-m3", alias="PINECONE_RERANK_MODEL")
    pinecone_namespace: str = Field("nom_sense", alias="PINECONE_NAMESPACE")

    chat_model: str = Field("gpt-4o-mini", alias="CHAT_MODEL")

    chunk_size: int = Field(1600, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(300, alias="CHUNK_OVERLAP")
    retriever_k: int = Field(25, alias="RETRIEVER_K")
    rerank_top_k: int = Field(4, alias="RERANK_TOP_K")

    serve_docs: bool = Field(True, alias="SERVE_DOCS")
    auto_ingest_on_startup: bool = Field(False, alias="AUTO_INGEST_ON_STARTUP")
    docs_mount_path: str = Field("/docs", alias="DOCS_MOUNT_PATH")
    allowed_origins: list[str] = Field(["*"], alias="ALLOWED_ORIGINS")

    model_config = SettingsConfigDict(
        env_file=(".env", "config/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir

    def ensure_env(self) -> None:
        os.environ.setdefault("OPENAI_API_KEY", self.openai_api_key)
        os.environ.setdefault("PINECONE_API_KEY", self.pinecone_api_key)
        # Ensure Pinecone client can find the key if it looks for env var


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_env()
    return settings
