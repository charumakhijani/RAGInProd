from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

def load_env(env_path: str = ".env") -> None:
    load_dotenv(env_path, override=False)

@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_rerank_model: str
    openai_embed_model: str

    reranker_type: str
    cross_encoder_model: str

    opensearch_host: str
    opensearch_user: str | None
    opensearch_pass: str | None
    opensearch_index: str

    chroma_persist_dir: str
    chroma_collection: str

    chunk_max_tokens: int
    chunk_overlap_tokens: int
    retr_k: int
    rerank_top_n: int
    max_context_tokens: int
    per_chunk_max_tokens: int

    otlp_endpoint: str | None
    service_name: str

def get_settings() -> Settings:
    def env(name: str, default: str | None = None) -> str:
        v = os.getenv(name, default)
        if v is None:
            raise RuntimeError(f"Missing required env var: {name}")
        return v

    def env_int(name: str, default: int) -> int:
        v = os.getenv(name)
        return int(v) if v is not None and v.strip() != "" else default

    return Settings(
        openai_api_key=env("OPENAI_API_KEY"),
        openai_model=env("OPENAI_MODEL", "gpt-5.2"),
        openai_rerank_model=env("OPENAI_RERANK_MODEL", env("OPENAI_MODEL", "gpt-5.2")),
        openai_embed_model=env("OPENAI_EMBED_MODEL", "text-embedding-3-large"),

        reranker_type=env("RERANKER_TYPE", "openai").lower(),
        cross_encoder_model=env("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),

        opensearch_host=env("OPENSEARCH_HOST", "http://localhost:9200"),
        opensearch_user=os.getenv("OPENSEARCH_USER") or None,
        opensearch_pass=os.getenv("OPENSEARCH_PASS") or None,
        opensearch_index=env("OPENSEARCH_INDEX", "rag_chunks"),

        chroma_persist_dir=env("CHROMA_PERSIST_DIR", "./chroma"),
        chroma_collection=env("CHROMA_COLLECTION", "rag_chunks"),

        chunk_max_tokens=env_int("CHUNK_MAX_TOKENS", 350),
        chunk_overlap_tokens=env_int("CHUNK_OVERLAP_TOKENS", 40),
        retr_k=env_int("RETR_K", 80),
        rerank_top_n=env_int("RERANK_TOP_N", 20),
        max_context_tokens=env_int("MAX_CONTEXT_TOKENS", 1200),
        per_chunk_max_tokens=env_int("PER_CHUNK_MAX_TOKENS", 350),

        otlp_endpoint=os.getenv("OTLP_ENDPOINT") or None,
        service_name=env("SERVICE_NAME", "prod-rag"),
    )
