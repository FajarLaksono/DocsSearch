from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

class Settings:
    """Typed, validated-on-read settings."""

    def __init__(self) -> None:
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
        self.judge_model: str = os.getenv("JUDGE_MODEL", "llama3.2:3b")
        self.embed_model: str = os.getenv("EMBED_MODEL", "all-minilm")
        self.embed_dim: int = int(os.getenv("EMBED_DIM", "384"))

        self.thinking: str = os.getenv("THINKING", "off").lower()
        self.num_ctx: int = int(os.getenv("NUM_CTX", "4096"))

        self.top_k: int = int(os.getenv("TOP_K", "4"))
        self.min_top_score: float = float(os.getenv("MIN_TOP_SCORE", "0.35"))
        self.chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "300"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

        self.ollama_base_url: str = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/")
        self.database_url: str = os.getenv(
            "DATABASE_URL", "postgresql://aidocsuser:aidocspassword@localhost:5432/aidocs"
        )


settings = Settings()