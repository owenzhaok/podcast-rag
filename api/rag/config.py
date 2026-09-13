"""Optional RAG settings; loading them never initializes a provider."""

import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    enabled: bool = False
    llm_configured: bool = False
    source_index: str = "podcast_clips"
    candidate_limit: int = 30
    max_sources: int = 6
    context_max_bytes: int = 16000

    @staticmethod
    def _bounded_int(name: str, default: int, maximum: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
            return value if 1 <= value <= maximum else default
        except ValueError:
            return default

    @classmethod
    def from_env(cls) -> "RagConfig":
        # Unknown values fail closed without preventing ordinary API startup.
        enabled = os.environ.get("RAG_ENABLED", "false").strip().lower() in {
            "true", "1", "yes", "on",
        }
        configured = all(
            os.environ.get(name, "").strip()
            for name in ("RAG_LLM_PROVIDER", "RAG_LLM_MODEL", "RAG_LLM_API_KEY")
        )
        # Retain only presence, never the credential itself.
        index = os.environ.get("RAG_SOURCE_INDEX", "podcast_clips")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,254}", index):
            index = "podcast_clips"
        return cls(
            enabled=enabled, llm_configured=configured, source_index=index,
            candidate_limit=cls._bounded_int("RAG_CANDIDATE_LIMIT", 30, 200),
            max_sources=cls._bounded_int("RAG_MAX_SOURCES", 6, 20),
            context_max_bytes=cls._bounded_int("RAG_CONTEXT_MAX_BYTES", 16000, 64000),
        )
