"""Optional RAG settings; loading them never initializes a provider."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    enabled: bool = False
    llm_configured: bool = False

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
        return cls(enabled=enabled, llm_configured=configured)
