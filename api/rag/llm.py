"""Provider-neutral request boundary and safe failure codes."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    system: str
    user: str
    model: str
    max_output_tokens: int
    timeout_seconds: float


class LLMProvider(Protocol):
    async def generate(self, request: GenerationRequest) -> str:
        """Return structured JSON text, never a provider-specific envelope."""
        ...


class ProviderFailure(Exception):
    ALLOWED = {"provider_timeout", "authentication_failed", "rate_limited", "quota_exceeded",
               "provider_unavailable", "provider_request_rejected"}

    def __init__(self, reason="provider_unavailable"):
        self.reason = reason if reason in self.ALLOWED else "provider_unavailable"
        super().__init__(self.reason)


class InvalidGeneration(Exception):
    ALLOWED = {"malformed_provider_response", "invalid_structured_output", "invalid_citations"}

    def __init__(self, reason="invalid_structured_output"):
        self.reason = reason if reason in self.ALLOWED else "invalid_structured_output"
        super().__init__(self.reason)
