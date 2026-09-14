"""API tests never use ambient LLM credentials or real HTTPX network transports."""

import os
import httpx
import pytest


@pytest.fixture(autouse=True)
def isolate_llm(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RAG_LLM_") or name in (
            "RAG_CONTEXT_MAX_TOKENS", "RAG_MAX_OUTPUT_TOKENS", "RAG_ANSWER_CACHE_TTL_SECONDS",
        ):
            monkeypatch.delenv(name, raising=False)

    attempts = []

    async def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Real HTTPX network requests are forbidden in API tests")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *args, **kwargs: pytest.fail(
        "Real HTTPX network requests are forbidden in API tests"))
    yield
    assert not attempts, "A test attempted a real HTTPX network request"
