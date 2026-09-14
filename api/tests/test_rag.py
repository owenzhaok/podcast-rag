from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import api.main as main
from api.models import SearchResponse
from api.rag.config import RagConfig


@pytest.fixture(autouse=True)
def clean_rag_env(monkeypatch):
    for name in (
        "RAG_ENABLED", "RAG_LLM_PROVIDER", "RAG_LLM_MODEL", "RAG_LLM_API_KEY",
        "RAG_EMBEDDING_PROVIDER", "RAG_EMBEDDING_MODEL", "RAG_EMBEDDING_API_KEY",
        "RAG_SOURCE_INDEX", "RAG_CANDIDATE_LIMIT", "RAG_MAX_SOURCES", "RAG_CONTEXT_MAX_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("value", [None, "false", "invalid", "true"])
@pytest.mark.asyncio
async def test_startup_without_optional_credentials(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("RAG_ENABLED", value)
    es, pool, redis = AsyncMock(), AsyncMock(), AsyncMock()
    es.search.return_value = {"hits": {"hits": []}}
    # Restore existing globals after exercising the real production lifespan.
    for name in ("_es_client", "_db_pool", "_cache_client"):
        monkeypatch.setattr(main, name, None)
    with (
        patch("elasticsearch.AsyncElasticsearch", return_value=es),
        patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=pool),
        patch("redis.asyncio.Redis.from_url", return_value=redis),
    ):
        app = main.create_app(use_lifespan=True)
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                health = await client.get("/health")
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                response = await client.post("/ask", json={"question": "What is discussed?"})
                assert response.status_code == 200
                expected = "insufficient_context" if value == "true" else "disabled"
                assert response.json()["status"] == expected
            if value == "true":
                es.search.assert_awaited_once()
            else:
                es.search.assert_not_called()
            pool.fetchrow.assert_not_called()
            redis.get.assert_not_called()
            redis.set.assert_not_called()
        es.close.assert_awaited_once()
        pool.close.assert_awaited_once()
        redis.aclose.assert_awaited_once()


@pytest.mark.parametrize("enabled,configured,reason", [
    ("false", False, "rag_disabled"),
])
@pytest.mark.asyncio
async def test_ask_unavailable_contract_and_no_io(monkeypatch, enabled, configured, reason):
    monkeypatch.setenv("RAG_ENABLED", enabled)
    if configured:
        for name in ("RAG_LLM_PROVIDER", "RAG_LLM_MODEL", "RAG_LLM_API_KEY"):
            monkeypatch.setenv(name, "test-placeholder")
    es, pool, cache = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(main, "_es_client", es)
    monkeypatch.setattr(main, "_db_pool", pool)
    monkeypatch.setattr(main, "_cache_client", cache)
    # ASGI requests need no TCP; prohibit accidental provider network calls.
    with patch("socket.socket.connect", side_effect=AssertionError("Unexpected network call")):
        app = main.create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/ask", json={"question": "  What is discussed?  "})
    assert response.status_code == 200
    assert response.json() == {
        "question": "What is discussed?",
        "status": "disabled" if enabled == "false" else "generation_unavailable",
        "reason": reason,
        "answer": None,
        "sources": [],
        "retrieval_mode": None,
        "degraded": False,
        "cached": False,
        "took_ms": 0,
    }
    assert "test-placeholder" not in response.text
    assert not es.mock_calls and not pool.mock_calls and not cache.mock_calls


@pytest.mark.parametrize("body", [
    {}, {"question": ""}, {"question": " \n\t "},
    {"question": None}, {"question": 123}, {"question": "x" * 2001},
])
@pytest.mark.asyncio
async def test_ask_rejects_invalid_question(body):
    app = main.create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ask", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize("enabled", ["false", "true"])
@pytest.mark.asyncio
async def test_existing_search_and_health_unchanged(monkeypatch, enabled):
    monkeypatch.setenv("RAG_ENABLED", enabled)
    expected = SearchResponse(query="test", total=0, clips=[], took_ms=5)
    with patch("api.main.execute_search", new_callable=AsyncMock, return_value=expected) as search:
        app = main.create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/search", params={"q": "test", "from": 10, "size": 5})
            assert response.status_code == 200
            assert response.json() == expected.model_dump()
            search.assert_awaited_once_with(
                q="test", from_=10, size=5,
                es=main._es_client, db_pool=main._db_pool, cache=main._cache_client,
            )
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}


def test_config_presence_and_secret_not_retained(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", " TRUE ")
    monkeypatch.setenv("RAG_LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("RAG_LLM_MODEL", "test-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", " ")
    assert RagConfig.from_env() == RagConfig(enabled=True, llm_configured=False,
                                           llm_provider="test-provider", llm_model="test-model")
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-placeholder")
    config = RagConfig.from_env()
    assert config == RagConfig(enabled=True, llm_configured=True,
                               llm_provider="test-provider", llm_model="test-model")
    assert "test-placeholder" not in repr(config)
