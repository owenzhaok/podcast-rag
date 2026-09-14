import json
import os
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis

import api.main as main
import api.rag.cache as cache_module
from api.cache import CacheClient
from api.rag.cache import GenerationCache, generation_key
from api.rag.config import RagConfig
from api.rag.generation import generate_answer
from api.rag.groq import get_generation_provider
from api.rag.llm import GenerationRequest, ProviderFailure
from api.tests.fake_llm import FakeProvider
from api.tests.test_rag_generation import answer, sources


@pytest_asyncio.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_miss_hit_payload_ttl_and_current_sources(redis):
    fake = FakeProvider(answer())
    config = replace(RagConfig(), answer_cache_ttl_seconds=123)
    first = await generate_answer("question", sources(), config, fake, redis)
    assert first.status == "answered" and not first.cached
    key = generation_key(fake.requests[0], config.llm_provider)
    assert 0 < await redis.ttl(key) <= 123
    assert set(json.loads(await redis.get(key))) == {"status", "paragraphs"}
    # Audio URL is not sent to the model; response must use today's source data.
    current = sources()
    current[0].audio_link = "https://example.com/current"
    second = await generate_answer("question", current, config, fake, redis)
    assert second.cached and second.answer == first.answer and second.sources == current
    assert len(fake.requests) == 1
    assert await redis.ttl(key) <= 123


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["question", "context", "order", "model", "provider", "output", "system", "version"])
async def test_generation_identity_changes_miss(redis, monkeypatch, change):
    config, evidence, question = RagConfig(), sources(), "question"
    fake = FakeProvider(answer())
    await generate_answer(question, evidence, config, fake, redis)
    if change == "question":
        question = "new question"
    elif change == "context":
        evidence[0].excerpt = "Changed transcript"
    elif change == "order":
        evidence.reverse()
    elif change == "model":
        config = replace(config, llm_model="another-model")
    elif change == "provider":
        config = replace(config, llm_provider="another-provider")
    elif change == "output":
        config = replace(config, max_output_tokens=1600)
    elif change == "system":
        monkeypatch.setattr("api.rag.generation.SYSTEM", "New application instructions")
    else:
        monkeypatch.setattr(cache_module, "GENERATION_VERSION", "next-version")
    result = await generate_answer(question, evidence, config, fake, redis)
    assert not result.cached and len(fake.requests) == 2


@pytest.mark.asyncio
async def test_cached_abstention(redis):
    fake = FakeProvider()
    first = await generate_answer("question", sources(), RagConfig(), fake, redis)
    second = await generate_answer("question", sources(), RagConfig(), fake, redis)
    assert not first.cached and second.cached and len(fake.requests) == 1
    assert second.status == "insufficient_context" and second.reason == "model_abstained"
    assert second.answer is None and second.sources == sources()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [b"not json", b"\xff", pytest.param(b"x" * 65537, id="oversized"),
    '{"status":"answered","paragraphs":[]}',
    '{"status":"answered","paragraphs":[{"text":"Fact","source_ids":[]}]}',
    '{"status":"answered","paragraphs":[{"text":"Fact","source_ids":["S9"]}]}',
])
async def test_corrupt_or_invalid_cached_output_is_miss(redis, bad):
    fake = FakeProvider(answer())
    await generate_answer("question", sources(), RagConfig(), fake, redis)
    key = generation_key(fake.requests[0], "")
    await redis.set(key, bad, ex=900)
    result = await generate_answer("question", sources(), RagConfig(), fake, redis)
    assert result.status == "answered" and not result.cached and len(fake.requests) == 2


@pytest.mark.asyncio
async def test_cache_hit_revalidates_current_citation_membership(redis):
    fake = FakeProvider(answer())
    await generate_answer("question", sources(), RagConfig(), fake, redis)
    key = generation_key(fake.requests[0], "")
    cache = GenerationCache(redis, 900)
    assert await cache.get(key, frozenset({"S1", "S2"})) is not None
    assert await cache.get(key, frozenset({"S1"})) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set"])
async def test_redis_failure_fails_open(operation):
    redis = AsyncMock()
    redis.get.return_value = None
    getattr(redis, operation).side_effect = RuntimeError("private redis credentials")
    result = await generate_answer("question", sources(), RagConfig(), FakeProvider(answer()), redis)
    assert result.status == "answered" and not result.cached
    assert "private" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_slow_redis_is_bounded(monkeypatch):
    import asyncio
    async def slow(*args, **kwargs):
        await asyncio.sleep(10)
    redis = AsyncMock()
    redis.get.side_effect = slow
    redis.set.side_effect = slow
    monkeypatch.setattr(cache_module, "CACHE_IO_TIMEOUT_SECONDS", 0.001)
    result = await generate_answer("question", sources(), RagConfig(), FakeProvider(answer()), redis)
    assert result.status == "answered" and not result.cached


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["provider", "invalid", "citations", "budget", "empty", "unconfigured", "unsupported", "disabled"])
async def test_noncacheable_states(failure):
    redis = AsyncMock()
    redis.get.return_value = None
    config, evidence, fake = RagConfig(), sources(), FakeProvider(answer())
    if failure == "provider": fake.error = ProviderFailure("provider_timeout")
    if failure == "invalid": fake.output = "invalid JSON"
    if failure == "citations": fake.output = answer([{"text": "Fact", "source_ids": ["S9"]}])
    if failure == "budget": config = replace(config, context_max_tokens=1)
    if failure == "empty": evidence = []
    if failure in ("unconfigured", "unsupported"):
        fake = None
        if failure == "unsupported": config = replace(config, llm_provider="unsupported")
    if failure == "disabled":
        from api.rag.routes import create_router
        from fastapi import FastAPI
        app = FastAPI()
        app.state.rag_config = config
        app.include_router(create_router(config, get_redis=lambda: redis))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/ask", json={"question": "question"})).json()["status"] == "disabled"
    else:
        result = await generate_answer("question", evidence, config, fake, redis)
        assert not result.cached and result.answer is None
    redis.set.assert_not_called()
    if failure in ("budget", "empty", "unconfigured", "unsupported", "disabled"):
        redis.get.assert_not_called()


@pytest.mark.asyncio
async def test_zero_ttl_disables_all_cache_io():
    redis, fake = AsyncMock(), FakeProvider(answer())
    for _ in range(2):
        result = await generate_answer("question", sources(), replace(RagConfig(), answer_cache_ttl_seconds=0), fake, redis)
        assert result.status == "answered" and not result.cached
    assert len(fake.requests) == 2
    redis.get.assert_not_called()
    redis.set.assert_not_called()


@pytest.mark.parametrize("value,expected", [(None, 900), ("0", 0), ("30", 30), ("-1", 900), ("bad", 900)])
def test_ttl_configuration(monkeypatch, value, expected):
    if value is not None: monkeypatch.setenv("RAG_ANSWER_CACHE_TTL_SECONDS", value)
    assert RagConfig.from_env().answer_cache_ttl_seconds == expected


def test_key_is_deterministic_namespaced_and_secret_free(monkeypatch):
    monkeypatch.setenv("RAG_LLM_API_KEY", "test-secret-never-in-key")
    request = GenerationRequest("system", "raw-question raw-transcript", "model", 800, 20)
    key = generation_key(request, "groq")
    assert key == generation_key(request, "groq")
    assert key.startswith("rag:answer:v1:") and len(key.removeprefix("rag:answer:v1:")) == 64
    assert key != CacheClient(None).cache_key("raw-question", 0, 10)
    for raw in ("raw-question", "raw-transcript", "test-secret-never-in-key"):
        assert raw not in key
    monkeypatch.setenv("RAG_LLM_API_KEY", "rotated-test-secret")
    assert generation_key(request, "groq") == key


async def route_roundtrip(redis, monkeypatch, question):
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setattr(main, "_cache_client", CacheClient(redis))
    evidence = sources()
    retrieval = AsyncMock(return_value=evidence)
    monkeypatch.setattr("api.rag.routes.retrieve", retrieval)
    fake = FakeProvider(answer())
    app = main.create_app()
    app.dependency_overrides[get_generation_provider] = lambda: fake
    key = None
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = (await client.post("/ask", json={"question": question})).json()
            key = generation_key(fake.requests[0], "")
            evidence[0].audio_link = "https://example.com/current"
            second = (await client.post("/ask", json={"question": question})).json()
        assert first["status"] == second["status"] == "answered"
        assert not first["cached"] and second["cached"]
        assert second["sources"][0]["audio_link"] == evidence[0].audio_link
        assert len(fake.requests) == 1 and retrieval.await_count == 2
        assert 0 < await redis.ttl(key) <= 900
    finally:
        if key is not None: await redis.delete(key)  # Only this test's exact key.


@pytest.mark.asyncio
async def test_route_retrieves_on_every_request(redis, monkeypatch):
    await route_roundtrip(redis, monkeypatch, "question")


@pytest.mark.skipif(os.environ.get("RAG_INTEGRATION_TESTS") != "1", reason="Opt-in Docker Redis test")
@pytest.mark.asyncio
async def test_docker_redis_generation_cache(monkeypatch):
    redis = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    try:
        await route_roundtrip(redis, monkeypatch, "cache integration " + uuid4().hex)
    finally:
        await redis.aclose()
