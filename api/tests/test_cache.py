import pytest
import pytest_asyncio
import fakeredis.aioredis
from api.cache import CacheClient
from api.models import SearchResponse


def _make_response(**kwargs) -> SearchResponse:
    defaults = {"query": "test", "total": 0, "clips": [], "took_ms": 10}
    defaults.update(kwargs)
    return SearchResponse(**defaults)


@pytest_asyncio.fixture
async def cache():
    redis = fakeredis.aioredis.FakeRedis()
    client = CacheClient(redis)
    yield client
    await redis.aclose()


class TestCacheKey:
    def test_cache_key_is_deterministic(self, cache):
        k1 = cache.cache_key("hello", 0, 10)
        k2 = cache.cache_key("hello", 0, 10)
        assert k1 == k2

    def test_cache_key_differs_for_different_queries(self, cache):
        k1 = cache.cache_key("hello", 0, 10)
        k2 = cache.cache_key("world", 0, 10)
        assert k1 != k2


class TestCacheGetSet:
    @pytest.mark.asyncio
    async def test_set_and_get_round_trips(self, cache):
        resp = _make_response(query="physics", total=5, took_ms=42)
        key = cache.cache_key("physics", 0, 10)
        await cache.set(key, resp)
        got = await cache.get(key)
        assert got is not None
        assert got.query == "physics"
        assert got.total == 5
        assert got.took_ms == 42

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, cache):
        result = await cache.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_is_set(self, cache):
        resp = _make_response()
        key = cache.cache_key("test", 0, 10)
        await cache.set(key, resp, ttl_s=3600)
        ttl = await cache._redis.ttl(key)
        assert ttl > 0
        assert ttl <= 3600
