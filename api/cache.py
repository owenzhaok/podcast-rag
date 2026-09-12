"""Async Redis cache wrapper for search responses."""

import hashlib
from redis.asyncio import Redis
from api.models import SearchResponse


class CacheClient:
    def __init__(self, redis: Redis):
        self._redis = redis

    def cache_key(self, q: str, from_: int, size: int) -> str:
        """SHA256 of (q, from_, size) truncated to 16 hex chars."""
        raw = f"{q}:{from_}:{size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def get(self, key: str) -> SearchResponse | None:
        """Return deserialized SearchResponse or None on miss."""
        data = await self._redis.get(key)
        if data is None:
            return None
        return SearchResponse.model_validate_json(data)

    async def set(self, key: str, value: SearchResponse, ttl_s: int = 3600) -> None:
        """Serialize and store with TTL."""
        await self._redis.set(key, value.model_dump_json(), ex=ttl_s)
