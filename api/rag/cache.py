"""Generation-only cache on the existing Redis connection; all I/O fails open."""

import asyncio
import hashlib
import json
from dataclasses import asdict

from api.rag.citations import GeneratedOutput, validate_output
from api.rag.llm import GenerationRequest


# Bump when the provider wire schema or fixed generation settings change.
GENERATION_VERSION = "strict-grounded-answer-v1"
CACHE_IO_TIMEOUT_SECONDS = 1.0


def generation_key(request: GenerationRequest, provider: str) -> str:
    identity = {
        "request": asdict(request), "provider": provider,
        "generation_version": GENERATION_VERSION,
        "validation_schema": GeneratedOutput.model_json_schema(),
    }
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "rag:answer:v1:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class GenerationCache:
    def __init__(self, redis, ttl_seconds: int):
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def get(self, key: str, source_ids: frozenset[str]) -> GeneratedOutput | None:
        if self.redis is None or self.ttl_seconds <= 0:
            return None
        try:
            raw = await asyncio.wait_for(self.redis.get(key), CACHE_IO_TIMEOUT_SECONDS)
            if isinstance(raw, bytes):
                if len(raw) > 65536:
                    return None
                raw = raw.decode("utf-8")
            return validate_output(raw, source_ids)
        except Exception:
            # Missing, corrupt, invalid or unreachable entries are misses. No deletion
            # is needed: a valid generation overwrites them, otherwise TTL expires.
            return None

    async def set(self, key: str, output: GeneratedOutput) -> None:
        if self.redis is None or self.ttl_seconds <= 0:
            return
        try:
            await asyncio.wait_for(
                self.redis.set(key, output.model_dump_json(), ex=self.ttl_seconds),
                CACHE_IO_TIMEOUT_SECONDS,
            )
        except Exception:
            pass  # A cache write must never discard successful generation.
