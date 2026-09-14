"""FastAPI application for podcast search."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import SearchResponse
from api.search import execute_search
from api.cache import CacheClient
from api.rag.config import RagConfig
from api.rag.routes import create_router


# Global references set during lifespan (or left None for testing)
_es_client = None
_db_pool = None
_cache_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _es_client, _db_pool, _cache_client
    from elasticsearch import AsyncElasticsearch
    import asyncpg
    from redis.asyncio import Redis

    es_host = os.environ.get("ES_HOST", "http://localhost:9200")
    postgres_dsn = os.environ.get("POSTGRES_DSN", "postgresql://podcast:podcast@localhost:5432/podcasts")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    _es_client = AsyncElasticsearch(es_host)
    _db_pool = await asyncpg.create_pool(postgres_dsn, min_size=2, max_size=10)
    _cache_client = CacheClient(Redis.from_url(redis_url))

    yield

    await _es_client.close()
    await _db_pool.close()
    await _cache_client._redis.aclose()


def create_app(use_lifespan: bool = False) -> FastAPI:
    """Create the FastAPI app. use_lifespan=False for testing."""
    app = FastAPI(title="Podcast Search", lifespan=lifespan if use_lifespan else None)
    app.state.rag_config = RagConfig.from_env()
    app.include_router(create_router(app.state.rag_config, lambda: (_es_client, _db_pool)))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/search", response_model=SearchResponse)
    async def search(
        q: str = Query(..., min_length=1),
        clip_minutes: float = Query(2.0),
        from_: int = Query(0, alias="from"),
        size: int = Query(10, le=100),
    ):
        result = await execute_search(
            q=q, from_=from_, size=size,
            es=_es_client, db_pool=_db_pool, cache=_cache_client,
        )
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app(use_lifespan=True)
