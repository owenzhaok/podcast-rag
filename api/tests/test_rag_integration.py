"""Opt-in Compose test: RAG_INTEGRATION_TESTS=1; no external model services.

Uses a unique disposable ES index and a rolled-back PostgreSQL transaction.
The legacy podcast_clips index and existing metadata are never written.
"""

import os
from uuid import uuid4

import asyncpg
import pytest
from elasticsearch import AsyncElasticsearch
from httpx import ASGITransport, AsyncClient

import api.main as main
from ingest.es_client import INDEX_MAPPING


@pytest.mark.skipif(os.environ.get("RAG_INTEGRATION_TESTS") != "1", reason="Opt-in Docker services test")
@pytest.mark.asyncio
async def test_real_bm25_source_resolution_and_staleness(monkeypatch):
    suffix = uuid4().hex
    index, show, episode = f"rag-test-{suffix}", f"show_{suffix}", f"episode_{suffix}"
    es = AsyncElasticsearch(os.environ.get("ES_HOST", "http://localhost:9200"))
    conn = None
    transaction = None
    created = False
    try:
        conn = await asyncpg.connect(os.environ.get(
            "POSTGRES_DSN", "postgresql://podcast:podcast@localhost:5432/podcasts"))
        transaction = conn.transaction()
        await transaction.start()
        await conn.execute("INSERT INTO shows(show_id, name) VALUES($1, $2)", show, "Integration Show")
        await conn.execute("INSERT INTO episodes(episode_id, show_id, name) VALUES($1, $2, $3)",
                           episode, show, "Integration Episode")
        await es.indices.create(index=index, body=INDEX_MAPPING)
        created = True
        text = "Machine learning helps identify patterns in research. " * 12
        doc = {"podcast_id": show, "episode_id": episode, "clip_index": 0,
               "clip_start_ms": 0, "clip_end_ms": 120000, "clip_text": text,
               "speakers": [1], "word_timestamps": []}
        # Legacy ingestion assigns distinct IDs to identical documents.
        await es.index(index=index, id="first", document=doc)
        await es.index(index=index, id="duplicate", document=doc)
        await es.index(index=index, id="overlap", document={
            **doc, "clip_index": 1, "clip_start_ms": 60000, "clip_end_ms": 180000})
        await es.indices.refresh(index=index)
        monkeypatch.setenv("RAG_ENABLED", "true")
        monkeypatch.setenv("RAG_SOURCE_INDEX", index)
        for name in ("RAG_LLM_API_KEY", "RAG_LLM_MODEL", "RAG_LLM_PROVIDER",
                     "RAG_CANDIDATE_LIMIT", "RAG_MAX_SOURCES", "RAG_CONTEXT_MAX_BYTES"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(main, "_es_client", es)
        monkeypatch.setattr(main, "_db_pool", conn)
        async with AsyncClient(transport=ASGITransport(app=main.create_app()), base_url="http://test") as client:
            response = await client.post("/ask", json={"question": "machine learning"})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["answer"] is None and body["retrieval_mode"] == "bm25"
            assert len(body["sources"]) == 1
            source = body["sources"][0]
            assert source["excerpt"] == text and source["show_name"] == "Integration Show"
            assert source["episode_name"] == "Integration Episode"
            url = "/sources/" + source["chunk_id"]
            assert (await client.get(url)).json() == source
            again = (await client.post("/ask", json={"question": "machine learning"})).json()
            assert again["sources"] == body["sources"]
            # New app instance proves resolution has no in-memory registry.
            async with AsyncClient(transport=ASGITransport(app=main.create_app()), base_url="http://test") as restarted:
                assert (await restarted.get(url)).status_code == 200
            empty = await client.post("/ask", json={"question": "zxqvnonexistentbaseline"})
            assert empty.json()["status"] == "insufficient_context"
            for doc_id in ("first", "duplicate", "overlap"):
                await es.delete(index=index, id=doc_id)
            await es.indices.refresh(index=index)
            assert (await client.get(url)).status_code == 404
            await es.indices.delete(index=index)
            created = False
            assert (await client.post("/ask", json={"question": "machine learning"})).status_code == 503
            assert (await client.get("/health")).status_code == 200
    finally:
        try:
            if created:
                await es.indices.delete(index=index)
        finally:
            await es.close()
            if conn is not None:
                try:
                    if transaction is not None:
                        await transaction.rollback()
                finally:
                    await conn.close()
