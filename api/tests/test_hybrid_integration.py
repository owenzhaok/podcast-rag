"""Opt-in real Elasticsearch kNN, fake query embeddings and generation only."""

import os
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
from elasticsearch import AsyncElasticsearch

import api.main as main
from api.rag.backfill import vector_document
from api.rag.embedding_input import DOCUMENT_INPUT_VERSION
from api.rag.groq import get_generation_provider
from api.rag.vector_index import VectorConfig, vector_mapping
from api.tests.fake_llm import FakeProvider
from api.tests.test_hybrid import clip, hit, VECTOR
from ingest.es_client import INDEX_MAPPING


@pytest.mark.skipif(os.environ.get("RAG_INTEGRATION_TESTS") != "1", reason="Opt-in Docker ES test")
@pytest.mark.asyncio
async def test_real_knn_hydration_api_and_no_index_writes(monkeypatch):
    suffix = uuid4().hex
    config = replace(VECTOR, source_index="rag-hybrid-source-" + suffix,
                     vector_index="rag-hybrid-vector-" + suffix)
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("RAG_SOURCE_INDEX", config.source_index)
    monkeypatch.setattr(VectorConfig, "from_env", classmethod(lambda cls: config))
    calls = []
    class FakeQuery:
        def embed(self, texts):
            calls.append(texts)
            return [[1.0] + [0.0] * 767]
    monkeypatch.setattr(VectorConfig, "make_provider", lambda self: FakeQuery())
    es = AsyncElasticsearch(os.environ.get("ES_HOST", "http://localhost:9200"))
    created = []
    try:
        await es.indices.create(index=config.source_index, body=INDEX_MAPPING)
        created.append(config.source_index)
        await es.indices.create(index=config.vector_index, mappings=vector_mapping(config),
                                settings={"number_of_shards": 1, "number_of_replicas": 0})
        created.append(config.vector_index)
        for i, evidence in enumerate([clip("a"), clip("b", "A semantic-only passage"), clip("stale")]):
            source_hit = hit(evidence)
            if i < 2:
                await es.index(index=config.source_index, id=str(i), document=source_hit["_source"])
            document_id, document = vector_document(source_hit, config)
            await es.index(index=config.vector_index, id=document_id, document={**document,
                "embedding": [1.0] + [0.0] * 767, "embedding_input_version": DOCUMENT_INPUT_VERSION,
                "embedding_input_hash": "test-only"})
        for index in created:
            await es.indices.refresh(index=index)
        async def snapshot():
            result = {}
            for index in created:
                raw = await es.search(index=index, size=10, seq_no_primary_term=True)
                result[index] = {h["_id"]: (h["_source"], h["_seq_no"], h["_primary_term"])
                                 for h in raw["hits"]["hits"]}
            return result
        before = await snapshot()
        monkeypatch.setattr(main, "_es_client", es)
        monkeypatch.setattr(main, "_db_pool", None)
        monkeypatch.setattr(main, "_cache_client", None)
        app = main.create_app()
        fake = FakeProvider('{"status":"answered","paragraphs":[{"text":"Supported.","source_ids":["S1"]}]}')
        app.dependency_overrides[get_generation_provider] = lambda: fake
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/ask", json={"question": "Machine learning"})
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "answered" and body["retrieval_mode"] == "hybrid"
            assert {s["episode_id"] for s in body["sources"]} == {"a", "b"}
            for source in body["sources"]:
                resolved = (await client.get("/sources/" + source["chunk_id"])).json()
                assert resolved["chunk_id"] == source["chunk_id"] and resolved["excerpt"] == source["excerpt"]
            assert (await client.get("/health")).status_code == 200
            assert len(calls) == 1 and len(fake.requests) == 1
            # An unavailable companion is a read-only fallback, without removal.
            monkeypatch.setattr(VectorConfig, "from_env", classmethod(
                lambda cls: replace(config, vector_index="rag-hybrid-missing-" + suffix)))
            fallback = (await client.post("/ask", json={"question": "Machine learning"})).json()
            assert fallback["retrieval_mode"] == "bm25" and fallback["degraded"]
            assert [s["episode_id"] for s in fallback["sources"]] == ["a"]
            assert len(calls) == 1  # Mapping failure avoids even a fake provider call.
        assert await snapshot() == before
    finally:
        for index in reversed(created):
            await es.indices.delete(index=index)  # Only unique indices created above.
        await es.close()
