import copy
import hashlib
import os
from dataclasses import replace
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4

import pytest
from elasticsearch import Elasticsearch
from httpx import ASGITransport, AsyncClient

import api.main as main
import api.rag.backfill as tool
from api.rag.embeddings import FakeEmbeddings, validate_embeddings
from api.rag.sources import Evidence
from api.rag.vector_index import VectorConfig, vector_mapping, create_index, check_index
from api.tests.test_rag_generation import sources
from api.models import SearchResponse
from ingest.es_client import INDEX_MAPPING


CONFIG = VectorConfig(provider="fake", model="fake-test", dimensions=8, batch_size=2)


def hit(text="Machine learning identifies patterns.", episode="episode"):
    return {"_id": "legacy-generated-id", "_source": {
        "podcast_id": "show", "episode_id": episode, "clip_start_ms": 0, "clip_end_ms": 120000,
        "clip_index": 0, "clip_text": text, "speakers": [1],
    }}


class CountingProvider(FakeEmbeddings):
    def __init__(self, config=CONFIG):
        super().__init__(config.dimensions, config.model, config.revision)
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return super().embed(texts)


@pytest.fixture
def store(monkeypatch):
    es, documents, source = MagicMock(), {}, [hit(), hit()]
    es.indices.get_mapping.side_effect = lambda index: (
        {index: {"mappings": vector_mapping(CONFIG)}} if index == CONFIG.vector_index else {index: INDEX_MAPPING})
    es.mget.side_effect = lambda index, ids, **kw: {"docs": [
        {"_id": i, "found": i in documents, "_source": documents.get(i)} for i in ids]}
    def bulk(client, actions, **kwargs):
        for action in actions:
            assert action["_index"] == CONFIG.vector_index
            documents[action["_id"]] = copy.deepcopy(action["_source"])
        return len(actions), []
    def scan(client, **kwargs):
        assert kwargs["index"] == CONFIG.source_index
        yield from copy.deepcopy(source)
    monkeypatch.setattr(tool.helpers, "scan", scan)
    monkeypatch.setattr(tool.helpers, "bulk", bulk)
    return es, documents, source


def test_create_and_check_mapping():
    es = MagicMock()
    es.indices.exists.return_value = False
    es.indices.get_mapping.return_value = {CONFIG.vector_index: {"mappings": vector_mapping(CONFIG)}}
    create_index(es, CONFIG)
    mapping = es.indices.create.call_args.kwargs["mappings"]
    assert mapping["dynamic"] == "strict"
    assert mapping["properties"]["embedding"] == {"type": "dense_vector", "element_type": "float", "dims": 8,
        "index": True, "similarity": "cosine", "index_options": {"type": "hnsw"}}
    assert mapping["properties"]["chunk_id"]["type"] == "keyword"
    es.indices.exists.return_value = True
    create_index(es, CONFIG)
    assert es.indices.create.call_count == 1
    es.indices.delete.assert_not_called()


def test_document_identity_reuses_stage2():
    expected = Evidence("show", "episode", 0, 120000, hit()["_source"]["clip_text"], 0, (1,)).chunk_id
    doc_id, doc = tool.vector_document(hit(), CONFIG)
    assert doc["chunk_id"] == expected
    assert doc_id == hashlib.sha256(expected.encode()).hexdigest()
    assert doc["content_hash"] == hashlib.sha256(doc["chunk_text"].encode()).hexdigest()
    duplicate = hit()
    duplicate["_id"] = "another-legacy-id"
    assert tool.vector_document(duplicate, CONFIG) == (doc_id, doc)
    assert tool.vector_document(hit(doc["chunk_text"] + " "), CONFIG)[0] != doc_id


def test_first_and_repeat_backfill_skip_duplicates(store):
    es, documents, source = store
    before = copy.deepcopy(source)
    provider = CountingProvider()
    first = tool.backfill(es, CONFIG, provider, 10, progress=lambda _: None)
    second = tool.backfill(es, CONFIG, provider, 10, progress=lambda _: None)
    assert first == {"scanned": 2, "written": 1, "skipped": 1}
    assert second == {"scanned": 2, "written": 0, "skipped": 2}
    assert len(documents) == 1 and len(provider.calls) == 1 and source == before
    es.indices.create.assert_not_called()
    es.indices.delete.assert_not_called()


@pytest.mark.parametrize("change", ["content", "model", "revision", "provider"])
def test_changes_reembed(store, change):
    es, documents, source = store
    tool.backfill(es, CONFIG, CountingProvider(), 10, progress=lambda _: None)
    config = CONFIG
    if change == "content":
        source[:] = [hit("Changed transcript")]
    else:
        config = replace(CONFIG, **{change: "changed"})
    provider = CountingProvider(config)
    result = tool.backfill(es, config, provider, 10, progress=lambda _: None)
    assert result["written"] == 1 and len(provider.calls) == 1
    assert len(documents) == (2 if change == "content" else 1)


def test_batch_limit_and_completed_batches_survive_failure(store):
    es, documents, source = store
    source[:] = [hit(episode=str(i)) for i in range(5)]
    provider = CountingProvider()
    original = provider.embed
    def fail_second(texts):
        if provider.calls:
            raise RuntimeError("secret-provider-error")
        return original(texts)
    provider.embed = fail_second
    with pytest.raises(ValueError, match="Embedding provider failed") as exc:
        tool.backfill(es, CONFIG, provider, 5, progress=lambda _: None)
    assert "secret" not in str(exc.value) and len(documents) == 2
    retry = CountingProvider()
    result = tool.backfill(es, CONFIG, retry, 5, progress=lambda _: None)
    assert result == {"scanned": 5, "written": 3, "skipped": 2}
    assert [len(batch) for batch in retry.calls] == [2, 1]
    assert len(documents) == 5


@pytest.mark.parametrize("vectors", [[[1.0]], [[0.0] * 8], [[float("nan")] * 8], [], [[True] * 8]])
def test_bad_vectors_rejected_before_writes(store, vectors):
    es, documents, _ = store
    provider = MagicMock()
    provider.embed.return_value = vectors
    with pytest.raises(ValueError):
        tool.backfill(es, CONFIG, provider, 2, progress=lambda _: None)
    assert not documents


@pytest.mark.parametrize("change", ["dims", "type", "element_type", "meta", "alias"])
def test_incompatible_mapping_fails_without_writes(store, change):
    es, documents, _ = store
    mapping = vector_mapping(CONFIG)
    if change == "dims": mapping["properties"]["embedding"]["dims"] = 16
    if change == "type": mapping["properties"]["chunk_text"]["type"] = "keyword"
    if change == "element_type": mapping["properties"]["embedding"]["element_type"] = "byte"
    if change == "meta": mapping["_meta"] = {}
    es.indices.get_mapping.side_effect = None
    es.indices.get_mapping.return_value = {
        "different-index" if change == "alias" else CONFIG.vector_index: {"mappings": mapping}}
    with pytest.raises(ValueError):
        tool.backfill(es, CONFIG, CountingProvider(), 2)
    assert not documents
    es.indices.delete.assert_not_called()
    es.indices.create.assert_not_called()


def test_explicit_limit_bounds_scan(store):
    es, documents, source = store
    source[:] = [hit(episode=str(i)) for i in range(6)]
    result = tool.backfill(es, CONFIG, CountingProvider(), 3, progress=lambda _: None)
    assert result["scanned"] == result["written"] == len(documents) == 3
    with pytest.raises(ValueError):
        tool.backfill(es, CONFIG, CountingProvider(), 0)


def test_partial_bulk_failure_can_resume(store, monkeypatch):
    es, documents, source = store
    source[:] = [hit(episode=str(i)) for i in range(2)]
    original = tool.helpers.bulk
    def partial(client, actions, **kwargs):
        original(client, actions[:1], **kwargs)
        return 1, [{"index": {"error": "private-write-error"}}]
    monkeypatch.setattr(tool.helpers, "bulk", partial)
    with pytest.raises(ValueError, match="Vector write failed") as exc:
        tool.backfill(es, CONFIG, CountingProvider(), 2, progress=lambda _: None)
    assert "private-write-error" not in str(exc.value) and len(documents) == 1
    monkeypatch.setattr(tool.helpers, "bulk", original)
    result = tool.backfill(es, CONFIG, CountingProvider(), 2, progress=lambda _: None)
    assert result["written"] == 1 and len(documents) == 2


def test_config_is_explicit_and_fake_deterministic(monkeypatch):
    for name in list(os.environ):
        if name.startswith("RAG_EMBEDDING_") or name == "RAG_VECTOR_INDEX": monkeypatch.delenv(name)
    assert VectorConfig.from_env().vector_index == "podcast_rag_v1"
    with pytest.raises(ValueError): VectorConfig.from_env().make_provider()
    with pytest.raises(ValueError): replace(CONFIG, source_index=CONFIG.vector_index).validate()
    with pytest.raises(ValueError): replace(CONFIG, provider="paid-vendor").make_provider()
    with pytest.raises(ValueError): replace(CONFIG, dimensions=4097).validate()
    assert CONFIG.make_provider().embed(["text"]) == CONFIG.make_provider().embed(["text"])


def test_no_secret_metadata_or_progress(store, monkeypatch):
    es, documents, _ = store
    monkeypatch.setenv("RAG_LLM_API_KEY", "private-value")
    messages = []
    tool.backfill(es, CONFIG, CountingProvider(), 2, progress=messages.append)
    assert all(set(__import__('json').loads(line)) == {"scanned", "written", "skipped"} for line in messages)
    assert "private-value" not in str(documents) + str(messages)
    assert "legacy-generated-id" not in str(documents)


@pytest.mark.asyncio
async def test_embedding_settings_do_not_affect_runtime(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "invalid-and-unused")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "")
    monkeypatch.setattr("api.rag.routes.retrieve", AsyncMock(return_value=sources()))
    monkeypatch.setattr(main, "execute_search", AsyncMock(return_value=SearchResponse(query="q", total=0, clips=[], took_ms=0)))
    async with AsyncClient(transport=ASGITransport(app=main.create_app()), base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/search?q=q")).status_code == 200
        result = (await client.post("/ask", json={"question": "question"})).json()
        assert result["retrieval_mode"] == "bm25" and result["reason"] == "llm_not_configured"


@pytest.mark.skipif(os.environ.get("RAG_INTEGRATION_TESTS") != "1", reason="Opt-in Docker ES test")
def test_real_vector_index_roundtrip():
    suffix = uuid4().hex
    config = replace(CONFIG, source_index=f"rag-vector-source-{suffix}", vector_index=f"rag-vector-test-{suffix}")
    created = []
    with Elasticsearch(os.environ.get("ES_HOST", "http://localhost:9200")) as es:
        try:
            es.indices.create(index=config.source_index, body=INDEX_MAPPING)
            created.append(config.source_index)
            for i in range(3):
                es.index(index=config.source_index, id=str(i), document=hit(episode=str(i % 2))["_source"])
            es.indices.refresh(index=config.source_index)
            before = es.search(index=config.source_index, size=10)["hits"]["hits"]
            create_index(es, config)
            created.append(config.vector_index)
            provider = CountingProvider(config)
            first = tool.backfill(es, config, provider, 10, progress=lambda _: None)
            es.indices.refresh(index=config.vector_index)
            ids = {h["_id"] for h in es.search(index=config.vector_index, size=10)["hits"]["hits"]}
            calls = len(provider.calls)
            second = tool.backfill(es, config, provider, 10, progress=lambda _: None)
            assert first["written"] == 2 and second["written"] == 0 and len(provider.calls) == calls
            assert es.count(index=config.vector_index)["count"] == 2
            after = es.search(index=config.vector_index, size=10)["hits"]["hits"]
            assert {h["_id"] for h in after} == ids
            for h in after:
                doc = h["_source"]
                validate_embeddings([doc["embedding"]], 1, 8)
                assert doc["chunking_version"] == "v1" and doc["podcast_id"] == "show"
                assert doc["embedding_model"] == config.model
                assert h["_id"] == hashlib.sha256(doc["chunk_id"].encode()).hexdigest()
            query = provider.embed(["Machine learning"])[0]
            assert es.search(index=config.vector_index, knn={"field": "embedding", "query_vector": query,
                             "k": 1, "num_candidates": 10})["hits"]["hits"]
            unchanged = es.search(index=config.source_index, size=10)["hits"]["hits"]
            assert {h["_id"]: h["_source"] for h in before} == {h["_id"]: h["_source"] for h in unchanged}
            assert es.count(index=config.source_index)["count"] == 3
        finally:
            for index in reversed(created):
                es.indices.delete(index=index)
