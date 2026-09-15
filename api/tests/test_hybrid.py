"""Stage 7: deterministic query vectors only; no external provider requests."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import httpx
import pytest

import api.main as main
import api.rag.hybrid as hybrid
from api.models import SearchResponse
from api.rag.config import RagConfig
from api.rag.context import select_context
from api.rag.embedding_input import query_input, QUERY_INPUT_VERSION, DOCUMENT_INPUT_VERSION
from api.rag.gemini import GeminiEmbeddings
from api.rag.groq import get_generation_provider
from api.rag.retrieval import resolve_evidence, RetrievalUnavailable
from api.rag.sources import Evidence, decode_chunk_id
from api.rag.vector_index import VectorConfig, vector_mapping
from api.tests.fake_llm import FakeProvider


CONFIG = RagConfig(enabled=True, retrieval_mode="hybrid")
VECTOR = VectorConfig(provider="gemini", model="gemini-embedding-2", dimensions=768)


def clip(episode="episode", text="Machine learning", start=0, score=1):
    return Evidence("show", episode, start, start + 120000, text, 0, (1,), score)


def hit(evidence):
    return {"_source": {"podcast_id": evidence.podcast_id, "episode_id": evidence.episode_id,
        "clip_start_ms": evidence.start_ms, "clip_end_ms": evidence.end_ms,
        "clip_index": evidence.clip_index, "speakers": list(evidence.speakers), "clip_text": evidence.text},
        "_score": evidence.score}


def vector_hit(evidence):
    return {"_source": {"chunk_id": evidence.chunk_id, "chunk_text": evidence.text,
        "content_hash": decode_chunk_id(evidence.chunk_id)[4], "source_index": CONFIG.source_index,
        "embedding_provider": "gemini", "embedding_model": VECTOR.model,
        "embedding_revision": "v1", "chunking_version": "v1",
        "embedding_input_version": DOCUMENT_INPUT_VERSION}}


@pytest.fixture
def vector_setup(monkeypatch):
    monkeypatch.setattr(VectorConfig, "from_env", classmethod(lambda cls: VECTOR))
    provider = MagicMock()
    provider.embed.return_value = [[0.1] * 768]
    monkeypatch.setattr(VectorConfig, "make_provider", lambda self: provider)
    es = AsyncMock()
    es.indices.get_mapping.return_value = {VECTOR.vector_index: {"mappings": vector_mapping(VECTOR)}}
    es.search.return_value = {"hits": {"hits": [vector_hit(clip())]}}
    monkeypatch.setattr(hybrid, "resolve_evidence", AsyncMock(return_value=clip()))
    return es, provider


@pytest.mark.parametrize("value, expected", [(None, "bm25"), ("bm25", "bm25"), ("hybrid", "hybrid"),
                                           ("unexpected", "bm25")])
def test_mode_configuration(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("RAG_RETRIEVAL_MODE", value)
    assert RagConfig.from_env().retrieval_mode == expected


def test_query_format():
    assert QUERY_INPUT_VERSION == "gemini-query-v1"
    assert DOCUMENT_INPUT_VERSION == "gemini-document-v1"
    assert query_input("Exact question?\n中文") == "task: question answering | query: Exact question?\n中文"


def test_rrf_scores_duplicates_and_ties():
    a, b, c = clip("a"), clip("b"), clip("c")
    ranked = hybrid.reciprocal_rank_fusion([a, a, b], [b, c])
    assert [e.episode_id for e in ranked] == ["b", "a", "c"]
    assert ranked[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert ranked[1].score == pytest.approx(1 / 61)
    tied = hybrid.reciprocal_rank_fusion([a], [c])
    assert [e.chunk_id for e in tied] == sorted([a.chunk_id, c.chunk_id])
    assert hybrid.reciprocal_rank_fusion([a, b], []) == hybrid.reciprocal_rank_fusion([a, b])
    assert len(hybrid.reciprocal_rank_fusion([], [c])) == 1


@pytest.mark.asyncio
async def test_knn_request_and_canonical_evidence(vector_setup):
    es, provider = vector_setup
    result = await hybrid.vector_candidates("What is discussed?", es, CONFIG)
    assert result == [clip()]
    provider.embed.assert_called_once_with([query_input("What is discussed?")])
    request = es.search.call_args.kwargs
    assert request["index"] == VECTOR.vector_index
    body = request["body"]
    assert body["knn"]["query_vector"] == [0.1] * 768
    assert body["knn"]["k"] == body["size"] == 30
    assert body["knn"]["num_candidates"] == 100
    assert {"term": {"embedding_input_version": DOCUMENT_INPUT_VERSION}} in body["knn"]["filter"]["bool"]["filter"]
    assert "embedding" not in body["_source"]
    es.indices.refresh.assert_not_called()
    es.index.assert_not_called()


@pytest.mark.asyncio
async def test_query_uses_existing_gemini_mocktransport(vector_setup, monkeypatch):
    es, _ = vector_setup
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"embedding": {"values": [0.1] * 768}})
    provider = GeminiEmbeddings(VECTOR.model, "test-placeholder", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(VectorConfig, "make_provider", lambda self: provider)
    await hybrid.vector_candidates("question", es, CONFIG)
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {"content": {"parts": [{"text": query_input("question")}]},
                                           "outputDimensionality": 768}


@pytest.mark.asyncio
@pytest.mark.parametrize("vectors", [[], [[1.0] * 767], [[float("nan")] * 768], [[0.0] * 768],
                                      [["private"] * 768], [[0.1] * 768] * 2])
async def test_invalid_query_embedding(vector_setup, vectors):
    es, provider = vector_setup
    provider.embed.return_value = vectors
    with pytest.raises(ValueError):
        await hybrid.vector_candidates("question", es, CONFIG)
    es.search.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "text", "hash", "provider", "revision", "source_index"])
async def test_stale_or_foreign_vectors_discarded(vector_setup, monkeypatch, change):
    es, _ = vector_setup
    doc = es.search.return_value["hits"]["hits"][0]["_source"]
    if change == "missing":
        monkeypatch.setattr(hybrid, "resolve_evidence", AsyncMock(return_value=None))
    else:
        field = {"text": "chunk_text", "hash": "content_hash", "provider": "embedding_provider",
                 "revision": "embedding_revision", "source_index": "source_index"}[change]
        doc[field] = "stale"
    assert await hybrid.vector_candidates("question", es, CONFIG) == []


@pytest.mark.asyncio
async def test_actual_canonical_resolution_rejects_changed_content():
    es = AsyncMock()
    es.search.return_value = {"hits": {"hits": [hit(clip(text="Changed exact transcript"))]}}
    assert await resolve_evidence(clip().chunk_id, es, CONFIG) is None


@pytest.mark.asyncio
async def test_incompatible_mapping_before_provider(vector_setup):
    es, provider = vector_setup
    es.indices.get_mapping.return_value[VECTOR.vector_index]["mappings"]["properties"]["embedding"]["dims"] = 8
    with pytest.raises(ValueError):
        await hybrid.vector_candidates("question", es, CONFIG)
    provider.embed.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_order_overlap_limits_and_vector_only(monkeypatch):
    a, b, c = clip("a", score=4), clip("b", score=3), clip("c")
    overlap = clip("b", start=60000)
    monkeypatch.setattr(hybrid, "retrieve_candidates", AsyncMock(return_value=[a, b]))
    monkeypatch.setattr(hybrid, "vector_candidates", AsyncMock(return_value=[b, overlap, c]))
    sources, mode, degraded = await hybrid.retrieve_hybrid("q", None, None, replace(CONFIG, max_sources=3))
    assert mode == "hybrid" and not degraded
    assert [s.episode_id for s in sources] == ["b", "a", "c"]
    assert [s.source_id for s in sources] == ["S1", "S2", "S3"]
    limited, _, _ = await hybrid.retrieve_hybrid("q", None, None, replace(CONFIG, context_max_bytes=len(b.text.encode())))
    assert len(limited) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError("private auth failure"), asyncio.TimeoutError(),
                                      RetrievalUnavailable(), OSError("private network")])
async def test_optional_failures_preserve_bm25(monkeypatch, failure):
    lexical = [clip("a", score=2), clip("b", score=1)]
    monkeypatch.setattr(hybrid, "retrieve_candidates", AsyncMock(return_value=lexical))
    monkeypatch.setattr(hybrid, "vector_candidates", AsyncMock(side_effect=failure))
    sources, mode, fallback = await hybrid.retrieve_hybrid("q", None, None, CONFIG)
    assert mode == "bm25" and fallback
    assert [s.chunk_id for s in sources] == [e.chunk_id for e in select_context(lexical, 6, 16000)]


@pytest.mark.asyncio
async def test_missing_key_falls_back(monkeypatch):
    monkeypatch.setattr(VectorConfig, "from_env", classmethod(lambda cls: VECTOR))
    monkeypatch.setattr(hybrid, "retrieve_candidates", AsyncMock(return_value=[clip()]))
    sources, mode, fallback = await hybrid.retrieve_hybrid("q", None, None, CONFIG)
    assert sources and mode == "bm25" and fallback


@pytest.mark.asyncio
async def test_no_vector_results_keep_lexical_order(monkeypatch):
    monkeypatch.setattr(hybrid, "retrieve_candidates", AsyncMock(return_value=[clip()]))
    monkeypatch.setattr(hybrid, "vector_candidates", AsyncMock(return_value=[]))
    sources, mode, fallback = await hybrid.retrieve_hybrid("q", None, None, CONFIG)
    assert sources and mode == "bm25" and not fallback


@pytest.mark.asyncio
async def test_vector_only_can_supply_evidence(monkeypatch):
    monkeypatch.setattr(hybrid, "retrieve_candidates", AsyncMock(return_value=[]))
    monkeypatch.setattr(hybrid, "vector_candidates", AsyncMock(return_value=[clip()]))
    sources, mode, fallback = await hybrid.retrieve_hybrid("q", None, None, CONFIG)
    assert len(sources) == 1 and mode == "hybrid" and not fallback


@pytest.mark.asyncio
async def test_knn_failure_falls_back_through_real_branch(vector_setup, monkeypatch):
    es, _ = vector_setup
    es.search.side_effect = OSError("private-index-error")
    monkeypatch.setattr(hybrid, "retrieve_candidates", AsyncMock(return_value=[clip()]))
    sources, mode, fallback = await hybrid.retrieve_hybrid("q", es, None, CONFIG)
    assert sources and mode == "bm25" and fallback


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(hybrid, "retrieve_candidates", AsyncMock(return_value=[clip()]))
    monkeypatch.setattr(hybrid, "vector_candidates", AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await hybrid.retrieve_hybrid("q", None, None, CONFIG)


@pytest.mark.asyncio
async def test_api_modes_current_retrieval_cache_and_legacy(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "true")
    a, b = clip("a"), clip("b")
    bm25 = AsyncMock(return_value=[a])
    vectors = AsyncMock(return_value=[b])
    monkeypatch.setattr("api.rag.retrieval.retrieve_candidates", bm25)
    monkeypatch.setattr(hybrid, "retrieve_candidates", bm25)
    monkeypatch.setattr(hybrid, "vector_candidates", vectors)
    search = AsyncMock(return_value=SearchResponse(query="q", total=0, clips=[], took_ms=0))
    monkeypatch.setattr(main, "execute_search", search)
    redis = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr(main, "_cache_client", MagicMock(_redis=redis))
    fake = FakeProvider('{"status":"answered","paragraphs":[{"text":"Supported.","source_ids":["S1"]}]}')
    async def ask(mode):
        monkeypatch.setenv("RAG_RETRIEVAL_MODE", mode)
        app = main.create_app()
        app.dependency_overrides[get_generation_provider] = lambda: fake
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/search?q=q")).status_code == 200
            return (await client.post("/ask", json={"question": "q"})).json()
    try:
        first = await ask("bm25")
        assert first["retrieval_mode"] == "bm25" and not first["cached"]
        vectors.assert_not_called()  # No Gemini/vector configuration necessary.
        second = await ask("hybrid")
        assert second["retrieval_mode"] == "hybrid" and not second["cached"]
        third = await ask("hybrid")
        assert third["cached"] and len(fake.requests) == 2
        assert bm25.await_count == 3 and vectors.await_count == 2
        vectors.side_effect = ValueError("private-provider-body")
        fallback = await ask("hybrid")
        assert fallback["retrieval_mode"] == "bm25" and fallback["cached"] and fallback["degraded"]
        assert fallback["reason"] == "hybrid_retrieval_unavailable"
        assert "private-provider-body" not in json.dumps(fallback)
        assert search.await_count == 4
    finally:
        await redis.aclose()
