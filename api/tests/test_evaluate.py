"""Small deterministic judgments; no Gemini/Groq network calls."""

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import api.rag.evaluate as evaluator
from api.rag.config import RagConfig
from api.rag.retrieval import RetrievalUnavailable


FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_smoke.jsonl"


def sources(*ids):
    return [SimpleNamespace(chunk_id=identity, excerpt="PRIVATE TRANSCRIPT") for identity in ids]


@pytest.mark.parametrize("extension", [".json", ".jsonl"])
def test_dataset_formats(tmp_path, extension):
    records = [{"query": " q ", "relevant_chunk_ids": ["a", "a", "b"], "notes": "optional"}]
    path = tmp_path / ("data" + extension)
    path.write_text(json.dumps(records) if extension == ".json" else json.dumps(records[0]) + "\n\n", encoding="utf-8")
    data = evaluator.load_dataset(path)
    assert data[0].query == "q" and data[0].relevant_chunk_ids == ["a", "b"]
    assert len(evaluator.load_dataset(FIXTURE)) == 2


@pytest.mark.parametrize("payload", [[], {}, {"query": "", "relevant_chunk_ids": []},
    {"query": "  ", "relevant_chunk_ids": []}, {"query": 123, "relevant_chunk_ids": []},
    {"query": "q", "relevant_chunk_ids": "a"}, {"query": "q", "relevant_chunk_ids": [1]},
    {"query": "q", "relevant_chunk_ids": [""]}, {"query": "q", "relevant_chunk_ids": [" a"]},
    {"query": "q", "relevant_chunk_ids": [], "extra": "PRIVATE"}])
def test_invalid_dataset(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid evaluation dataset") as exc:
        evaluator.load_dataset(path)
    assert "PRIVATE" not in str(exc.value)


def test_malformed_oversized_and_single_record(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    path.write_text('{"query":"q","relevant_chunk_ids":[]}', encoding="utf-8")
    assert evaluator.load_dataset(path)[0].relevant_chunk_ids == []
    monkeypatch.setattr(evaluator, "MAX_DATASET_BYTES", 2)
    with pytest.raises(ValueError): evaluator.load_dataset(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError): evaluator.load_dataset(path)


def test_binary_metrics_multiple_relevant_and_duplicate_ranks():
    metrics = evaluator.retrieval_metrics(["x", "a", "a", "b"], ["a", "b", "b", "c"])
    assert metrics["hit_rate@1"] == 0
    assert metrics["hit_rate@3"] == metrics["hit_rate@5"] == 1
    assert metrics["recall@1"] == 0
    assert metrics["recall@3"] == metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg@5"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) /
                                              (1 + 1 / math.log2(3) + 1 / math.log2(4)))


@pytest.mark.parametrize("ranked,relevant", [([], ["a"]), (["x"], ["a"]), (["a"], [])])
def test_empty_missing_or_unanswerable(ranked, relevant):
    assert all(value == 0 for value in evaluator.retrieval_metrics(ranked, relevant).values())


def test_fewer_than_k_and_mrr_beyond_five():
    assert all(value == 1 for value in evaluator.retrieval_metrics(["a"], ["a"]).values())
    metrics = evaluator.retrieval_metrics(["x1", "x2", "x3", "x4", "x5", "a"], ["a"])
    assert metrics["mrr"] == pytest.approx(1 / 6) and metrics["ndcg@5"] == 0


@pytest.mark.asyncio
async def test_both_modes_serialization_privacy_and_latency(monkeypatch, tmp_path):
    data = evaluator.load_dataset(FIXTURE)
    bm25 = AsyncMock(side_effect=[sources("smoke-a"), sources()])
    hybrid = AsyncMock(side_effect=[(sources("smoke-b", "smoke-a"), "hybrid", False),
                                   (sources("smoke-c"), "bm25", True)])
    monkeypatch.setattr(evaluator, "retrieve", bm25)
    monkeypatch.setattr(evaluator, "retrieve_hybrid", hybrid)
    for name, value in {"RAG_EMBEDDING_PROVIDER": "gemini", "RAG_EMBEDDING_MODEL": "gemini-embedding-2",
                        "RAG_EMBEDDING_DIMENSIONS": "768", "RAG_EMBEDDING_API_KEY": "PRIVATE KEY",
                        "RAG_LLM_API_KEY": "PRIVATE GROQ"}.items():
        monkeypatch.setenv(name, value)
    ticks = iter([0, .010, 1, 1.030, 2, 2.050, 3, 3.070])
    report = await evaluator.evaluate(data, "both", None, RagConfig(), clock=lambda: next(ticks))
    a, b = [report["runs"][name]["summary"] for name in ("bm25", "hybrid")]
    assert a["metrics"]["mean_latency_ms"] == pytest.approx(30)
    assert a["metrics"]["median_latency_ms"] == pytest.approx(30)
    assert a["metrics"]["recall@5"] == .25  # Macro average (.5 + 0) / 2.
    assert b["metrics"]["mean_latency_ms"] == pytest.approx(50)
    assert b["successful_hybrid_count"] == b["bm25_fallback_count"] == b["vector_failure_count"] == 1
    assert b["successful_hybrid_rate"] == b["bm25_fallback_rate"] == .5
    assert b["no_usable_vector_count"] == 0
    assert bm25.await_args_list[0].args[0] == hybrid.await_args_list[0].args[0] == data[0].query
    metadata = report["configuration"]
    assert metadata["rrf_k"] == 60 and metadata["embedding"]["dimensions"] == 768
    path = tmp_path / "results" / "report.json"
    evaluator.write_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    output = path.read_text(encoding="utf-8") + evaluator.console_report(report)
    for private in ["PRIVATE", data[0].query, data[0].notes]:
        assert private not in output
    assert "functional evaluation only" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
async def test_single_mode_dispatch(monkeypatch, mode):
    lexical, vector = AsyncMock(return_value=[]), AsyncMock(return_value=([], "bm25", False))
    monkeypatch.setattr(evaluator, "retrieve", lexical)
    monkeypatch.setattr(evaluator, "retrieve_hybrid", vector)
    report = await evaluator.evaluate(evaluator.load_dataset(FIXTURE), mode, None, RagConfig())
    assert set(report["runs"]) == {mode}
    assert (lexical.await_count, vector.await_count) == ((2, 0) if mode == "bm25" else (0, 2))
    if mode == "hybrid":
        summary = report["runs"][mode]["summary"]
        assert summary["bm25_fallback_rate"] == 1 and summary["vector_failure_count"] == 0
        assert summary["no_usable_vector_count"] == 2


@pytest.mark.asyncio
async def test_errors_are_counted_without_raw_bodies(monkeypatch):
    monkeypatch.setattr(evaluator, "retrieve", AsyncMock(side_effect=RetrievalUnavailable("PRIVATE")))
    report = await evaluator.evaluate(evaluator.load_dataset(FIXTURE), "bm25", None, RagConfig())
    assert report["runs"]["bm25"]["summary"]["error_count"] == 2
    assert report["runs"]["bm25"]["summary"]["metrics"]["hit_rate@5"] == 0
    assert "PRIVATE" not in json.dumps(report)


def test_median_odd_and_mean_are_distinct():
    rows = [{"latency_ms": value, "metrics": {"mrr": 0}, "error": None} for value in (1, 2, 90)]
    result = evaluator.summarize(rows, "bm25")["metrics"]
    assert result["median_latency_ms"] == 2 and result["mean_latency_ms"] == 31


def test_cli_defaults_do_not_load_hybrid_or_generation(monkeypatch, tmp_path, capsys):
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
    monkeypatch.setattr(evaluator, "AsyncElasticsearch", lambda *args: Client())
    monkeypatch.setattr(evaluator, "retrieve", AsyncMock(return_value=sources("smoke-a")))
    forbidden = AsyncMock(side_effect=AssertionError("Hybrid must not run"))
    monkeypatch.setattr(evaluator, "retrieve_hybrid", forbidden)
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["evaluate", "--dataset", str(FIXTURE), "--output", str(output)])
    evaluator.main()
    assert set(json.loads(output.read_text())["runs"]) == {"bm25"}
    forbidden.assert_not_called()
    assert "functional evaluation only" in capsys.readouterr().out


@pytest.mark.skipif(os.environ.get("RAG_INTEGRATION_TESTS") != "1", reason="Opt-in Docker ES test")
@pytest.mark.asyncio
async def test_real_retrieval_evaluation_leaves_indices_unchanged(monkeypatch):
    from elasticsearch import AsyncElasticsearch
    from api.rag.backfill import vector_document
    from api.rag.embedding_input import DOCUMENT_INPUT_VERSION
    from api.rag.vector_index import VectorConfig, vector_mapping
    from api.tests.test_hybrid import clip, hit, VECTOR
    from ingest.es_client import INDEX_MAPPING

    suffix = uuid4().hex
    config = replace(VECTOR, source_index="rag-eval-source-" + suffix, vector_index="rag-eval-vector-" + suffix)
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
        evidence = clip()
        source = hit(evidence)
        await es.index(index=config.source_index, id="one", document=source["_source"], refresh=True)
        identity, document = vector_document(source, config)
        await es.index(index=config.vector_index, id=identity, document={**document,
            "embedding": [1.0] + [0.0] * 767, "embedding_input_version": DOCUMENT_INPUT_VERSION,
            "embedding_input_hash": "test-only"}, refresh=True)
        async def snapshot():
            return [dict(await es.get(index=index, id=doc_id))
                    for index, doc_id in ((config.source_index, "one"), (config.vector_index, identity))]
        before = await snapshot()
        dataset = [evaluator.EvaluationQuery(query="Machine learning", relevant_chunk_ids=[evidence.chunk_id])]
        report = await evaluator.evaluate(dataset, "both", es, replace(RagConfig(), source_index=config.source_index))
        for mode in ("bm25", "hybrid"):
            summary = report["runs"][mode]["summary"]
            assert summary["metrics"]["recall@5"] == summary["metrics"]["ndcg@5"] == 1
            assert summary["error_count"] == 0
        assert report["runs"]["hybrid"]["summary"]["successful_hybrid_count"] == 1
        assert len(calls) == 1
        assert await snapshot() == before
    finally:
        for index in reversed(created):
            await es.indices.delete(index=index)  # Only the unique test indices above.
        await es.close()
