"""Offline retrieval evaluation; never invokes generation, Redis, or index writes."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from api.rag.config import RagConfig
from api.rag.embedding_input import QUERY_INPUT_VERSION, DOCUMENT_INPUT_VERSION
from api.rag.hybrid import retrieve_hybrid, RRF_K, VECTOR_CANDIDATES, KNN_CANDIDATES
from api.rag.retrieval import retrieve
from api.rag.vector_index import VectorConfig


KS = (1, 3, 5)
MAX_DATASET_BYTES = 5 * 1024 * 1024


class EvaluationQuery(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    relevant_chunk_ids: list[str] = Field(max_length=10000)
    notes: str = Field(default="", max_length=4000)

    @field_validator("query")
    @classmethod
    def question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()  # Same normalization as AskRequest.

    @field_validator("relevant_chunk_ids")
    @classmethod
    def identities(cls, values):
        if any(not value.strip() or value != value.strip() or len(value) > 2048 for value in values):
            raise ValueError("Invalid chunk ID")
        return list(dict.fromkeys(values))


def load_dataset(path: Path) -> list[EvaluationQuery]:
    """JSON array/single record, or one JSON object per nonblank JSONL line."""
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_DATASET_BYTES + 1)
        if len(data) > MAX_DATASET_BYTES:
            raise ValueError("Dataset exceeds 5 MiB")
        text = data.decode("utf-8-sig")
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = json.loads(text)
            if isinstance(records, dict):
                records = [records]
        if not isinstance(records, list) or not 1 <= len(records) <= 10000:
            raise ValueError("Dataset must contain 1-10000 query records")
        return [EvaluationQuery.model_validate(record) for record in records]
    except (OSError, UnicodeError, ValueError, ValidationError, RecursionError):
        # Validation errors may contain entire input records; never echo them.
        raise ValueError("Invalid evaluation dataset: use UTF-8 JSON/JSONL query records (max 5 MiB)") from None


def retrieval_metrics(retrieved: list[str], relevant: list[str]) -> dict[str, float]:
    """Binary judgments; unique ranks; macro-average these per-query metrics."""
    ranked = list(dict.fromkeys(retrieved))
    labels = set(relevant)
    scores = {}
    for k in KS:
        hits = len(set(ranked[:k]) & labels)
        scores[f"hit_rate@{k}"] = float(hits > 0)
        scores[f"recall@{k}"] = hits / len(labels) if labels else 0.0
    scores["mrr"] = next((1 / rank for rank, identity in enumerate(ranked, 1) if identity in labels), 0.0)
    dcg = sum(1 / math.log2(rank + 1) for rank, identity in enumerate(ranked[:5], 1) if identity in labels)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(labels)) + 1))
    scores["ndcg@5"] = dcg / ideal if ideal else 0.0
    return scores


def summarize(rows: list[dict], mode: str) -> dict:
    count = len(rows)
    metrics = {key: mean(row["metrics"][key] for row in rows) for key in rows[0]["metrics"]}
    metrics.update(mean_latency_ms=mean(row["latency_ms"] for row in rows),
                   median_latency_ms=median(row["latency_ms"] for row in rows))
    result = {"query_count": count, "metrics": metrics,
              "error_count": sum(row["error"] is not None for row in rows)}
    if mode == "hybrid":
        successful = sum(row["actual_mode"] == "hybrid" for row in rows)
        fallback = sum(row["actual_mode"] == "bm25" for row in rows)
        result.update(successful_hybrid_count=successful, successful_hybrid_rate=successful / count,
                      bm25_fallback_count=fallback, bm25_fallback_rate=fallback / count,
                      vector_failure_count=sum(row["vector_failed"] for row in rows),
                      no_usable_vector_count=sum(row["actual_mode"] == "bm25" and not row["vector_failed"] for row in rows))
    return result


def configuration(config: RagConfig, modes: tuple[str, ...]) -> dict:
    # Explicit allowlist: never serialize os.environ, connection URLs, keys or LLM settings.
    metadata = {"source_index": config.source_index, "bm25_candidate_limit": config.candidate_limit,
                "max_sources": config.max_sources, "context_max_bytes": config.context_max_bytes,
                "vector_candidate_limit": VECTOR_CANDIDATES, "knn_num_candidates": KNN_CANDIDATES,
                "rrf_k": RRF_K, "embedding": None}
    if "hybrid" in modes:
        try:
            vector = VectorConfig.from_env()
            vector.validate()
            metadata["embedding"] = {"provider": vector.provider, "model": vector.model,
                "dimensions": vector.dimensions, "revision": vector.revision,
                "vector_index": vector.vector_index, "timeout_seconds": vector.timeout_seconds,
                "query_input_version": QUERY_INPUT_VERSION, "document_input_version": DOCUMENT_INPUT_VERSION}
        except ValueError:
            metadata["embedding"] = {"configuration": "missing_or_invalid"}
    return metadata


async def evaluate(dataset: list[EvaluationQuery], mode: str, es, config: RagConfig, clock=perf_counter) -> dict:
    if mode not in ("bm25", "hybrid", "both") or not dataset:
        raise ValueError("Select bm25, hybrid or both with a nonempty dataset")
    modes = ("bm25", "hybrid") if mode == "both" else (mode,)
    identity = [{"query": item.query, "relevant_chunk_ids": sorted(item.relevant_chunk_ids)} for item in dataset]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    report = {"report_version": 1, "label": "functional evaluation only", "dataset_sha256": digest,
              "query_count": len(dataset), "configuration": configuration(config, modes),
              "evaluation_scope": "selected sources after dedupe/overlap/byte limits, before generation context",
              "latency_scope": "retrieval including query embedding and hydration; excludes PostgreSQL enrichment and generation",
              "run_order": "dataset order; bm25 then hybrid per query when both selected; no warmup",
              "runs": {mode: {"queries": []} for mode in modes}}
    for number, item in enumerate(dataset, 1):
        for requested_mode in modes:
            started = clock()
            actual, vector_failed, error, ids = requested_mode, False, None, []
            try:
                # Metadata does not affect retrieval ranking; pool=None avoids
                # PostgreSQL dependency and makes the latency scope explicit.
                if requested_mode == "hybrid":
                    sources, actual, vector_failed = await retrieve_hybrid(item.query, es, None, config)
                else:
                    sources = await retrieve(item.query, es, None, config)
                ids = list(dict.fromkeys(source.chunk_id for source in sources))
            except Exception:
                # Infrastructure failures are scored as empty retrieval, not silently
                # dropped from the denominator. Never serialize provider/ES bodies.
                actual, error = None, "retrieval_unavailable"
            latency = (clock() - started) * 1000
            report["runs"][requested_mode]["queries"].append({
                "query_number": number, "relevant_chunk_ids": item.relevant_chunk_ids,
                "retrieved_chunk_ids": ids, "actual_mode": actual, "vector_failed": vector_failed,
                "error": error, "latency_ms": latency, "metrics": retrieval_metrics(ids, item.relevant_chunk_ids),
            })
    for name, run in report["runs"].items():
        run["summary"] = summarize(run["queries"], name)
    return report


def write_report(report: dict, output: Path) -> None:
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")


def console_report(report: dict) -> str:
    lines = [report["label"], f"Queries: {report['query_count']}"]
    for mode, run in report["runs"].items():
        summary = run["summary"]
        lines.append(f"{mode}: errors={summary['error_count']}")
        lines.extend(f"  {name}: {value:.4f}" for name, value in summary["metrics"].items())
        if mode == "hybrid":
            lines.append(f"  hybrid={summary['successful_hybrid_count']} ({summary['successful_hybrid_rate']:.1%}); "
                         f"BM25 fallback={summary['bm25_fallback_count']} ({summary['bm25_fallback_rate']:.1%}); "
                         f"vector failures={summary['vector_failure_count']}; no usable vectors={summary['no_usable_vector_count']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("bm25", "hybrid", "both"), default="bm25")
    parser.add_argument("--output", type=Path, default=Path(".local-eval/report.json"))
    args = parser.parse_args()
    try:
        if args.output.resolve() == args.dataset.resolve():
            raise ValueError("Output must differ from dataset")
        dataset = load_dataset(args.dataset)
        async def run():
            async with AsyncElasticsearch(os.environ.get("ES_HOST", "http://localhost:9200")) as es:
                return await evaluate(dataset, args.mode, es, RagConfig.from_env())
        report = asyncio.run(run())
        write_report(report, args.output)
    except Exception:
        parser.exit(2, "Evaluation could not start or save its report; check dataset, configuration and output path.\n")
    print(console_report(report))
    if any(run["summary"]["error_count"] for run in report["runs"].values()):
        parser.exit(1, "Report saved with retrieval errors.\n")


if __name__ == "__main__":
    main()
