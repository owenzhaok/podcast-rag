"""Explicit offline CLI: python -m api.rag.backfill {create,check,backfill}."""

import argparse
import hashlib
import json
import os
from itertools import islice

from elasticsearch import Elasticsearch, helpers

from api.rag.embeddings import EmbeddingProvider, validate_embeddings
from api.rag.embedding_input import DOCUMENT_INPUT_VERSION, document_input, episode_titles
from api.rag.retrieval import FIELDS, parse_hit
from api.rag.sources import decode_chunk_id
from api.rag.vector_index import VectorConfig, check_index, create_index


MAX_CLIP_BYTES = 262144


def vector_document(hit: dict, config: VectorConfig) -> tuple[str, dict]:
    evidence = parse_hit(hit)
    if evidence is None or len(evidence.text.encode("utf-8")) > MAX_CLIP_BYTES:
        raise ValueError("Invalid or oversized source clip; backfill stopped")
    chunk_id = evidence.chunk_id
    # ES _id has a 512-byte limit. Hash the existing ID, retaining it verbatim in _source.
    document_id = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
    return document_id, {
        "chunk_id": chunk_id, "podcast_id": evidence.podcast_id, "episode_id": evidence.episode_id,
        "clip_start_ms": evidence.start_ms, "clip_end_ms": evidence.end_ms,
        "chunk_text": evidence.text, "content_hash": decode_chunk_id(chunk_id)[4],
        "speakers": list(evidence.speakers), "chunking_version": "v1",
        "embedding_provider": config.provider, "embedding_model": config.model,
        "embedding_revision": config.revision, "source_index": config.source_index,
    }


def process_batch(es, config: VectorConfig, provider: EmbeddingProvider, hits: list[dict], title_lookup=None) -> dict:
    documents = dict(vector_document(hit, config) for hit in hits)
    inputs = {doc_id: doc["chunk_text"] for doc_id, doc in documents.items()}
    if config.provider == "gemini":
        titles = title_lookup(sorted({doc["episode_id"] for doc in documents.values()})) if title_lookup else {}
        for doc_id, doc in documents.items():
            inputs[doc_id] = document_input(doc, titles.get(doc["episode_id"]))
            doc["embedding_input_version"] = DOCUMENT_INPUT_VERSION
            doc["embedding_input_hash"] = hashlib.sha256(inputs[doc_id].encode("utf-8")).hexdigest()
    existing = es.mget(index=config.vector_index, ids=list(documents), realtime=True)["docs"]
    if len(existing) != len(documents) or any("error" in doc for doc in existing):
        raise ValueError("Unable to check existing vector documents")
    current = {doc["_id"]: doc.get("_source") for doc in existing if doc.get("found")}
    pending = {}
    for doc_id, document in documents.items():
        old = current.get(doc_id) or {}
        if all(old.get(field) == value for field, value in document.items()):
            try:
                validate_embeddings([old.get("embedding")], 1, config.dimensions)
                continue
            except ValueError:
                pass
        pending[doc_id] = document
    if pending:
        try:
            vectors = provider.embed([inputs[doc_id] for doc_id in pending])
        except Exception:
            raise ValueError("Embedding provider failed; completed batches remain intact") from None
        validate_embeddings(vectors, len(pending), config.dimensions)
        actions = [{"_op_type": "index", "_index": config.vector_index, "_id": doc_id,
                    "_source": {**doc, "embedding": vector}}
                   for (doc_id, doc), vector in zip(pending.items(), vectors)]
        _, errors = helpers.bulk(es, actions, chunk_size=config.batch_size, raise_on_error=False)
        if errors:
            raise ValueError("Vector write failed; rerun safely to complete partial batch")
    return {"scanned": len(hits), "written": len(pending), "skipped": len(hits) - len(pending)}


def backfill(es, config: VectorConfig, provider: EmbeddingProvider, limit: int, progress=print, title_lookup=None) -> dict:
    config.validate()
    if limit < 1:
        raise ValueError("An explicit positive document limit is required")
    check_index(es, config)  # Backfill never implicitly creates an index.
    if set(es.indices.get_mapping(index=config.source_index)) != {config.source_index}:
        raise ValueError("Source must be one concrete index, not an alias")
    totals = {"scanned": 0, "written": 0, "skipped": 0}
    stream = helpers.scan(es, index=config.source_index, size=config.batch_size, scroll="30m",
                          query={"query": {"match_all": {}}, "_source": FIELDS}, clear_scroll=True)
    try:
        while totals["scanned"] < limit:
            hits = list(islice(stream, min(config.batch_size, limit - totals["scanned"])))
            if not hits:
                break
            result = process_batch(es, config, provider, hits, title_lookup)
            for key, value in result.items():
                totals[key] += value
            progress(json.dumps(totals))  # Counts only; no transcript or provider error bodies.
    finally:
        stream.close()
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "check", "backfill"))
    parser.add_argument("--limit", type=int, help="Required explicit maximum source documents to scan")
    args = parser.parse_args()
    try:
        config = VectorConfig.from_env()
        config.validate()
        provider = config.make_provider() if args.command == "backfill" else None
        if args.command == "backfill" and (args.limit is None or args.limit < 1):
            raise ValueError("Backfill requires --limit with a positive maximum document count")
        with Elasticsearch(os.environ.get("ES_HOST", "http://localhost:9200"), request_timeout=30) as es:
            if args.command == "create":
                create_index(es, config)
            elif args.command == "check":
                check_index(es, config)
            else:
                backfill(es, config, provider, args.limit, title_lookup=episode_titles)
            print("Vector preparation complete; active retrieval remains BM25.")
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
    except Exception:
        parser.exit(1, "Elasticsearch preparation failed; no automatic recreation attempted.\n")


if __name__ == "__main__":
    main()
