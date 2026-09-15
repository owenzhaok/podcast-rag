"""Optional read-only vector candidates and application-side reciprocal rank fusion."""

import asyncio
from dataclasses import replace

from api.rag.config import RagConfig
from api.rag.context import select_context
from api.rag.embedding_input import DOCUMENT_INPUT_VERSION, query_input
from api.rag.embeddings import validate_embeddings
from api.rag.retrieval import enrich, retrieve_candidates, resolve_evidence, search_index
from api.rag.sources import Evidence, decode_chunk_id
from api.rag.vector_index import VectorConfig


VECTOR_CANDIDATES = 30
KNN_CANDIDATES = 100
RRF_K = 60
HYDRATION_CONCURRENCY = 4
VECTOR_IO_BUDGET_SECONDS = 15


def reciprocal_rank_fusion(*rankings: list[Evidence]) -> list[Evidence]:
    """One contribution per chunk per list; unique ranks start at one.

    Input order is ranking order. Canonical chunk ID breaks fused score ties.
    Prefer the first (BM25) canonical record when identical clips occur twice.
    """
    records, scores = {}, {}
    for ranking in rankings:
        seen = set()
        for clip in ranking:
            identity = clip.chunk_id
            if identity in seen:
                continue
            seen.add(identity)
            records.setdefault(identity, clip)
            scores[identity] = scores.get(identity, 0.0) + 1 / (RRF_K + len(seen))
    return [replace(records[key], score=scores[key])
            for key in sorted(records, key=lambda key: (-scores[key], key))]


async def vector_candidates(question: str, es, config: RagConfig) -> list[Evidence]:
    # Lazy configuration: BM25 never loads embedding settings or credentials.
    settings = VectorConfig.from_env()
    settings.validate()
    if (settings.provider != "gemini" or settings.model != "gemini-embedding-2"
            or settings.dimensions != 768 or settings.source_index != config.source_index):
        raise ValueError("Incompatible hybrid embedding configuration")
    provider = settings.make_provider()

    async def run():
        # Do not call the offline check_index helper: it refreshes the index.
        actual = await es.indices.get_mapping(index=settings.vector_index)
        if set(actual) != {settings.vector_index}:
            raise ValueError("Hybrid requires a concrete vector index")
        mapping = actual[settings.vector_index]["mappings"]
        vector = mapping.get("properties", {}).get("embedding", {})
        if (mapping.get("_meta") != {"rag_vector_version": 1}
                or vector.get("type") != "dense_vector" or vector.get("dims") != 768
                or vector.get("index") is not True or vector.get("similarity") != "cosine"):
            raise ValueError("Incompatible hybrid vector mapping")
        # Reuse the synchronous adapter off the event loop. The adapter also has
        # its own HTTP timeout; no retries or online index writes occur here.
        vectors = await asyncio.wait_for(
            asyncio.to_thread(provider.embed, [query_input(question)]), settings.timeout_seconds)
        validate_embeddings(vectors, 1, 768)
        filters = {"source_index": config.source_index, "embedding_provider": settings.provider,
                   "embedding_model": settings.model, "embedding_revision": settings.revision,
                   "embedding_input_version": DOCUMENT_INPUT_VERSION, "chunking_version": "v1"}
        raw = await search_index(es, settings.vector_index, {
            "size": VECTOR_CANDIDATES,
            "_source": ["chunk_id", "chunk_text", "content_hash", *filters],
            "sort": [{"_score": "desc"}, {"chunk_id": "asc"}],
            "knn": {"field": "embedding", "query_vector": vectors[0],
                    "k": VECTOR_CANDIDATES, "num_candidates": KNN_CANDIDATES,
                    "filter": {"bool": {"filter": [{"term": {key: value}}
                                                   for key, value in filters.items()]}}},
        })
        semaphore = asyncio.Semaphore(HYDRATION_CONCURRENCY)

        async def hydrate(hit):
            doc = hit.get("_source", {})
            chunk_id = doc.get("chunk_id")
            if not isinstance(chunk_id, str) or any(doc.get(k) != v for k, v in filters.items()):
                return None
            identity = decode_chunk_id(chunk_id)
            if identity is None or doc.get("content_hash") != identity[4]:
                return None
            async with semaphore:
                clip = await resolve_evidence(chunk_id, es, config)
            # Companion text is never used as evidence; reject stale/mismatching
            # content even when a vector happens to retain a valid corpus ID.
            return clip if clip is not None and clip.text == doc.get("chunk_text") else None

        clips = await asyncio.gather(*(hydrate(hit) for hit in raw["hits"]["hits"][:VECTOR_CANDIDATES]))
        return [clip for clip in clips if clip is not None]

    return await asyncio.wait_for(run(), settings.timeout_seconds + VECTOR_IO_BUDGET_SECONDS)


async def retrieve_hybrid(question: str, es, pool, config: RagConfig):
    # Lexical infrastructure failures retain the existing /ask 503 behavior.
    lexical = await retrieve_candidates(question, es, config)
    mode, fallback = "bm25", False
    candidates = lexical
    try:
        vectors = await vector_candidates(question, es, config)
        if vectors:
            candidates = reciprocal_rank_fusion(lexical, vectors)
            mode = "hybrid"
    except Exception:
        # Optional provider/index boundary: never expose exception bodies, keys,
        # headers or context. Cancellation still propagates (BaseException).
        fallback = True
    sources = await enrich(select_context(candidates, config.max_sources, config.context_max_bytes), pool)
    return sources, mode, fallback
