"""Read-only lexical retrieval and corpus-bound source resolution."""

import asyncio
import math

import asyncpg
from elasticsearch import ApiError
from elastic_transport import TransportError

from api.db import get_episode_info
from api.search import build_query
from api.rag.config import RagConfig
from api.rag.context import select_context
from api.rag.models import RagSource
from api.rag.sources import Evidence, decode_chunk_id


FIELDS = ["podcast_id", "episode_id", "clip_index", "clip_start_ms", "clip_end_ms",
          "clip_text", "speakers"]
SORT = [{"podcast_id": "asc"}, {"episode_id": "asc"}, {"clip_start_ms": "asc"},
        {"clip_end_ms": "asc"}, {"clip_index": "asc"}, {"clip_text.raw": "asc"}]


class RetrievalUnavailable(Exception):
    pass


async def search_index(es, index: str, body: dict) -> dict:
    if es is None:
        raise RetrievalUnavailable()
    try:
        raw = await asyncio.wait_for(es.search(index=index, body=body), timeout=5)
        if raw.get("timed_out") or raw.get("_shards", {}).get("failed", 0):
            raise RetrievalUnavailable()
        return raw
    except (ApiError, TransportError, OSError, asyncio.TimeoutError) as exc:
        raise RetrievalUnavailable() from exc


def parse_hit(hit: dict) -> Evidence | None:
    try:
        src = hit["_source"]
        show, episode, text = src["podcast_id"], src["episode_id"], src["clip_text"]
        start, end, index = src["clip_start_ms"], src["clip_end_ms"], src["clip_index"]
        if not all(isinstance(x, str) and x.strip() for x in (show, episode, text)):
            return None
        if any(type(x) is not int for x in (start, end, index)) or not 0 <= start < end or index < 0:
            return None
        speakers = tuple(sorted(set(src.get("speakers") or [])))
        if any(type(x) is not int for x in speakers):
            return None
        score = float(hit.get("_score") or 0)
        if not math.isfinite(score):
            return None
        evidence = Evidence(show, episode, start, end, text, index, speakers, score)
        return evidence if len(evidence.chunk_id) <= 2048 else None
    except (KeyError, TypeError, ValueError):
        return None


async def enrich(evidence: list[Evidence], pool) -> list[RagSource]:
    metadata = {}
    sources = []
    for clip in evidence:
        if clip.episode_id not in metadata:
            info = None
            if pool is not None:
                try:
                    info = await asyncio.wait_for(get_episode_info(pool, clip.episode_id), timeout=2)
                except (asyncpg.PostgresError, OSError, asyncio.TimeoutError):
                    pass
            metadata[clip.episode_id] = info
        info = metadata[clip.episode_id] or {}
        sources.append(RagSource(
            source_id=f"S{len(sources) + 1}", chunk_id=clip.chunk_id,
            podcast_id=clip.podcast_id, episode_id=clip.episode_id,
            clip_index=clip.clip_index, clip_start_ms=clip.start_ms, clip_end_ms=clip.end_ms,
            excerpt=clip.text, speakers=list(clip.speakers),
            show_name=info.get("show_name") or "Unknown Show",
            episode_name=info.get("episode_name") or "Unknown Episode",
            audio_link=info.get("audio_link"), metadata_available=bool(info),
        ))
    return sources


async def retrieve_candidates(question: str, es, config: RagConfig) -> list[Evidence]:
    # Preserve legacy query semantics, but never use its highlight as evidence.
    query = build_query(question, 0, config.candidate_limit)
    query.pop("highlight", None)
    query.update({"_source": FIELDS, "sort": [{"_score": "desc"}, *SORT]})
    raw = await search_index(es, config.source_index, query)
    return [clip for hit in raw["hits"]["hits"] if (clip := parse_hit(hit)) is not None]


async def retrieve(question: str, es, pool, config: RagConfig) -> list[RagSource]:
    candidates = await retrieve_candidates(question, es, config)
    return await enrich(select_context(candidates, config.max_sources, config.context_max_bytes), pool)


async def resolve_evidence(chunk_id: str, es, config: RagConfig) -> Evidence | None:
    locator = decode_chunk_id(chunk_id)
    if locator is None:
        return None
    show, episode, start, end, _ = locator
    filters = [{"term": {field: value}} for field, value in (
        ("podcast_id", show), ("episode_id", episode),
        ("clip_start_ms", start), ("clip_end_ms", end),
    )]
    # Bound work for pathological repeated imports. Never claim not-found when
    # the safety cap prevents checking all matching corpus records.
    for offset in range(0, 1000, 100):
        raw = await search_index(es, config.source_index, {
            "query": {"bool": {"filter": filters}}, "_source": FIELDS,
            "from": offset, "size": 100, "sort": SORT,
        })
        hits = raw["hits"]["hits"]
        for hit in hits:
            clip = parse_hit(hit)
            if clip is not None and clip.chunk_id == chunk_id:
                return clip
        if len(hits) < 100:
            return None
    raise RetrievalUnavailable()


async def resolve(chunk_id: str, es, pool, config: RagConfig) -> RagSource | None:
    clip = await resolve_evidence(chunk_id, es, config)
    return (await enrich([clip], pool))[0] if clip is not None else None
