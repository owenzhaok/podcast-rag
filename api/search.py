"""Search query builder and result assembler."""

from api.models import SearchResponse, ClipResult, WordTimestamp
from api.cache import CacheClient


def build_query(q: str, from_: int, size: int) -> dict:
    """Return the ES query dict."""
    return {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": q,
                            "fields": ["clip_text"],
                            "type": "best_fields",
                            "fuzziness": "AUTO",
                        }
                    }
                ]
            }
        },
        "highlight": {
            "fields": {
                "clip_text": {
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                    "fragment_size": 300,
                    "number_of_fragments": 1,
                }
            }
        },
        "from": from_,
        "size": size,
    }


async def execute_search(
    q: str,
    from_: int,
    size: int,
    es,
    db_pool,
    cache: CacheClient,
) -> SearchResponse:
    """Run the full search pipeline and return assembled results."""
    # Check cache
    cache_key = cache.cache_key(q, from_, size)
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached

    # Query ES
    query = build_query(q, from_, size)
    raw = await es.search(index="podcast_clips", body=query)

    total = raw["hits"]["total"]["value"]
    took_ms = raw["took"]

    # Assemble results with local episode cache for this request
    clips = []
    episode_cache: dict[str, dict | None] = {}

    for hit in raw["hits"]["hits"]:
        src = hit["_source"]
        episode_id = src["episode_id"]

        if episode_id not in episode_cache:
            info = await db_pool.fetchrow(
                "SELECT s.name as show_name, e.name as episode_name, e.audio_link "
                "FROM episodes e JOIN shows s ON e.show_id = s.show_id "
                "WHERE e.episode_id = $1",
                episode_id,
            )
            episode_cache[episode_id] = dict(info) if info else None

        info = episode_cache[episode_id]
        highlight_text = ""
        if "highlight" in hit and "clip_text" in hit["highlight"]:
            highlight_text = hit["highlight"]["clip_text"][0]

        clips.append(
            ClipResult(
                podcast_id=src["podcast_id"],
                episode_id=episode_id,
                show_name=info["show_name"] if info else "Unknown Show",
                episode_name=info["episode_name"] if info else "Unknown Episode",
                clip_start_ms=src["clip_start_ms"],
                clip_end_ms=src["clip_end_ms"],
                score=hit["_score"],
                highlight=highlight_text,
                word_timestamps=[WordTimestamp(**wt) for wt in src.get("word_timestamps", [])],
                speakers=src.get("speakers", []),
                audio_link=info.get("audio_link") if info else None,
            )
        )

    response = SearchResponse(query=q, total=total, clips=clips, took_ms=took_ms)
    await cache.set(cache_key, response)
    return response
