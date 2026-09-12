"""Elasticsearch index management and bulk indexing."""

from dataclasses import dataclass, field
from elasticsearch import Elasticsearch, BadRequestError, helpers


INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "podcast_id": {"type": "keyword"},
            "episode_id": {"type": "keyword"},
            "clip_index": {"type": "integer"},
            "clip_start_ms": {"type": "long"},
            "clip_end_ms": {"type": "long"},
            "clip_text": {
                "type": "text",
                "analyzer": "english",
                "fields": {"raw": {"type": "keyword"}},
            },
            "word_timestamps": {"type": "object", "enabled": False},
            "speakers": {"type": "integer"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}


@dataclass
class BulkResult:
    indexed: int
    failed: int
    errors: list[dict] = field(default_factory=list)


def create_index(es: Elasticsearch, index: str) -> None:
    """Create index if it does not exist. Idempotent."""
    try:
        es.indices.create(index=index, body=INDEX_MAPPING)
    except BadRequestError as e:
        if "resource_already_exists_exception" in str(e):
            return
        raise


def bulk_index(es: Elasticsearch, index: str, docs: list[dict]) -> BulkResult:
    """Bulk index documents. Returns BulkResult with counts."""
    actions = [{"_index": index, "_source": doc} for doc in docs]
    success, errors = helpers.bulk(es, actions, chunk_size=500, raise_on_error=False)
    return BulkResult(
        indexed=success,
        failed=len(errors),
        errors=errors,
    )
