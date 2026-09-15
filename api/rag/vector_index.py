"""Explicit companion-index preparation, never imported by API startup."""

import os
import re
from dataclasses import dataclass

from api.rag.embeddings import FakeEmbeddings


@dataclass(frozen=True)
class VectorConfig:
    source_index: str = "podcast_clips"
    vector_index: str = "podcast_rag_v1"
    provider: str = ""
    model: str = ""
    dimensions: int = 0
    batch_size: int = 32
    revision: str = "v1"
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls):
        try:
            return cls(
                source_index=os.environ.get("RAG_SOURCE_INDEX", "podcast_clips"),
                vector_index=os.environ.get("RAG_VECTOR_INDEX", "podcast_rag_v1"),
                provider=os.environ.get("RAG_EMBEDDING_PROVIDER", ""),
                model=os.environ.get("RAG_EMBEDDING_MODEL", ""),
                dimensions=int(os.environ.get("RAG_EMBEDDING_DIMENSIONS", "") or "0"),
                batch_size=int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE", "32")),
                revision=os.environ.get("RAG_EMBEDDING_REVISION", "v1"),
                timeout_seconds=int(os.environ.get("RAG_EMBEDDING_TIMEOUT_SECONDS", "20")),
            )
        except ValueError:
            raise ValueError("Invalid vector preparation numeric configuration") from None

    def validate(self):
        for name in (self.source_index, self.vector_index):
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,254}", name):
                raise ValueError("Use concrete index names without wildcards or aliases")
        if self.source_index == self.vector_index:
            raise ValueError("Companion index must differ from source index")
        if not 1 <= self.dimensions <= 4096 or not 1 <= self.batch_size <= 200:
            raise ValueError("Dimensions must be 1-4096 and batch size 1-200")
        if not 1 <= self.timeout_seconds <= 120:
            raise ValueError("Embedding timeout must be 1-120 seconds")
        if self.provider == "gemini" and (self.dimensions != 768 or "/" in self.model):
            raise ValueError("Gemini preparation requires 768 dimensions and an unprefixed model identifier")
        for value in (self.provider, self.model, self.revision):
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]{0,127}", value):
                raise ValueError("Embedding provider, model and revision must be configured identifiers")

    def make_provider(self):
        self.validate()
        if self.provider == "gemini":
            from api.rag.gemini import GeminiEmbeddings
            return GeminiEmbeddings(self.model, os.environ.get("RAG_EMBEDDING_API_KEY", "").strip(),
                                    self.dimensions, self.timeout_seconds)
        if self.provider != "fake":
            raise ValueError("Only the explicit fake provider is available; select a real adapter separately")
        return FakeEmbeddings(self.dimensions, self.model, self.revision)


def vector_mapping(config: VectorConfig) -> dict:
    config.validate()
    properties = {name: {"type": "keyword"} for name in (
        "chunk_id", "podcast_id", "episode_id", "content_hash", "embedding_provider",
        "embedding_model", "embedding_revision", "chunking_version", "source_index",
    )}
    properties.update({
        "clip_start_ms": {"type": "long"}, "clip_end_ms": {"type": "long"},
        "chunk_text": {"type": "text", "index": False},
        "speakers": {"type": "integer"},
        "embedding": {"type": "dense_vector", "element_type": "float", "dims": config.dimensions,
                      "index": True, "similarity": "cosine", "index_options": {"type": "hnsw"}},
    })
    if config.provider == "gemini":
        properties.update({"embedding_input_hash": {"type": "keyword"},
                           "embedding_input_version": {"type": "keyword"}})
    return {"dynamic": "strict", "_meta": {"rag_vector_version": 1}, "properties": properties}


def check_index(es, config: VectorConfig) -> None:
    expected = vector_mapping(config)
    actual = es.indices.get_mapping(index=config.vector_index)
    # Refuse aliases even if they resolve to one compatible index.
    if set(actual) != {config.vector_index}:
        raise ValueError("Vector target must be a concrete companion index")
    mapping = actual[config.vector_index]["mappings"]
    if mapping.get("dynamic") != "strict" or mapping.get("_meta") != expected["_meta"]:
        raise ValueError("Incompatible vector index metadata; use a new versioned index")
    for field, definition in expected["properties"].items():
        stored = mapping.get("properties", {}).get(field, {})
        for key, value in definition.items():
            if key == "element_type" and stored.get(key, "float") == value:
                continue
            if key == "index_options":
                if stored.get(key, {}).get("type") == value["type"]:
                    continue
            elif stored.get(key) == value:
                continue
            raise ValueError("Incompatible vector mapping/dimensions; no index was recreated")
    if config.provider == "gemini" or "embedding_input_version" in mapping.get("properties", {}):
        # Refresh only the already-validated companion index so foreign unrefreshed
        # documents cannot be missed. Run one backfill per target at a time.
        es.indices.refresh(index=config.vector_index)
        incompatible = es.count(index=config.vector_index, query={"bool": {"must_not": [{"bool": {
            "filter": [{"term": {"embedding_provider": config.provider}},
                       {"term": {"embedding_model": config.model}}]}}]}})
        if incompatible["count"]:
            raise ValueError("Target contains fake/other model vectors; use a new versioned index, no overwrite performed")
        if config.provider == "gemini":
            from api.rag.embedding_input import DOCUMENT_INPUT_VERSION
            old_format = es.count(index=config.vector_index, query={"bool": {
                "filter": [{"term": {"embedding_revision": config.revision}}],
                "must_not": [{"term": {"embedding_input_version": DOCUMENT_INPUT_VERSION}}],
            }})
            if old_format["count"]:
                raise ValueError("Document format changed; bump embedding revision before re-embedding")


def create_index(es, config: VectorConfig) -> None:
    config.validate()
    if not es.indices.exists(index=config.vector_index):
        es.indices.create(index=config.vector_index, mappings=vector_mapping(config),
                          settings={"number_of_shards": 1, "number_of_replicas": 0})
    check_index(es, config)
