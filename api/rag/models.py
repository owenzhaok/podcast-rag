"""Evidence-only question answering contracts; generation remains unavailable."""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, strict=True)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class RagSource(BaseModel):
    source_id: str
    chunk_id: str
    podcast_id: str
    episode_id: str
    show_name: str
    episode_name: str
    clip_index: int = 0
    clip_start_ms: int
    clip_end_ms: int
    excerpt: str
    audio_link: str | None = None
    speakers: list[int] = Field(default_factory=list)
    metadata_available: bool = True


class AskResponse(BaseModel):
    question: str
    status: Literal["disabled", "generation_unavailable", "insufficient_context"]
    reason: Literal["rag_disabled", "llm_not_configured", "not_implemented", "no_usable_evidence"]
    answer: None = None
    sources: list[RagSource] = Field(default_factory=list)
    retrieval_mode: Literal["bm25"] | None = None
    degraded: bool = False
    cached: bool = False
    took_ms: int = 0
