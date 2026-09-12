"""Stage 1 request and unavailable-response contracts."""

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
    clip_start_ms: int
    clip_end_ms: int
    excerpt: str
    audio_link: str | None = None


class AskResponse(BaseModel):
    question: str
    status: Literal["disabled", "generation_unavailable"]
    reason: Literal["rag_disabled", "llm_not_configured", "not_implemented"]
    answer: None = None
    sources: list[RagSource] = Field(default_factory=list)
    retrieval_mode: None = None
    degraded: bool = False
    cached: bool = False
    took_ms: int = 0
