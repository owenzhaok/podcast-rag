"""Question answering, validated paragraphs, and evidence response contracts."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnswerParagraph(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    source_ids: list[str] = Field(min_length=1, max_length=20)

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value):
        if not value.strip():
            raise ValueError("Empty paragraph")
        return value


class Answer(BaseModel):
    paragraphs: list[AnswerParagraph]


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
    status: Literal["disabled", "generation_unavailable", "insufficient_context", "answered", "invalid_generation"]
    reason: str | None = None
    answer: Answer | None = None
    sources: list[RagSource] = Field(default_factory=list)
    retrieval_mode: Literal["bm25", "hybrid"] | None = None
    degraded: bool = False
    cached: bool = False
    took_ms: int = 0
