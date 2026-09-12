"""Pydantic request/response models for the search API."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    q: str
    clip_minutes: float = 2.0
    from_: int = Field(0, alias="from")
    size: int = 10


class WordTimestamp(BaseModel):
    word: str
    start_ms: int
    end_ms: int


class ClipResult(BaseModel):
    podcast_id: str
    episode_id: str
    show_name: str
    episode_name: str
    clip_start_ms: int
    clip_end_ms: int
    score: float
    highlight: str
    word_timestamps: list[WordTimestamp]
    speakers: list[int]
    audio_link: str | None


class SearchResponse(BaseModel):
    query: str
    total: int
    clips: list[ClipResult]
    took_ms: int
