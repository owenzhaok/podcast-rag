"""Segment a word list into overlapping clips."""

from dataclasses import dataclass, field
from ingest.parser import WordRecord


@dataclass
class Clip:
    start_ms: int
    end_ms: int
    text: str
    word_timestamps: list[dict] = field(default_factory=list)
    speakers: list[int] = field(default_factory=list)


def segment(
    words: list[WordRecord],
    clip_duration_s: int = 120,
    overlap_s: int = 60,
) -> list[Clip]:
    """Produce overlapping clips from a word list."""
    if not words:
        return []

    clip_duration_ms = clip_duration_s * 1000
    overlap_ms = overlap_s * 1000
    step_ms = clip_duration_ms - overlap_ms

    max_start_ms = max(w.start_ms for w in words)

    clips = []
    window_start = 0
    while window_start <= max_start_ms:
        window_end = window_start + clip_duration_ms
        window_words = [w for w in words if window_start <= w.start_ms < window_end]
        if not window_words:
            window_start += step_ms
            continue

        clips.append(
            Clip(
                start_ms=window_words[0].start_ms,
                end_ms=window_words[-1].end_ms,
                text=" ".join(w.word for w in window_words),
                word_timestamps=[
                    {"word": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms}
                    for w in window_words
                ],
                speakers=sorted(set(w.speaker_tag for w in window_words)),
            )
        )
        window_start += step_ms

    return clips
