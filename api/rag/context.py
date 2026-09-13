"""Deterministic evidence selection without a tokenizer or text truncation."""

from api.rag.sources import Evidence


def evidence_size(text: str) -> int:
    """UTF-8 bytes: conservative token proxy, NOT a model token guarantee.

    This bounds exact transcript evidence only, excluding metadata/JSON framing.
    A future tokenizer can replace this function and its configured budget.
    """
    return len(text.encode("utf-8"))


def overlaps(left: Evidence, right: Evidence) -> bool:
    if (left.podcast_id, left.episode_id) != (right.podcast_id, right.episode_id):
        return False
    shared = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    shorter = min(left.end_ms - left.start_ms, right.end_ms - right.start_ms)
    # 45% catches nominal 50% windows whose endpoints follow actual words.
    return shared / shorter >= 0.45


def select_context(candidates: list[Evidence], max_sources: int, max_bytes: int) -> list[Evidence]:
    ranked = sorted(candidates, key=lambda c: (-c.score, c.chunk_id, c.clip_index, c.speakers))
    selected, seen = [], set()
    remaining = max_bytes
    for candidate in ranked:
        identity = candidate.chunk_id
        if identity in seen:
            continue
        seen.add(identity)
        cost = evidence_size(candidate.text)
        if cost > remaining or any(overlaps(candidate, other) for other in selected):
            continue
        selected.append(candidate)
        remaining -= cost
        if len(selected) >= max_sources:
            break
    return selected
