"""Parse Spotify podcast transcript JSON files into word records."""

from dataclasses import dataclass
import json
import re


class TranscriptParseError(Exception):
    pass


@dataclass
class WordRecord:
    word: str
    start_ms: int
    end_ms: int
    speaker_tag: int


_TIME_RE = re.compile(r"^(\d+(?:\.\d+)?)s$")


def _parse_time(time_str: str) -> int:
    """Convert '1.200s' to 1200 (milliseconds)."""
    m = _TIME_RE.match(time_str)
    if not m:
        raise TranscriptParseError(f"Malformed time string: {time_str!r}")
    return int(float(m.group(1)) * 1000)


def parse_transcript(filepath: str) -> list[WordRecord]:
    """Parse a Spotify transcript JSON file.

    Returns a list of WordRecord(word, start_ms, end_ms, speaker_tag).
    Uses the last element of results[] as the canonical source (has speaker tags).
    Falls back to the first element if there is only one.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    if "results" not in data:
        raise TranscriptParseError("Missing 'results' key in transcript JSON")

    results = data["results"]
    if not results:
        raise TranscriptParseError("Empty 'results' array in transcript JSON")

    entry = results[-1]
    alternatives = entry.get("alternatives", [])
    if not alternatives:
        raise TranscriptParseError("No alternatives in results entry")

    words_raw = alternatives[0].get("words", [])
    records = []
    for w in words_raw:
        records.append(
            WordRecord(
                word=w["word"],
                start_ms=_parse_time(w["startTime"]),
                end_ms=_parse_time(w["endTime"]),
                speaker_tag=w.get("speakerTag", 0),
            )
        )
    return records
