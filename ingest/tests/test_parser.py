import json
import tempfile
import os
import pytest
from ingest.parser import parse_transcript, TranscriptParseError


def _write_json(data):
    """Write data to a temp JSON file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _make_transcript(words_in_last, chunks=None):
    """Build a minimal valid transcript JSON structure.

    words_in_last: list of dicts with keys word, startTime, endTime, and optionally speakerTag.
    chunks: optional list of earlier 30s chunk entries (alternatives with words).
    """
    results = []
    if chunks:
        for chunk_words in chunks:
            results.append({"alternatives": [{"transcript": "chunk", "words": chunk_words}]})
    # Last entry is the full-episode word list
    results.append({"alternatives": [{"words": words_in_last}]})
    return {"results": results}


class TestParseTranscript:
    def test_parse_valid_transcript(self):
        words = [
            {"startTime": "0.600s", "endTime": "0.800s", "word": "Hello", "speakerTag": 1},
            {"startTime": "0.800s", "endTime": "1.200s", "word": "world", "speakerTag": 2},
            {"startTime": "1.200s", "endTime": "1.500s", "word": "test", "speakerTag": 1},
        ]
        path = _write_json(_make_transcript(words))
        try:
            result = parse_transcript(path)
            assert len(result) == 3
            assert result[0].word == "Hello"
            assert result[0].start_ms == 600
            assert result[0].end_ms == 800
            assert result[0].speaker_tag == 1
            assert result[1].word == "world"
            assert result[1].speaker_tag == 2
            assert result[2].start_ms == 1200
            assert result[2].end_ms == 1500
        finally:
            os.unlink(path)

    def test_parse_time_conversion(self):
        words = [
            {"startTime": "1.200s", "endTime": "2s", "word": "a", "speakerTag": 1},
            {"startTime": "0s", "endTime": "0.100s", "word": "b", "speakerTag": 1},
            {"startTime": "61.050s", "endTime": "62s", "word": "c", "speakerTag": 1},
        ]
        path = _write_json(_make_transcript(words))
        try:
            result = parse_transcript(path)
            assert result[0].start_ms == 1200
            assert result[0].end_ms == 2000
            assert result[1].start_ms == 0
            assert result[1].end_ms == 100
            assert result[2].start_ms == 61050
            assert result[2].end_ms == 62000
        finally:
            os.unlink(path)

    def test_missing_speaker_tag_defaults_to_zero(self):
        words = [
            {"startTime": "0s", "endTime": "1s", "word": "hello"},
        ]
        path = _write_json(_make_transcript(words))
        try:
            result = parse_transcript(path)
            assert result[0].speaker_tag == 0
        finally:
            os.unlink(path)

    def test_empty_words_returns_empty_list(self):
        path = _write_json(_make_transcript([]))
        try:
            result = parse_transcript(path)
            assert result == []
        finally:
            os.unlink(path)

    def test_malformed_time_raises(self):
        words = [
            {"startTime": "abc", "endTime": "1s", "word": "hello", "speakerTag": 1},
        ]
        path = _write_json(_make_transcript(words))
        try:
            with pytest.raises(TranscriptParseError):
                parse_transcript(path)
        finally:
            os.unlink(path)

    def test_missing_results_key_raises(self):
        path = _write_json({"no_results": []})
        try:
            with pytest.raises(TranscriptParseError):
                parse_transcript(path)
        finally:
            os.unlink(path)

    def test_single_results_entry_uses_first_chunk(self):
        """When only one entry exists in results (no separate full-episode list),
        use the words from that entry."""
        data = {
            "results": [
                {
                    "alternatives": [
                        {
                            "transcript": "hello world",
                            "words": [
                                {"startTime": "0s", "endTime": "1s", "word": "hello"},
                                {"startTime": "1s", "endTime": "2s", "word": "world"},
                            ],
                        }
                    ]
                }
            ]
        }
        path = _write_json(data)
        try:
            result = parse_transcript(path)
            assert len(result) == 2
            assert result[0].word == "hello"
            assert result[1].word == "world"
        finally:
            os.unlink(path)
