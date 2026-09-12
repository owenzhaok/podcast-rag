from ingest.parser import WordRecord
from ingest.segmenter import segment


def _make_words(count: int, interval_ms: int = 500) -> list[WordRecord]:
    """Create count words spaced interval_ms apart starting at 0."""
    return [
        WordRecord(word=f"word{i}", start_ms=i * interval_ms, end_ms=i * interval_ms + 300, speaker_tag=(i % 2) + 1)
        for i in range(count)
    ]


class TestSegment:
    def test_segment_basic_clip_count(self):
        # 300s of words (600 words * 500ms each), 120s clips, 60s overlap
        # Step = 120 - 60 = 60s. Starts: 0, 60, 120, 180, 240 -> 5 clips
        words = _make_words(600, interval_ms=500)  # 0ms to 299_500ms = ~300s
        clips = segment(words, clip_duration_s=120, overlap_s=60)
        assert len(clips) == 5

    def test_clip_overlap_contains_shared_words(self):
        words = _make_words(600, interval_ms=500)
        clips = segment(words, clip_duration_s=120, overlap_s=60)
        # Clip 0 covers [0, 120_000), Clip 1 covers [60_000, 180_000)
        clip0_starts = {w["start_ms"] for w in clips[0].word_timestamps}
        clip1_starts = {w["start_ms"] for w in clips[1].word_timestamps}
        shared = clip0_starts & clip1_starts
        assert len(shared) > 0

    def test_clip_text_is_space_joined(self):
        words = _make_words(100, interval_ms=500)
        clips = segment(words, clip_duration_s=120, overlap_s=60)
        expected_words = [w.word for w in words if w.start_ms < 120_000]
        assert clips[0].text == " ".join(expected_words)

    def test_single_clip_when_audio_shorter_than_duration(self):
        words = _make_words(100, interval_ms=500)  # 50s of audio
        clips = segment(words, clip_duration_s=120, overlap_s=60)
        assert len(clips) == 1

    def test_empty_words_returns_empty_list(self):
        assert segment([], clip_duration_s=120, overlap_s=60) == []

    def test_clip_start_end_match_word_boundaries(self):
        words = _make_words(600, interval_ms=500)
        clips = segment(words, clip_duration_s=120, overlap_s=60)
        clip = clips[0]
        assert clip.start_ms == 0
        last_word_in_window = [w for w in words if w.start_ms < 120_000][-1]
        assert clip.end_ms == last_word_in_window.end_ms

    def test_speakers_deduplicated(self):
        words = _make_words(600, interval_ms=500)  # alternates speaker 1 and 2
        clips = segment(words, clip_duration_s=120, overlap_s=60)
        assert sorted(clips[0].speakers) == [1, 2]

    def test_custom_duration_and_overlap(self):
        # 600s of words (1200 * 500ms), 300s clips, 120s overlap -> step=180s
        # max_start_ms = 599_500. Starts: 0, 180k, 360k, 540k -> 4 clips
        words = _make_words(1200, interval_ms=500)
        clips = segment(words, clip_duration_s=300, overlap_s=120)
        assert len(clips) == 4
