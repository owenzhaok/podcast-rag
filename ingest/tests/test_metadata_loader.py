import os
import sqlite3
import tempfile
import pytest
from ingest.metadata_loader import load_metadata, MetadataLoadError


def _create_db():
    """Create an in-memory SQLite DB with the same schema as Postgres."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE shows (
            show_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            publisher TEXT,
            rss_link TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE episodes (
            episode_id TEXT PRIMARY KEY,
            show_id TEXT REFERENCES shows(show_id),
            name TEXT,
            description TEXT,
            audio_link TEXT,
            duration_s INTEGER,
            language TEXT
        )
    """)
    conn.commit()
    return conn


HEADER = "show_uri\tshow_name\tshow_description\tpublisher\tlanguage\trss_link\tepisode_uri\tepisode_name\tepisode_description\tduration\tshow_filename_prefix\tepisode_filename_prefix"

ROW1 = "spotify:show:abc123\tMy Show\tA show\tPublisher1\ten\thttp://rss1\tspotify:episode:ep001\tEpisode 1\tDesc 1\t3600\tshow_abc123\tep001"
ROW2 = "spotify:show:abc123\tMy Show\tA show\tPublisher1\ten\thttp://rss1\tspotify:episode:ep002\tEpisode 2\tDesc 2\t1800\tshow_abc123\tep002"
ROW3 = "spotify:show:def456\tOther Show\tAnother\tPublisher2\tes\thttp://rss2\tspotify:episode:ep003\tEpisode 3\tDesc 3\t7200\tshow_def456\tep003"


def _write_tsv(rows: list[str]) -> str:
    """Write TSV lines to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
    f.write(HEADER + "\n")
    for row in rows:
        f.write(row + "\n")
    f.close()
    return f.name


class TestLoadMetadata:
    def test_load_valid_tsv(self):
        conn = _create_db()
        path = _write_tsv([ROW1, ROW2, ROW3])
        try:
            result = load_metadata(path, conn)
            assert result.shows_upserted == 2
            assert result.episodes_upserted == 3

            shows = conn.execute("SELECT * FROM shows ORDER BY show_id").fetchall()
            assert len(shows) == 2

            episodes = conn.execute("SELECT * FROM episodes ORDER BY episode_id").fetchall()
            assert len(episodes) == 3
        finally:
            os.unlink(path)

    def test_upsert_is_idempotent(self):
        conn = _create_db()
        path = _write_tsv([ROW1])
        try:
            load_metadata(path, conn)
            load_metadata(path, conn)
            shows = conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]
            episodes = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            assert shows == 1
            assert episodes == 1
        finally:
            os.unlink(path)

    def test_missing_columns_raises(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
        f.write("col1\tcol2\n")
        f.write("a\tb\n")
        f.close()
        try:
            conn = _create_db()
            with pytest.raises(MetadataLoadError):
                load_metadata(f.name, conn)
        finally:
            os.unlink(f.name)

    def test_duration_parsed_as_integer(self):
        conn = _create_db()
        path = _write_tsv([ROW1])
        try:
            load_metadata(path, conn)
            dur = conn.execute("SELECT duration_s FROM episodes WHERE episode_id='ep001'").fetchone()[0]
            assert dur == 3600
            assert isinstance(dur, int)
        finally:
            os.unlink(path)
