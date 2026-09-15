"""Synthetic archives only: never copy dataset transcripts/metadata into fixtures."""

from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import tarfile
from unittest.mock import MagicMock
from uuid import uuid4
import zipfile

import pytest

import ingest.spotify_sample as sample
from api.rag.retrieval import parse_hit
from api.rag.backfill import vector_document
from api.rag.vector_index import VectorConfig


def row(show="show_a", episode="ep1", language="['en']"):
    return dict(show_uri="spotify:show:" + show, show_name="Synthetic Show", show_description="Test description",
                publisher="Test", language=language, rss_link="https://example.invalid/rss",
                episode_uri="spotify:episode:" + episode, episode_name="Synthetic Episode", episode_description="Test",
                duration="240.5", show_filename_prefix=show, episode_filename_prefix=episode)


def word(text="hello", start="0.600s", end="0.800s", **kwargs):
    return {"word": text, "startTime": start, "endTime": end, **kwargs}


def transcript(words=None):
    return {"results": [{"alternatives": [{"words": words if words is not None else [word()]}]}]}


def archive_file(tmp_path, rows, files):
    tsv = io.StringIO()
    writer = csv.DictWriter(tsv, fieldnames=sorted(sample.REQUIRED_COLUMNS), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w:gz") as handle:
        for name, payload in files:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))
    path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("dataset/metadata.tsv", tsv.getvalue())
        handle.writestr("dataset/" + sample.DEFAULT_ARCHIVE, tar.getvalue())
        handle.writestr("unused.tar.gz", b"never opened")
    return path


@pytest.mark.parametrize("value,expected", [("0.600s", 600), ("2s", 2000), ("1.001s", 1001),
                                           ("0s", 0), ("2.1239s", 2123)])
def test_timestamps(value, expected):
    assert sample.milliseconds(value) == expected


@pytest.mark.parametrize("value", [None, 1, "NaNs", "-1s", "1ms", "1", "900000s", "1e3s"])
def test_bad_timestamps(value):
    with pytest.raises(ValueError): sample.milliseconds(value)


def test_reconstruction_and_aggregate():
    a, b = word("Hello"), word("world", "1s", "2s")
    data = {"results": [transcript([a])["results"][0], transcript([b])["results"][0],
                        transcript([{**a, "speakerTag": 1}, {**b, "speakerTag": 2}])["results"][0]]}
    counts = Counter()
    words = sample.transcript_words(data, counts)
    assert [w.word for w in words] == ["Hello", "world"]
    assert [w.speaker_tag for w in words] == [1, 2]
    assert not counts["malformed_transcripts"]
    data["results"].pop()
    assert [w.word for w in sample.transcript_words(data, Counter())] == ["Hello", "world"]


def test_defensive_words_preserve_order():
    counts = Counter()
    words = sample.transcript_words(transcript([word(), None, word(start="bad"), word("later", "2s", "3s"),
                                               word("out of order", "1s", "2s")]), counts)
    assert [w.word for w in words] == ["hello", "later", "out of order"]
    assert counts["malformed_transcripts"] == 1 and counts["malformed_words"] == 2
    assert counts["timestamp_order_anomalies"] == 1


def test_zero_duration_asr_words_are_preserved():
    counts = Counter()
    words = sample.transcript_words(transcript([word("punctuation", "1s", "1s")]), counts)
    assert len(words) == 1 and not counts["malformed_words"]


@pytest.mark.parametrize("data", [None, [], {}, {"results": []}, {"results": [None]},
                                   {"results": [{"alternatives": []}]}, transcript([])])
def test_malformed_structure(data):
    counts = Counter()
    assert sample.transcript_words(data, counts) == [] and counts["malformed_transcripts"] == 1


def test_window_identity_and_provenance():
    words = sample.transcript_words(transcript([word(str(i), f"{i}s", f"{i+1}s") for i in (0, 60, 119, 120, 180)]), Counter())
    pairs = list(sample.documents(words, row(), sample.DEFAULT_ARCHIVE, "show_a/ep1.json", 120, 60))
    assert [d["clip_text"] for _, d in pairs] == ["0 60 119", "60 119 120", "120 180", "180"]
    for identity, doc in pairs:
        assert doc["chunk_id"] == parse_hit({"_source": doc}).chunk_id
        vector_id, _ = vector_document({"_source": doc}, VectorConfig())
        assert identity == vector_id
        assert doc["episode_uri"] == row()["episode_uri"] and doc["show_filename_prefix"] == "show_a"
        assert doc["source_dataset"] == "spotify-podcasts-2020"
    assert pairs == list(sample.documents(words, row(), sample.DEFAULT_ARCHIVE, "show_a/ep1.json", 120, 60))


def test_streaming_sampling_and_dry_run(tmp_path, monkeypatch):
    rows = [row(episode="a"), row(episode="b"), row("show_b", "c"), row("show_c", "d")]
    path = archive_file(tmp_path, rows, [(f"prefix/{r['show_filename_prefix']}/{r['episode_filename_prefix']}.json", transcript()) for r in rows])
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda *args: pytest.fail("No extraction"))
    monkeypatch.setattr(tarfile.TarFile, "extractall", lambda *args: pytest.fail("No extraction"))
    opened = []
    original = tarfile.open
    def open_stream(*args, **kwargs):
        opened.append(kwargs["mode"])
        return original(*args, **kwargs)
    monkeypatch.setattr(sample.tarfile, "open", open_stream)
    es, conn = MagicMock(), MagicMock()
    first = sample.ingest_sample(path, episodes=2, per_show=1, dry_run=True, es=es, conn=conn, progress=lambda _: None)
    second = sample.ingest_sample(path, episodes=2, per_show=1, progress=lambda _: None)
    assert first == second and opened == ["r|gz", "r|gz"]
    assert first["episodes_scanned"] == 3 and first["episodes_accepted"] == 2
    assert first["show_limit_skips"] == 1 and first["chunks_produced"] == 2
    assert not es.mock_calls and not conn.mock_calls


def test_skips_malformed_missing_and_nonenglish(tmp_path):
    rows = [row(episode="bad"), row(episode="spanish", language="es"), row(episode="good")]
    files = [("show_a/unknown.json", transcript()), ("show_a/spanish.json", transcript()),
             ("show_a/bad.json", b"not json"), ("show_a/good.json", transcript())]
    path = archive_file(tmp_path, rows, files)
    result = sample.ingest_sample(path, episodes=1, progress=lambda _: None)
    assert result["metadata_misses"] == 2 and result["malformed_transcripts"] == 1
    assert result["episodes_scanned"] == 4 and result["episodes_skipped"] == 3
    assert result["non_english_metadata"] == 1 and result["sample_complete"]


@pytest.mark.parametrize("index", ["podcast_clips", "podcast_rag_v1", "*", "podcast_clips_real_*",
                                   "podcast_clips_real_a,b", "../podcast_clips", "_all"])
def test_protected_targets_before_any_io(index):
    with pytest.raises(ValueError): sample.ingest_sample("nonexistent.zip", index=index)


def test_idempotent_writer_and_resume(tmp_path, monkeypatch):
    path = archive_file(tmp_path, [row()], [("show_a/ep1.json", transcript())])
    documents, metadata = {}, []
    def bulk(es, actions, **kwargs):
        for action in actions:
            assert action["_index"] == "podcast_clips_real_v1" and action["_op_type"] == "create"
            exists = action["_id"] in documents
            documents.setdefault(action["_id"], action["_source"])
            yield not exists, {"create": {"status": 409 if exists else 201}}
    monkeypatch.setattr(sample.helpers, "streaming_bulk", bulk)
    monkeypatch.setattr(sample, "ensure_index", lambda *args: None)
    monkeypatch.setattr(sample, "save_metadata", lambda conn, row: metadata.append(row))
    first = sample.ingest_sample(path, episodes=1, dry_run=False, es=object(), conn=object(), progress=lambda _: None)
    second = sample.ingest_sample(path, episodes=1, dry_run=False, es=object(), conn=object(), progress=lambda _: None)
    assert first["chunks_written"] == 1 and second["chunks_written"] == 0 and second["chunks_existing"] == 1
    assert len(documents) == 1 and metadata[0]["episode_name"] == "Synthetic Episode"


def test_mapping_creation_and_incompatible_target():
    es = MagicMock()
    es.indices.exists.return_value = False
    es.indices.get_mapping.return_value = {"podcast_clips_real_v1": {"mappings": sample.mapping({"version": 1})["mappings"]}}
    sample.ensure_index(es, "podcast_clips_real_v1", {"version": 1})
    es.indices.create.assert_called_once()
    es.indices.exists.return_value = True
    with pytest.raises(ValueError): sample.ensure_index(es, "podcast_clips_real_v1", {"version": 2})
    es.indices.get_mapping.return_value = {"podcast_clips": {"mappings": {}}}
    with pytest.raises(ValueError): sample.ensure_index(es, "podcast_clips_real_v1", {"version": 1})
    es.indices.delete.assert_not_called()


def test_pg_metadata_conflict_is_not_overwritten():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ("conflicting",)
    with pytest.raises(ValueError, match="conflicts"):
        sample.save_metadata(conn, row())
    sql = " ".join(call.args[0] for call in cursor.execute.call_args_list)
    assert "ON CONFLICT DO NOTHING" in sql and "DO UPDATE" not in sql


def test_cli_dry_run_never_connects(tmp_path, monkeypatch):
    path = archive_file(tmp_path, [row()], [("show_a/ep1.json", transcript())])
    monkeypatch.setattr("sys.argv", ["sample", "--zip", str(path), "--episodes", "1", "--dry-run"])
    monkeypatch.setattr(sample, "Elasticsearch", lambda *args: pytest.fail("No ES in dry run"))
    monkeypatch.setattr(sample.psycopg2, "connect", lambda *args, **kwargs: pytest.fail("No PG in dry run"))
    sample.main()


def test_partial_write_can_resume(monkeypatch):
    doc_pairs = [("a", {"clip_text": "Synthetic A"}), ("b", {"clip_text": "Synthetic B"})]
    stored = set()
    def interrupted(es, actions, **kwargs):
        for action in actions:
            stored.add(action["_id"])
            yield True, {"create": {"status": 201}}
            raise OSError("Synthetic interruption")
    monkeypatch.setattr(sample.helpers, "streaming_bulk", interrupted)
    with pytest.raises(OSError): sample.write_documents(None, "podcast_clips_real_test", doc_pairs, Counter())
    def resume(es, actions, **kwargs):
        for action in actions:
            exists = action["_id"] in stored
            stored.add(action["_id"])
            yield not exists, {"create": {"status": 409 if exists else 201}}
    monkeypatch.setattr(sample.helpers, "streaming_bulk", resume)
    counts = Counter()
    sample.write_documents(None, "podcast_clips_real_test", doc_pairs, counts)
    assert counts["chunks_existing"] == counts["chunks_written"] == 1 and stored == {"a", "b"}


@pytest.mark.skipif(os.environ.get("RAG_INTEGRATION_TESTS") != "1", reason="Opt-in ES/PostgreSQL integration")
def test_real_services_with_synthetic_archive(tmp_path):
    # PG temp tables shadow public tables only on this test connection, and are
    # dropped by PostgreSQL on close. No real corpus metadata is modified.
    index = "podcast_clips_real_test_" + uuid4().hex
    path = archive_file(tmp_path, [row()], [("show_a/ep1.json", transcript())])
    conn = sample.psycopg2.connect(os.environ.get("POSTGRES_DSN", "postgresql://podcast:podcast@localhost:5432/podcasts"))
    es = sample.Elasticsearch(os.environ.get("ES_HOST", "http://localhost:9200"))
    created = False
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE TEMP TABLE shows (LIKE public.shows INCLUDING ALL)")
            cursor.execute("CREATE TEMP TABLE episodes (LIKE public.episodes INCLUDING ALL)")
        conn.commit()
        # Own the unique test index before the importer checks it, so cleanup is exact.
        with zipfile.ZipFile(path) as archive:
            signature = {"version": 1, "dataset": sample.DATASET, "archive": sample.DEFAULT_ARCHIVE,
                "archive_crc": archive.getinfo("dataset/" + sample.DEFAULT_ARCHIVE).CRC,
                "metadata_crc": archive.getinfo("dataset/metadata.tsv").CRC,
                "duration": 120, "overlap": 60, "max_per_show": 5, "language": "english-only"}
        es.indices.create(index=index, body=sample.mapping(signature))
        created = True
        first = sample.ingest_sample(path, episodes=1, index=index, dry_run=False, es=es, conn=conn, progress=lambda _: None)
        second = sample.ingest_sample(path, episodes=1, index=index, dry_run=False, es=es, conn=conn, progress=lambda _: None)
        es.indices.refresh(index=index)
        assert first["chunks_written"] == second["chunks_existing"] == 1 and second["chunks_written"] == 0
        assert es.count(index=index)["count"] == 1
        doc = es.search(index=index)["hits"]["hits"][0]["_source"]
        assert parse_hit({"_source": doc}).chunk_id == doc["chunk_id"]
        with conn.cursor() as cursor:
            cursor.execute("SELECT name,show_id FROM episodes WHERE episode_id='ep1'")
            assert cursor.fetchone() == ("Synthetic Episode", "show_a")
        # Existing metadata remains untouched when a later row conflicts.
        with pytest.raises(ValueError): sample.save_metadata(conn, {**row(), "episode_name": "Conflicting"})
        with conn.cursor() as cursor:
            cursor.execute("SELECT name FROM episodes WHERE episode_id='ep1'")
            assert cursor.fetchone()[0] == "Synthetic Episode"
    finally:
        if created: es.indices.delete(index=index)
        es.close()
        conn.close()
