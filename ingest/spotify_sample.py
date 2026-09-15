"""Bounded Spotify 2020 ZIP/tar stream ingestion into a separate real source index."""

import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import PurePosixPath
import re
import sqlite3
import tarfile
import tempfile
import zipfile

from elasticsearch import Elasticsearch, helpers
import psycopg2

from ingest.es_client import INDEX_MAPPING
from ingest.metadata_loader import REQUIRED_COLUMNS
from ingest.parser import WordRecord
from ingest.segmenter import segment
from api.rag.retrieval import parse_hit


DATASET = "spotify-podcasts-2020"
DEFAULT_ARCHIVE = "podcasts-transcripts-0to2.tar.gz"
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024
MAX_WORDS = 50000
MAX_TIME_MS = 24 * 60 * 60 * 1000


def milliseconds(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,6}(?:\.\d{1,9})?s", value):
        raise ValueError("Invalid timestamp")
    result = int(Decimal(value[:-1]) * 1000)  # Floor sub-ms precision without float rounding.
    if result > MAX_TIME_MS:
        raise ValueError("Timestamp exceeds safety bound")
    return result


def transcript_words(data, counts):
    bad = False
    groups = []
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        counts["malformed_transcripts"] += 1
        return []
    total = 0
    for result in results:
        alternatives = result.get("alternatives") if isinstance(result, dict) else None
        words = alternatives[0].get("words") if isinstance(alternatives, list) and alternatives and isinstance(alternatives[0], dict) else None
        if not isinstance(words, list) or not words:
            bad = True
            continue
        group = []
        total += len(words)
        if total > MAX_WORDS * 2:  # Includes the optional repeated aggregate.
            counts["malformed_transcripts"] += 1
            return []
        for word in words:
            try:
                text = word["word"]
                start, end = milliseconds(word["startTime"]), milliseconds(word["endTime"])
                # ASR legitimately emits zero-duration words (e.g. punctuation).
                if not isinstance(text, str) or not text.strip() or end < start:
                    raise ValueError("Invalid word")
                speaker = word.get("speakerTag", 0)
                if type(speaker) is not int or speaker < 0:
                    speaker = 0
                group.append(WordRecord(text, start, end, speaker))
            except (KeyError, TypeError, ValueError):
                counts["malformed_words"] += 1
                bad = True
        groups.append(group)
    # Spotify's final diarized aggregate can repeat the earlier segment words.
    # Detect the exact repetition, retaining the final speaker-tagged records.
    preceding = [word for group in groups[:-1] for word in group]
    signature = lambda word: (word.word, word.start_ms, word.end_ms)
    if len(groups) > 1 and list(map(signature, preceding)) == list(map(signature, groups[-1])):
        words = groups[-1]
    else:
        words = [word for group in groups for word in group]
    # Overlapping ASR/speaker turns may move backwards in time. Retain these
    # valid records in original order; do not silently sort or drop the passage.
    counts["timestamp_order_anomalies"] += sum(b.start_ms < a.start_ms for a, b in zip(words, words[1:]))
    accepted = words
    if len(accepted) > MAX_WORDS:
        accepted, bad = [], True
    if bad or not accepted:
        counts["malformed_transcripts"] += 1
    return accepted


def english(value):
    try:
        languages = ast.literal_eval(value) if value.startswith("[") else [value]
        return isinstance(languages, list) and any(str(x).lower() in ("en", "en-us", "en-gb", "english") for x in languages)
    except (ValueError, SyntaxError):
        return False


def member_name(archive, basename):
    matches = [name for name in archive.namelist() if PurePosixPath(name).name == basename]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous archive member")
    return matches[0]


def metadata_index(archive, name, database, counts):
    database.execute("CREATE TABLE metadata (episode TEXT PRIMARY KEY, show TEXT, payload TEXT)")
    with archive.open(name) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
        reader = csv.DictReader(text, delimiter="\t")
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError("Missing metadata columns")
        for row in reader:
            if any(not isinstance(row.get(key), str) for key in REQUIRED_COLUMNS):
                counts["malformed_metadata"] += 1
                continue
            if not english(row["language"]):
                counts["non_english_metadata"] += 1
                continue
            if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", row[key]) for key in ("show_filename_prefix", "episode_filename_prefix")):
                counts["malformed_metadata"] += 1
                continue
            cursor = database.execute("INSERT OR IGNORE INTO metadata VALUES (?, ?, ?)",
                (row["episode_filename_prefix"], row["show_filename_prefix"], json.dumps(row)))
            if not cursor.rowcount:
                counts["duplicate_metadata"] += 1
    database.commit()


def validate_target(index):
    if not re.fullmatch(r"podcast_clips_real_[a-z0-9][a-z0-9_-]{0,200}", index):
        raise ValueError("Target must be a concrete podcast_clips_real_* index; legacy/vector targets are forbidden")


def mapping(signature):
    body = deepcopy(INDEX_MAPPING)
    body["mappings"]["_meta"] = {"spotify_sample": signature}
    body["mappings"]["properties"].update({key: {"type": "keyword"} for key in (
        "chunk_id", "show_filename_prefix", "episode_filename_prefix", "show_uri", "episode_uri",
        "source_dataset", "source_archive", "source_member", "chunking_version")})
    return body


def ensure_index(es, index, signature):
    validate_target(index)
    expected = mapping(signature)
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body=expected)
    actual = es.indices.get_mapping(index=index)
    if set(actual) != {index} or actual[index]["mappings"].get("_meta") != expected["mappings"]["_meta"]:
        raise ValueError("Index provenance/configuration mismatch; use a new real versioned index")
    for field, spec in expected["mappings"]["properties"].items():
        if actual[index]["mappings"].get("properties", {}).get(field) != spec:
            raise ValueError("Incompatible source mapping; no index was recreated")


def save_metadata(conn, row):
    """Same PostgreSQL tables/ID convention; never overwrite existing metadata."""
    try:
        duration = int(Decimal(row["duration"]))
        if not 0 <= duration <= 2147483647:
            duration = None
    except Exception:
        duration = None
    show = (row["show_filename_prefix"], row["show_name"], row["show_description"], row["publisher"], row["rss_link"])
    episode = (row["episode_filename_prefix"], show[0], row["episode_name"], row["episode_description"], None, duration, row["language"])
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO shows (show_id,name,description,publisher,rss_link) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", show)
            cursor.execute("SELECT show_id,name,description,publisher,rss_link FROM shows WHERE show_id=%s", (show[0],))
            if cursor.fetchone() != show:
                raise ValueError("Existing show metadata conflicts; no overwrite performed")
            cursor.execute("INSERT INTO episodes (episode_id,show_id,name,description,audio_link,duration_s,language) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", episode)
            cursor.execute("SELECT episode_id,show_id,name,description,audio_link,duration_s,language FROM episodes WHERE episode_id=%s", (episode[0],))
            if cursor.fetchone() != episode:
                raise ValueError("Existing episode metadata conflicts; no overwrite performed")


def documents(words, row, archive, member, duration, overlap):
    for number, clip in enumerate(segment(words, duration, overlap)):
        doc = {"podcast_id": row["show_filename_prefix"], "episode_id": row["episode_filename_prefix"],
               "clip_index": number, "clip_start_ms": clip.start_ms, "clip_end_ms": clip.end_ms,
               "clip_text": clip.text, "word_timestamps": clip.word_timestamps, "speakers": clip.speakers}
        evidence = parse_hit({"_source": doc})
        if evidence is None:
            continue
        doc.update(chunk_id=evidence.chunk_id, show_filename_prefix=doc["podcast_id"],
                   episode_filename_prefix=doc["episode_id"], show_uri=row["show_uri"], episode_uri=row["episode_uri"],
                   source_dataset=DATASET, source_archive=archive, source_member=member, chunking_version="v1")
        yield hashlib.sha256(evidence.chunk_id.encode("utf-8")).hexdigest(), doc


def write_documents(es, index, pairs, counts):
    validate_target(index)
    for success, item in helpers.streaming_bulk(es, (
        {"_op_type": "create", "_index": index, "_id": identity, "_source": doc} for identity, doc in pairs),
        chunk_size=100, max_chunk_bytes=5 * 1024 * 1024, raise_on_error=False):
        status = item["create"]["status"]
        if success:
            counts["chunks_written"] += 1
        elif status == 409:
            counts["chunks_existing"] += 1
        else:
            raise ValueError("Source write failed; rerun the same sample safely")


def ingest_sample(path, episodes=20, per_show=5, index="podcast_clips_real_v1", archive_name=DEFAULT_ARCHIVE,
                  dry_run=True, es=None, conn=None, duration=120, overlap=60, progress=print):
    validate_target(index)
    if not 1 <= episodes <= 1000 or not 1 <= per_show <= 1000 or not 0 <= overlap < duration <= 3600:
        raise ValueError("Invalid sample or window limits")
    if archive_name not in (DEFAULT_ARCHIVE, "podcasts-transcripts-3to5.tar.gz", "podcasts-transcripts-6to7.tar.gz"):
        raise ValueError("Select a supported transcript archive")
    counts = Counter({key: 0 for key in ("episodes_scanned", "episodes_accepted", "episodes_skipped", "chunks_produced",
        "chunks_written", "chunks_existing", "malformed_transcripts", "malformed_words", "metadata_misses",
        "malformed_metadata", "non_english_metadata", "duplicate_metadata", "show_limit_skips", "duplicate_episodes",
        "timestamp_order_anomalies")})
    seen, shows = set(), Counter()
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix="spotify-sample-") as temporary:
        metadata = member_name(archive, "metadata.tsv")
        tar_name = member_name(archive, archive_name)
        signature = {"version": 1, "dataset": DATASET, "archive": archive_name,
                     "archive_crc": archive.getinfo(tar_name).CRC, "metadata_crc": archive.getinfo(metadata).CRC,
                     "duration": duration, "overlap": overlap, "max_per_show": per_show, "language": "english-only"}
        database = sqlite3.connect(os.path.join(temporary, "metadata.sqlite"))
        try:
            metadata_index(archive, metadata, database, counts)
            if not dry_run:
                if es is None or conn is None:
                    raise ValueError("Both Elasticsearch and PostgreSQL are required for writes")
                ensure_index(es, index, signature)
            with archive.open(tar_name) as raw, tarfile.open(fileobj=raw, mode="r|gz") as stream:
                for member in stream:
                    # Python 3.12 retains visited TarInfo headers even in stream
                    # mode. No name lookup is needed, so release that history.
                    stream.members.clear()
                    name = PurePosixPath(member.name)
                    if not member.isfile() or name.suffix.lower() != ".json":
                        continue
                    counts["episodes_scanned"] += 1
                    found = database.execute("SELECT show,payload FROM metadata WHERE episode=?", (name.stem,)).fetchone()
                    if found is None or name.parent.name != found[0] or ".." in name.parts:
                        counts["metadata_misses"] += 1
                        counts["episodes_skipped"] += 1
                        continue
                    if name.stem in seen or shows[found[0]] >= per_show:
                        counts["duplicate_episodes" if name.stem in seen else "show_limit_skips"] += 1
                        counts["episodes_skipped"] += 1
                        continue
                    try:
                        if not 0 < member.size <= MAX_TRANSCRIPT_BYTES:
                            raise ValueError("Transcript size limit")
                        with stream.extractfile(member) as transcript:
                            data = json.load(transcript)
                        words = transcript_words(data, counts)
                    except (ValueError, UnicodeError, RecursionError):
                        counts["malformed_transcripts"] += 1
                        words = []
                    row = json.loads(found[1])
                    pairs = list(documents(words, row, archive_name, member.name, duration, overlap))
                    if not pairs:
                        counts["episodes_skipped"] += 1
                        continue
                    counts["chunks_produced"] += len(pairs)
                    if not dry_run:
                        save_metadata(conn, row)
                        write_documents(es, index, pairs, counts)
                    seen.add(name.stem)
                    shows[found[0]] += 1
                    counts["episodes_accepted"] += 1
                    progress(json.dumps(dict(counts), sort_keys=True))
                    if counts["episodes_accepted"] == episodes:
                        break
        finally:
            database.close()
    counts["sample_complete"] = counts["episodes_accepted"] == episodes
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-per-show", type=int, default=5)
    parser.add_argument("--index", default="podcast_clips_real_v1")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clip-duration", type=int, default=int(os.environ.get("CLIP_DURATION_DEFAULT", "120")))
    parser.add_argument("--overlap", type=int, default=int(os.environ.get("CLIP_OVERLAP", "60")))
    args = parser.parse_args()
    es = conn = None
    try:
        validate_target(args.index)
        if not args.dry_run:
            es = Elasticsearch(os.environ.get("ES_HOST", "http://localhost:9200"))
            conn = psycopg2.connect(os.environ.get("POSTGRES_DSN", "postgresql://podcast:podcast@localhost:5432/podcasts"),
                                    connect_timeout=5, options="-c statement_timeout=10000")
        counts = ingest_sample(args.zip, args.episodes, args.max_per_show, args.index, args.archive,
                               args.dry_run, es, conn, args.clip_duration, args.overlap)
        print(json.dumps(counts, sort_keys=True))
    except Exception:
        parser.exit(2, "Sample ingestion stopped. Check archive, metadata conflicts, target compatibility and services; completed chunks remain safe to resume.\n")
    finally:
        if conn is not None: conn.close()
        if es is not None: es.close()
    if not counts["sample_complete"]:
        parser.exit(1, "Archive exhausted before the requested English sample was collected.\n")


if __name__ == "__main__":
    main()
