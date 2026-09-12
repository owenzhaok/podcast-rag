"""CLI entry point for the podcast ingest pipeline."""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from elasticsearch import Elasticsearch
import psycopg2
from tqdm import tqdm

from ingest.parser import parse_transcript, TranscriptParseError
from ingest.segmenter import segment
from ingest.es_client import create_index, bulk_index


def process_transcript(filepath: str, show_id: str, episode_id: str,
                       clip_duration: int, overlap: int) -> list[dict]:
    """Parse and segment one transcript file. Returns clip docs ready for ES."""
    words = parse_transcript(filepath)
    if not words:
        return []
    clips = segment(words, clip_duration_s=clip_duration, overlap_s=overlap)
    docs = []
    for i, clip in enumerate(clips):
        docs.append({
            "podcast_id": show_id,
            "episode_id": episode_id,
            "clip_index": i,
            "clip_start_ms": clip.start_ms,
            "clip_end_ms": clip.end_ms,
            "clip_text": clip.text,
            "word_timestamps": clip.word_timestamps,
            "speakers": clip.speakers,
        })
    return docs


def find_transcripts(transcripts_dir: str) -> list[tuple[str, str, str]]:
    """Walk the transcript directory tree.

    Returns list of (filepath, show_id, episode_id) tuples.
    Expected structure: .../show_{id}/{episode_id}.json
    """
    results = []
    for root, dirs, files in os.walk(transcripts_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(root, fname)
            episode_id = fname.replace(".json", "")
            show_dir = os.path.basename(root)
            results.append((filepath, show_dir, episode_id))
    return results


def _load_metadata_pg(tsv_path: str, conn):
    """Load metadata using Postgres ON CONFLICT syntax."""
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cursor = conn.cursor()
        shows_seen = set()
        count = 0
        for row in reader:
            show_id = row["show_filename_prefix"]
            episode_id = row["episode_filename_prefix"]
            if show_id not in shows_seen:
                cursor.execute(
                    "INSERT INTO shows (show_id, name, description, publisher, rss_link) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (show_id) DO UPDATE SET name=EXCLUDED.name, description=EXCLUDED.description, "
                    "publisher=EXCLUDED.publisher, rss_link=EXCLUDED.rss_link",
                    (show_id, row["show_name"], row["show_description"], row["publisher"], row["rss_link"]),
                )
                shows_seen.add(show_id)
            cursor.execute(
                "INSERT INTO episodes (episode_id, show_id, name, description, audio_link, duration_s, language) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (episode_id) DO UPDATE SET name=EXCLUDED.name, description=EXCLUDED.description, "
                "duration_s=EXCLUDED.duration_s, language=EXCLUDED.language",
                (episode_id, show_id, row["episode_name"], row["episode_description"],
                 None, int(float(row["duration"])), row["language"]),
            )
            count += 1
        conn.commit()
        print(f"  Loaded {len(shows_seen)} shows, {count} episodes.")


def main():
    parser = argparse.ArgumentParser(description="Ingest podcast transcripts")
    parser.add_argument("--transcripts-dir", required=True, help="Root directory of extracted transcripts")
    parser.add_argument("--metadata-tsv", required=True, help="Path to metadata.tsv")
    parser.add_argument("--clip-duration", type=int, default=int(os.environ.get("CLIP_DURATION_DEFAULT", 120)))
    parser.add_argument("--overlap", type=int, default=int(os.environ.get("CLIP_OVERLAP", 60)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--es-host", default=os.environ.get("ES_HOST", "http://localhost:9200"))
    parser.add_argument("--es-index", default=os.environ.get("ES_INDEX", "podcast_clips"))
    parser.add_argument("--postgres-dsn", default=os.environ.get("POSTGRES_DSN", "postgresql://podcast:podcast@localhost:5432/podcasts"))
    parser.add_argument("--batch-size", type=int, default=500, help="Docs per bulk index call")
    args = parser.parse_args()

    # Load metadata into Postgres
    print("Loading metadata...")
    conn = psycopg2.connect(args.postgres_dsn)
    _load_metadata_pg(args.metadata_tsv, conn)
    conn.close()
    print("Metadata loaded.")

    # Set up Elasticsearch
    es = Elasticsearch(args.es_host)
    create_index(es, args.es_index)

    # Find all transcript files
    transcripts = find_transcripts(args.transcripts_dir)
    print(f"Found {len(transcripts)} transcript files.")

    # Process in chunks to limit memory usage.
    # Submit only CHUNK_SIZE futures at a time so we don't hold all results in memory.
    chunk_size = args.workers * 50  # ~200 files in flight at once
    batch = []
    total_indexed = 0
    total_failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pbar = tqdm(total=len(transcripts), desc="Processing")
        for chunk_start in range(0, len(transcripts), chunk_size):
            chunk = transcripts[chunk_start:chunk_start + chunk_size]
            futures = {
                executor.submit(
                    process_transcript, fp, show_id, ep_id,
                    args.clip_duration, args.overlap
                ): ep_id
                for fp, show_id, ep_id in chunk
            }
            for future in as_completed(futures):
                ep_id = futures[future]
                try:
                    docs = future.result()
                    batch.extend(docs)
                    if len(batch) >= args.batch_size:
                        result = bulk_index(es, args.es_index, batch)
                        total_indexed += result.indexed
                        total_failed += result.failed
                        batch = []
                except TranscriptParseError as e:
                    print(f"  Parse error for {ep_id}: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"  Error for {ep_id}: {e}", file=sys.stderr)
                pbar.update(1)
            # Clear references to completed futures
            del futures
        pbar.close()

    # Index remaining batch
    if batch:
        result = bulk_index(es, args.es_index, batch)
        total_indexed += result.indexed
        total_failed += result.failed

    print(f"Done. Indexed: {total_indexed}, Failed: {total_failed}")


if __name__ == "__main__":
    main()
