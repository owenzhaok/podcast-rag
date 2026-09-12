"""End-to-end integration test. Requires docker-compose services running."""

import json
import os
import sys
import time
import subprocess
import requests
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
FIXTURE_DIR = "/tmp/podcast_test_fixtures"
API_BASE = "http://localhost:8000"
ES_HOST = "http://localhost:9200"
ES_INDEX = "podcast_clips"


def wait_for_service(url: str, timeout: int = 30):
    """Wait for a service to respond."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(url, timeout=2)
            return
        except requests.ConnectionError:
            time.sleep(1)
    raise TimeoutError(f"Service at {url} did not start within {timeout}s")


def main():
    print("1. Generating fixtures...")
    sys.path.insert(0, str(SCRIPT_DIR))
    from sample_data import generate_fixtures
    episodes = generate_fixtures(FIXTURE_DIR, count=5)

    print("2. Waiting for services...")
    wait_for_service(ES_HOST)

    print("3. Running ingest...")
    subprocess.run(
        [
            sys.executable, "-m", "ingest.ingest",
            "--transcripts-dir", os.path.join(FIXTURE_DIR, "transcripts"),
            "--metadata-tsv", os.path.join(FIXTURE_DIR, "metadata.tsv"),
            "--clip-duration", "120",
            "--overlap", "60",
            "--workers", "2",
        ],
        cwd=str(PROJECT_DIR),
        check=True,
    )

    # Wait for ES to refresh
    requests.post(f"{ES_HOST}/{ES_INDEX}/_refresh")
    time.sleep(1)

    print("4. Checking ES document count...")
    resp = requests.get(f"{ES_HOST}/{ES_INDEX}/_count")
    count = resp.json()["count"]
    print(f"   Documents in ES: {count}")
    assert count > 0, f"Expected clips in ES, got {count}"

    print("5. Testing search via ES directly...")
    # 'machine' and 'learning' are in our sample word list
    search_body = {
        "query": {"match": {"clip_text": "machine learning"}},
        "size": 5,
    }
    resp = requests.post(f"{ES_HOST}/{ES_INDEX}/_search", json=search_body)
    hits = resp.json()["hits"]["total"]["value"]
    print(f"   ES search hits: {hits}")
    assert hits > 0, f"Expected search results, got {hits}"

    print("\nAll integration tests passed!")


if __name__ == "__main__":
    main()
