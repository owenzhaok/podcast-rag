# Podcast Search

## Optional RAG evidence (Stage 2)

Set `RAG_ENABLED=true` in the process environment (or use Uvicorn's
`--env-file .env`) to enable `POST /ask` with `{"question": "..."}`. No LLM or
embedding credentials are required for retrieval. Generation is not implemented:
successful retrieval returns `answer: null`, full passages in `sources`,
`retrieval_mode: "bm25"`, and `status: "generation_unavailable"`. The reason is
`llm_not_configured` when generation settings are absent and `not_implemented`
when present. An empty or unusable candidate set returns `insufficient_context`.
Disabled RAG retains the Stage 1 disabled response and performs no I/O.

The RAG path reuses the lexical query builder but reads `_source.clip_text`, never
search highlights. It requests 30 candidates by default, ranks by BM25 score with
stable provenance tie-breaks, deduplicates identical source identities, and skips
clips overlapping at least 45% of the shorter selected clip in the same episode.
It selects at most 6 sources with a combined 16,000 UTF-8 bytes of transcript text.
This is a conservative token proxy, not a model token guarantee or a total HTTP
response-size limit. Oversized clips are skipped, not truncated. Candidate,
source, and byte limits are configurable in `.env.example`. A bounded candidate
window may underfill the source list when repeated imports dominate results.

`chunk_id` is `v1.` followed by unpadded base64url of compact UTF-8 JSON containing
`[podcast_id, episode_id, start_ms, end_ms, sha256(exact_clip_text)]`. The content
hash preserves whitespace; Elasticsearch document IDs and clip indexes are not
identity inputs. Thus identical reimports collapse without modifying legacy data.
`source_id` (`S1`, `S2`, ...) is only a response-local display label.

`GET /sources/{chunk_id}` searches only the server's `RAG_SOURCE_INDEX` and checks
the content hash before returning exact text and current PostgreSQL metadata.
It needs no in-memory registry and survives API restarts. Invalid, unknown, stale,
or disabled sources return 404. Elasticsearch errors/partial results return 503.
Resolution checks up to 1,000 matching time-range records before returning 503
rather than falsely reporting not-found. Missing or unavailable metadata retains
identifiers/timestamps with unknown display names and `metadata_available=false`.
Evidence responses flag `degraded=true` when metadata is unavailable.

Run unit tests with `python -m pytest ingest/tests api/tests`. To also run the
isolated real-service test in PowerShell:

```powershell
$env:RAG_INTEGRATION_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest api/tests/test_rag_integration.py -v
```

The integration test creates/deletes a unique Elasticsearch index and rolls back
its PostgreSQL metadata transaction. Existing `/search`, Redis caching, ingestion,
database schemas, and frontend behavior are unchanged.

A full-stack search engine over the [Spotify Podcast Dataset](https://podcastsdataset.byspotify.com/) that lets users find clips from podcast episodes matching a free-text query. Results display ranked clip cards with highlighted transcript excerpts and speaker attribution.

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌───────────────┐
│  React   │────▶│   FastAPI    │────▶│ Elasticsearch │
│ Frontend │     │   /search    │     │  (clip index) │
└──────────┘     │   /health    │     └───────────────┘
                 │              │────▶┌───────────────┐
                 │              │     │  PostgreSQL    │
                 │              │     │  (metadata)    │
                 │              │     └───────────────┘
                 │              │────▶┌───────────────┐
                 │              │     │    Redis       │
                 └──────────────┘     │   (cache)      │
                                      └───────────────┘

┌──────────────────────────────────────────────────────┐
│              Ingest Pipeline (CLI)                    │
│  Transcript JSON ──▶ Parser ──▶ Segmenter ──▶ ES     │
│  metadata.tsv    ──▶ Metadata Loader ──▶ Postgres    │
└──────────────────────────────────────────────────────┘
```

**Ingest pipeline** parses Spotify transcript JSONs into overlapping 2-minute clips (with 1-minute overlap), indexes them into Elasticsearch with the English analyzer and fuzzy matching.

**Search API** (FastAPI) queries ES, enriches results with show/episode metadata from Postgres, and caches responses in Redis (1-hour TTL).

**Frontend** (React + TypeScript + Tailwind) displays clip cards with highlighted excerpts, speaker badges, timestamp ranges, and a "Load more" button for pagination.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Search index | Elasticsearch 8.13 |
| Metadata store | PostgreSQL 16 |
| Cache | Redis 7 |
| Backend | Python 3.12, FastAPI |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Ingest | Python, concurrent.futures for parallelism |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+
- Node.js 18+

### 1. Start infrastructure

```bash
cp .env.example .env
docker compose up -d
```

This starts Elasticsearch (port 9200), PostgreSQL (port 5432), and Redis (port 6379).

### 2. Install Python dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r ingest/requirements.txt
pip install -r api/requirements.txt
```

### 3. Extract transcripts

The dataset ships as tar.gz archives. Extract at least one:

```bash
cd podcasts-no-audio-13GB
tar -xzf podcasts-transcripts-0to2.tar.gz
```

This creates `spotify-podcasts-2020/podcasts-transcripts/` with JSON files organized by `{0-7}/{A-Z}/show_{id}/{episode_id}.json`.

### 4. Run the ingest pipeline

```bash
python -m ingest.ingest \
  --transcripts-dir podcasts-no-audio-13GB/spotify-podcasts-2020/podcasts-transcripts \
  --metadata-tsv podcasts-no-audio-13GB/metadata.tsv \
  --clip-duration 120 \
  --overlap 60 \
  --workers 2
```

This parses transcripts, segments them into overlapping clips, loads metadata into Postgres, and bulk-indexes clips into Elasticsearch.

### 5. Start the API

```bash
uvicorn api.main:app --port 8000
```

Endpoints:
- `GET /search?q=machine+learning&from=0&size=10` — search clips
- `GET /health` — health check

### 6. Start the frontend

```bash
cd ui
npm install
npm run dev
```

Open http://localhost:5173 in your browser. The Vite dev server proxies `/search` and `/health` to the API on port 8000.

## Project Structure

```
podcast-search/
├── docker-compose.yml          # ES, Postgres, Redis
├── .env.example                # Environment variables template
├── init.sql                    # Postgres schema (shows, episodes)
├── ingest/
│   ├── parser.py               # Transcript JSON → WordRecord list
│   ├── segmenter.py            # WordRecord list → overlapping Clips
│   ├── es_client.py            # ES index creation + bulk indexing
│   ├── metadata_loader.py      # metadata.tsv → Postgres
│   ├── ingest.py               # CLI entry point (parallel processing)
│   └── tests/                  # 23 unit tests
├── api/
│   ├── models.py               # Pydantic request/response schemas
│   ├── search.py               # ES query builder + result assembler
│   ├── cache.py                # Redis async cache wrapper
│   ├── db.py                   # Postgres async queries
│   ├── main.py                 # FastAPI app
│   └── tests/                  # 19 unit tests
├── ui/
│   ├── src/
│   │   ├── App.tsx             # Main app with search state
│   │   ├── components/
│   │   │   ├── SearchBar.tsx   # Search input + duration selector
│   │   │   ├── ClipCard.tsx    # Result card with highlights
│   │   │   ├── ResultsList.tsx # Results list with load more
│   │   │   └── DurationSelector.tsx
│   │   ├── api/client.ts       # Fetch wrapper for /search
│   │   └── types.ts            # TypeScript interfaces
│   └── tests/                  # 17 component tests
└── scripts/
    ├── sample_data.py          # Generate synthetic test fixtures
    └── integration_test.py     # End-to-end test
```

## Running Tests

```bash
# Backend tests (42 tests)
source .venv/bin/activate
python -m pytest ingest/tests/ api/tests/ -v

# Frontend tests (17 tests)
cd ui && npx vitest run

# Integration test (requires docker-compose services running)
python scripts/integration_test.py
```

## Key Design Decisions

- **50% clip overlap** — clips overlap by half their duration so relevant passages near boundaries are always fully captured in at least one clip
- **Fuzzy matching (`fuzziness: AUTO`)** — compensates for the ~18% ASR word error rate in the transcripts
- **`word_timestamps` stored but not indexed** (`"enabled": false`) — keeps the ES index lean; timestamps are only used by the frontend for potential audio seek
- **Metadata enriched at query time** — joining against Postgres at search time means metadata updates don't require re-indexing
- **All time values in milliseconds (integers)** — avoids floating-point formatting inconsistencies across the stack

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ES_HOST` | `http://localhost:9200` | Elasticsearch URL |
| `ES_INDEX` | `podcast_clips` | ES index name |
| `POSTGRES_USER` | `podcast` | Postgres user |
| `POSTGRES_PASSWORD` | `podcast` | Postgres password |
| `POSTGRES_DB` | `podcasts` | Postgres database |
| `POSTGRES_DSN` | `postgresql://podcast:podcast@localhost:5432/podcasts` | Full DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL |
| `CLIP_DURATION_DEFAULT` | `120` | Default clip duration (seconds) |
| `CLIP_OVERLAP` | `60` | Clip overlap (seconds) |
| `API_PORT` | `8000` | API server port |

## Dataset

This project uses the [Spotify Podcasts Dataset](https://podcastsdataset.byspotify.com/) (~105K episodes). The dataset includes:

- **Transcripts** — Google Speech-to-Text ASR output in JSON format, with word-level timestamps and speaker diarization
- **Metadata** — TSV with show/episode names, descriptions, publishers, durations, and RSS links
