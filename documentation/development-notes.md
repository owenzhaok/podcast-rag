# Development and operations

[Project overview](../README.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md)

This guide retains useful preparation and validation procedures from the incremental development notes. Commands are PowerShell examples run from the repository root unless stated otherwise. They are operational instructions, not an invitation to overwrite an existing corpus.

## Configuration and local startup

Use Python 3.12, Node.js 24 and Docker Compose. The current frontend uses React 19 and Vite 8, so the old Node 18 setup is obsolete. Install Python requirements from both `ingest/requirements.txt` and `api/requirements.txt`; install frontend packages with `npm ci` in `ui/`.

Python tools read process environment. Uvicorn can additionally load ignored `.env` using `--env-file .env`; ingestion, vector backfill and evaluation CLIs do not automatically load that file. Compose uses its own `.env` substitution. Keep PostgreSQL connection configuration consistent with Compose, and never print credentials or connection strings during troubleshooting.

| Setting | Behavior/default |
|---|---|
| `RAG_ENABLED` | Disabled by default; true/1/yes/on enable Ask. |
| `RAG_RETRIEVAL_MODE` | `bm25` default; explicit `hybrid` enables the vector branch. Unknown values use BM25. |
| `RAG_SOURCE_INDEX` | `podcast_clips` default; controls Ask, source resolution and offline tooling, not `/search`. |
| `RAG_CANDIDATE_LIMIT`, `RAG_MAX_SOURCES`, `RAG_CONTEXT_MAX_BYTES` | Defaults 30, 6, 16000; configured bounds 1–200, 1–20, 1–64000. |
| `RAG_LLM_PROVIDER`, `RAG_LLM_MODEL`, `RAG_LLM_API_KEY` | Groq provider selection, configurable strict-output model and server-only credential. |
| `RAG_CONTEXT_MAX_TOKENS` | 8000 estimated total tokens by default. |
| `RAG_MAX_OUTPUT_TOKENS` | Code default 800; `.env.example` documents 1600 as a previously validated starting value. |
| `RAG_LLM_TIMEOUT_SECONDS` | 20-second whole-generation timeout by default. |
| `RAG_ANSWER_CACHE_TTL_SECONDS` | 900 seconds; 0 disables generation caching. Search TTL remains separate. |

`openai/gpt-oss-20b` was previously live-validated with estimated context/output budgets 8000/1600 and a 20-second timeout. Model selection remains environmental; those settings are not universal optima. No speculative reasoning setting is required by the adapter.

For evidence-only validation, use a separate terminal and explicitly leave generation unconfigured:

```powershell
$env:RAG_ENABLED = 'true'
$env:RAG_RETRIEVAL_MODE = 'bm25'
$env:RAG_LLM_PROVIDER = ''
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

With a populated source index, a valid `/ask` request returns sources and `answer: null` with `generation_unavailable` / `llm_not_configured`. An empty result returns `insufficient_context`. Stop this temporary API with Ctrl+C. Restore intended environment settings before normal startup.

## Persistence and corpus separation

Compose runs Elasticsearch 8.13.0, PostgreSQL 16 and Redis 7. Elasticsearch's named volume must mount at its effective data directory, `/usr/share/elasticsearch/data`, as the current Compose file specifies. PostgreSQL uses its own named volume. Redis has no named persistent volume in this Compose configuration; it is a cache, not canonical storage.

Before migrations, compare container mounts with Elasticsearch `_nodes/stats/fs`. Never recreate a container with unverified overlay-only data. Preserve existing backups. The prior persistence repair was a separate migration, not something normal setup or ingestion repeats. Do not remove volumes or recreate indexes as a troubleshooting shortcut.

Keep these roles distinct:

- `podcast_clips`: legacy Search/default RAG source; the synthetic workflow remains supported.
- `podcast_rag_v1`: default companion vector target; do not mix fake and real embedding spaces.
- `podcast_clips_real_v1`: default bounded real-source ingestion target; ingestion does not switch runtime retrieval.
- Real-data vector/evaluation targets: explicitly select the matching versioned index in your local configuration. No particular local evaluation index name is implied by the published aggregate results.

## Bounded real ingestion (Stage 8.5A)

The Spotify ZIP remains outside the repository. The CLI streams the nested `podcasts-transcripts-0to2.tar.gz` by default and stops after the requested accepted sample. Sampling follows archive order, uses English metadata and defaults to at most five episodes per show. It is deterministic for the same inputs/options, not random or representative.

Metadata is streamed into an automatically cleaned temporary SQLite lookup. Memory is bounded to one transcript; temporary disk usage grows with metadata size. Limits are 16 MiB JSON, 50000 retained words (100000 input entries allowing a repeated aggregate), and 24-hour timestamps. No tarball is extracted to disk, and retained tar-header history is cleared.

The parser uses `alternatives[0].words`, exact decimal millisecond conversion, original word order and optional speaker tags. It preserves zero-duration words and counts backwards-time anomalies without sorting. When the final diarized aggregate exactly repeats earlier words, it replaces them rather than duplicating text. Missing/empty results can increment malformed counters while the episode remains usable. Clips reuse 120-second windows with 60-second overlap and actual word endpoints.

```powershell
$dataset = Read-Host 'Path to your local Spotify dataset ZIP'
.\.venv\Scripts\python.exe -m ingest.spotify_sample --zip $dataset --episodes 20 --max-per-show 5 --clip-duration 120 --overlap 60 --dry-run
# Only after reviewing the dry-run counters and confirming target/services:
.\.venv\Scripts\python.exe -m ingest.spotify_sample --zip $dataset --episodes 20 --max-per-show 5 --clip-duration 120 --overlap 60 --index podcast_clips_real_v1
```

Dry runs make no Elasticsearch/PostgreSQL connections. Real runs require both. Selected metadata goes into the existing PostgreSQL tables using filename-prefix IDs. Matching rows are reused; conflicting existing rows stop the run without overwrite. Metadata commits before Elasticsearch writes, so interruption can leave metadata or a partially indexed episode; rerun the same command to finish.

The target must be a concrete `podcast_clips_real_*` name. Protected synthetic/vector targets, aliases, wildcards and incompatible mapping/provenance are rejected. There is no automatic delete/recreate operation. Index metadata records archive/metadata ZIP CRCs, parser version, language, window settings and show cap. CRCs detect input changes, not malicious tampering.

Writes are create-only with deterministic IDs. A rerun reports existing IDs rather than duplicates; `chunks_produced` counts windows, so stored count can be smaller when identical windows share an ID. Increasing the episode limit extends the deterministic sample; reducing it never removes data. To change sampling/window/input configuration, choose a new versioned real index. Resume scans from the start of gzip, not a persisted cursor. Use one importer per target.

Filename prefixes, URIs, dataset identity and archive/member provenance remain attached to documents for future evaluation and compliance handling. Do not commit real rows or passages. Keep retraction requirements in the operational process; no automatic retraction workflow is implemented.

## Vector preparation (Stages 6 and 6.5)

The explicit CLI supports `create`, `check` and `backfill --limit N`. A compatible existing index is required for backfill; no API-startup backfill occurs. Use the existing source and a separate companion target, with an embedding credential already configured privately:

```powershell
$env:RAG_SOURCE_INDEX = 'podcast_clips_real_v1'
$env:RAG_VECTOR_INDEX = Read-Host 'Concrete companion index for this corpus'
$env:RAG_EMBEDDING_PROVIDER = 'gemini'
$env:RAG_EMBEDDING_MODEL = 'gemini-embedding-2'
$env:RAG_EMBEDDING_DIMENSIONS = '768'
$env:RAG_EMBEDDING_REVISION = 'v1'
$env:RAG_EMBEDDING_BATCH_SIZE = '32'
$env:RAG_EMBEDDING_TIMEOUT_SECONDS = '20'
# RAG_EMBEDDING_API_KEY must already be supplied in the local environment.
.\.venv\Scripts\python.exe -m api.rag.backfill create
if ($LASTEXITCODE -ne 0) { throw 'Stop: incompatible target or configuration' }
.\.venv\Scripts\python.exe -m api.rag.backfill check
if ($LASTEXITCODE -ne 0) { throw 'Stop: vector preflight failed' }
# Example limit for a verified 730-document source; adjust to the intended corpus.
.\.venv\Scripts\python.exe -m api.rag.backfill backfill --limit 730
```

These backfill commands make real Gemini requests. Start with a tiny uniquely named test source/companion before a full intended sample. Never put fake vectors in a real target. The deterministic fake provider is reserved for tests; it measures plumbing, not semantics.

Preflight checks dimensions, mapping and provenance. Gemini rejects fake/other-model vectors in the target. Change provider/model/dimensions by using a new versioned companion, not by silently replacing an old one. `check` may refresh the already validated companion; it does not recreate it. The lexical source is read-only to backfill.

Unchanged content/provider/model/revision/chunking/input identity with valid vectors is skipped. Title changes affect Gemini's input hash. Changed content gets a new stable identity; old documents are retained. Revision changes can re-embed existing identities. Finish the full intended scan before using a partially migrated index. A limited rerun begins at the start, not a saved cursor.

The Gemini adapter uses bounded HTTP requests, no SDK or retries, sequential requests within batches, and validates the whole embedding batch before writes. Failed bulk writes can still leave partial completed documents; rerun safely. Source clips above 256 KiB are rejected by vector preparation; canonical Gemini inputs above 8192 UTF-8 bytes are also rejected rather than truncated. Response size is bounded to 128 KiB. Earlier completed batches survive failures.

After backfill, refresh only the selected companion if immediate visibility is needed. Check count, stable IDs, 768 dimensions and provider/model/revision metadata; repeat the same intended scan and verify `written=0`. Source counts, mappings and contents should remain unchanged. Use one backfill per target at a time.

## Validation

`pytest.ini` discovers `ingest/tests` and `api/tests`. Test coverage includes parsing/windowing, synthetic nested archives, protected indexes, stable identities, vector preparation, mocked Gemini/Groq calls, structured generation, citation rejection, prompt isolation, cache behavior, Hybrid fallback and evaluation formulas. Frontend tests cover Search/pagination, Ask states, citations, errors and cancellation/stale requests.

```powershell
.\.venv\Scripts\python.exe -m pytest
$env:RAG_INTEGRATION_TESTS = '1'
.\.venv\Scripts\python.exe -m pytest api/tests/test_vector_preparation.py api/tests/test_gemini_embeddings.py api/tests/test_hybrid_integration.py api/tests/test_evaluate.py ingest/tests/test_spotify_sample.py
Remove-Item Env:RAG_INTEGRATION_TESTS
```

The opt-in tests use unique Elasticsearch indexes and isolated PostgreSQL data, including transaction rollbacks or connection-local temporary tables. Normal provider tests prohibit real HTTPX network transports. The legacy `scripts/integration_test.py` is not part of default pytest discovery: it seeds the legacy index and uses a separate undeclared `requests` dependency. Prefer isolated opt-in pytest tests when preserving a working corpus.

For a manual Hybrid smoke check, use prepared vectors and an evidence-only API (`RAG_ENABLED=true`, `RAG_LLM_PROVIDER` empty), explicitly select `RAG_RETRIEVAL_MODE=hybrid`, and submit a locally chosen question. Check actual mode and resolve returned chunk IDs directly through backend `/sources/{chunk_id}`. To test fallback without deleting data, restart only that temporary API with a fresh nonexistent companion name; mapping failure occurs before embedding and should return BM25 evidence if available. Restore intended settings afterwards. A cache hit still performs retrieval, including the Hybrid branch when selected.

Record counts/mappings/IDs before and after ingestion-independent tests. For any full content snapshot, keep it local; never add it to documentation. The offline evaluator and full reproduction limitations are documented separately in [evaluation](evaluation.md).

## Diagnostics and publication boundaries

Groq diagnostics remain server-side and sanitized. They record safe error fields and request facts, not request headers or full provider bodies. Only `json_validate_failed` includes a sanitized, single-line `failed_generation` representation bounded to 2000 characters. These details are not exposed through `/ask`.

Never paste API keys, connection strings, transcripts, copied metadata or individual judgments into commits or public troubleshooting notes. `.env`, `.local-eval/`, local backups and dataset archive patterns are ignored; that does not make arbitrary copied data safe to commit. Provider input disclosure remains part of deployment review.

Public guides live in `documentation/` because the existing `.gitignore` reserves `docs/` for internal plans. This keeps publication separate without changing ignore rules or exposing private artifacts.
