# Architecture

[Project overview](../README.md) · [Evaluation](evaluation.md) · [Operations](development-notes.md)

## Boundaries and entry points

| Area | Entry points | Responsibility |
|---|---|---|
| Browser | `ui/src/App.tsx`, `ui/src/api/client.ts` | Search/Ask tabs, pagination, cancellation and source presentation |
| API | `api/main.py`, `api/rag/routes.py` | Service lifecycle, search, optional QA and source resolution |
| Retrieval | `api/search.py`, `api/rag/retrieval.py`, `api/rag/hybrid.py` | Lexical candidates, canonical evidence, vector retrieval and RRF |
| Generation | `api/rag/generation.py`, `groq.py`, `citations.py` | Bounded requests, provider isolation and output validation |
| Preparation | `ingest/spotify_sample.py`, `api/rag/backfill.py` | Explicit source ingestion and companion-vector preparation |
| Evaluation | `api/rag/evaluate.py` | Offline retrieval metrics without generation |

FastAPI initializes Elasticsearch, a PostgreSQL pool and the existing Redis client during lifespan startup. RAG configuration loads without initializing a provider. Missing LLM/embedding credentials do not prevent startup; PostgreSQL is still a startup dependency. `/health` does not probe these services.

The browser calls FastAPI only. Ask uses `AbortController` and request identity checks to suppress stale responses. Canceling browser work does not guarantee cancellation of a provider request already underway. Source cards use data already returned by `/ask`, avoiding repeated source-resolution requests. Vite proxies `/search`, `/ask` and `/health`; direct source-resolution checks use the backend URL.

## Search is independent

`GET /search` uses Elasticsearch `multi_match` over `clip_text`, `fuzziness: AUTO`, pagination and a 300-character highlight fragment. It queries the fixed `podcast_clips` index, joins episode/show metadata from PostgreSQL and caches the complete search response in Redis for one hour. RAG provider failures cannot enter this code path. The search cache is distinct from the generation cache and does not share its fail-open handling.

`clip_minutes` is accepted for compatibility but not used by `execute_search`; windows are determined during ingestion. `ES_INDEX` is used by legacy ingestion, not to select the `/search` index.

## RAG retrieval

BM25 is the default. It reuses lexical query semantics but removes highlighting and reads full `_source.clip_text`. Default lexical candidate count is 30, with provenance-based tie-breaking. `RAG_SOURCE_INDEX` defaults to `podcast_clips` and can select a separate real corpus without affecting Search.

Hybrid is explicitly enabled by `RAG_RETRIEVAL_MODE=hybrid`:

1. Retrieve lexical candidates first.
2. Lazily validate Gemini configuration and the concrete companion index mapping.
3. Embed `task: question answering | query: <exact validated question>` (`gemini-query-v1`) using `gemini-embedding-2`, exactly 768 dimensions.
4. Request 30 kNN hits with `num_candidates=100`. Filters require matching source index, provider, model, revision, document input version and chunking version.
5. Resolve vector chunk IDs against the canonical lexical source and compare hashes/text. Discard stale, missing or mismatching hits; companion text alone is never trusted as evidence.
6. Fuse rankings, then run existing selection and PostgreSQL enrichment.

RRF gives each unique chunk one contribution per list:

```text
score(chunk) = sum(1 / (60 + rank_in_list))
```

Ranks start at one after duplicate removal. Shared chunks receive both contributions; chunk ID breaks ties. This avoids comparing BM25 and vector score scales directly. Fusion is deterministic for fixed rankings; approximate kNN does not guarantee identical candidates across changed indexes.

Selection suppresses exact duplicates and same-episode clips sharing at least 45% of the shorter clip's duration. Defaults allow six sources totaling 16,000 UTF-8 bytes of transcript. Oversized passages are skipped, not truncated. A second model-context budget may omit additional complete passages before generation.

## Provenance and preparation

The lexical schema retains show/episode IDs, clip index, start/end milliseconds, full text, word timestamps and speakers. PostgreSQL `shows` and `episodes` remain authoritative for display titles and embedding-title lookup.

Stable `chunk_id` is `v1.` plus unpadded base64url of compact UTF-8 JSON:

```text
[podcast_id, episode_id, start_ms, end_ms, sha256(exact_clip_text)]
```

Whitespace is significant. Elasticsearch-generated IDs, clip indexes and speaker tags are not identity inputs. Real ingestion and vector preparation use SHA-256 of this chunk ID as Elasticsearch `_id` to fit the 512-byte limit. Response-local labels such as `S1` are different from durable chunk IDs.

The companion index defaults to `podcast_rag_v1`, with a strict mapping and indexed float `dense_vector`, cosine similarity and HNSW. It stores provenance, text/hash, provider/model/revision, chunking version and source-index identity. Gemini documents also store input format version and input hash.

Document input is `title: <episode title> | text: <exact transcript>` (`gemini-document-v1`); unavailable titles fall back to `<podcast_id>/<episode_id>`. Backfill reads the source, skips unchanged compatible vectors, validates complete embedding batches and upserts deterministic IDs. It never changes lexical mappings or runs on API startup. Fake embeddings test plumbing only. [Safe preparation](development-notes.md#vector-preparation-stages-6-and-65).

## Generation and validation

`generation.py` depends on a small provider interface. Groq uses async HTTPX and strict JSON Schema output with configured model, output limit and timeout. No large orchestration framework is required.

The system prompt requires evidence-only answers, paragraph citations and abstention. Question, metadata and transcript content are explicitly untrusted data. Python validates the closed JSON contract, status/paragraph consistency, nonblank text and citation membership against the subset actually sent. An invalid paragraph rejects the whole generation; no partially invalid answer is returned.

The context estimator counts one UTF-8 byte as one approximate token, adds a 256-token framing allowance and reserves output tokens. Defaults are 8000 total estimated tokens, 800 output tokens in code and a 20-second generation timeout. `.env.example` documents 1600 output tokens as a validated starting setting. This is not an exact tokenizer guarantee.

## Generation cache and failures

Retrieval precedes every cache lookup. `rag:answer:v1:<sha256>` hashes the effective generation request: system/user prompts, selected text/order/labels/metadata, provider, model, output limit, timeout and generation/validation versions. Different evidence cannot accidentally reuse an old full response; identical effective prompts may safely share generation across retrieval modes.

Only validated answers or model abstentions are cached, for 900 seconds by default; `RAG_ANSWER_CACHE_TTL_SECONDS=0` disables this. Hits are revalidated and returned with current sources and mode. Errors, invalid output, disabled/unconfigured RAG, no evidence and context-budget failures are not cached. Redis GET/SET failures fail open; hits do not renew TTL. There is no concurrent-request coalescing, so simultaneous misses can each generate.

| Condition | Behavior |
|---|---|
| RAG disabled | `disabled`; no retrieval/provider activity |
| No usable evidence | `insufficient_context`; no LLM call |
| No generation configuration | `generation_unavailable`; preserve evidence |
| Model abstention | `insufficient_context`, reason `model_abstained` |
| Invalid structure/citations | `invalid_generation`; no answer, preserve sources |
| Provider timeout/auth/rate/quota/network failure | Safe `generation_unavailable` reason; preserve sources |
| Hybrid vector branch fails | BM25 fallback, actual mode `bm25`, `degraded=true` |
| Vector branch succeeds but has no usable candidates | Preserve BM25 order/mode without a vector-failure flag |
| Lexical infrastructure fails | HTTP 503; Hybrid cannot replace canonical storage |
| Metadata unavailable during retrieval | Keep IDs/text/timestamps with unknown names; mark degraded |

On successful generation after vector failure, reason is `hybrid_retrieval_unavailable`; existing generation/no-evidence reasons take precedence. Hybrid work is bounded by the embedding timeout plus 15 seconds, with four canonical lookups concurrently at most. The synchronous Gemini adapter runs in a worker thread, which may finish after cancellation until its HTTP timeout.

`GET /sources/{chunk_id}` resolves corpus-bound evidence without an in-memory registry. Unknown/stale IDs return 404; infrastructure failure or reaching its 1000-record safety cap returns 503. This is source identity validation, not access control.
