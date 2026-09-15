# Podcast Search

## Optional vector preparation (Stages 6 / 6.5)

Stage 6 is offline preparation only. `/search` is unchanged; `/ask` still uses
BM25, the Stage 5 generation cache, and Groq. No vector configuration is loaded by
API startup and no vector state is added to the active generation-cache identity.
Stage 7 optionally enables hybrid retrieval as documented below; BM25 remains default.

The companion index defaults to `RAG_VECTOR_INDEX=podcast_rag_v1`. It uses a strict
mapping with an indexed float `dense_vector`, explicit dimensions, cosine
similarity, and HNSW. It stores the exact Stage 2 `chunk_id`, show/episode IDs,
timestamps, exact `chunk_text`, content hash, speakers, source index, provider,
model, revision, and chunking version. Legacy Elasticsearch document IDs are not
needed: duplicates share the same stable provenance. The vector document `_id`
is SHA-256 of the existing Stage 2 `chunk_id` (to respect Elasticsearch's 512-byte
ID limit); the original chunk ID remains available in the document.

`RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL`, and `RAG_EMBEDDING_DIMENSIONS` must
be explicitly configured for the tool. Batch size defaults to 32 (1–200), revision
to `v1`. Fake dimensions may be 1–4096; Gemini is configured for exactly 768.
The `fake` provider's deterministic normalized vectors test plumbing, **not semantic
similarity**. Stage 6.5 also supports `gemini` / `gemini-embedding-2` through HTTPX,
without a new SDK or dependency. Never put keys in model/revision identifiers or
commit credentials. Real Gemini usage may incur provider charges.

Commands use the process environment and existing `ES_HOST`/`RAG_SOURCE_INDEX`:

```powershell
.\.venv\Scripts\python.exe -B -m api.rag.backfill create
.\.venv\Scripts\python.exe -B -m api.rag.backfill check
.\.venv\Scripts\python.exe -B -m api.rag.backfill backfill --limit 10
```

Creation is explicit and idempotent. Backfill requires an existing compatible
index and an explicit positive scan limit; it never creates/recreates indexes.
Concrete index names only: aliases, wildcard targets, and equal source/vector
names are rejected. Incompatible mappings/dimensions stop safely. Do not create
fake vectors in a companion index intended for real semantic retrieval.

Backfill scans source clips in bounded batches using a scroll snapshot, validates
whole embedding batches before writing, and upserts deterministic IDs. It checks
existing documents in real time, skipping matching content/provenance and
provider/model/revision/chunking version with valid vectors. Progress logs contain
only scanned/written/skipped counts. Source clips over 256 KiB or invalid clips
stop the run. Previously completed batches survive a failure; a bulk-write failure
can leave part of its batch complete, and rerunning safely fills the remainder.
No global in-memory deduplication set or automatic startup work is used.

Changed exact transcript content has a new identity and creates a new document;
old identities are retained, never silently deleted. Revision changes re-embed
existing identities. Gemini refuses an index containing fake or different-model
vectors: provider/model/dimension migrations require a new versioned index.
Fake-only Stage 6 model-change tests retain their existing behavior.
Run one backfill per target at a time; after interruption or
a model change, finish a full intended scan before using that index in future
retrieval. A limited rerun begins from the start and is not a persistent cursor.

### Safe manual Stage 6 validation (PowerShell)

Run from the repository root with existing Docker services running. This creates
only two uniquely named tiny test indexes; it does not write `podcast_clips`, touch
backups, or delete any data. It intentionally uses a test companion name rather
than populating the production default with fake vectors.

```powershell
$es = 'http://localhost:9200'
$originalCount = (Invoke-RestMethod "$es/podcast_clips/_count").count
$suffix = [guid]::NewGuid().ToString('N')
$env:RAG_SOURCE_INDEX = "rag-stage6-source-$suffix"
$env:RAG_VECTOR_INDEX = "podcast_rag_v1_smoke_$suffix"
$env:RAG_EMBEDDING_PROVIDER = 'fake'
$env:RAG_EMBEDDING_MODEL = 'fake-smoke'
$env:RAG_EMBEDDING_DIMENSIONS = '8'
$env:RAG_EMBEDDING_BATCH_SIZE = '2'
$env:RAG_EMBEDDING_REVISION = 'v1'
$sourceUrl = "$es/$env:RAG_SOURCE_INDEX"
$vectorUrl = "$es/$env:RAG_VECTOR_INDEX"
Invoke-RestMethod -Method Put -Uri $sourceUrl -ContentType 'application/json' `
  -Body '{"settings":{"number_of_shards":1,"number_of_replicas":0}}'
$doc = @{podcast_id='demo-show'; episode_id='demo-episode'; clip_index=0;
  clip_start_ms=0; clip_end_ms=120000; speakers=@(1);
  clip_text='Machine learning identifies patterns in research.'} | ConvertTo-Json
Invoke-RestMethod -Method Put -Uri "$sourceUrl/_doc/one?refresh=true" `
  -ContentType 'application/json' -Body $doc
Invoke-RestMethod -Method Put -Uri "$sourceUrl/_doc/duplicate?refresh=true" `
  -ContentType 'application/json' -Body $doc
$beforeSource = Invoke-RestMethod "$sourceUrl/_search?size=10"
.\.venv\Scripts\python.exe -B -m api.rag.backfill create
.\.venv\Scripts\python.exe -B -m api.rag.backfill check
.\.venv\Scripts\python.exe -B -m api.rag.backfill backfill --limit 10
Invoke-RestMethod -Method Post "$vectorUrl/_refresh"
$first = Invoke-RestMethod "$vectorUrl/_search?size=10"
.\.venv\Scripts\python.exe -B -m api.rag.backfill backfill --limit 10
Invoke-RestMethod -Method Post "$vectorUrl/_refresh"
$second = Invoke-RestMethod "$vectorUrl/_search?size=10"
if ($first.hits.total.value -ne 1 -or $second.hits.total.value -ne 1) { throw 'Count mismatch' }
if ($first.hits.hits[0]._id -ne $second.hits.hits[0]._id) { throw 'ID changed' }
if ($second.hits.hits[0]._source.embedding.Count -ne 8) { throw 'Wrong dimension' }
Invoke-RestMethod "$vectorUrl/_mapping" | ConvertTo-Json -Depth 15
$afterSource = Invoke-RestMethod "$sourceUrl/_search?size=10"
$before = $beforeSource.hits.hits | Sort-Object _id | ConvertTo-Json -Depth 15 -Compress
$after = $afterSource.hits.hits | Sort-Object _id | ConvertTo-Json -Depth 15 -Compress
if ($before -ne $after) { throw 'Test source changed' }
if ((Invoke-RestMethod "$es/podcast_clips/_count").count -ne $originalCount) { throw 'Corpus count changed' }
# Leave the tiny test indexes intact for inspection; no deletion commands.
# Reset source selection before starting any API in this shell.
$env:RAG_SOURCE_INDEX = 'podcast_clips'
```

Expected backfill summaries: first `written=1`, second `written=0`, both
`scanned=2`. To verify BM25 without a Groq call, start an evidence-only API on a
separate port (keep existing application processes unchanged):

```powershell
$env:RAG_ENABLED = 'true'
$env:RAG_LLM_PROVIDER = ''
.\.venv\Scripts\python.exe -B -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

In another PowerShell window:

```powershell
$result = Invoke-RestMethod -Method Post 'http://localhost:8001/ask' `
  -ContentType 'application/json' -Body '{"question":"machine learning"}'
$result | Select-Object status, reason, retrieval_mode
if ($result.retrieval_mode -ne 'bm25') { throw 'Unexpected retrieval mode' }
```

Stop the temporary API with Ctrl+C. The opt-in automated integration test uses
temporary indices and fake vectors, checks kNN queryability and source integrity,
and cleans up only its own unique indices:

```powershell
$env:RAG_INTEGRATION_TESTS = '1'
.\.venv\Scripts\python.exe -B -m pytest api/tests/test_vector_preparation.py -q
```

### Gemini preparation (Stage 6.5): manual live validation

The adapter sends `POST .../models/<configured-model>:embedContent`, with the key
only in `x-goog-api-key`, and requests `outputDimensionality=768`. See the
[Gemini embedding REST documentation](https://ai.google.dev/gemini-api/docs/embeddings).
It makes sequential requests inside each existing backfill batch, validates all
vectors before writing that batch, and performs no automatic retries. The timeout
is `RAG_EMBEDDING_TIMEOUT_SECONDS=20` per HTTP operation. Responses are bounded to
128 KiB; HTTP/network errors contain only safe reasons, never provider bodies.
The scroll lease is 30 minutes to accommodate sequential requests. Use small
batches during initial validation; complete previous batches survive failures.

Gemini's canonical input is `title: <episode title> | text: <exact transcript>`.
The CLI reads episode titles from PostgreSQL in a bounded read-only batch lookup;
missing/unavailable metadata uses `<podcast_id>/<episode_id>` as the title.
`gemini-document-v1` and the SHA-256 of this exact input are stored alongside the
unchanged Stage 2 identity. Title changes therefore invalidate skip eligibility.
Changing formatting requires a format-version and embedding-revision bump.
Do not reuse a revision with a different format; preflight rejects this case.
Fake providers retain the Stage 6 raw-transcript convention. Stage 6.5 preparation
does not itself activate hybrid retrieval. To avoid silent truncation, canonical Gemini inputs above 8192
UTF-8 bytes stop rather than being shortened; this is a conservative token proxy.

Use your existing `RAG_EMBEDDING_API_KEY` environment variable. Do not paste it
into commands, source files, examples, or output. These commands make **real**
Gemini requests only when you run them manually. Automated tests never do.

**Phase A — two duplicate sample clips, one real vector:**

```powershell
if (-not $env:RAG_EMBEDDING_API_KEY) { throw 'Configure the key privately in the local environment first' }
$es = 'http://localhost:9200'
$env:RAG_EMBEDDING_PROVIDER = 'gemini'
$env:RAG_EMBEDDING_MODEL = 'gemini-embedding-2'
$env:RAG_EMBEDDING_DIMENSIONS = '768'
$env:RAG_EMBEDDING_REVISION = 'v1'
$env:RAG_EMBEDDING_BATCH_SIZE = '2'
$env:RAG_EMBEDDING_TIMEOUT_SECONDS = '20'
$suffix = [guid]::NewGuid().ToString('N')
$env:RAG_SOURCE_INDEX = "rag-gemini-source-$suffix"
$env:RAG_VECTOR_INDEX = "rag-gemini-vector-$suffix"
$sourceUrl = "$es/$env:RAG_SOURCE_INDEX"
$vectorUrl = "$es/$env:RAG_VECTOR_INDEX"
function Invoke-VectorTool([string]$Action, [int]$Limit = 0) {
  if ($Action -eq 'backfill') {
    & .\.venv\Scripts\python.exe -B -m api.rag.backfill backfill --limit $Limit
  } else {
    & .\.venv\Scripts\python.exe -B -m api.rag.backfill $Action
  }
  if ($LASTEXITCODE -ne 0) { throw 'Vector tool failed; stop and inspect the safe error' }
}
Invoke-RestMethod -Method Put $sourceUrl -ContentType 'application/json' `
  -Body '{"settings":{"number_of_shards":1,"number_of_replicas":0}}'
$sample = @{podcast_id='gemini-demo'; episode_id='gemini-demo-episode'; clip_index=0;
  clip_start_ms=0; clip_end_ms=120000; speakers=@(1);
  clip_text='Machine learning identifies patterns in research.'} | ConvertTo-Json
foreach ($id in @('one','duplicate')) {
  Invoke-RestMethod -Method Put "$sourceUrl/_doc/${id}?refresh=true" `
    -ContentType 'application/json' -Body $sample
}
Invoke-VectorTool create
Invoke-VectorTool check
Invoke-VectorTool backfill 10
Invoke-RestMethod -Method Post "$vectorUrl/_refresh"
$first = Invoke-RestMethod "$vectorUrl/_search?size=100"
if ($first.hits.total.value -ne 1) { throw 'Expected one unique vector' }
if ($first.hits.hits[0]._source.embedding.Count -ne 768) { throw 'Wrong dimension' }
Invoke-VectorTool backfill 10 # Must report written=0.
Invoke-RestMethod -Method Post "$vectorUrl/_refresh"
$second = Invoke-RestMethod "$vectorUrl/_search?size=100"
if ($second.hits.total.value -ne 1 -or $first.hits.hits[0]._id -ne $second.hits.hits[0]._id) { throw 'Idempotence failed' }
$phaseAPassed = $true
```

**Phase B — only after Phase A succeeds:**

```powershell
if (-not $phaseAPassed) { throw 'Finish Phase A first' }
$env:RAG_SOURCE_INDEX = 'podcast_clips'
$env:RAG_VECTOR_INDEX = 'podcast_rag_v1'
$vectorUrl = "$es/$env:RAG_VECTOR_INDEX"
$originalCount = (Invoke-RestMethod "$es/podcast_clips/_count").count
if ($originalCount -ne 25) { throw 'Corpus changed; review the intended scan limit first' }
$original = Invoke-RestMethod "$es/podcast_clips/_search?size=100"
$originalMapping = Invoke-RestMethod "$es/podcast_clips/_mapping"
# Compute expected UNIQUE stable IDs, not the number of possibly duplicated source clips.
$expected = & .\.venv\Scripts\python.exe -B -c 'from elasticsearch import Elasticsearch; from api.rag.retrieval import parse_hit; es=Elasticsearch("http://localhost:9200"); hits=es.search(index="podcast_clips",size=100)["hits"]["hits"]; clips=[parse_hit(h) for h in hits]; assert all(clips); print(len({c.chunk_id for c in clips})); es.close()'
if ($LASTEXITCODE -ne 0) { throw 'Expected identity count failed' }
Invoke-VectorTool create # Creates only if absent; otherwise checks without recreation.
Invoke-VectorTool check  # Rejects incompatible mapping, fake/other models, or old format.
# If either fails, STOP. Use a new versioned index after review; never delete the old index.
Invoke-VectorTool backfill 25
Invoke-RestMethod -Method Post "$vectorUrl/_refresh"
$vectors = Invoke-RestMethod "$vectorUrl/_search?size=100"
if ($vectors.hits.total.value -ne [int]$expected) { throw 'Unexpected vector count; investigate without deleting data' }
foreach ($hit in $vectors.hits.hits) {
  $doc = $hit._source
  if ($doc.embedding.Count -ne 768 -or $doc.embedding_provider -ne 'gemini' -or
      $doc.embedding_model -ne 'gemini-embedding-2' -or $doc.embedding_revision -ne 'v1') {
    throw 'Vector metadata mismatch'
  }
}
Invoke-VectorTool backfill 25 # Must report written=0.
Invoke-RestMethod -Method Post "$vectorUrl/_refresh"
$again = Invoke-RestMethod "$vectorUrl/_search?size=100"
if (($vectors.hits.hits._id | Sort-Object | ConvertTo-Json) -ne
    ($again.hits.hits._id | Sort-Object | ConvertTo-Json)) { throw 'Vector IDs changed' }
$after = Invoke-RestMethod "$es/podcast_clips/_search?size=100"
if (($original.hits.hits | Sort-Object _id | ConvertTo-Json -Depth 30 -Compress) -ne
    ($after.hits.hits | Sort-Object _id | ConvertTo-Json -Depth 30 -Compress)) { throw 'Source changed' }
if ((Invoke-RestMethod "$es/podcast_clips/_count").count -ne $originalCount) { throw 'Source count changed' }
if (($originalMapping | ConvertTo-Json -Depth 30 -Compress) -ne
    ((Invoke-RestMethod "$es/podcast_clips/_mapping") | ConvertTo-Json -Depth 30 -Compress)) { throw 'Source mapping changed' }
```

Use the evidence-only API on port 8001 described above to check `/ask` still
reports `retrieval_mode=bm25`, without invoking Groq. No commands delete indices,
Docker volumes, or the retained Elasticsearch backup. The Phase A indices remain
available for inspection. Do not use fake embeddings in the production companion.

## Optional hybrid retrieval (Stage 7)

`RAG_RETRIEVAL_MODE=bm25` remains the default. Only `hybrid` opts into Gemini query
embeddings, Elasticsearch kNN and application-side Reciprocal Rank Fusion (RRF).
Unknown mode values safely select BM25. `/search` and its cache are unchanged.
BM25 startup and retrieval need no Gemini key, vector index or embedding settings.

Hybrid uses the existing `gemini` / `gemini-embedding-2` adapter, exactly 768
dimensions, `RAG_EMBEDDING_REVISION=v1`, and `podcast_rag_v1` by default. The query
convention is **gemini-query-v1**: `task: question answering | query: <exact user question>`.
The **gemini-document-v1** preparation format is unchanged. Credentials stay on
the server; no browser provider requests, backfills, index refreshes or writes occur.

BM25 requests `RAG_CANDIDATE_LIMIT` (default 30) candidates; vector retrieval takes
30 with `num_candidates=100`. kNN filters provider, model, revision, input format,
chunking version and source index. The vector mapping is checked before embedding.
Vector hits must resolve to the same exact text/hash in the canonical source index;
missing, stale or mismatching hits are discarded. Metadata comes from PostgreSQL.
Each unique chunk contributes `1 / (60 + rank)` per list, with ranks starting at
one after duplicate removal. Contributions are summed by stable Stage 2 chunk ID;
ties use chunk ID. Existing overlap suppression, source counts, evidence-byte
limits and generation context budgets apply after fusion.

Failures in Gemini, configuration, vector mapping, kNN or canonical hydration
fail open to BM25 with `retrieval_mode="bm25"` and `degraded=true`. A successful
answer in that case has safe reason `hybrid_retrieval_unavailable`; existing
generation/no-evidence reasons take precedence. No usable vector hits also leaves
the BM25 order/mode unchanged, without marking an otherwise successful empty
vector search as a provider failure. `hybrid` means validated vector candidates
participated in fusion. Lexical infrastructure failures retain the existing 503.
Vector work is bounded by the embedding timeout plus 15 seconds, with up to four
canonical lookups at once. The synchronous Gemini adapter runs off the event loop;
a canceled request can leave its worker finishing until the HTTP timeout.

Retrieval still runs before every generation-cache lookup. Its identity already
hashes the effective prompt, including evidence text, order, labels and metadata.
Different hybrid context cannot reuse an old BM25 generation. Identical effective
generation requests can safely share a cached result across modes; responses use
current sources and the actual current retrieval mode. No cache-key migration is needed.

Stage 8 will evaluate quality before considering a different default. The current
25-clip synthetic corpus supports functional validation only, not quality claims.

### Manual hybrid validation (PowerShell; real Gemini, no Groq)

Use two terminals in the repository root. Keep your existing Gemini key in the
server terminal's local environment; do not print or paste it into these commands.
The following requests use Gemini quota. They do not write either index.

Terminal 1 — start a separate evidence-only API (no `--env-file` needed):

```powershell
if (-not $env:RAG_EMBEDDING_API_KEY) { throw 'Set your existing key locally first; do not print it' }
$env:RAG_ENABLED = 'true'
$env:RAG_RETRIEVAL_MODE = 'hybrid'
$env:RAG_LLM_PROVIDER = ''
$env:RAG_SOURCE_INDEX = 'podcast_clips'
$env:RAG_VECTOR_INDEX = 'podcast_rag_v1'
$env:RAG_EMBEDDING_PROVIDER = 'gemini'
$env:RAG_EMBEDDING_MODEL = 'gemini-embedding-2'
$env:RAG_EMBEDDING_DIMENSIONS = '768'
$env:RAG_EMBEDDING_REVISION = 'v1'
$env:RAG_EMBEDDING_TIMEOUT_SECONDS = '20'
.\.venv\Scripts\python.exe -B -m uvicorn api.main:app --host 127.0.0.1 --port 8007
```

Terminal 2 — record data, then verify hybrid evidence and source resolution:

```powershell
$es = 'http://localhost:9200'
$api = 'http://127.0.0.1:8007'
foreach ($index in @('podcast_clips', 'podcast_rag_v1')) {
  if ((Invoke-RestMethod "$es/$index/_count").count -ne 25) { throw "Unexpected count: $index" }
}
$mapping = Invoke-RestMethod "$es/podcast_rag_v1/_mapping"
if ($mapping.podcast_rag_v1.mappings.properties.embedding.dims -ne 768) { throw 'Wrong dimension' }
function Get-CorpusSnapshot {
  $records = foreach ($index in @('podcast_clips', 'podcast_rag_v1')) {
    $data = Invoke-RestMethod "$es/$index/_search?size=100&seq_no_primary_term=true"
    foreach ($item in ($data.hits.hits | Sort-Object _id)) {
      [ordered]@{ index=$index; id=$item._id; seq=$item._seq_no; term=$item._primary_term; source=$item._source }
    }
  }
  ConvertTo-Json -InputObject @($records) -Depth 30 -Compress
}
$before = Get-CorpusSnapshot
Invoke-RestMethod "$api/health"
Invoke-RestMethod "$api/search?q=machine%20learning"
foreach ($question in @('What do the speakers say about machine learning?', 'How is AI used?', 'What challenges are discussed?')) {
  $body = @{question=$question} | ConvertTo-Json
  $result = Invoke-RestMethod "$api/ask" -Method Post -ContentType 'application/json' -Body $body
  $result | Select-Object status, retrieval_mode, reason, degraded
  if ($result.retrieval_mode -ne 'hybrid' -or $result.sources.Count -eq 0) { throw 'Hybrid evidence not returned' }
  if ($null -ne $result.answer -or $result.reason -ne 'llm_not_configured') { throw 'Expected evidence-only response' }
  foreach ($source in $result.sources) {
    $id = [uri]::EscapeDataString($source.chunk_id)
    $resolved = Invoke-RestMethod "$api/sources/$id"
    if ($resolved.chunk_id -ne $source.chunk_id -or $resolved.excerpt -ne $source.excerpt) { throw 'Source mismatch' }
  }
}
if ((Get-CorpusSnapshot) -ne $before) { throw 'Corpus changed' }
```

For a safe fallback test, stop only this test API with Ctrl+C in Terminal 1,
point it at a nonexistent name, and restart. This does not create/delete an index
and fails before contacting Gemini:

```powershell
$env:RAG_VECTOR_INDEX = 'rag-unavailable-' + [guid]::NewGuid().ToString('N')
.\.venv\Scripts\python.exe -B -m uvicorn api.main:app --host 127.0.0.1 --port 8007
```

Terminal 2:

```powershell
$body = @{question='machine learning'} | ConvertTo-Json
$fallback = Invoke-RestMethod "$api/ask" -Method Post -ContentType 'application/json' -Body $body
$fallback | Select-Object status, retrieval_mode, degraded, reason
if ($fallback.retrieval_mode -ne 'bm25' -or -not $fallback.degraded -or $fallback.sources.Count -eq 0) { throw 'Fallback failed' }
if ((Get-CorpusSnapshot) -ne $before) { throw 'Corpus changed' }
```

Stop the test API afterwards. Restore `$env:RAG_VECTOR_INDEX='podcast_rag_v1'` and
`$env:RAG_RETRIEVAL_MODE='bm25'` in Terminal 1 before your next normal startup.
No data or volume cleanup is required.

## RAG generation cache (Stage 5)

`RAG_ANSWER_CACHE_TTL_SECONDS=900` enables a 15-minute generation cache using the
existing Redis connection. Set `0` to disable it; restart the API after changes.
BM25 retrieval, metadata enrichment, and context selection still run on every
`/ask`. Only validated structured answers and model abstentions are cached, never
the complete response or old source metadata. `cached=true` means generation was
loaded from this cache and revalidated against the current context's source IDs.

Keys use `rag:answer:v1:<sha256>`, separate from the unchanged search cache and its
TTL. The hash covers the actual system/user prompts (including evidence order),
provider, model, output limit, timeout, validation schema, and generation version.
Raw questions, transcripts, and credentials are not placed in keys. The payload
contains only `status` and `paragraphs`; Redis therefore stores generated text.
Provider wire-schema/fixed-setting changes must bump `GENERATION_VERSION`.

Failures, invalid outputs, disabled/unconfigured RAG, missing evidence, and context
budget failures are not cached. Corrupt entries are misses and may be overwritten
by a valid generation; otherwise they expire. Redis GET/SET errors fail open, with
a one-second bound per cache operation, and never expose Redis diagnostics.
Hits do not refresh TTL. Concurrent misses may each generate; there is no locking
or request coalescing in this stage. Equivalent means identical effective prompts,
not semantic similarity. API keys are excluded from cache identity, so key rotation
does not invalidate otherwise identical generation.

## Optional RAG answers (Stage 3)

Set `RAG_ENABLED=true` in the process environment (or use Uvicorn's
`--env-file .env`) to enable `POST /ask` with `{"question": "..."}`. No LLM or
embedding credentials are required for default BM25 retrieval. Without generation settings,
retrieval returns `answer: null`, full passages in `sources`,
`retrieval_mode: "bm25"`, and `status: "generation_unavailable"` with reason
`llm_not_configured`. An unsupported provider returns `unsupported_provider`.
An empty or unusable candidate set returns `insufficient_context`.
Disabled RAG retains the Stage 1 disabled response and performs no I/O.

For optional live generation, configure only server-side environment variables:

```text
RAG_ENABLED=true
RAG_LLM_PROVIDER=groq
RAG_LLM_MODEL=openai/gpt-oss-20b
RAG_LLM_API_KEY=
RAG_CONTEXT_MAX_TOKENS=8000
RAG_MAX_OUTPUT_TOKENS=1600
RAG_LLM_TIMEOUT_SECONDS=20
```

Supply the key privately in your local environment. These are live-validated
starting values for this project, not universal optimal values. The model is
selected solely through `RAG_LLM_MODEL`; application logic has no hard-coded model.
Live validation returned a grounded answer with validated citations using these settings.

No default model or SDK is supplied. The adapter uses the existing async HTTPX
client and Groq's `POST https://api.groq.com/openai/v1/chat/completions`, with
`response_format.type=json_schema`, `json_schema.strict=true`, and
`max_completion_tokens`. Select a model supporting strict Structured Outputs.
The closed schema requires `status` and `paragraphs`; each paragraph requires
`text` and `source_ids`. Application validation remains authoritative for citation
membership and semantic status consistency. See [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs).
Provider credentials are read only from `RAG_LLM_API_KEY`, never put in prompts,
serialized config, error messages, or frontend responses. Clients close after each
request; redirects, ambient HTTP proxies, streaming, and automatic retries are off.

System instructions require evidence-only answers, paragraph-level source IDs,
abstention when unsupported, and ignoring instructions in transcript text. The
user message contains JSON-encoded question and `TRANSCRIPT_EVIDENCE` data.
Successful responses contain `status: "answered"` and
`answer: {"paragraphs":[{"text":"...","source_ids":["S1"]}]}`. Every paragraph
must be nonblank and cite supplied evidence; unknown IDs, extra fields, duplicate
JSON keys, wrong types, malformed JSON, and missing citations reject the entire
answer as `invalid_generation`. No regex or Markdown citation extraction is used.
Citation validation verifies source membership, not semantic entailment.

`RAG_CONTEXT_MAX_TOKENS` defaults to 8,000, including instructions, question,
JSON source labels/metadata, evidence, a 256-token framing margin, and reserved
output (`RAG_MAX_OUTPUT_TOKENS`: documented starting value 1600; code fallback 800
when absent). One UTF-8 byte counts as one estimated
token. This intentionally conservative approximation is isolated in `prompt.py`;
it cannot guarantee the limits of every model/tokenizer. Choose a total budget
within the selected model's context window. Whole passages that do not fit are
omitted from the model request, never truncated; returned sources remain intact.
Citations are checked against the subset actually sent, not all retrieved sources.
No fitting evidence returns `insufficient_context` / `context_budget_exceeded`
without calling the provider. Model abstention returns `model_abstained`.

Timeouts (`RAG_LLM_TIMEOUT_SECONDS`, default 20), authentication errors, rate/quota
limits, provider 5xx/network failures and rejected requests return safe reason
codes with `generation_unavailable`, `answer: null`, retained sources, and
`degraded: true`. Invalid generation also retains sources and is degraded. No LLM
configuration or network failure affects startup, `/search`, `/health`, or sources.
No real Groq validation is part of automated tests; live validation is separate.
Safe server-side `Groq provider error:` diagnostics retain only sanitized error
fields and request facts (model, format, output limit, message byte lengths, and
source count). Only `json_validate_failed` includes a sanitized `failed_generation`,
bounded to 2000 characters with escaped line breaks; credential-like content is
redacted. Diagnostics never dump request headers or complete provider bodies and
are not exposed through `/ask`.

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

Run unit tests with `python -m pytest` (or explicit `ingest/tests api/tests`).
`pytest.ini` restricts discovery to test directories, excluding the legacy CLI
`scripts/integration_test.py`, whose separate `requests` dependency is undeclared.
Generation tests use a deterministic fake provider or HTTPX MockTransport, clear
ambient LLM settings, and prohibit real HTTPX network transports. To also run the
isolated real-service test in PowerShell:

```powershell
$env:RAG_INTEGRATION_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest api/tests/test_rag_integration.py -v
```

The integration tests exercise evidence-only and fake-generation modes; they
create/delete unique Elasticsearch indexes and roll back PostgreSQL metadata
transactions. Existing `/search`, Redis caching, ingestion,
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
