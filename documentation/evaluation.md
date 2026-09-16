# Retrieval evaluation

[Project overview and results table](../README.md#real-data-evaluation) · [Architecture](architecture.md)

## Final local real-data run

The reported run used 20 real Spotify podcast episodes, 730 chunks from 120-second windows with 60-second overlap, and 730 Gemini embedding-2 document vectors at 768 dimensions. BM25 and Hybrid used the same fixed corpus and 20 **AI-assisted, human-reviewed and approved** queries/relevance judgments. They are not described as purely human-authored judgments.

Only the supplied aggregate results are published; transcripts, individual questions, labels and local reports are not included. These results describe a completed local run, not a benchmark reproduced by the repository's synthetic fixtures.

Dataset fingerprint:

```text
8455e4af572fae84fbd4fefda494a60247d9b8452d0e55fedd3eb0b6bcfa4d6b
```

The fingerprint identifies normalized queries and sorted relevance sets in dataset order; notes are excluded. It does **not** fingerprint Elasticsearch data. Reproduction requires matching judgments and a fixed corpus/index configuration.

The [full metric table](../README.md#real-data-evaluation) shows higher Hit@3, Hit@5, MRR and nDCG@5 for Hybrid, together with substantially higher retrieval latency. Both runs recorded zero retrieval errors. Hybrid reported:

| Operational result | Count / rate |
|---|---:|
| Successful Hybrid retrieval | 20/20 (100%) |
| BM25 fallback | 0/20 (0%) |
| Vector failures | 0 |
| No usable vector cases | 0 |

These findings support a quality/latency trade-off on this small set, not a universal ranking of methods. The application default remains `RAG_RETRIEVAL_MODE=bm25`.

## What the evaluator measures

`api/rag/evaluate.py` reuses actual BM25 and Hybrid retrieval. It scores the final selected sources after deduplication, overlap suppression, source-count and evidence-byte limits, but **before** generation's model-context budgeting. It does not call Groq, use the generation cache or score answers.

Judgments are binary. Retrieved IDs are deduplicated while retaining first occurrence and compressing ranks; relevant IDs are a set. Metrics are computed per query and macro-averaged, so every row has equal weight.

| Metric | Definition |
|---|---|
| Hit Rate@1/@3/@5 | Fraction of queries with at least one relevant chunk in the first k unique results. |
| Recall@1/@3/@5 | Relevant chunks in top k divided by all judged relevant chunks for that query, then macro-averaged. |
| MRR | Mean reciprocal rank of the first relevant result across the entire returned list, normally capped at six sources; zero if none. |
| Binary nDCG@5 | Sum of `1/log2(rank+1)` for relevant top-five hits, divided by the ideal sum for `min(5, relevant count)` hits. |
| Mean/median latency | Milliseconds around each retrieval attempt, aggregated per requested mode. |

Empty/nonmatching retrieval scores zero. Queries with no relevant labels also score zero and remain in averages. Fewer-than-k results do not change recall's denominator. Infrastructure failures are counted, scored as empty retrieval and retained in latency aggregates rather than silently excluded.

Latency includes Hybrid query embedding, kNN, canonical hydration and selection. It excludes PostgreSQL enrichment (no database pool is passed), generation and report writing. Client construction is outside the timer, but first-request connection establishment may still contribute. These are **retrieval timings**, not browser-to-answer latency or a sustained-load benchmark.

## Hybrid accounting

Successful Hybrid means valid vector candidates participated in fusion. All attempted queries are the rate denominator. BM25 fallback can reflect a vector/provider failure or a successful branch with no usable vector evidence; `vector_failure_count` and `no_usable_vector_count` distinguish these. A total retrieval error is neither successful Hybrid nor BM25 fallback.

Hybrid quality metrics include fallbacks, matching the behavior users receive. Inspect these counters alongside scores; a run dominated by BM25 fallback is not evidence of semantic-retrieval quality.

## Reproduction

Use UTF-8 JSON (list or single object) or JSONL (one object per nonblank line). This is an illustrative schema, not a real question or judgment:

```json
{"query":"<reviewed question>","relevant_chunk_ids":["<stable chunk ID>"],"notes":"<judgment provenance>"}
```

`query` must be nonblank and at most 2000 characters; it is trimmed as in `/ask`. Use stable chunk IDs, not `S1` labels or Elasticsearch `_id` hashes. `notes` is optional. Unknown fields and malformed records are rejected. Limits are 5 MiB, 10000 queries, 10000 relevance IDs per query, 2048 characters per ID and 4000 characters for notes.

Keep real datasets/reports in ignored `.local-eval/` or outside the repository. Set `RAG_SOURCE_INDEX` to the intended frozen corpus and `RAG_VECTOR_INDEX` to its matching companion. Keep selection settings identical; do not ingest/backfill during comparison. Gemini settings are in [operations](development-notes.md#vector-preparation-stages-6-and-65). Never copy credentials into dataset fields.

```powershell
# Repository root; local process environment already configured.
.\.venv\Scripts\python.exe -m api.rag.evaluate --dataset .local-eval/queries.jsonl --mode bm25 --output .local-eval/bm25.json
.\.venv\Scripts\python.exe -m api.rag.evaluate --dataset .local-eval/queries.jsonl --mode hybrid --output .local-eval/hybrid.json
```

Alternatively use `--mode both --output .local-eval/comparison.json`; it processes each query with BM25 then Hybrid, with no warmup. Running both procedures makes extra Gemini calls. The CLI defaults to BM25, does not change server mode, and reads process environment rather than loading `.env` automatically.

```powershell
$bm = Get-Content .local-eval/bm25.json -Raw | ConvertFrom-Json
$hy = Get-Content .local-eval/hybrid.json -Raw | ConvertFrom-Json
if ($bm.dataset_sha256 -ne $hy.dataset_sha256) { throw 'Judgments differ' }
$bm.runs.bm25.summary
$hy.runs.hybrid.summary
```

Reports include configuration, fingerprints, query numbers, relevance/result IDs, actual modes, safe errors, metrics and latency. They omit query text, notes, transcripts, connection URLs and credentials, but still contain corpus identifiers: keep them local. The CLI labels reports `functional evaluation only`; interpreting a real-data run requires the scope and judgment provenance above.

Exit codes: 0 for completion (including valid fallback), 1 when saved results include retrieval errors, 2 for input/startup/output failures. The committed smoke fixture is for deterministic test doubles; it cannot reproduce the real-data results.

## Interpretation limits

- Twenty queries and twenty episodes limit coverage; no statistical significance or generalization claim is made.
- AI assistance, reviewer choices and potentially incomplete relevance sets can affect judgments. Record provenance rather than implying wholly manual authorship.
- Window overlap, ASR quality, source limits and chunk boundaries affect recall; this is not episode-level relevance evaluation.
- Network conditions, provider availability, index warmth and fixed run order affect latency. Aggregates do not isolate each component's cost.
- Retrieval metrics do not measure answer factuality, citation entailment, user satisfaction or privacy/compliance outcomes.
- Keep the 25-clip synthetic workflow for functional regressions; it is distinct from this 730-chunk real-data run.
