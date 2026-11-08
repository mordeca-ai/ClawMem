# Query pipeline

The `query` MCP tool runs the full hybrid retrieval pipeline, combining BM25, vector search, query expansion, and cross-encoder reranking in a single call.

## Pipeline stages

```
User Query + optional intent hint
  │
  ▼
BM25 Probe
  │ Top hit score >= 0.85 with gap >= 0.15?
  │ (disabled when intent is provided)
  ├── YES → Skip expansion, use BM25 results directly
  │
  ▼
Query Expansion (LLM)
  │ Generates text variants (lex/vec/hyde)
  │ Intent steers the expansion prompt
  │
  ▼
Parallel Search (typed routing)
  │ BM25(original) + Vector(original)   ← original fans out to both
  │ + BM25(lex expansions)              ← lex → BM25 only
  │ + Vector(vec/hyde expansions)       ← vec/hyde → vector only
  │
  ▼
Reciprocal Rank Fusion (k=60)
  │ Original query lists get 2x positional weight
  │ Expanded query lists get 1x weight
  │ Top candidateLimit results (default 30)
  │
  ▼
Intent-Aware Chunk Selection
  │ Intent terms at 0.5x weight
  │ Query terms at 1.0x weight
  │
  ▼
Cross-Encoder Reranking
  │ 4000 char context per doc
  │ Intent prepended to rerank query
  │ Chunk dedup (identical texts share single call)
  │ Batch cap = 4
  │
  ▼
Rerank / RRF Blend (blendRerank)
  │ 0.9 · reranker + 0.1 · normalized-RRF tiebreaker
  │ Reranker is the dominant signal; can promote a doc over RRF #1
  │ Falls back to pure RRF order if the reranker is unavailable / degenerate
  │ (no score above RERANK_DEGENERATE_FLOOR ≈ 1e-4); emits a rate-limited warning on fallback
  │ (the reranker-health guard surfaces this — see `clawmem doctor` / `clawmem rerank-health`)
  │
  ▼
Composite Scoring
  │ (relevance * 0.70 + recency * 0.15 + confidence * 0.15)   ← query-tuned weights (v0.13.0)
  │ recency-intent queries instead use 0.10 / 0.70 / 0.20
  │ × quality multiplier × co-activation boost
  │
  ▼
MMR Diversity Filter
  │ Jaccard bigram similarity > 0.6 → demoted
  │
  ▼
Results
```

## Key parameters

| Parameter | Effect | Default |
|-----------|--------|---------|
| `intent` | Steers 5 stages: expansion, reranking, chunk selection, snippet extraction, bypass | — |
| `candidateLimit` | How many candidates reach reranking | 30 |
| `compact` | Snippet vs full content output | true |
| `collection` | Filter by collection (comma-separated) | all |

## BM25 strong-signal bypass

When the top BM25 hit scores >= 0.85 and the gap to the second hit is >= 0.15, the pipeline skips query expansion entirely. This is a fast path for queries with obvious keyword matches.

The bypass is disabled when `intent` is provided — intent implies the query is ambiguous, so keyword confidence alone is insufficient.

Functional as of v0.23.0: through v0.22.0 a clamp bug flattened every FTS score to a constant 1.0, so with two or more hits the gap was always 0 (the bypass never fired) and with exactly one hit the gap was always 1.0 (it fired even on a garbage match). The scores are now the monotonic `|bm25|/(1+|bm25|)` transform — 0.85 corresponds to |bm25| ≥ 5.67 — and the check itself lives in one shared helper (`hasStrongFtsSignal`) used by both this pipeline and the CLI `query` command. Threshold tuning is deliberately deferred to a judged query-pipeline A/B (BACKLOG 49.3).

Ops escape hatch (v0.24.0): `CLAWMEM_DISABLE_FTS_BYPASS=true` forces the full expansion path on every query at BOTH consumers (the MCP pipeline and the CLI `query`), via the shared `ftsBypassEnabled()` read in `search-utils.ts`. Intended for A/B harnesses and incident triage; `hasStrongFtsSignal` itself stays pure.

## Query expansion

The LLM generates lex (keyword), vec (semantic), and hyde (hypothetical answer) variants of the query, each carried as a typed `ExpandedQuery`. Variants are **routed by type**: `lex` expansions are searched on BM25 only, `vec` and `hyde` on vector only. The original query is the only leg that fans out to *both* backends. All legs are searched in parallel and fused; the original query's two lists receive 2x weight in RRF, ensuring it anchors the ranking.

Routing by type matters because a keyword expansion ("auth token refresh") is useless to a dense vector model, and a hypothetical-answer passage (hyde) is mostly stopword noise to BM25. Searching every variant on both backends — the earlier behavior — fed each leg the input it handles worst, diluting RRF with low-quality lists.

## Vector-leg cap

Each vec or hyde expansion runs its own vector search. On sqlite an unscoped vector search scans the whole `vectors_vec` table: about 0.7-1.4s per leg on a 341k-fragment vault. A query with two vec legs and one hyde leg pays that four times. A collection-scoped leg the scoped scan serves (below) costs far less, so the cap matters mainly for unscoped and large-scope queries.

`CLAWMEM_QUERY_VEC_EXPANSION_LEGS=<n>` caps the expansion vector legs at `n` in both the MCP pipeline and the CLI `query`. Lex legs and the original query's vector leg always run. The cap is off by default; an unset or empty value, or anything other than a non-negative integer, leaves it off.

The cap keeps hyde legs first, then vec legs, each in generation order. A vec leg paraphrases the query, so its hits mostly repeat the original leg's. The hyde passage reaches answer-shaped text that the query embedding misses, and it is the one expansion the anchor gate never drops. On the 25-question library baseline (vn4rz.92), `n=1` matched the uncapped recall@10 and MRR. Keeping a vec leg instead lowered MRR by 0.023; `n=0` lowered it by 0.063.

When the cap drops legs, stderr shows `[query] vec-leg cap <n>: running <kept>/<total> expansion vector legs`.

`CLAWMEM_QUERY_EXPAND=off` turns expansion off for the process. `expandQuery` returns no legs without calling the LLM or reading the cache, so the query runs only as written. Unsetting `CLAWMEM_LLM_URL` does not do this, because `bin/clawmem` applies its default URL unless this switch is `off`.

## Parallel vector legs (CLI `query` and MCP `query`)

Both consumers run their vector legs (the original query plus each vec/hyde expansion that survives the cap) in parallel read-only Bun Workers, each with its own connection to the vault. `CLAWMEM_QUERY_VEC_PARALLEL=<n>` sets the pool size: default 4, clamped to 1-8, and `1` restores the old sequential in-process loop. Results are merged back in the original leg order, so the fused ranking is identical to the sequential run. A leg whose worker fails reruns in-process, so a fatal vector error still surfaces as before. The cap above applies first; the pool only sees the legs that survive it.

- **CLI `query`**: builds a pool per invocation (`createWorkerLegPool`) and closes it when the legs finish.
- **MCP `query`** (vn4rz.96): the server owns one long-lived pool, created lazily on the first multi-leg query, kept warm across calls (no per-call worker spawn or sqlite-vec load) and closed with the stores on shutdown. Workers are spawned on demand, never more than `CLAWMEM_QUERY_VEC_PARALLEL`, and concurrent `query` calls share that bound. Each leg carries its own options (`collections`, `dateRange`, `excludeCollections`, limit) and returns the `searchVecDetailed` result, so the `degraded` markers on the response are unchanged. A single-leg query, an in-memory store, or a store whose file is missing runs in-process. A worker opens each vault path read-only at most once, so named vaults share the pool.

## Expansion overlap (CLI and MCP `query`)

The original query's vector leg does not depend on the expansion. When the expansion is a live LLM call, the CLI `query` (vn4rz.97) and MCP `query` tool (vn4rz.101) start that leg on the worker pool at once and warm the remaining workers (`min(CLAWMEM_QUERY_VEC_PARALLEL - 1, CLAWMEM_QUERY_VEC_EXPANSION_LEGS)`). The expansion legs start on warm workers as soon as the expansion lands. Legs and their order are unchanged, so the fused ranking is identical.

The old path still runs in three cases: a strong BM25 signal (no expansion), an `llm_cache` hit (the leg list is known at once), and `CLAWMEM_QUERY_VEC_PARALLEL=1`.

### Deadline on the vector legs

`CLAWMEM_QUERY_VEC_DEADLINE_MS=<n>` bounds the vector stage of the CLI `query` and of `POST /query` on `clawmem serve` (vn4rz.131). Unset, `0` or anything other than a positive integer leaves it off, and the paths described here run unchanged. The MCP `query` tool has its own pipeline and ignores it.

When it is on, every leg runs on a worker, including the single-leg and scoped paths that otherwise run in process. A scan on the main thread would block the timer meant to end it. `clawmem serve` keeps one worker warm across requests. Legs and their order are unchanged, so a query that finishes in time ranks exactly as it does with the deadline off. Scoped legs read the scope's vectors on their worker's own thread rather than on `CLAWMEM_QUERY_VEC_PARALLEL` threads, so a large scope loads more slowly under the deadline.

On expiry the workers are terminated, the vector results are dropped, and the query continues with FTS, fusion and rerank. It never falls back to an in-process leg. `POST /query` then answers `{"degraded": true, "degraded_reason": "vec_deadline_expired", "hits": [...]}` instead of the bare array. The CLI prints the usual array and notes the drop on stderr. With `CLAWMEM_QUERY_TIMING=1`, `counts` carries `vec_deadline_expired`, `vec_deadline_shed` and `vec_leg_failed`.

`Worker.terminate()` does not stop a scan already inside SQLite. Measured on Bun 1.3.13, it returns in under 10 ms, but the thread keeps scanning until `sqlite3_step` returns, and only then does its `close` event fire. So an abandoned scan still uses CPU for its full length. Once 2 abandoned scans are still running, later queries skip the vector stage and answer degraded with `vec_deadline_shed`, so they cannot stack up.

A worker failure other than expiry also drops the vector results, as `vec_leg_failed`. A fatal vector error, such as a model mismatch, still fails the query as it does with the deadline off.

### FTS beside the vector stage (`clawmem serve` with a request budget)

The original query's vector leg does not depend on FTS: FTS only decides whether expansion runs. When `CLAWMEM_QUERY_BUDGET_MS` and `CLAWMEM_QUERY_VEC_DEADLINE_MS` are both on, `POST /query` starts the vector stage (vn4rz.150) before it awaits FTS, so the two run side by side and a request costs about the longer of the two instead of their sum. `CLAWMEM_QUERY_FTS_VEC_OVERLAP=0` (or `false`/`off`/`no`) restores the serial order; the default is on.

- The vector deadline is `min(CLAWMEM_QUERY_VEC_DEADLINE_MS, what the budget has left)` read when the stage starts, which is now before FTS, so the deadline also covers the wait for FTS and the expansion that follows it.
- Expansion still waits for FTS. `expandQuery` starts only when FTS reports no strong signal and the budget is not spent, so the strong-signal bypass still saves the LLM call. The expansion vector legs join the stage once the expansion lands (the lex legs still run after it, as before); a strong or spent FTS gives the stage no expansion. Workers for the expansion legs are not pre-warmed on this path, since FTS has not yet said whether there will be any.
- When FTS is cut by the budget, the vector results that arrived within the deadline stay in the answer. The reply is still marked `query_budget_expired`. The "vector stage skipped" log line now appears only when the stage never started (budget already spent on arrival, or the overlap is off).
- Without a budget, with the vector deadline off, or with the knob at `0`, the stages run in the old order. The CLI `query` never overlaps.
- With `CLAWMEM_QUERY_TIMING=1`, a budgeted row carries `fts_vec_overlap` (1 when the vector stage started beside FTS, 0 otherwise). The `fts` and `vec_legs` stage windows then overlap; `vec_legs` ends when the vector stage ends, not when FTS lets the request reach it.

## Scoped vector scan (collection-scoped legs)

sqlite-vec reads every vector chunk before it applies a `hash_seq IN (...)` filter, so a scoped vector leg cost a scan of the whole vault (1.4 GB for 350k x 768 vectors) whatever the collection. For an include scope (`-c`, `collections`, `dateRange`) small enough to pre-filter, the leg now reads only the vec0 chunks that hold the scope's fragments, straight from the shadow tables, and computes cosine distances in JS with sqlite-vec's float32 operation order (vn4rz.100). The distances are bit-identical to vec0's and cover every in-scope fragment, so the results equal the vec0 path's. On the live vault, `docs` (15.7k fragments) sits in 172 of 459 chunks: about 0.54 GB is read once, and each further leg is a ~25 ms scan.

- **When it applies**: the table has clawmem's canonical `vec0(hash_seq TEXT PRIMARY KEY, embedding float[N] distance_metric=cosine)` shape and the scope spans at most half of the table's chunks. Otherwise the leg uses vec0 MATCH as before (`-c monchujo`, for example, is too large to pre-filter).
- **Safety**: every slot is checked before its vector is used (blob sizes, the validity bit, and the chunk's rowid at that slot equal to the enumerated rowid). Any mismatch abandons the load and the leg uses vec0 MATCH. vec0 rowids are AUTOINCREMENT and clawmem never updates a vector in place, so a matching rowid is the same vector.
- **CLI `query`**: when the scope qualifies, the legs run in-process on one read-only store instead of the worker pool. The scope's chunks are read once, on `CLAWMEM_QUERY_VEC_PARALLEL` threads, while the expansion is generated; the leg embeds still overlap. That store caches the scope (enumeration and vectors) until another connection commits (`PRAGMA data_version`) or the schema changes.
- **Other callers** (MCP `query` workers, `vsearch`, writable stores) use the scan per call and keep nothing, so a long-lived process holds no scope vectors.
- **Knob**: `CLAWMEM_QUERY_VEC_SCOPED_SCAN=0` (or `false`/`off`/`no`) turns it off; every scoped leg then uses vec0 MATCH.
- **Resident scope (uvnhf.17)**: `CLAWMEM_QUERY_RESIDENT_SCOPE_MB=<whole MiB>` makes the main thread keep loaded scope vectors between queries, for `clawmem serve`, which opens a new leg store per query. Off when unset, `0` or invalid. A hit needs the same file (path, device, inode, birth time), scope key, dim and `schema_version`, an element-wise equal fragment list, and an entry younger than 6 hours (`CLAWMEM_QUERY_RESIDENT_SCOPE_MAX_AGE_S` overrides it for tests). Enumeration still runs on every query. Worker threads never use it. The timing row gains `scope_resident_hits` and `scope_resident_misses`, counted per lookup, so a query can report 2.

Read-only connections also map the vault into memory (`PRAGMA mmap_size`). Without it, a vec0 scan copied each 4 KiB page through a `pread` call, and system time was the larger part of a leg. The CLI scoped scan uses 128 MiB per read connection (vn4rz.102); its chunk-targeted reads do not need a whole-vault map. Other read-only paths retain the 2047 MiB default. `CLAWMEM_SQLITE_MMAP_MB` explicitly overrides either size, including `0` to disable mmap. Writable connections are unchanged. On the live `docs` scope, the 128 MiB setting measured 522 MiB maximum RSS over 30 queries and 1.819s p50 against 2.119s at the old setting (30 alternating pairs, shared cached expansion, host load recorded).

## Fixed per-query costs

- **Enrich**: `enrichResults` looks each result up by `(collection, path)` through the documents unique index. The old `(collection || '/' || path) = ?` match scanned the documents table once per result, 0.25-1.1s for a 30-candidate query on a 46k-document vault. A miss falls back to the old expression, so a hand-edited collection name containing `/` still resolves.
- **Rerank cache writes**: the scores from one rerank call are written to `llm_cache` in one transaction. Before, each score was its own autocommit insert, about 8ms each.
- **Vector hydrate**: the document body is read once per returned document (vn4rz.100). The hydrate query used to select `content.doc` on every fragment row of every requested hash, which for one 60-document `-c docs` leg was 2,399 rows and about 91 MB of text (200-380 ms).

## Stage timing

`CLAWMEM_QUERY_TIMING=1` makes the CLI `query` print one line to stderr when it finishes:

```
[query-timing] {"startup_ms":422,"total_ms":3775,"stages":{"expand":{"start_ms":540,"ms":498},...},"counts":{"vec_legs":4,"lex_legs":0}}
```

`start_ms` is measured from process start, so overlapping stages show their real window and the durations need not sum to `total_ms`. Stages: `store_open`, `fts`, `expand`, `vec_legs`, `lex_legs`, `fuse_rerank`, `rerank`, `enrich`, `score`, `dispose`; a scoped query on the scoped scan adds `scope_plan` (open the read-only store and enumerate the scope) and `scope_load` (read its chunks), with counts `scope_scans`, `scope_chunks_read` and `scope_load_fallbacks`. Off by default.

## Chunk selection

For reranking, each document's content is windowed into overlapping chunks. Chunks are scored by overlap with query terms (1.0x weight) and intent terms (0.5x weight). The best chunk per document is sent to the cross-encoder.

## When to use query vs alternatives

| Tool | Cost | When |
|------|------|------|
| `search` | 0 GPU calls | Known exact terms, spot-check |
| `vsearch` | 1 GPU call | Conceptual, don't know vocabulary |
| `query` | 3+ GPU calls | General recall, unsure which signal matters |
| `intent_search` | Hybrid + graph | Why/entity chains, temporal queries |
| `query_plan` | Hybrid + decomposition | Multi-topic queries |
