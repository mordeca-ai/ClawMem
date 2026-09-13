# `src/pg` — the PostgreSQL write path (master-harness-vn4rz.7)

Additive. This module does **not** touch `src/store.ts` and does not change the sqlite
path's behavior. The two writers coexist until `master-harness-vn4rz.11` retires sqlite —
the transient window [ADR-0162](../../../claude/master-harness/intelligence/decisions/0162-knowledge-substrate-cutover-postgres-pgvector-source-corpus-tier.md) §6 sanctions.
ADR-0162 forbids a *permanent* pluggable backend, not this.

## Configure

Either one URL:

```bash
export CLAWMEM_PG_URL="postgres://clawmem:<pw>@127.0.0.1:5433/clawmem"
```

…or the discrete vars: `CLAWMEM_PG_HOST`, `CLAWMEM_PG_PORT` (default `5433`),
`CLAWMEM_PG_DATABASE`, `CLAWMEM_PG_USER`, `CLAWMEM_PG_PASSWORD`.

**There is no default and no sqlite fallback.** With nothing configured every entry point
throws. A write path that quietly re-routed to the old store on a config typo would give
you a green run and an empty database.

**Port 5433 is the `clawmem-pg` container. The host cluster on 5432 is a different
database.** Passwords live in `master-harness/.secrets/local/clawmem-pg/<role>.pw`.

## Run

```bash
bun src/pg/cli.ts migrate                                  # idempotent; safe to re-run
bun src/pg/cli.ts status                                   # dim, vec models, per-collection counts
bun src/pg/cli.ts reindex --collection memory-topics       # one collection
bun src/pg/cli.ts reindex --collection foo --limit 1       # validate small first
bun src/pg/cli.ts reindex --no-embed                       # schema/parity smoke, no yoshiee traffic
```

Parity gate (in master-harness):

```bash
tools/clawmem-pg-parity --counts
```

## The read path: two arms and one fusion

| Module | Entry point | What it is |
|---|---|---|
| `search.ts` | `pgSearchVecDetailed` | The **vector** arm — pgvector ANN, bounded, model-fenced |
| `search-fts.ts` | `pgSearchFtsDetailed` | The **lexical** arm — `websearch_to_tsquery` + weighted `ts_rank_cd` |
| `search-hybrid.ts` | `pgSearchHybridDetailed` | **RRF fusion** of the two |

Three rulings are encoded in the fusion, and each has a test that goes red if it is undone:

**Fusion consumes RANKS, not scores.** There is no cross-backend score normalization, no
blend weight and no scaling between the arms. `1 - cosine_distance` and `ts_rank_cd` are not
on a shared scale and no comparability between them has been *measured* on this corpus, so
inventing one would encode a relationship we have no evidence for. RRF needs only each arm's
ORDER, which is the part we have grounds to trust. The arms are equally weighted and that is
**not** configurable — a tuning knob with no eval behind it is the same unmeasured claim.

**A fused result NEVER silently presents itself as two-arm.** `PgHybridSearchResult.arms` is
an always-present discriminator:

| `arms` | `degraded` | Meaning |
|---|---|---|
| `"vec+fts"` | `false` | Both arms ran and are trustworthy. The **only** value on which a caller may say "hybrid". Includes the genuine-empty answer (`results: []`, nothing degraded — "nothing matched"). |
| `"vec-only"` / `"fts-only"` | `true` | One arm degraded or threw; the healthy arm's ranking is returned, in its own order. Not an error, not empty — **not hybrid**. |
| `"none"` | `true` | Neither arm could look. `results: []`. Rendering this as "no matches" is the lie both arms' degraded channels exist to prevent. |

Both arms rejecting is a **throw**, never an empty answer. `armFailures[]` carries each
non-contributing arm with its own arm-native reason (`kind: "degraded"`) or the error it
raised (`kind: "threw"`); the two reason vocabularies are deliberately not merged, because
`embed-unavailable` and `empty-tsquery` are not the same kind of fact.

**The arms run SEQUENTIALLY, and that is a correctness constraint.** `PgQueryable` is one
connection and a PostgreSQL connection holds one transaction at a time; both arms wrap each
SQL leg in `BEGIN` / `SET LOCAL` / `COMMIT`. Running them concurrently on one client
interleaves two transaction blocks and the loser gets SQLSTATE **25P01** (*"ROLLBACK TO
SAVEPOINT can only be used in transaction blocks"*) — measured, not theorised. Consequence:
`timeoutMs` is a **per-arm** budget, so a hybrid call's worst-case wall clock is the *sum* of
the two arms', not the max. Concurrency would need two pooled clients.

Nothing in `src/` calls the hybrid path yet — caller migration is a separate step.

### The rerank stage on top: `search-reranked.ts`

| Module | Entry point | What it is |
|---|---|---|
| `search-reranked.ts` | `pgSearchRerankedDetailed` | The **cross-encoder rerank** stage — composes the hybrid, re-orders its head |

It **composes** the hybrid rather than editing it: the call returns
`{ results, hybrid, rerank, timings, … }` with `hybrid` carrying
`PgHybridSearchResult` *verbatim*, so slice 5's three-way invariant still holds over that
object (both test tiers re-assert it there). Folding rerank state into `degraded` / `arms`
would break it — and a rerank problem is not an arm problem: retrieval succeeded either way.

**`rerank` is an always-present discriminator, exactly like `arms`.** For every value other
than `"applied"`, `results` **is** `hybrid.results` — the same array, byte-identical order.
A rerank problem degrades to fusion, never to garbage, never to empty, never to a reordering
nobody chose.

| `rerank` | Meaning |
|---|---|
| `"applied"` | Ran, returned usable scores, the blend used them. The only value on which the order differs from the fusion. |
| `"skipped-no-reranker"` | None injected. A **caller choice**, not a failure. |
| `"skipped-budget"` | Less than `PG_RERANK_MIN_BUDGET_MS` (250) left of the deadline; we chose not to start. |
| `"skipped-no-text"` | Nothing to rank — every candidate body empty, or no candidates at all. A degenerate *input*. |
| `"degenerate"` | Responded, but no score cleared the degenerate floor (an empty response included). Surfaced by `blendRerank`'s `onFallback`. |
| `"failed"` | Threw, or blew the remaining budget. `rerankError` carries the cause. |

**A rerank problem NEVER throws the search.** Both arms failing still throws (slice 5's
"we could not look" ≠ "nothing matched"), and the hybrid call here is deliberately *not*
wrapped in a try/catch — this stage degrades rerank problems, not retrieval problems.

**The budget is ONE deadline that DECREMENTS — not a third per-stage timeout.** `timeoutMs`
already bounds *each arm*, and the arms run sequentially, so the hybrid's worst case is
vec + fts. An independent `rerankTimeoutMs` would make it vec + fts + rerank and put the
`hook p95 ≤ 1800 ms` clause out of reach *by construction*. Instead `deadlineMs` (default
**1500**, unmeasured, chosen to sit under 1800 with caller headroom) is an **overall**
wall-clock budget: `t0` once, hybrid, then `remaining = deadlineMs − elapsed` handed to the
reranker as its `timeoutMs` — and raced against the same instant, so a reranker that honours
neither `timeoutMs` nor `signal` still cannot overrun. Below the floor we skip. Measured
`timings: { hybridMs, rerankMs, totalMs }` come back on the result so the parity run **reads**
the additive budget instead of inferring it.

**The reranker is INJECTED, like the embedder** — `src/pg/` does not import `src/store.ts`.
The `CLAWMEM_RERANK_URL` / `CLAWMEM_RERANK_API_KEY` contract, the remote-GPU→local fallback
chain, the batch-of-4 VRAM cap and the d0hz per-backend cache-key namespacing all live in
`store.ts`'s `rerank()` and are inherited through injection, **never re-read here** (a second
read of that env inside `src/pg/` is exactly the drift the "reuse the helper, don't restate
it" ruling exists to prevent). Wire it as
`(q, docs, o) => store.rerank(q, docs, DEFAULT_RERANK_MODEL, intent, o)`.

**The blend is `blendRerank`, and there is no third blender.** Two correctness reasons, not
preferences: it maps over **candidates**, so partial rerank coverage can never *drop* a
document (`blendFusionAndRerank` maps over the rerank output and would); and it has the
degenerate floor plus the `onFallback` hook that becomes the `"degenerate"` status here.
`blendFusionAndRerank` also still applies `rrfWeight = 0.75` at rrfRank ≤ 3 — the very defect
`blendRerank`'s own docstring names (RRF rank-1 mathematically immovable by the reranker).

**Known duplication, stated rather than refactored:** the candidate-cap expression
`Math.max(limit, 30)` and the 4000-char text truncation now exist both here and in
`src/clawmem.ts` (~1549 / ~1608, the sqlite reference implementation of this same stage).
Unifying them means touching the sqlite read path, which is a separate change.

Inside `src/`, only the `retrieve` verb calls these functions.

### The CLI over all of it: `retrieve.ts`

`bun src/pg/cli.ts retrieve --mode <search|vsearch|query> --query <text>` prints exactly
one JSON object (`clawmem-pg-retrieve/v1`). It is the first place the reranker injection
actually runs. The contract, exit codes and wiring are in
[`README-retrieve.md`](README-retrieve.md).

## The three things that will bite you

**1. The embedding dimension has exactly ONE home.** `EMBED_DIM` in `config.ts`. pgvector
fixes the dimension at `CREATE TABLE`; sqlite's `vec0` table discovered it at runtime from
the first embed response, and that discovery is gone. The migration runner substitutes it
into `:EMBED_DIM`, and `assertSchemaGeometry` compares it against the live column before
any write. A mismatch is a loud refusal — never a truncate, never a pad.

**2. The geometry preflight runs INSIDE the write transaction.** sqlite's
`assertWriteEmbedModelConsistent` memoized on `PRAGMA data_version`; Postgres has no
equivalent, so **the memo is dropped and only the transactional atomicity is kept** — the
half that matters. Doing the check outside the transaction, or after the insert, re-opens
the foreign-model vector poisoning `master-harness-p2ib3` and `vn4rz.21` were written to
close. A transaction-scoped advisory lock closes the check-then-write race that READ
COMMITTED leaves open between two concurrent writers on an empty table.

**3. Reindex, never copy.** Every document is re-read and re-embedded from its
file-authored source. sqlite is read *only* as a count baseline by `clawmem-pg-parity`,
never as a data source.

## Known consequences, stated rather than papered over

- **FTS ranking SHIFTS.** sqlite used FTS5 `porter unicode61`; this uses
  `to_tsvector('english', …)` with `setweight()` (title `A`, description/tags `B`, body
  `D`) and GIN. Different stemmer, different scoring model — a retrieval-quality change
  with no exception thrown. Postgres FTS is a Boolean match model with `ts_rank` bolted on
  (ADR-0162 §2), which is why the lexical arm should feed an RRF fusion rather than be
  trusted as a ranked result on its own. **The configuration is named explicitly at both
  ends**; a bare `to_tsquery` at query time against an `'english'` write side fails
  silently.
- **The vector index family is NOT decided.** `002_vector_index_hnsw.sql` ships ADR-0162
  §2's own *seeded* starting point (`m=16, ef_construction=128, ef_search=100`, AWS
  guidance). `master-harness-vn4rz.32` runs the probe that actually chooses it. These
  numbers are not measured on our data.
- **Today's sqlite vector search is brute force** (`vec0`, no ANN index at all). Any
  pgvector index is therefore a strict improvement, not a parity requirement, and recall
  below 1.0 is a new tradeoff for vn4rz.32 to price — not a regression.
- **`content_type` is the closed ADR-0058 enum.** Values outside it land as `'unknown'`
  with the raw string preserved in `content_type_raw`, and the reindexer prints the retag
  backlog per collection. Nothing is silently discarded, and §5's "gaps resolve by retag"
  stays a query rather than an archaeology exercise.
- **No `vault` column anywhere.** ADR-0162 §1 fixes vault topology as database-per-vault
  (`clawmem` vs `clawmem_nsfw` are separate DATABASES), so sqlite's `entity_nodes.vault` is
  dropped on purpose.
