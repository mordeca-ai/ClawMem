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
