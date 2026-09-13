# `pg retrieve` — the machine-readable PG read path (master-harness-2wx75 slice 7)

The first caller of the four read functions. One verb, three modes, **exactly one JSON
object on stdout** (logs go to stderr), so a parity eval can run the PG arm next to the
sqlite `clawmem search|vsearch|query` CLI and normalize both the same way.

```bash
bun src/pg/cli.ts retrieve --mode <search|vsearch|query> --query <text> \
  [--limit N (default 10)] [--collection a,b] [--vault public|nsfw] \
  [--deadline-ms N] [--no-rerank]
```

| `--mode` | Calls | `--deadline-ms` becomes |
|---|---|---|
| `search` | `pgSearchFtsDetailed` (lexical arm) | `timeoutMs` |
| `vsearch` | `pgSearchVecDetailed` (vector arm) | `timeoutMs` |
| `query` | `pgSearchRerankedDetailed` (RRF hybrid + cross-encoder) | `deadlineMs` (default 1500) |

`--vault public` is the `sfw` vault (both spellings work). Unknown flags are **refused**:
a typo'd `--colection` that quietly searched everything would be a green run over the
wrong scope.

## Output contract — `clawmem-pg-retrieve/v1` (pinned)

```json
{"schema":"clawmem-pg-retrieve/v1","mode":"query","query":"...","limit":10,
 "results":[{"file":"<collection>/<path>","score":0.123}],
 "degraded":false,"degradedReason":null,
 "rerankStatus":"applied","timings":{"totalMs":123.4,"hybridMs":80.1,"rerankMs":30.2,"embedMs":150.6},
 "error":null}
```

Field names are the contract. Renaming one means a new `v2` schema, not an edit to v1.

- **`file`** is `SearchResult.displayPath` (`<collection>/<relative path>`), the same key
  the sqlite `clawmem query --json` prints. Checked live on 2026-09-12: both CLIs put
  `research/cpo1-game-architectures/define/define-ingestion-strategy.md` first for
  *"hybrid search reciprocal rank fusion"*.
- **`score`** uses whatever scale the arm produces (ts_rank_cd, `1 − cosine distance`, RRF, or
  the blended rerank score). It is **not** comparable across modes or with the sqlite
  scores. Compare by rank.
- **`degradedReason`**: the arm's own reason for `search`/`vsearch`. For `query` it is
  `"<arm>:<reason>"` or `"<arm>:threw:<Type>: <msg>"` for each arm that didn't contribute,
  joined with `; `.
- **`rerankStatus`** is `null` for `search`/`vsearch`. For `query` it is the reranked module's
  status, passed through unchanged. Any non-`applied` value means the fused order. The reason
  goes to stderr as `[pg-retrieve] rerank <status>: <reason>`.
- **`timings`**: `totalMs` is always measured. `hybridMs`/`rerankMs` are `query`-only.
  `embedMs` is measured around the embed call (`vsearch`/`query`).

| Exit | Meaning |
|---|---|
| `0` | The call completed. That includes degraded results (single-arm, or degraded-empty with `degraded: true`) and a real empty answer. |
| `2` | The call threw (statement timeout, PG unreachable, vault not configured) **or** the flags didn't parse. You still get one JSON object, with `results: []` and `error: "<Type>: <message>"`. |

## The reranker wiring

`query` mode is where the injection documented in `search-reranked.ts` ruling 5 first
runs for real:

```ts
(q, docs, o) => store.rerank(q, docs, DEFAULT_RERANK_MODEL, undefined, o)
```

`store.rerank` reads `CLAWMEM_RERANK_URL` / `CLAWMEM_RERANK_API_KEY` on its own. `intent`
is `undefined` because the sqlite `clawmem query` passes none.

**The rerank cache is an in-memory sqlite (`createStore(":memory:")`).** If it used the
operator's sqlite index, a PG read would write to the store it is replacing and replay
sqlite-era cached scores into a PG measurement. The cost: every call scores live, with no
cache warm-up.

**Fallback, measured:** with `CLAWMEM_RERANK_URL=http://127.0.0.1:9` (a connection that
*hangs* on this WSL host rather than being refused), `rerankStatus` is `"failed"` once the
deadline runs out: `totalMs` ≈ 1546 at the default 1500 ms deadline, ≈ 10052 at
`--deadline-ms 10000`. Results are the fused order and the exit code is `0`. The reranked
module reports `rerankMs: 0` on `failed`, even though the time was spent.

## Running it live

The URL is built the same way `master-harness/tools/lib/clawmem_pg_env.py` builds it
(`postgres://clawmem:<pw>@127.0.0.1:5433/clawmem`, password from
`.secrets/local/clawmem-pg/clawmem.pw`). `vsearch`/`query` also need the
`CLAWMEM_EMBED_*` / `CLAWMEM_RERANK_URL` environment (`~/.config/environment.d/50-clawmem.conf`).
