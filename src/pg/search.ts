/**
 * PG VECTOR READ PATH — the search arm (master-harness-2wx75, ADR-0162 §6).
 *
 * The write path landed first (src/pg/write.ts, src/pg/reindex.ts); until now
 * nothing could READ what it wrote. This module is the vector half of that read
 * path, and it exists to return the SAME SHAPE src/store.ts's searchVec()
 * returns (`SearchResult`), so a caller consuming vector hits does not care
 * which backend served them.
 *
 * THREE THINGS ARE LOAD-BEARING HERE:
 *
 *  1. THE QUERY FORMATTING. The stored vectors were produced by
 *     formatDocForEmbedding(); a query must go through formatQueryForEmbedding()
 *     (`task: search result | query: ...`) or the two live in different corners
 *     of the same vector space and every distance is quietly wrong. We IMPORT
 *     that helper from src/llm.ts rather than restating the template, because a
 *     restated template is a template that drifts.
 *
 *  2. THE MODEL-IDENTITY FENCE. content_vectors.model records which model
 *     produced each row. Embedding a query with a DIFFERENT model is not a
 *     degraded search, it is a meaningless one — cosine distance between two
 *     vector spaces is a number with no interpretation. The write path already
 *     refuses this (PgVecWriteModelMismatchError); the read path refuses it here
 *     with PgVecReadModelMismatchError, naming both models. It must never
 *     degrade to "zero rows", which is indistinguishable from "no matches".
 *
 *  3. THE ANN PREDICATE. `ORDER BY cv.embedding <=> $1::vector LIMIT n` is the
 *     only shape content_vectors_embedding_hnsw_idx (vector_cosine_ops) can
 *     serve. Any rewrite that computes the distance in a projection and sorts on
 *     the alias, or that wraps the column, silently becomes a seq scan over
 *     every fragment in the vault.
 *
 * MEASURED PLANS (2026-09-09, live vault, pgvector 0.8.6, EXPLAIN COSTS OFF):
 *   - NO collection filter  -> `Index Scan using content_vectors_embedding_hnsw_idx`
 *     feeding a Memoize'd lookup of documents. The ANN index serves the order-by,
 *     which is the whole point of the shape above.
 *   - `--collection research` (141 docs / 8.5k fragments) -> the planner instead
 *     PRE-filters (`documents_collection_active_idx`) and sorts exactly. That is
 *     the RIGHT choice at that selectivity — it is exact rather than approximate —
 *     and it is the planner's to make; do not hint it away. The `<=>` order-by is
 *     what keeps BOTH plans available.
 *
 * FILTERING + RECALL. The collection/active predicates sit in the SAME query as
 * the ANN order-by, which makes this a POST-FILTER against the HNSW index: the
 * index walks in distance order and the filter discards. On a narrow collection
 * filter that can under-fill. pgvector 0.8's iterative index scan is the
 * sanctioned answer (`hnsw.iterative_scan`), so it is turned on for the duration
 * of the search and reset afterwards — `strict_order` so the rows we LIMIT are
 * genuinely the nearest, not merely near. See pgvector-readme#… / the
 * filtered-ANN notes in the postgres-ops canon. NOTE: not measured on our corpus
 * yet (master-harness-vn4rz.32 is the probe that would turn this into evidence).
 *
 * BOUNDING + GUC LIFETIME (slice 2, GAP 4 + GAP 8). Both GUCs are now `SET
 * LOCAL` inside an explicit transaction, alongside a `statement_timeout` that
 * bounds the SQL leg against this path's p95 <= 1800 ms hook budget. `SET` +
 * `finally`-`RESET` on a POOLED client leaks the setting onto the next checkout
 * the first time anything skips the finally; a transaction-local setting unwinds
 * on commit AND on rollback, with no finally to forget. A cancelled statement
 * surfaces as PgVecSearchTimeoutError — never as a raw driver throw and never as
 * an empty result, which a caller cannot tell apart from "nothing matched".
 */

import type { SearchResult } from "../store.ts";
import { formatQueryForEmbedding, getDefaultLlamaCpp } from "../llm.ts";
import { toVectorLiteral, withClient } from "./client.ts";
import type { Vault } from "./vaults.ts";
import { PgVecReadModelMismatchError, PgVecSearchTimeoutError } from "./errors.ts";

/**
 * The narrowest thing this module needs from a `pg` client: a parameterized
 * query. Typed structurally (not as PoolClient) so the unit layer can drive the
 * REAL exported functions against a recording fake, without a live database —
 * a PoolClient is satisfied by this by construction.
 */
export interface PgQueryable {
  query<R extends Record<string, unknown> = Record<string, unknown>>(
    text: string,
    values?: unknown[],
  ): Promise<{ rows: R[] }>;
}

/**
 * The narrowest thing this module needs from an embedding backend.
 *
 * Structurally satisfied by `LlamaCpp` (src/llm.ts), so production passes
 * nothing and gets `getDefaultLlamaCpp()`. It is an OPTION rather than a hard
 * import because the integration tier has to drive the REAL pgSearchVec against
 * the live cluster: a test that cannot control the query vector cannot assert
 * distance ORDER, and one that depends on a remote GPU endpoint being up is a
 * test that reports the endpoint's health instead of this module's behaviour.
 */
export interface PgVecEmbedder {
  embed(
    text: string,
    options?: { isQuery?: boolean; signal?: AbortSignal },
  ): Promise<{ embedding: number[]; model: string } | null>;
}

/**
 * Default `statement_timeout` for BOTH SQL legs, in ms.
 *
 * Derived, not picked: this read path serves a p95 <= 1800 ms hook budget
 * (master-harness-2wx75), and that budget has to cover the embed round trip to
 * the remote endpoint plus the SQL leg plus the mapping. 1200 ms leaves ~600 ms
 * for everything that is not SQL while still being several hundred times the
 * measured ANN latency on the live vault — so it bounds a pathological scan
 * without being reachable by a healthy one. A caller with a different budget
 * passes `statementTimeoutMs`; a caller that genuinely wants no bound passes 0.
 */
export const DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS = 1200;

export interface PgSearchVecOptions {
  /** One collection name, a list of them, or nothing = every collection. */
  collections?: string | string[];
  /** Max documents returned (deduped by path). Default 20, same as sqlite. */
  limit?: number;
  /** Wall-clock budget in ms for the embed leg + the SQL leg. */
  timeoutMs?: number;
  /**
   * Server-side bound on EACH SQL leg, in ms. Default
   * DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS; 0 disables the bound entirely
   * (PostgreSQL's own meaning for statement_timeout = 0).
   */
  statementTimeoutMs?: number;
  /** Embedding backend. Defaults to getDefaultLlamaCpp(). */
  embedder?: PgVecEmbedder;
  /**
   * Fragments the ANN pass considers before the JOIN filters and the per-document
   * dedup thin them. Mirrors the sqlite path's `limit * 3` overfetch, widened
   * because the collection filter here is a POST-filter.
   */
  overfetch?: number;
}

/** What one ANN row looks like on the wire. Exported for the unit layer. */
export type PgVecRow = {
  hash: string;
  seq: number;
  pos: number;
  fragment_type: string | null;
  fragment_label: string | null;
  collection: string;
  path: string;
  title: string | null;
  modified_at: Date | string | null;
  body: string;
  distance: number | string;
};

/** Normalize the collection filter to a list, or null for "no filter". */
export function normalizeCollections(c: string | string[] | undefined): string[] | null {
  if (c === undefined) return null;
  const list = (Array.isArray(c) ? c : [c]).map(s => s.trim()).filter(s => s.length > 0);
  return list.length > 0 ? list : null;
}

/**
 * Build the ANN query. Pure — no client, no I/O — so the SQL contract
 * (the `<=>` order-by, the `active` fence, the collection filter, the limit)
 * is assertable in the unit tier rather than only observable in a live plan.
 *
 * `pg` uses $1-style placeholders and NOTHING is interpolated into the text:
 * the vector arrives as a bind parameter cast to ::vector, the collections as a
 * text[] fed to `= ANY(...)`.
 */
export function buildVecSearchQuery(
  vectorLiteral: string,
  collections: string[] | null,
  fragmentLimit: number,
): { text: string; values: unknown[] } {
  const values: unknown[] = [vectorLiteral];
  let filter = "";
  if (collections !== null) {
    values.push(collections);
    filter = `\n    AND d.collection = ANY($${values.length}::text[])`;
  }
  values.push(fragmentLimit);
  const limitParam = `$${values.length}`;
  const text = `
    SELECT
      cv.hash,
      cv.seq,
      cv.pos,
      cv.fragment_type,
      cv.fragment_label,
      d.collection,
      d.path,
      d.title,
      d.modified_at,
      content.doc AS body,
      cv.embedding <=> $1::vector AS distance
    FROM content_vectors cv
    JOIN documents d ON d.hash = cv.hash
    JOIN content ON content.hash = cv.hash
    WHERE d.active = true
    AND d.invalidated_at IS NULL${filter}
    ORDER BY cv.embedding <=> $1::vector
    LIMIT ${limitParam}
  `;
  return { text, values };
}

/**
 * Which models produced the stored vectors visible to this search.
 *
 * Scoped to the SAME rows the search will read (active documents, and the
 * requested collections when there are any): a mismatch that only exists in a
 * collection nobody asked about is not this query's problem, and refusing on it
 * would make an unrelated corner of the vault able to block every search.
 */
export async function getStoredVecModels(
  c: PgQueryable,
  collections: string[] | null,
): Promise<string[]> {
  const values: unknown[] = [];
  let filter = "";
  if (collections !== null) {
    values.push(collections);
    filter = ` AND d.collection = ANY($1::text[])`;
  }
  const { rows } = await c.query<{ model: string }>(
    `SELECT DISTINCT cv.model AS model
     FROM content_vectors cv
     JOIN documents d ON d.hash = cv.hash
     WHERE d.active = true AND d.invalidated_at IS NULL${filter}
     ORDER BY 1`,
    values,
  );
  return rows.map(r => r.model);
}

/**
 * THE FENCE. Throws unless the query model is the single model behind every
 * stored row in scope.
 *
 * No-ops on an EMPTY stored set only — a collection with no vectors yet cannot
 * disagree with anything, and the search will legitimately return nothing. A
 * heterogeneous stored set is refused outright: there is no "mostly comparable".
 */
export function assertQueryModelMatchesStored(
  storedModels: string[],
  queryModel: string,
  scope: string,
): void {
  if (storedModels.length === 0) return;
  if (storedModels.length === 1 && storedModels[0] === queryModel) return;
  throw new PgVecReadModelMismatchError(storedModels, queryModel, scope);
}

/** PostgreSQL's SQLSTATE for "cancelled by statement_timeout" (or pg_cancel_backend). */
const QUERY_CANCELED = "57014";

export function isQueryCanceled(e: unknown): boolean {
  return typeof e === "object" && e !== null && (e as { code?: unknown }).code === QUERY_CANCELED;
}

/**
 * Run `fn`'s statements inside ONE transaction whose GUCs are `SET LOCAL`
 * (master-harness-2wx75 slice 2, GAP 4 + GAP 8).
 *
 * WHY A TRANSACTION AND NOT `SET` + `finally`-`RESET`. The previous shape set
 * `hnsw.iterative_scan` on the session and reset it in a `finally`. On a POOLED
 * client that is a leak waiting for the first thing that skips the finally — a
 * process exit, a connection-level error that kills the client mid-flight, a
 * future early return added between the two — and the leaked GUC then applies
 * to whatever unrelated caller checks that connection out next. `SET LOCAL`
 * inside an explicit transaction cannot leak: the settings unwind on COMMIT and
 * on ROLLBACK alike, including the ROLLBACK the server performs for us when the
 * connection dies. There is no finally to forget because there is no reset.
 *
 * The `statement_timeout` rides the same mechanism, which is the point: the
 * bound and the recall knob have identical lifetimes.
 *
 * `timeoutMs` is INTERPOLATED, not bound — `SET LOCAL` takes no parameters — so
 * it is asserted to be a non-negative integer first. 0 is PostgreSQL's own
 * "no bound".
 */
export interface BoundedTxOptions {
  /**
   * Set `hnsw.iterative_scan = strict_order` for the transaction. TRUE for the
   * vector arm (it is the sanctioned answer to post-filtered ANN under-fill);
   * FALSE for the lexical arm, which has no ANN index and for which the GUC
   * would be pure noise on the connection.
   */
  hnswIterativeScan?: boolean;
  /**
   * How a SQLSTATE 57014 cancellation becomes a typed error. Defaults to
   * PgVecSearchTimeoutError; the FTS arm passes its own so a caller reading
   * `err.name` learns WHICH arm gave up.
   */
  onCanceled?: (timeoutMs: number, scope: string, stage: string, cause: unknown) => Error;
}

export async function withBoundedTx<T>(
  c: PgQueryable,
  timeoutMs: number,
  scope: string,
  stage: string,
  fn: () => Promise<T>,
  txOpts: BoundedTxOptions = {},
): Promise<T> {
  if (!Number.isInteger(timeoutMs) || timeoutMs < 0) {
    throw new Error(
      `statementTimeoutMs must be a non-negative integer (0 = no bound), got ${JSON.stringify(timeoutMs)}`,
    );
  }
  await c.query("BEGIN");
  try {
    await c.query(`SET LOCAL statement_timeout = ${timeoutMs}`);
    // Iterative index scan for the post-filtered case (pgvector >= 0.8). Older
    // pgvector has no such GUC; post-filter recall degrades, correctness does
    // not, so it is deliberately not fatal. The SAVEPOINT is what makes
    // "not fatal" true INSIDE a transaction — an unrecognized-parameter error
    // aborts the transaction, and every later statement would fail with
    // "current transaction is aborted" if we merely swallowed it.
    if (txOpts.hnswIterativeScan ?? true) {
      await c.query("SAVEPOINT clawmem_hnsw_guc");
      try {
        await c.query("SET LOCAL hnsw.iterative_scan = strict_order");
        await c.query("RELEASE SAVEPOINT clawmem_hnsw_guc");
      } catch {
        await c.query("ROLLBACK TO SAVEPOINT clawmem_hnsw_guc");
      }
    }
    const out = await fn();
    await c.query("COMMIT");
    return out;
  } catch (e) {
    // Best-effort: on a dead connection this throws too, and the server has
    // already rolled the transaction (and its SET LOCALs) back for us.
    await c.query("ROLLBACK").catch(() => {});
    if (isQueryCanceled(e)) {
      const make = txOpts.onCanceled
        ?? ((ms, sc, st, cause) => new PgVecSearchTimeoutError(ms, sc, st, cause));
      throw make(timeoutMs, scope, stage, e);
    }
    throw e;
  }
}

/** Cosine distance → the sqlite path's score. Identical formula on purpose. */
function scoreFromDistance(distance: number): number {
  return 1 - distance;
}

/**
 * Why a search returned nothing, when it returned nothing for a reason OTHER
 * than "we looked and nothing matched".
 *
 * Each member corresponds 1:1 to one early-exit in pgSearchVecDetailed. There is
 * deliberately NO member for "empty" — a genuine empty answer is
 * `degraded: false` with `results: []`, and keeping that OUT of this union is
 * the whole point (master-harness-2wx75 GAP 7).
 */
export type PgVecDegradedReason =
  | "no-stored-vectors"           // the model fence found nothing embedded in scope
  | "budget-exhausted-pre-embed"  // timeoutMs already spent before the embed leg
  | "embed-unavailable"           // the embed endpoint returned no embedding
  | "budget-exhausted-pre-sql";   // timeoutMs spent after the embed, before the ANN scan

/**
 * The typed result of a vector search — the degraded channel this path needs so
 * that "we could not look" stops being the same value as "nothing matched".
 *
 * A CALLER MUST HANDLE THREE DISTINCT OUTCOMES, and a caller that cannot tell
 * them apart is the bug this shape closes:
 *
 *  1. DEGRADED-EMPTY — `degraded: true`, `results: []`, `degradedReason` naming
 *     which of the four non-answers happened. Nothing was searched (or nothing
 *     could be). Presenting this to a user as "no matches" is a lie; it belongs
 *     in a "search unavailable / not indexed yet" surface, and
 *     "no-stored-vectors" in particular is an instruction to run the reindex.
 *  2. GENUINE-EMPTY — `degraded: false`, `results: []`. The fence passed, the
 *     embed leg produced a vector, the ANN scan ran, and zero rows matched. This
 *     is a real, trustworthy answer: "nothing matched."
 *  3. THROWN — the call rejects. A query-model mismatch throws
 *     PgVecReadModelMismatchError (a meaningless search is refused, never
 *     degraded), and a SQL leg that exceeded its SERVER-SIDE bound throws
 *     PgVecSearchTimeoutError. Those stay throws ON PURPOSE and are NOT folded
 *     into `degradedReason`: the fields below describe a call that completed,
 *     and a cancelled statement did not.
 *
 * KNOWN WRINKLE, carried forward from slice 2 rather than fixed here:
 * `isQueryCanceled()` keys on SQLSTATE 57014, which `pg_cancel_backend()` also
 * emits — so an OPERATOR-initiated cancel is reported as a timeout. Stated, not
 * addressed; distinguishing the two needs more than the SQLSTATE.
 */
export interface PgVecSearchDetailedResult {
  results: SearchResult[];
  degraded: boolean;
  degradedReason?: PgVecDegradedReason;
  /** Distinct embedding models the fence found in scope (0 ⇔ "no-stored-vectors"). */
  storedModels: number;
  /** ANN rows returned BEFORE the per-document dedup thinned them. 0 on every degraded path. */
  scannedFragments: number;
  /** The model the embed leg reported, when the embed leg ran and returned one. */
  embedModel?: string;
}

function degraded(
  reason: PgVecDegradedReason,
  storedModels: number,
  embedModel?: string,
): PgVecSearchDetailedResult {
  return {
    results: [],
    degraded: true,
    degradedReason: reason,
    storedModels,
    scannedFragments: 0,
    ...(embedModel === undefined ? {} : { embedModel }),
  };
}

/**
 * Vector search over the PG vault, WITH the degraded channel.
 *
 * The behaviour is identical to pgSearchVec (which is now a thin wrapper over
 * this); the difference is entirely in what the caller can LEARN about an empty
 * result. See PgVecSearchDetailedResult for the three outcomes.
 *
 * `results` is deduped to the best-scoring fragment per document and totally
 * ordered by (distance, filepath) so ties do not resolve by whatever order the
 * plan happened to emit.
 */
export async function pgSearchVecDetailed(
  c: PgQueryable,
  query: string,
  opts: PgSearchVecOptions = {},
): Promise<PgVecSearchDetailedResult> {
  const limit = opts.limit ?? 20;
  const collections = normalizeCollections(opts.collections);
  const scope = collections ? collections.join(", ") : "(all collections)";
  const fragmentLimit = opts.overfetch ?? Math.max(limit * 8, 64);
  const deadline = opts.timeoutMs === undefined ? undefined : Date.now() + opts.timeoutMs;
  const statementTimeoutMs = opts.statementTimeoutMs ?? DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS;

  // The fence FIRST: a cheap DISTINCT beats paying for an embed we are about to
  // refuse. It also means a mismatched endpoint reports the mismatch rather than
  // an embed timeout. It is bounded too — a DISTINCT is cheap work but it can
  // still wait an unbounded time on a lock.
  const storedModels = await withBoundedTx(c, statementTimeoutMs, scope, "model-fence", () =>
    getStoredVecModels(c, collections));
  if (storedModels.length === 0) return degraded("no-stored-vectors", 0);

  const llm = opts.embedder ?? getDefaultLlamaCpp();
  let signal: AbortSignal | undefined;
  if (deadline !== undefined) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) return degraded("budget-exhausted-pre-embed", storedModels.length);
    signal = AbortSignal.timeout(remaining);
  }
  // isQuery + formatQueryForEmbedding: BOTH, exactly as store.ts's getEmbedding
  // does. The flag selects the endpoint's query-side params; the formatting is
  // what puts the vector in the same space as the stored fragments.
  const embedded = await llm.embed(formatQueryForEmbedding(query), { isQuery: true, signal });
  if (!embedded?.embedding) return degraded("embed-unavailable", storedModels.length);

  // THE FENCE'S CALL SITE. Covered directly by
  // tests/unit/pg-search-vec.test.ts §"pgSearchVec wires the fence" and by
  // tests/integration/pg-search-vec.test.ts against a live cluster: delete this
  // line and BOTH go red. The pure function having its own unit cases is not
  // enough — a fence nobody calls is a fence with a gate next to it.
  assertQueryModelMatchesStored(storedModels, embedded.model, scope);

  if (deadline !== undefined && Date.now() >= deadline) {
    return degraded("budget-exhausted-pre-sql", storedModels.length, embedded.model);
  }

  const { text, values } = buildVecSearchQuery(
    toVectorLiteral(embedded.embedding),
    collections,
    fragmentLimit,
  );

  const rows = await withBoundedTx(c, statementTimeoutMs, scope, "ann-scan", async () => {
    const { rows } = await c.query<PgVecRow>(text, values);
    return rows;
  });

  // GENUINE-EMPTY lives here: zero rows is `degraded: false`. We searched.
  return {
    results: dedupeToSearchResults(rows, limit),
    degraded: false,
    storedModels: storedModels.length,
    scannedFragments: rows.length,
    embedModel: embedded.model,
  };
}

/**
 * Vector search over the PG vault. Returns `SearchResult[]` — the same shape
 * src/store.ts's searchVec() returns.
 *
 * BACK-COMPAT WRAPPER over pgSearchVecDetailed, byte-identical in observable
 * behaviour. Prefer pgSearchVecDetailed for anything that needs to tell
 * "nothing matched" from "we could not look"; this signature cannot express the
 * difference, which is exactly why the detailed one exists.
 */
export async function pgSearchVec(
  c: PgQueryable,
  query: string,
  opts: PgSearchVecOptions = {},
): Promise<SearchResult[]> {
  return (await pgSearchVecDetailed(c, query, opts)).results;
}

/**
 * Fragment rows → per-document SearchResults. Pure, and exported so the mapping
 * (dedup by filepath, best distance wins, total order) is testable without a DB.
 */
export function dedupeToSearchResults(rows: PgVecRow[], limit: number): SearchResult[] {
  const best = new Map<string, { row: PgVecRow; distance: number }>();
  for (const row of rows) {
    const filepath = `clawmem://${row.collection}/${row.path}`;
    const distance = typeof row.distance === "string" ? Number(row.distance) : row.distance;
    const existing = best.get(filepath);
    if (!existing || distance < existing.distance) best.set(filepath, { row, distance });
  }
  return [...best.entries()]
    .sort((a, b) => a[1].distance - b[1].distance || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, limit)
    .map(([filepath, { row, distance }]) => {
      const body = row.body ?? "";
      const modifiedAt = row.modified_at instanceof Date
        ? row.modified_at.toISOString()
        : (row.modified_at ?? "");
      return {
        filepath,
        displayPath: `${row.collection}/${row.path}`,
        title: row.title ?? row.path,
        // The PG schema has no folder-context table yet; sqlite's
        // getContextForFile() has no counterpart. NULL is the honest answer, not
        // an empty string pretending a lookup happened.
        context: null,
        hash: row.hash,
        docid: row.hash.slice(0, 6),
        collectionName: row.collection,
        modifiedAt,
        bodyLength: body.length,
        body,
        score: scoreFromDistance(distance),
        source: "vec" as const,
        chunkPos: row.pos,
        fragmentType: row.fragment_type ?? undefined,
        fragmentLabel: row.fragment_label ?? undefined,
      } satisfies SearchResult;
    });
}

/** Convenience: run a search on a client checked out of a vault's pool. */
export async function pgSearchVecInVault(
  vault: Vault,
  query: string,
  opts: PgSearchVecOptions = {},
): Promise<SearchResult[]> {
  return withClient(vault, c => pgSearchVec(c, query, opts));
}

/**
 * Convenience: run a DETAILED search on a client checked out of a vault's pool.
 * Mirrors pgSearchVecInVault; returns the degraded channel with it.
 */
export async function pgSearchVecDetailedInVault(
  vault: Vault,
  query: string,
  opts: PgSearchVecOptions = {},
): Promise<PgVecSearchDetailedResult> {
  return withClient(vault, c => pgSearchVecDetailed(c, query, opts));
}
