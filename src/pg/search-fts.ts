/**
 * PG LEXICAL (FTS) READ PATH — the second arm of the PG read path
 * (master-harness-2wx75 slice 4, ADR-0162 §2).
 *
 * Slice 2/3 built the VECTOR arm (src/pg/search.ts). This is its lexical
 * counterpart: it returns the SAME `SearchResult` shape src/store.ts's
 * searchFTS() returns, tagged `source: "fts"`, so a caller consuming lexical
 * hits does not care which backend served them.
 *
 * WHY IT LIVES IN ITS OWN FILE. src/pg/search.ts is already ~560 lines of
 * vector-specific doctrine (the ANN predicate, the model-identity fence, the
 * embed leg). The two arms share PLUMBING, not concerns — so the plumbing is
 * IMPORTED from there (`PgQueryable`, `normalizeCollections`, `withBoundedTx`,
 * `DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS`) and nothing is copied. If a third
 * arm arrives, the shared helpers move to one module; two arms do not justify
 * the churn yet.
 *
 * FOUR THINGS ARE LOAD-BEARING HERE:
 *
 *  1. THE MATCH PREDICATE IS `d.fts @@ q`. `documents.fts` is a materialized,
 *     trigger-maintained, setweight()-ed tsvector (migrations/001 §116), and
 *     `documents_fts_idx` is a GIN index on that exact column. `@@` against the
 *     bare column is the only shape that index can serve; computing
 *     to_tsvector() at query time, or wrapping the column, throws the index away
 *     AND changes the answer (the query-time vector would carry no weights).
 *
 *  2. THE RANK WEIGHTS ARE DERIVED, NOT PICKED. `ts_rank_cd('{0.1,0.2,0.4,1.0}',
 *     ...)` is PostgreSQL's DEFAULT {D,C,B,A} weight array, which gives
 *     A:D = 1.0:0.1 = 10:1 — the SAME title:body ratio as the sqlite arm's
 *     `bm25(documents_fts, 10.0, 1.0)`. That is cross-backend rank parity by
 *     construction rather than a magic array, and it is what
 *     tests/integration/pg-search-fts.test.ts's WEIGHT PROOF measures. Cover
 *     density (`ts_rank_cd`) rather than plain `ts_rank`: it accounts for term
 *     proximity, which is what a lexical arm over prose bodies wants.
 *
 *  3. THE QUERY CONSTRUCTOR mirrors sqlite's buildFTS5Query(): Unicode-aware
 *     token boundaries, curated stopword relaxation, uppercase OR groups, and
 *     one server-stemmed prefix tsquery node per token. This avoids phrase
 *     nodes for hyphenated input and restores the prefix semantics of FTS5's
 *     `"term"*`; all user text remains in bind values.
 *
 *  4. THE COLLECTION FILTER IS IN SQL. `d.collection = ANY($n::text[])` sits in
 *     the same WHERE as the match, so `limit` is satisfied with eligible rows by
 *     construction. A post-filter over a fixed overfetch can be STARVED by
 *     higher-ranked ineligible documents — the same reasoning the sqlite arm's
 *     own comment gives.
 *
 * WHY THIS ARM IS THE FOUNDATION OF THE yoshiee-DOWN LEG. Its degraded-reason
 * union has exactly TWO members, against the vector arm's four, because it has
 * NO embed leg and NO model-identity fence: there is no remote endpoint to be
 * unavailable and no vector space to disagree about. That asymmetry is the
 * point — an FTS query needs no embedding endpoint, so when yoshiee is down
 * this is the arm that still answers. (The fallback WIRING itself is a later
 * slice; this slice only makes the arm exist.)
 *
 * NOT IN THIS SLICE, stated so its absence is not read as an oversight: the
 * `documents_title_trgm_idx` typo/trigram path, and RRF fusion of this arm with
 * the vector arm. Both are still owed by the bead.
 */

import type { SearchResult } from "../store.ts";
import { withClient } from "./client.ts";
import type { Vault } from "./vaults.ts";
import { PgFtsSearchTimeoutError } from "./errors.ts";
import { buildPgFtsQuery, type PgFtsQueryFragment } from "./fts-query.ts";
import {
  DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS,
  normalizeCollections,
  withBoundedTx,
  type PgQueryable,
} from "./search.ts";

export { DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS };

/**
 * PostgreSQL's DEFAULT ts_rank weight array, {D, C, B, A}.
 *
 * A = 1.0 (title), D = 0.1 (body) => a 10:1 title:body ratio, identical to the
 * sqlite arm's `bm25(documents_fts, 10.0, 1.0)`. Interpolated as a LITERAL and
 * not a bind parameter because it is a compile-time constant of this module —
 * no caller input ever reaches it.
 */
const RANK_WEIGHTS = "{0.1, 0.2, 0.4, 1.0}";

export interface PgSearchFtsOptions {
  /** One collection name, a list of them, or nothing = every collection. */
  collections?: string | string[];
  /** Max documents returned. Default 20 — same as the vec arm and sqlite. */
  limit?: number;
  /** Wall-clock budget in ms for the whole call. */
  timeoutMs?: number;
  /**
   * Server-side bound on the SQL leg, in ms. Default
   * DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS; 0 disables the bound entirely
   * (PostgreSQL's own meaning for statement_timeout = 0).
   */
  statementTimeoutMs?: number;
}

/** What one ranked FTS row looks like on the wire. Exported for the unit layer. */
export type PgFtsRow = {
  hash: string;
  collection: string;
  path: string;
  title: string | null;
  modified_at: Date | string | null;
  body: string;
  rank: number | string;
};

/**
 * Why an FTS search returned nothing, when it returned nothing for a reason
 * OTHER than "we looked and nothing matched".
 *
 * TWO members, and there is deliberately no third: this arm has no embed leg
 * and no model fence, so `embed-unavailable` / `no-stored-vectors` have no
 * analogue here. As with the vector arm (GAP 7) there is NO member for
 * "empty" — a genuine empty answer is `degraded: false` with `results: []`,
 * and keeping that OUT of this union is the whole point.
 *
 * This union is INTENTIONALLY NOT `PgVecDegradedReason` widened. Sharing one
 * union would tell every caller that an FTS search might report
 * `embed-unavailable`, which is unrepresentable here — and it would erase the
 * very asymmetry that makes this the yoshiee-DOWN arm.
 */
export type PgFtsDegradedReason =
  | "empty-tsquery"       // the query normalized to zero lexemes (all stopwords/punctuation)
  | "budget-exhausted";   // the wall-clock budget was spent before the SQL leg ran

/**
 * The typed result of a lexical search — the same degraded channel slice 3 gave
 * the vector arm, so that "we could not look" stops being the same value as
 * "nothing matched".
 *
 * THREE OUTCOMES a caller must handle:
 *
 *  1. DEGRADED-EMPTY — `degraded: true`, `results: []`, `degradedReason` naming
 *     which non-answer happened. `empty-tsquery` in particular is an
 *     instruction to the UI ("your query was all stopwords"), NOT "no matches".
 *  2. GENUINE-EMPTY — `degraded: false`, `results: []`. The tsquery had lexemes,
 *     the scan ran, zero documents matched. A trustworthy "nothing matched".
 *  3. THROWN — a SQL leg cancelled by its server-side bound throws
 *     PgFtsSearchTimeoutError. That stays a throw ON PURPOSE and is NOT folded
 *     into `degradedReason`: these fields describe a call that COMPLETED, and a
 *     cancelled statement did not.
 */
export interface PgFtsSearchDetailedResult {
  results: SearchResult[];
  degraded: boolean;
  degradedReason?: PgFtsDegradedReason;
  /** Ranked rows the SQL leg returned, before the limit/mapping thinned them. */
  scannedRows: number;
}

function degraded(reason: PgFtsDegradedReason): PgFtsSearchDetailedResult {
  return { results: [], degraded: true, degradedReason: reason, scannedRows: 0 };
}

/**
 * Build the ranked FTS query. Pure — no client, no I/O — so the SQL contract
 * (the `@@` match against the bare column, the weighted `ts_rank_cd`, the
 * active/invalidated fence, the in-SQL collection filter, the total order) is
 * assertable in the unit tier rather than only observable in a live plan.
 *
 * The parameterized prefix-node expression comes from buildPgFtsQuery and is
 * reused byte-for-byte for ranking and matching. No user text is interpolated.
 *
 * TOTAL ORDER: `ORDER BY rank DESC, d.collection ASC, d.path ASC`. Rank alone
 * leaves ties to whatever order the plan emitted — and with a weighted tsvector
 * over short documents ties are COMMON, not exotic. (collection, path) is the
 * UNIQUE key of `documents`, so this is a genuinely total order, and it sorts
 * identically to the `displayPath` the mapping produces. The vector arm made
 * exactly this choice; matching it means a hybrid caller sees one tie-break rule.
 */
export function buildFtsSearchQuery(
  query: string,
  collections: string[] | null,
  limit: number,
): { text: string; values: unknown[]; tsquery: PgFtsQueryFragment } | null {
  const tsquery = buildPgFtsQuery(query);
  if (tsquery === null) return null;
  const values: unknown[] = [...tsquery.values];
  let filter = "";
  if (collections !== null) {
    values.push(collections);
    filter = `\n      AND d.collection = ANY($${values.length}::text[])`;
  }
  values.push(limit);
  const limitParam = `$${values.length}`;
  const text = `
    SELECT
      d.hash,
      d.collection,
      d.path,
      d.title,
      d.modified_at,
      content.doc AS body,
      ts_rank_cd('${RANK_WEIGHTS}', d.fts, ${tsquery.text}) AS rank
    FROM documents d
    JOIN content ON content.hash = d.hash
    WHERE d.fts @@ ${tsquery.text}
      AND d.active = true
      AND d.invalidated_at IS NULL${filter}
    ORDER BY rank DESC, d.collection ASC, d.path ASC
    LIMIT ${limitParam}
  `;
  return { text, values, tsquery };
}

/**
 * Is the query text going to produce a tsquery with zero lexemes?
 *
 * DECIDED IN SQL, on purpose, and as a cheap PRE-CHECK inside the same bounded
 * transaction as the scan:
 *
 *  - IN SQL because PostgreSQL's english dictionary has its own stopwords in
 *     addition to the curated sqlite-parity relaxation in fts-query.ts.
 *  - AS A PRE-CHECK rather than a column on the main query, because the main
 *     query returns ZERO ROWS in exactly the case we need to report — there is
 *     no row on which to carry the flag. Folding it in would need a LATERAL or
 *     a UNION whose only job is to survive the empty case.
 *  - `numnode(...) = 0` rather than `= ''::tsquery`, because casting the empty
 *     string to tsquery emits a NOTICE on every call ("text-search query
 *     doesn't contain lexemes"), which would spam the server log for a check
 *     that runs on every search.
 *
 * It touches no table, so its cost is a parse plus a dictionary lookup.
 */
async function isEmptyTsquery(c: PgQueryable, query: PgFtsQueryFragment): Promise<boolean> {
  const { rows } = await c.query<{ n: number | string }>(
    `SELECT numnode(${query.text}) AS n`,
    query.values,
  );
  const n = rows[0]?.n;
  return (typeof n === "string" ? Number(n) : (n ?? 0)) === 0;
}

/**
 * Lexical search over the PG vault, WITH the degraded channel.
 *
 * See PgFtsSearchDetailedResult for the three outcomes. Both SQL statements
 * (the emptiness pre-check and the ranked scan) run inside ONE bounded
 * transaction: `SET LOCAL statement_timeout` unwinds on COMMIT and on ROLLBACK
 * alike, so a pooled connection cannot inherit the bound — the same reasoning
 * slice 2 wrote up for the vector arm, reusing the SAME helper rather than
 * opening a second convention.
 */
export async function pgSearchFtsDetailed(
  c: PgQueryable,
  query: string,
  opts: PgSearchFtsOptions = {},
): Promise<PgFtsSearchDetailedResult> {
  const limit = opts.limit ?? 20;
  const collections = normalizeCollections(opts.collections);
  const scope = collections ? collections.join(", ") : "(all collections)";
  const statementTimeoutMs = opts.statementTimeoutMs ?? DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS;

  // The wall-clock budget is checked BEFORE the transaction opens: opening one
  // just to discover we have no time left is a round trip spent on nothing.
  if (opts.timeoutMs !== undefined && opts.timeoutMs <= 0) return degraded("budget-exhausted");
  const deadline = opts.timeoutMs === undefined ? undefined : Date.now() + opts.timeoutMs;

  const built = buildFtsSearchQuery(query, collections, limit);
  if (built === null) return degraded("empty-tsquery");
  const { text, values, tsquery } = built;

  const outcome = await withBoundedTx(
    c,
    statementTimeoutMs,
    scope,
    "fts-scan",
    async (): Promise<PgFtsSearchDetailedResult> => {
      if (await isEmptyTsquery(c, tsquery)) return degraded("empty-tsquery");
      if (deadline !== undefined && Date.now() >= deadline) return degraded("budget-exhausted");
      const { rows } = await c.query<PgFtsRow>(text, values);
      // GENUINE-EMPTY lives here: zero rows is `degraded: false`. We looked.
      return { results: toSearchResults(rows, limit), degraded: false, scannedRows: rows.length };
    },
    {
      // No ANN index on this path, so the pgvector recall knob would be noise.
      hnswIterativeScan: false,
      // A cancelled FTS statement must say FTS, not "Vector search".
      onCanceled: (ms, sc, st, cause) => new PgFtsSearchTimeoutError(ms, sc, st, cause),
    },
  );
  return outcome;
}

/**
 * Lexical search over the PG vault. Returns `SearchResult[]` — the same shape
 * src/store.ts's searchFTS() returns.
 *
 * BACK-COMPAT WRAPPER over pgSearchFtsDetailed, byte-identical in observable
 * behaviour. Prefer the detailed one for anything that needs to tell "nothing
 * matched" from "your query had no searchable words".
 */
export async function pgSearchFts(
  c: PgQueryable,
  query: string,
  opts: PgSearchFtsOptions = {},
): Promise<SearchResult[]> {
  return (await pgSearchFtsDetailed(c, query, opts)).results;
}

/**
 * Ranked rows → SearchResults. Pure, and exported so the mapping is testable
 * without a DB.
 *
 * One row per document already (the match is document-level, unlike the vector
 * arm's per-fragment rows), so there is no dedup step — but the `limit` is
 * applied here as well as in SQL so the mapping is honest on its own.
 */
export function toSearchResults(rows: PgFtsRow[], limit: number): SearchResult[] {
  return rows.slice(0, limit).map(row => {
    const body = row.body ?? "";
    const modifiedAt = row.modified_at instanceof Date
      ? row.modified_at.toISOString()
      : (row.modified_at ?? "");
    return {
      filepath: `clawmem://${row.collection}/${row.path}`,
      displayPath: `${row.collection}/${row.path}`,
      title: row.title ?? row.path,
      // The PG schema has no folder-context table yet; sqlite's
      // getContextForFile() has no counterpart. NULL is the honest answer, not
      // an empty string pretending a lookup happened. (Same call as the vec arm.)
      context: null,
      hash: row.hash,
      docid: row.hash.slice(0, 6),
      collectionName: row.collection,
      modifiedAt,
      bodyLength: body.length,
      body,
      // ts_rank_cd's raw value: HIGHER is better, unbounded above. Deliberately
      // NOT normalized to the sqlite arm's ftsScoreFromBm25() range — the two
      // are different scoring functions and a fake normalization would imply a
      // comparability we have not measured. What IS shared is the rank ORDER
      // (10:1 title:body); cross-backend SCORE calibration is a fusion concern
      // (RRF, a later slice) and RRF consumes ranks, not scores.
      score: typeof row.rank === "string" ? Number(row.rank) : row.rank,
      source: "fts" as const,
    } satisfies SearchResult;
  });
}

/** Convenience: run a lexical search on a client checked out of a vault's pool. */
export async function pgSearchFtsInVault(
  vault: Vault,
  query: string,
  opts: PgSearchFtsOptions = {},
): Promise<SearchResult[]> {
  return withClient(vault, c => pgSearchFts(c, query, opts));
}

/** Convenience: the DETAILED lexical search on a vault-pooled client. */
export async function pgSearchFtsDetailedInVault(
  vault: Vault,
  query: string,
  opts: PgSearchFtsOptions = {},
): Promise<PgFtsSearchDetailedResult> {
  return withClient(vault, c => pgSearchFtsDetailed(c, query, opts));
}
