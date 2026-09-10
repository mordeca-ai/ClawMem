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
 * FILTERING + RECALL. The collection/active predicates sit in the SAME query as
 * the ANN order-by, which makes this a POST-FILTER against the HNSW index: the
 * index walks in distance order and the filter discards. On a narrow collection
 * filter that can under-fill. pgvector 0.8's iterative index scan is the
 * sanctioned answer (`hnsw.iterative_scan`), so it is turned on for the duration
 * of the search and reset afterwards — `strict_order` so the rows we LIMIT are
 * genuinely the nearest, not merely near. See pgvector-readme#… / the
 * filtered-ANN notes in the postgres-ops canon. NOTE: not measured on our corpus
 * yet (master-harness-vn4rz.32 is the probe that would turn this into evidence).
 */

import type { SearchResult } from "../store.ts";
import { formatQueryForEmbedding, getDefaultLlamaCpp } from "../llm.ts";
import { toVectorLiteral, withClient } from "./client.ts";
import type { Vault } from "./vaults.ts";
import { PgVecReadModelMismatchError } from "./errors.ts";

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

export interface PgSearchVecOptions {
  /** One collection name, a list of them, or nothing = every collection. */
  collections?: string | string[];
  /** Max documents returned (deduped by path). Default 20, same as sqlite. */
  limit?: number;
  /** Wall-clock budget in ms for the embed leg + the SQL leg. */
  timeoutMs?: number;
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

/** Cosine distance → the sqlite path's score. Identical formula on purpose. */
function scoreFromDistance(distance: number): number {
  return 1 - distance;
}

/**
 * Vector search over the PG vault. Returns `SearchResult[]` — the same shape
 * src/store.ts's searchVec() returns, deduped to the best-scoring fragment per
 * document and totally ordered by (distance, filepath) so ties do not resolve
 * by whatever order the plan happened to emit.
 */
export async function pgSearchVec(
  c: PgQueryable,
  query: string,
  opts: PgSearchVecOptions = {},
): Promise<SearchResult[]> {
  const limit = opts.limit ?? 20;
  const collections = normalizeCollections(opts.collections);
  const scope = collections ? collections.join(", ") : "(all collections)";
  const fragmentLimit = opts.overfetch ?? Math.max(limit * 8, 64);
  const deadline = opts.timeoutMs === undefined ? undefined : Date.now() + opts.timeoutMs;

  // The fence FIRST: a cheap DISTINCT beats paying for an embed we are about to
  // refuse. It also means a mismatched endpoint reports the mismatch rather than
  // an embed timeout.
  const storedModels = await getStoredVecModels(c, collections);
  if (storedModels.length === 0) return [];

  const llm = getDefaultLlamaCpp();
  let signal: AbortSignal | undefined;
  if (deadline !== undefined) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) return [];
    signal = AbortSignal.timeout(remaining);
  }
  // isQuery + formatQueryForEmbedding: BOTH, exactly as store.ts's getEmbedding
  // does. The flag selects the endpoint's query-side params; the formatting is
  // what puts the vector in the same space as the stored fragments.
  const embedded = await llm.embed(formatQueryForEmbedding(query), { isQuery: true, signal });
  if (!embedded?.embedding) return [];

  assertQueryModelMatchesStored(storedModels, embedded.model, scope);

  if (deadline !== undefined && Date.now() >= deadline) return [];

  const { text, values } = buildVecSearchQuery(
    toVectorLiteral(embedded.embedding),
    collections,
    fragmentLimit,
  );

  // Iterative index scan for the post-filtered case (pgvector >= 0.8). Session
  // GUCs leak across a POOLED checkout, so the reset is in a finally.
  let rows: PgVecRow[];
  await c.query("SET hnsw.iterative_scan = strict_order").catch(() => {
    // Older pgvector has no such GUC. Post-filter recall degrades; correctness
    // does not. Deliberately not fatal.
  });
  try {
    ({ rows } = await c.query<PgVecRow>(text, values));
  } finally {
    await c.query("RESET hnsw.iterative_scan").catch(() => {});
  }

  return dedupeToSearchResults(rows, limit);
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
