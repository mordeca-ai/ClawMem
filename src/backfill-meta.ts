/**
 * `clawmem backfill-meta` — populate documents.description / documents.keywords for rows
 * indexed before those columns existed, and queue them for re-embed (master-harness-65amk).
 *
 * The s1lli (description) and 65amk (keywords) columns are additive: rows indexed before
 * them stay NULL until their source body changes, because the indexer's content hash is of
 * the frontmatter-STRIPPED body — a frontmatter-only edit never moves it, so the embed
 * worklist never re-selects the document either. This verb closes that gap:
 *
 *   1. For each ACTIVE, filesystem-backed document in scope (origin 'fs' or legacy NULL;
 *      'api' rows have no source file by design), re-read the source file and re-parse its
 *      frontmatter through the same parseDocument the indexer uses.
 *   2. A field is planned for writing only when the source has a NON-EMPTY value that
 *      differs from the stored one (stored NULL differs only from a non-empty source). A
 *      source that lost its description/keywords does NOT clear the stored value — the
 *      indexer's own "undefined = leave untouched" rule, kept so this verb never deletes.
 *   3. Rows whose source body no longer hashes to the stored hash are skipped
 *      (`source_changed`): the next index pass owns them, and attaching new metadata to old
 *      content would invalidate vectors that are about to be replaced anyway.
 *   4. --apply writes the planned fields via updateDocumentMeta and deletes ALL of each
 *      affected hash's fragment rows, in one transaction fenced by the embedding lease
 *      (store.ts applyDocMetaBackfill), so the next `clawmem embed` re-embeds the whole
 *      document. The lease is the one cmdEmbed holds for its entire run, so the deletes can
 *      never race a running embed; if it is held, --apply refuses (exit 1) and writes
 *      nothing.
 *
 * Shared hashes (one content hash under several paths, each with its own frontmatter):
 * every row gets its own metadata, but the hash is invalidated only when its canonical
 * alias — MIN(collection || '/' || path) over its active rows, the row whose metadata
 * getHashesNeedingFragments hands the embedder — was written. A write to a non-canonical
 * alias alone cannot change the embedded fragments, so it is counted
 * (`noncanonical_only_hashes`) and left embedded.
 *
 * Default is a DRY RUN; the CLI opens the store read-only for it, so it cannot write.
 */
import { existsSync, readFileSync } from "fs";
import type { Database } from "bun:sqlite";
import type { NamedCollection } from "./collections.ts";
import { hashContent, parseDocument } from "./indexer.ts";
import {
  applyDocMetaBackfill,
  canonicalActiveDocIdForHash,
  EmbedLeaseLostError,
  type DocMetaBackfillUpdate,
  type Store,
} from "./store.ts";
import { EMBED_LEASE_NAME, EMBED_LEASE_TTL_MS } from "./taint-clear.ts";
import { acquireWorkerLease, releaseWorkerLease } from "./worker-lease.ts";

export type BackfillMetaOptions = {
  apply: boolean;
  /** Restrict to one collection (by name). */
  collection?: string;
  /** Examine at most N in-scope rows (ordered by collection, path). */
  limit?: number;
  /** Collection config (name → root path, frontmatter options); injected for testability. */
  collections: NamedCollection[];
};

/** Counter names are the key=value summary line's keys, in print order. */
export type BackfillMetaCounts = {
  scanned: number;
  missing_source: number;
  unconfigured_collection: number;
  read_error: number;
  parse_failed: number;
  source_changed: number;
  would_update_description: number;
  would_update_keywords: number;
  rows_to_update: number;
  hashes_to_invalidate: number;
  vectors_to_delete: number;
  noncanonical_only_hashes: number;
  skipped_api: number;
};

export type BackfillMetaApplied = {
  applied: number;
  raced: number;
  invalidated_hashes: number;
  deleted_vectors: number;
};

export type BackfillMetaOutcome = {
  mode: "dry-run" | "apply";
  exitCode: 0 | 1;
  counts: BackfillMetaCounts;
  applied?: BackfillMetaApplied;
  /** Set when --apply refused (lease held / lost); nothing was written. */
  refused?: string;
};

export type BackfillMetaPlan = {
  counts: BackfillMetaCounts;
  updates: DocMetaBackfillUpdate[];
};

function hasColumn(db: Database, table: string, column: string): boolean {
  return (db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[]).some(
    (c) => c.name === column,
  );
}

/**
 * Read-only planning pass: every query is a SELECT, so it is safe on a read-only handle.
 * Tolerates a pre-65amk database (no `keywords` column yet — every stored value reads as
 * NULL), which is what a dry run against an un-migrated vault sees.
 */
export function planBackfillMeta(
  db: Database,
  opts: Omit<BackfillMetaOptions, "apply">,
): BackfillMetaPlan {
  const counts: BackfillMetaCounts = {
    scanned: 0,
    missing_source: 0,
    unconfigured_collection: 0,
    read_error: 0,
    parse_failed: 0,
    source_changed: 0,
    would_update_description: 0,
    would_update_keywords: 0,
    rows_to_update: 0,
    hashes_to_invalidate: 0,
    vectors_to_delete: 0,
    noncanonical_only_hashes: 0,
    skipped_api: 0,
  };
  const hasOrigin = hasColumn(db, "documents", "origin");
  const keywordsSel = hasColumn(db, "documents", "keywords") ? "keywords" : "NULL AS keywords";
  const descriptionSel = hasColumn(db, "documents", "description")
    ? "description"
    : "NULL AS description";

  const where: string[] = ["active = 1"];
  const params: (string | number)[] = [];
  if (opts.collection) {
    where.push("collection = ?");
    params.push(opts.collection);
  }
  const fsOnly = hasOrigin ? " AND (origin IS NULL OR origin != 'api')" : "";
  if (hasOrigin) {
    counts.skipped_api = (
      db
        .prepare(
          `SELECT count(*) AS n FROM documents WHERE ${where.join(" AND ")} AND origin = 'api'`,
        )
        .get(...params) as { n: number }
    ).n;
  }
  const limitSql = opts.limit !== undefined ? ` LIMIT ${Math.floor(opts.limit)}` : "";
  const rows = db
    .prepare(
      `SELECT id, collection, path, hash, ${descriptionSel}, ${keywordsSel}
       FROM documents WHERE ${where.join(" AND ")}${fsOnly}
      ORDER BY collection, path${limitSql}`,
    )
    .all(...params) as {
    id: number;
    collection: string;
    path: string;
    hash: string;
    description: string | null;
    keywords: string | null;
  }[];

  const byName = new Map(opts.collections.map((c) => [c.name, c]));
  const updates: DocMetaBackfillUpdate[] = [];
  for (const row of rows) {
    counts.scanned++;
    const col = byName.get(row.collection);
    if (!col) {
      counts.unconfigured_collection++;
      continue;
    }
    const abs = `${col.path}/${row.path}`;
    if (!existsSync(abs)) {
      counts.missing_source++;
      continue;
    }
    let content: string;
    try {
      content = readFileSync(abs, "utf-8");
    } catch {
      counts.read_error++;
      continue;
    }

    const parsed = parseDocument(content, row.path, col.content_type, col.frontmatter_map);
    if (parsed.frontmatter === "failed") {
      counts.parse_failed++;
      continue;
    }
    if (hashContent(parsed.body) !== row.hash) {
      counts.source_changed++;
      continue;
    }

    const srcDescription = parsed.meta.description;
    const srcKeywords = parsed.meta.keywords;
    const writeDescription = srcDescription !== undefined && srcDescription !== row.description;
    const writeKeywords = srcKeywords !== undefined && srcKeywords !== row.keywords;
    if (!writeDescription && !writeKeywords) continue;
    if (writeDescription) counts.would_update_description++;
    if (writeKeywords) counts.would_update_keywords++;
    updates.push({
      id: row.id,
      hash: row.hash,
      ...(writeDescription ? { description: srcDescription } : {}),
      ...(writeKeywords ? { keywords: srcKeywords } : {}),
    });
  }
  counts.rows_to_update = updates.length;

  // Same canonical-alias rule applyDocMetaBackfill enforces at write time.
  const updatedIds = new Set(updates.map((u) => u.id));
  const hashes = [...new Set(updates.map((u) => u.hash))].sort();
  const seqCount = db.prepare(`SELECT count(*) AS n FROM content_vectors WHERE hash = ?`);
  for (const hash of hashes) {
    const canonicalId = canonicalActiveDocIdForHash(db, hash);
    if (canonicalId !== null && updatedIds.has(canonicalId)) {
      counts.hashes_to_invalidate++;
      counts.vectors_to_delete += (seqCount.get(hash) as { n: number }).n;
    } else {
      counts.noncanonical_only_hashes++;
    }
  }
  return { counts, updates };
}

/**
 * Plan, then (with apply) write under the embedding lease. Never throws for a lease
 * conflict: a held or lost lease is a refusal with exitCode 1 and nothing written.
 */
export function runBackfillMeta(store: Store, opts: BackfillMetaOptions): BackfillMetaOutcome {
  const plan = planBackfillMeta(store.db, opts);
  if (!opts.apply) return { mode: "dry-run", exitCode: 0, counts: plan.counts };

  if (plan.updates.length === 0) {
    return {
      mode: "apply",
      exitCode: 0,
      counts: plan.counts,
      applied: { applied: 0, raced: 0, invalidated_hashes: 0, deleted_vectors: 0 },
    };
  }
  const lease = acquireWorkerLease(store, EMBED_LEASE_NAME, EMBED_LEASE_TTL_MS);
  if (!lease.acquired || !lease.token) {
    return {
      mode: "apply",
      exitCode: 1,
      counts: plan.counts,
      refused:
        "an embed is running (embedding lease held), so nothing was written. Re-run when it finishes.",
    };
  }
  try {
    const r = applyDocMetaBackfill(store.db, plan.updates, {
      workerName: EMBED_LEASE_NAME,
      token: lease.token,
    });
    return {
      mode: "apply",
      exitCode: 0,
      counts: plan.counts,
      applied: {
        applied: r.applied,
        raced: r.raced,
        invalidated_hashes: r.invalidatedHashes.length,
        deleted_vectors: r.deletedVectors,
      },
    };
  } catch (err) {
    if (err instanceof EmbedLeaseLostError) {
      return {
        mode: "apply",
        exitCode: 1,
        counts: plan.counts,
        refused:
          "the embedding lease was lost before the write, so nothing was written. Re-run it.",
      };
    }
    throw err;
  } finally {
    releaseWorkerLease(store, EMBED_LEASE_NAME, lease.token);
  }
}

/** The structured last line: `backfill-meta: mode=<m> key=value ...` (stable key order). */
export function formatBackfillMetaSummary(outcome: BackfillMetaOutcome): string {
  const pairs: string[] = [`mode=${outcome.mode}`];
  for (const [k, v] of Object.entries(outcome.counts)) pairs.push(`${k}=${v}`);
  if (outcome.applied)
    for (const [k, v] of Object.entries(outcome.applied)) pairs.push(`${k}=${v}`);
  if (outcome.refused) pairs.push("refused=1");
  pairs.push(`exit=${outcome.exitCode}`);
  return `backfill-meta: ${pairs.join(" ")}`;
}
