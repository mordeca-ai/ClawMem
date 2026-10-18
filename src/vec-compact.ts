/**
 * `clawmem vec-compact` — rewrite the vec0 table in collection-clustered order
 * (master-harness-vn4rz.110).
 *
 * vec0 stores rows in chunks in INSERT order. After months of incremental re-embeds a small
 * collection's vectors are scattered over most of the table's chunks, so a collection-scoped
 * search (vn4rz.100's chunk-targeted scan, or a vec0 MATCH) reads far more chunk blobs than the
 * collection owns. This module rebuilds `vectors_vec` with the rows inserted in
 * (owning collection, hash, seq) order so each collection spans ~ceil(n / chunk_size) chunks.
 *
 * Ordering key per hash_seq: the MIN collection name among ACTIVE documents whose content hash
 * owns the vector (documents.hash = content_vectors.hash, hash_seq = hash || '_' || seq), then
 * hash, then seq numerically. Vectors without an active owner go last. A vector shared by several
 * collections is placed with the lexicographically first one; every collection's span is still
 * reported (a shared vector counts for each owner).
 *
 * The rebuild is ONE `BEGIN IMMEDIATE` transaction and is streamed SQL-side: the pairs are copied
 * into a TEMP table (disk-backed, SQLite spills it to its temp dir), the vec0 table is dropped
 * and re-created with the SAME DDL (same dimension, cosine metric, chunk_size), and refilled with
 * `INSERT ... SELECT ... ORDER BY plan position`. Before COMMIT every (hash_seq, embedding) pair
 * is compared bit-for-bit against the copy. Any error rolls the whole thing back; vec0's shadow
 * tables are ordinary tables, so the original table returns intact.
 */
import type { Database } from "bun:sqlite";

/** vec0's default chunk_size when the DDL does not name one. */
export const VEC0_DEFAULT_CHUNK_SIZE = 1024;

/** A refusal to compact (missing / non-canonical table). Nothing was written. */
export class VecCompactRefusal extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VecCompactRefusal";
  }
}

/**
 * The canonical clawmem table shape, optionally with an explicit `chunk_size` (vec-scope-scan's
 * `canonicalVecTableDim` is stricter and declines a custom chunk_size; compaction must preserve
 * one, so it accepts exactly that one extra option and nothing else).
 */
const CANONICAL_WITH_CHUNK =
  /^CREATE VIRTUAL TABLE vectors_vec USING vec0\(hash_seq TEXT PRIMARY KEY, embedding float\[(\d+)\] distance_metric=cosine(?:, chunk_size=(\d+))?\)$/i;

export interface VecTableShape {
  dim: number;
  chunkSize: number;
  ddl: string;
}

/** The table's shape, or throws VecCompactRefusal when it is missing or non-canonical. */
export function readCanonicalVecShape(db: Database): VecTableShape {
  const row = db
    .prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'`)
    .get() as { sql: string } | null;
  if (!row) throw new VecCompactRefusal("vectors_vec is missing; nothing to compact");
  const m = row.sql.trim().match(CANONICAL_WITH_CHUNK);
  if (!m)
    throw new VecCompactRefusal(
      `vectors_vec has a non-canonical schema; refusing to rewrite it: ${row.sql}`,
    );
  const dim = parseInt(m[1]!, 10);
  const chunkSize = m[2] ? parseInt(m[2], 10) : VEC0_DEFAULT_CHUNK_SIZE;
  if (!(dim > 0) || !(chunkSize > 0))
    throw new VecCompactRefusal(`vectors_vec has an invalid dimension/chunk_size: ${row.sql}`);
  for (const t of ["vectors_vec_rowids", "vectors_vec_chunks"]) {
    const has = db.prepare(`SELECT 1 FROM sqlite_master WHERE type='table' AND name=?`).get(t);
    if (!has) throw new VecCompactRefusal(`vec0 shadow table ${t} is missing; refusing to compact`);
  }
  return { dim, chunkSize, ddl: row.sql.trim() };
}

/**
 * Rows in the target order: one row per (vector, active owning collection) pair, vectors in
 * placement order. `rid` is the vec0 rowid of the vector's row, `coll` an owning collection (or
 * NULL), `chunk` the chunk it lives in now, and `placed` the owner it is placed with (the MIN).
 * The same ORDER BY feeds the rebuild, so the projection cannot drift from what is built.
 */
const ORDERED_VECTORS_SQL = `
  WITH owner AS (
    SELECT hash, MIN(collection) AS placed FROM documents WHERE active = 1 GROUP BY hash
  ), ids AS (
    SELECT r.id AS hash_seq, r.chunk_id AS chunk,
           CASE WHEN instr(r.id, '_') > 0 THEN substr(r.id, 1, instr(r.id, '_') - 1) ELSE r.id END AS h,
           CAST(CASE WHEN instr(r.id, '_') > 0 THEN substr(r.id, instr(r.id, '_') + 1) ELSE '0' END AS INTEGER) AS q
    FROM vectors_vec_rowids r
  )
  SELECT ids.hash_seq AS hash_seq, ids.chunk AS chunk, owner.placed AS placed, ids.h AS h, ids.q AS q
  FROM ids LEFT JOIN owner ON owner.hash = ids.h
  ORDER BY (owner.placed IS NULL), owner.placed, ids.h, ids.q, ids.hash_seq`;

export interface CollectionSpan {
  name: string;
  vectors: number;
  currentSpan: number;
  projectedSpan: number;
  /** Measured after the rebuild; absent on a dry run. */
  afterSpan?: number;
}

export interface VecCompactReport {
  dryRun: boolean;
  dim: number;
  chunkSize: number;
  totalVectors: number;
  /** Vectors with no active owning document (placed last). */
  unownedVectors: number;
  totalChunksBefore: number;
  /** Chunks after compaction: measured on a real run, ceil(n / chunk_size) projected on a dry run. */
  totalChunksAfter: number;
  collectionCount: number;
  collections: CollectionSpan[];
  elapsedMs: number;
}

export type VecCompactStage =
  "planned" | "copied" | "dropped" | "recreated" | "inserted" | "verified";

export interface VecCompactOptions {
  dryRun?: boolean;
  /** Collections to always include in the report (in addition to the top-N by vector count). */
  collections?: string[];
  /** How many of the largest collections to report. Default 20. */
  top?: number;
  /** Test seam: called after each stage inside the transaction; a throw aborts and rolls back. */
  onStage?: (stage: VecCompactStage) => void;
}

interface Analysis {
  totalVectors: number;
  unowned: number;
  totalChunks: number;
  perCollection: Map<string, { vectors: number; current: Set<number>; projected: Set<number> }>;
}

/**
 * One streaming pass over the ordered ids (no vectors are read): per collection, the set of
 * chunks its vectors occupy now and the set they would occupy at their placement position.
 */
function analyze(db: Database, chunkSize: number): Analysis {
  const totalChunks = (
    db.prepare(`SELECT count(*) AS c FROM vectors_vec_chunks`).get() as { c: number }
  ).c;
  const owners = db.prepare(`SELECT DISTINCT hash, collection FROM documents WHERE active = 1`);
  const ownersByHash = new Map<string, string[]>();
  for (const o of owners.iterate() as Iterable<{ hash: string; collection: string }>) {
    const list = ownersByHash.get(o.hash);
    if (list) list.push(o.collection);
    else ownersByHash.set(o.hash, [o.collection]);
  }
  const perCollection: Analysis["perCollection"] = new Map();
  let pos = 0;
  let unowned = 0;
  for (const row of db.prepare(ORDERED_VECTORS_SQL).iterate() as Iterable<{
    chunk: number;
    h: string;
    placed: string | null;
  }>) {
    const projectedChunk = Math.floor(pos / chunkSize);
    pos++;
    const cols = ownersByHash.get(row.h);
    if (!cols) {
      unowned++;
      continue;
    }
    for (const name of cols) {
      let e = perCollection.get(name);
      if (!e) {
        e = { vectors: 0, current: new Set(), projected: new Set() };
        perCollection.set(name, e);
      }
      e.vectors++;
      e.current.add(row.chunk);
      e.projected.add(projectedChunk);
    }
  }
  return { totalVectors: pos, unowned, totalChunks, perCollection };
}

function selectCollections(an: Analysis, opts: VecCompactOptions): CollectionSpan[] {
  const top = opts.top ?? 20;
  const names = [...an.perCollection.entries()]
    .sort((a, b) => b[1].vectors - a[1].vectors || (a[0] < b[0] ? -1 : 1))
    .slice(0, top)
    .map(([n]) => n);
  for (const n of opts.collections ?? []) if (!names.includes(n)) names.push(n);
  return names.map((name) => {
    const e = an.perCollection.get(name);
    return {
      name,
      vectors: e?.vectors ?? 0,
      currentSpan: e?.current.size ?? 0,
      projectedSpan: e?.projected.size ?? 0,
    };
  });
}

function ensureBusyTimeout(db: Database, minMs: number): void {
  const row = db.prepare(`PRAGMA busy_timeout`).get() as {
    timeout?: number;
    busy_timeout?: number;
  } | null;
  const cur = row?.timeout ?? row?.busy_timeout ?? 0;
  if (cur < minMs) db.exec(`PRAGMA busy_timeout = ${minMs}`);
}

/**
 * Compact (or, with `dryRun`, only project) the vec0 table. Throws VecCompactRefusal when the
 * table is missing or non-canonical; any other error leaves the original table intact.
 */
export function compactVectors(db: Database, opts: VecCompactOptions = {}): VecCompactReport {
  const t0 = performance.now();
  const dryRun = !!opts.dryRun;

  if (dryRun) {
    const shape = readCanonicalVecShape(db);
    const an = analyze(db, shape.chunkSize);
    return {
      dryRun: true,
      dim: shape.dim,
      chunkSize: shape.chunkSize,
      totalVectors: an.totalVectors,
      unownedVectors: an.unowned,
      totalChunksBefore: an.totalChunks,
      totalChunksAfter: Math.ceil(an.totalVectors / shape.chunkSize),
      collectionCount: an.perCollection.size,
      collections: selectCollections(an, opts),
      elapsedMs: Math.round(performance.now() - t0),
    };
  }

  ensureBusyTimeout(db, 15000);
  const stage = (s: VecCompactStage) => opts.onStage?.(s);
  let report!: VecCompactReport;

  const run = db.transaction(() => {
    // Shape is read INSIDE the write lock so it cannot change between check and rewrite.
    const shape = readCanonicalVecShape(db);
    const before = analyze(db, shape.chunkSize);
    db.exec(`DROP TABLE IF EXISTS temp.vc_plan; DROP TABLE IF EXISTS temp.vc_copy`);
    db.exec(
      `CREATE TEMP TABLE vc_plan(pos INTEGER PRIMARY KEY AUTOINCREMENT, hash_seq TEXT NOT NULL UNIQUE)`,
    );
    db.exec(`INSERT INTO temp.vc_plan(hash_seq) SELECT hash_seq FROM (${ORDERED_VECTORS_SQL})`);
    stage("planned");
    db.exec(`CREATE TEMP TABLE vc_copy(hash_seq TEXT PRIMARY KEY, embedding BLOB NOT NULL)`);
    db.exec(
      `INSERT INTO temp.vc_copy(hash_seq, embedding) SELECT hash_seq, embedding FROM main.vectors_vec`,
    );
    const copied = (db.prepare(`SELECT count(*) AS c FROM temp.vc_copy`).get() as { c: number }).c;
    if (copied !== before.totalVectors) {
      throw new Error(
        `vec-compact: copied ${copied} vectors but the table has ${before.totalVectors}`,
      );
    }
    stage("copied");
    db.exec(`DROP TABLE main.vectors_vec`);
    stage("dropped");
    db.exec(shape.ddl);
    stage("recreated");
    db.exec(
      `INSERT INTO main.vectors_vec(hash_seq, embedding)
       SELECT c.hash_seq, c.embedding FROM temp.vc_plan p JOIN temp.vc_copy c ON c.hash_seq = p.hash_seq
       ORDER BY p.pos`,
    );
    stage("inserted");
    const newCount = (
      db.prepare(`SELECT count(*) AS c FROM main.vectors_vec_rowids`).get() as { c: number }
    ).c;
    const matching = (
      db
        .prepare(
          `SELECT count(*) AS c FROM main.vectors_vec v JOIN temp.vc_copy c ON c.hash_seq = v.hash_seq
           WHERE v.embedding = c.embedding`,
        )
        .get() as { c: number }
    ).c;
    if (newCount !== copied || matching !== copied) {
      throw new Error(
        `vec-compact: verification failed (rows ${newCount}, bit-identical ${matching}, expected ${copied})`,
      );
    }
    stage("verified");
    const after = analyze(db, shape.chunkSize);
    db.exec(`DROP TABLE temp.vc_plan; DROP TABLE temp.vc_copy`);
    const collections = selectCollections(before, opts).map((c) => ({
      ...c,
      afterSpan: after.perCollection.get(c.name)?.current.size ?? 0,
    }));
    report = {
      dryRun: false,
      dim: shape.dim,
      chunkSize: shape.chunkSize,
      totalVectors: before.totalVectors,
      unownedVectors: before.unowned,
      totalChunksBefore: before.totalChunks,
      totalChunksAfter: after.totalChunks,
      collectionCount: before.perCollection.size,
      collections,
      elapsedMs: 0,
    };
  });
  run.immediate();
  report.elapsedMs = Math.round(performance.now() - t0);
  return report;
}

/** Human-readable before/after table for the CLI. */
export function formatVecCompactReport(r: VecCompactReport): string {
  const lines: string[] = [];
  lines.push(
    `${r.dryRun ? "DRY RUN: " : ""}vectors_vec: ${r.totalVectors} vectors (${r.unownedVectors} without an active document), ` +
      `dim ${r.dim}, chunk_size ${r.chunkSize}, chunks ${r.totalChunksBefore} -> ${r.totalChunksAfter}${r.dryRun ? " (projected)" : ""}`,
  );
  const afterHeader = r.dryRun ? "projected" : "after";
  const nameW = Math.max(10, ...r.collections.map((c) => c.name.length));
  lines.push(
    `${"collection".padEnd(nameW)}  ${"vectors".padStart(9)}  ${"current".padStart(8)}  ${afterHeader.padStart(9)}`,
  );
  for (const c of r.collections) {
    const after = r.dryRun ? c.projectedSpan : (c.afterSpan ?? c.projectedSpan);
    lines.push(
      `${c.name.padEnd(nameW)}  ${String(c.vectors).padStart(9)}  ${String(c.currentSpan).padStart(8)}  ${String(after).padStart(9)}`,
    );
  }
  if (r.collectionCount > r.collections.length)
    lines.push(`(${r.collectionCount - r.collections.length} more collection(s) not shown)`);
  lines.push(`elapsed ${r.elapsedMs} ms`);
  return lines.join("\n");
}
