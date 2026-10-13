/**
 * Chunk-targeted exact vector scan for include-scoped queries (master-harness-vn4rz.100).
 *
 * sqlite-vec's vec0 KNN reads every vector chunk blob before it applies a `hash_seq IN (...)`
 * filter (sqlite-vec.c vec0Filter_knn_chunks_iter: sqlite3_blob_read, then the rowid bitmap).
 * On the live vault that is ~1.4 GB per leg (350k x 768 float32) whatever the collection, so a
 * `-c docs` leg (15.7k fragments in 172 of 459 chunks) paid for the whole vault.
 *
 * This module reads only the chunks that hold in-scope fragments, straight from the vec0 shadow
 * tables, and computes the distances in JS with the same float32 operation order as sqlite-vec's
 * scalar `distance_cosine_float`. The result is bit-identical (verified on the live vault:
 * 0 mismatches over 4096 compared distances); the unit test pins it against vec0 MATCH.
 *
 * Every chunk is verified (blob sizes, validity bit, rowid at the slot) before a vector is used.
 * Any mismatch returns null, and the caller falls back to vec0 MATCH, so a layout change in a
 * future sqlite-vec degrades to the old speed, never to a wrong answer.
 */
import type { Database } from "bun:sqlite";

/** One in-scope fragment and where vec0 stores its vector. */
export interface VecScopeFragment {
  hashSeq: string;
  rowid: number;
  chunkId: number;
  offset: number;
}

/** In-scope vectors, row-aligned: row i is `hashSeqs[i]`, magnitude `mags[i]`. */
export interface ScopedVectors {
  dim: number;
  hashSeqs: string[];
  vectors: Float32Array;
  /** float32 sum of squares, accumulated in sqlite-vec order (aMag in distance_cosine_float). */
  mags: Float32Array;
  chunksRead: number;
  bytesRead: number;
}

/** Process-wide counters (observability for CLAWMEM_QUERY_TIMING and tests). */
export interface VecScopeScanStats {
  /** Scope loads that produced vectors (in-thread or parallel). */
  loads: number;
  /** Of those, loads spread over worker threads. */
  parallelLoads: number;
  /** Scopes the plan declined (non-canonical table, or too large a share of the chunks). */
  declined: number;
  /** Loads abandoned on a chunk verification failure (the caller then used vec0 MATCH). */
  verifyFailures: number;
  /** Distance scans served from loaded vectors. */
  scans: number;
  chunksRead: number;
  bytesRead: number;
}

const stats: VecScopeScanStats = {
  loads: 0,
  parallelLoads: 0,
  declined: 0,
  verifyFailures: 0,
  scans: 0,
  chunksRead: 0,
  bytesRead: 0,
};

export function vecScopeScanStats(): VecScopeScanStats {
  return { ...stats };
}

export function resetVecScopeScanStats(): void {
  for (const k of Object.keys(stats) as (keyof VecScopeScanStats)[]) stats[k] = 0;
}

function countLoad(sv: ScopedVectors, parallel: boolean): void {
  stats.loads++;
  if (parallel) stats.parallelLoads++;
  stats.chunksRead += sv.chunksRead;
  stats.bytesRead += sv.bytesRead;
}

/** Above this share of the table's chunks, one vec0 MATCH is as cheap as reading the chunks here. */
export const VEC_SCOPED_SCAN_MAX_CHUNK_FRACTION = 0.5;

/** Only the canonical clawmem table shape is served; anything else (int8/bit, L2, partition or
 * metadata columns, a custom chunk_size) stays on vec0 MATCH. */
const CANONICAL_VEC_TABLE =
  /^CREATE VIRTUAL TABLE vectors_vec USING vec0\(hash_seq TEXT PRIMARY KEY, embedding float\[(\d+)\] distance_metric=cosine\)$/i;

/** CLAWMEM_QUERY_VEC_SCOPED_SCAN: on unless set to 0/false/off/no. */
export function scopedScanEnabled(env: Record<string, string | undefined> = process.env): boolean {
  const raw = env.CLAWMEM_QUERY_VEC_SCOPED_SCAN?.trim().toLowerCase();
  if (!raw) return true;
  return !["0", "false", "off", "no"].includes(raw);
}

/** The vector dimension when vectors_vec has the canonical shape this scan understands, else null. */
export function canonicalVecTableDim(db: Database): number | null {
  const row = db
    .prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'`)
    .get() as { sql: string } | null;
  if (!row) return null;
  const m = row.sql.trim().match(CANONICAL_VEC_TABLE);
  return m ? parseInt(m[1]!, 10) : null;
}

const fr = Math.fround;

/** float32 sum of squares in index order — sqlite-vec's `aMag`/`bMag` accumulation. */
export function f32SumSquares(v: Float32Array, start: number, dim: number): number {
  let acc = 0;
  for (let i = 0; i < dim; i++) {
    const x = v[start + i]!;
    acc = fr(acc + fr(x * x));
  }
  return acc;
}

/**
 * Cosine distance with sqlite-vec's float32 semantics:
 * `dot`, `aMag`, `bMag` are float32 accumulators updated in index order, then
 * `1 - dot / (sqrt(aMag) * sqrt(bMag))` in double, returned as float32.
 * A float32 product or sum computed in double and rounded once is the correctly rounded
 * float32 result (53 >= 2*24 + 2), so this matches the C code bit for bit.
 */
export function cosineDistanceF32(
  base: Float32Array,
  baseStart: number,
  baseMag: number,
  query: Float32Array,
  queryMag: number,
  dim: number,
): number {
  let dot = 0;
  for (let i = 0; i < dim; i++) {
    dot = fr(dot + fr(base[baseStart + i]! * query[i]!));
  }
  return fr(1 - dot / (Math.sqrt(baseMag) * Math.sqrt(queryMag)));
}

/** View `bytes` as a typed array, copying only when the blob is not suitably aligned. */
function aligned<T>(bytes: Uint8Array, align: number, view: (b: Uint8Array) => T): T {
  return view(bytes.byteOffset % align === 0 ? bytes : bytes.slice());
}

function bitSet(bitmap: Uint8Array, i: number): boolean {
  return ((bitmap[i >> 3]! >> (i & 7)) & 1) === 1;
}

/** Which chunks a scope needs, or null when the scoped scan should not serve it. */
export interface ScopedScanPlan {
  dim: number;
  byChunk: Map<number, number[]>;
  totalChunks: number;
}

/**
 * Decide whether `frags` can be served by the chunk-targeted scan, without reading any vector.
 * Null when the table shape is not canonical or the scope spans more than `maxChunkFraction`
 * of the table's chunks (then a vec0 MATCH reads about as much and needs no JS copy).
 */
export function planScopedScan(
  db: Database,
  frags: VecScopeFragment[],
  opts: { maxChunkFraction?: number } = {},
): ScopedScanPlan | null {
  const dim = canonicalVecTableDim(db);
  if (dim === null || dim <= 0) {
    stats.declined++;
    return null;
  }
  const byChunk = new Map<number, number[]>();
  frags.forEach((f, i) => {
    const list = byChunk.get(f.chunkId);
    if (list) list.push(i);
    else byChunk.set(f.chunkId, [i]);
  });
  const totalChunks = (
    db.prepare(`SELECT count(*) AS c FROM vectors_vec_chunks`).get() as { c: number }
  ).c;
  const maxFraction = opts.maxChunkFraction ?? VEC_SCOPED_SCAN_MAX_CHUNK_FRACTION;
  if (frags.length > 0 && (totalChunks <= 0 || byChunk.size / totalChunks > maxFraction)) {
    stats.declined++;
    return null;
  }
  return { dim, byChunk, totalChunks };
}

/** One chunk's share of a load: copy slot `offset` (holding `rowid`) into output row `row`. */
export interface ChunkReadTask {
  chunkId: number;
  slots: { row: number; offset: number; rowid: number }[];
}

/**
 * Copy the tasks' vectors into `vectors` (row-major, `dim` wide) and their float32 magnitudes
 * into `mags`. Every slot is verified first: blob sizes consistent with one chunk_size, the
 * validity bit set, and the chunk's rowid at that slot equal to the enumerated rowid (vec0 rowids
 * are AUTOINCREMENT and clawmem never UPDATEs a vector in place, so a matching rowid means the
 * same vector). Returns the bytes read, or -1 on any mismatch.
 */
export function readChunkTasks(
  db: Database,
  dim: number,
  tasks: ChunkReadTask[],
  vectors: Float32Array,
  mags: Float32Array,
): number {
  const chunkMeta = db.prepare(
    `SELECT validity, rowids FROM vectors_vec_chunks WHERE chunk_id = ?`,
  );
  const chunkVectors = db.prepare(
    `SELECT vectors FROM vectors_vec_vector_chunks00 WHERE rowid = ?`,
  );
  let bytesRead = 0;
  for (const task of tasks) {
    const meta = chunkMeta.get(task.chunkId) as { validity: Uint8Array; rowids: Uint8Array } | null;
    const vrow = chunkVectors.get(task.chunkId) as { vectors: Uint8Array } | null;
    if (!meta || !vrow) return -1;
    const chunkSize = meta.validity.byteLength * 8;
    if (meta.rowids.byteLength !== chunkSize * 8) return -1;
    if (vrow.vectors.byteLength !== chunkSize * dim * 4) return -1;
    bytesRead += vrow.vectors.byteLength;
    const rowids = aligned(
      meta.rowids,
      8,
      (b) => new BigInt64Array(b.buffer, b.byteOffset, b.byteLength / 8),
    );
    const chunk = aligned(
      vrow.vectors,
      4,
      (b) => new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4),
    );
    for (const slot of task.slots) {
      if (slot.offset < 0 || slot.offset >= chunkSize) return -1;
      if (!bitSet(meta.validity, slot.offset)) return -1;
      if (rowids[slot.offset] !== BigInt(slot.rowid)) return -1;
      const start = slot.offset * dim;
      vectors.set(chunk.subarray(start, start + dim), slot.row * dim);
      mags[slot.row] = f32SumSquares(vectors, slot.row * dim, dim);
    }
  }
  return bytesRead;
}

function tasksFor(
  frags: VecScopeFragment[],
  chunkIds: number[],
  byChunk: Map<number, number[]>,
  rowOf: (i: number) => number,
): ChunkReadTask[] {
  return chunkIds.map((chunkId) => ({
    chunkId,
    slots: byChunk
      .get(chunkId)!
      .map((i) => ({ row: rowOf(i), offset: frags[i]!.offset, rowid: frags[i]!.rowid })),
  }));
}

/**
 * Read the vectors of `frags` from the vec0 shadow tables, touching only their chunks.
 * Returns null (caller uses vec0 MATCH) when planScopedScan declines the scope or any chunk
 * fails verification. Call inside one read transaction so the enumeration and the chunk
 * reads see one snapshot.
 */
export function loadScopedVectors(
  db: Database,
  frags: VecScopeFragment[],
  opts: { maxChunkFraction?: number } = {},
): ScopedVectors | null {
  const plan = planScopedScan(db, frags, opts);
  if (!plan) return null;
  const { dim, byChunk } = plan;
  const vectors = new Float32Array(frags.length * dim);
  const mags = new Float32Array(frags.length);
  const bytesRead = readChunkTasks(
    db,
    dim,
    tasksFor(frags, [...byChunk.keys()], byChunk, (i) => i),
    vectors,
    mags,
  );
  if (bytesRead < 0) {
    stats.verifyFailures++;
    return null;
  }
  const sv: ScopedVectors = {
    dim,
    hashSeqs: frags.map((f) => f.hashSeq),
    vectors,
    mags,
    chunksRead: byChunk.size,
    bytesRead,
  };
  countLoad(sv, false);
  return sv;
}

/** Request/response of src/vec-scope-read-worker.ts. */
export interface ChunkReadRequest {
  dbPath: string;
  mmapBytes: number;
  dim: number;
  rows: number;
  tasks: ChunkReadTask[];
}
export type ChunkReadResponse =
  | { ok: true; vectors: Float32Array; mags: Float32Array; bytesRead: number }
  | { ok: false; error: string };

/**
 * loadScopedVectors split over `workers` threads, each with its own read-only connection.
 * One thread copies ~0.5 GB of chunk blobs in ~0.6 s on the live vault; the copy is the cost,
 * so it is spread. Each worker reads its chunks in one read transaction and verifies every slot
 * against the enumerated rowid, which is what makes reading from separate snapshots safe.
 * Resolves null on any worker failure (the caller then loads in-thread or uses vec0 MATCH).
 */
export async function loadScopedVectorsParallel(
  dbPath: string,
  frags: VecScopeFragment[],
  plan: ScopedScanPlan,
  workers: number,
  mmapBytes: number,
): Promise<ScopedVectors | null> {
  const { dim, byChunk } = plan;
  const chunkIds = [...byChunk.keys()].sort((a, b) => a - b);
  const n = Math.max(1, Math.min(Math.floor(workers), chunkIds.length));
  const groups: number[][] = Array.from({ length: n }, () => []);
  chunkIds.forEach((id, k) => groups[k % n]!.push(id));
  const vectors = new Float32Array(frags.length * dim);
  const mags = new Float32Array(frags.length);
  const spawned: Worker[] = [];
  try {
    const parts = await Promise.all(
      groups.map((group) => {
        const globalRows: number[] = [];
        const local = new Map<number, number>();
        for (const id of group)
          for (const i of byChunk.get(id)!) {
            local.set(i, globalRows.length);
            globalRows.push(i);
          }
        const req: ChunkReadRequest = {
          dbPath,
          mmapBytes,
          dim,
          rows: globalRows.length,
          tasks: tasksFor(frags, group, byChunk, (i) => local.get(i)!),
        };
        return new Promise<{ res: ChunkReadResponse; globalRows: number[] }>((resolve) => {
          let w: Worker;
          try {
            w = new Worker(new URL("./vec-scope-read-worker.ts", import.meta.url).href);
          } catch (e) {
            resolve({ res: { ok: false, error: String(e) }, globalRows });
            return;
          }
          spawned.push(w);
          w.onmessage = (ev: MessageEvent) =>
            resolve({ res: ev.data as ChunkReadResponse, globalRows });
          w.onerror = (ev: ErrorEvent) =>
            resolve({ res: { ok: false, error: ev.message || "worker error" }, globalRows });
          w.postMessage(req);
        });
      }),
    );
    let bytesRead = 0;
    for (const { res, globalRows } of parts) {
      if (!res.ok) {
        stats.verifyFailures++;
        return null;
      }
      bytesRead += res.bytesRead;
      globalRows.forEach((g, r) => {
        vectors.set(res.vectors.subarray(r * dim, (r + 1) * dim), g * dim);
        mags[g] = res.mags[r]!;
      });
    }
    const sv: ScopedVectors = {
      dim,
      hashSeqs: frags.map((f) => f.hashSeq),
      vectors,
      mags,
      chunksRead: chunkIds.length,
      bytesRead,
    };
    countLoad(sv, true);
    return sv;
  } finally {
    for (const w of spawned) {
      try {
        w.terminate();
      } catch {
        /* already gone */
      }
    }
  }
}

/**
 * Exact distances from `query` to every scoped vector (optionally only the `allowed` hash_seqs),
 * nearest first; equal distances order by hash_seq so the output does not depend on read order.
 */
export function scanScopedVectors(
  sv: ScopedVectors,
  query: Float32Array,
  allowed?: Set<string>,
): { hash_seq: string; distance: number }[] {
  if (query.length !== sv.dim) return [];
  stats.scans++;
  const qMag = f32SumSquares(query, 0, sv.dim);
  const out: { hash_seq: string; distance: number }[] = [];
  for (let i = 0; i < sv.hashSeqs.length; i++) {
    const hs = sv.hashSeqs[i]!;
    if (allowed && !allowed.has(hs)) continue;
    out.push({
      hash_seq: hs,
      distance: cosineDistanceF32(sv.vectors, i * sv.dim, sv.mags[i]!, query, qMag, sv.dim),
    });
  }
  // NaN (a zero vector) sorts last instead of poisoning the comparator.
  out.sort((a, b) => {
    const an = Number.isNaN(a.distance),
      bn = Number.isNaN(b.distance);
    if (an || bn)
      return an === bn
        ? a.hash_seq < b.hash_seq
          ? -1
          : a.hash_seq > b.hash_seq
            ? 1
            : 0
        : an
          ? 1
          : -1;
    return (
      a.distance - b.distance || (a.hash_seq < b.hash_seq ? -1 : a.hash_seq > b.hash_seq ? 1 : 0)
    );
  });
  return out;
}
