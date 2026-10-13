import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * master-harness-vn4rz.100 — chunk-targeted exact scan for include-scoped vector legs.
 *
 * sqlite-vec reads every vector chunk before it applies a `hash_seq IN (...)` filter, so a
 * `-c docs` leg paid a scan of the whole vault. The scoped scan reads only the chunks holding
 * the scope's fragments and computes the distances in JS with sqlite-vec's float32 order.
 *
 * Pinned here:
 *  - distances are bit-identical to vec0 MATCH (768-d random vectors, cosine);
 *  - only the scope's chunks are read (a collection in 1 of 3 chunks reads 1 chunk);
 *  - searchVecDetailed* returns the same docs and scores as the vec0 path;
 *  - a slot whose rowid does not match is refused (fallback to vec0, still correct);
 *  - the read-only connection's cache never serves stale data after another connection writes;
 *  - the parallel loader assembles exactly what the in-thread loader reads.
 */

import {
  createStore,
  searchVecDetailed,
  searchVecDetailedWithVector,
  prepareScopedVecScan,
  prewarmScopedVecScan,
  enableVecScopeCache,
  type Store,
} from "../../src/store.ts";
import {
  loadScopedVectors,
  loadScopedVectorsParallel,
  planScopedScan,
  scanScopedVectors,
  scopedScanEnabled,
  vecScopeScanStats,
  resetVecScopeScanStats,
  type VecScopeFragment,
} from "../../src/vec-scope-scan.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";
import { hashContent } from "../../src/indexer.ts";

const MODEL = "scope-scan-test-model";

// Deterministic PRNG (mulberry32) so a failure reproduces.
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomVec(r: () => number, dim: number): Float32Array {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = r() * 2 - 1;
  return v;
}

function fragmentsOf(db: Store["db"], collection: string): VecScopeFragment[] {
  const rows = db
    .prepare(
      `
    SELECT cv.hash || '_' || cv.seq AS hashSeq, r.rowid AS rowid, r.chunk_id AS chunkId, r.chunk_offset AS offset
    FROM documents d JOIN content_vectors cv ON cv.hash = d.hash
    JOIN vectors_vec_rowids r ON r.id = cv.hash || '_' || cv.seq
    WHERE d.active = 1 AND d.collection = ?
  `,
    )
    .all(collection) as VecScopeFragment[];
  return [...new Map(rows.map((r) => [r.hashSeq, r])).values()];
}

/** Seed `docs` documents of `frags` fragments each into `col`, in one transaction. */
function seedCollection(
  store: Store,
  col: string,
  docs: number,
  frags: number,
  dim: number,
  r: () => number,
): void {
  store.ensureVecTable(dim);
  const now = "2026-01-15T00:00:00.000Z";
  store.db.transaction(() => {
    for (let i = 0; i < docs; i++) {
      const path = `${col}-${String(i).padStart(4, "0")}.md`;
      const body = `body of ${col}/${path}`;
      const hash = hashContent(body);
      store.insertContent(hash, body, now);
      store.insertDocument(col, path, path, hash, now, now);
      store.insertEmbeddingsBatch(
        Array.from({ length: frags }, (_, seq) => ({
          hash,
          seq,
          pos: seq,
          embedding: randomVec(r, dim),
          model: MODEL,
          embeddedAt: now,
          fragmentType: "section",
        })),
      );
    }
  })();
}

let dir: string;
let dbPath: string;
let writer: Store;
let reader: Store;
const DIM = 16;

beforeAll(() => {
  setDefaultLlamaCpp({
    embed: async () => ({ embedding: Array.from(QUERY), model: MODEL }),
    query: async () => null,
    generate: async () => null,
    expandQuery: async () => [],
  } as any);
  dir = mkdtempSync(join(tmpdir(), "clawmem-scope-scan-"));
  dbPath = join(dir, "index.sqlite");
  writer = createStore(dbPath);
  const r = rng(100);
  // 2,048 "big" fragments fill vec0 chunks 1-2 (chunk_size 1024); the 300 "small" fragments all
  // land in chunk 3, so the small scope spans 1 of 3 chunks.
  seedCollection(writer, "big", 512, 4, DIM, r);
  seedCollection(writer, "small", 100, 3, DIM, r);
  reader = createStore(dbPath, { readonly: true });
  expect(enableVecScopeCache(reader.db)).toBe(true);
});

afterAll(() => {
  setDefaultLlamaCpp(null);
  reader?.close();
  writer?.close();
  if (dir) rmSync(dir, { recursive: true, force: true });
});

const QUERY = randomVec(rng(7), DIM);
const qv = () => ({ embedding: QUERY, endpointModel: MODEL });

describe("scoped scan distances (vn4rz.100)", () => {
  it("are bit-identical to vec0 MATCH for 768-d cosine vectors", () => {
    const s = createStore(":memory:");
    try {
      const r = rng(42);
      seedCollection(s, "c", 400, 3, 768, r);
      const frags = fragmentsOf(s.db, "c");
      const sv = loadScopedVectors(s.db, frags, { maxChunkFraction: 1 });
      expect(sv).not.toBeNull();
      const q = randomVec(rng(9), 768);
      const ours = scanScopedVectors(sv!, q);
      const vec0 = s.db
        .prepare(`SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ?`)
        .all(q, frags.length) as { hash_seq: string; distance: number }[];
      expect(ours.length).toBe(1200);
      const vecMap = new Map(vec0.map((x) => [x.hash_seq, x.distance]));
      const mismatches = ours.filter((x) => vecMap.get(x.hash_seq) !== x.distance);
      expect(mismatches).toEqual([]);
      // nearest-first, same order as vec0 (ties broken by hash_seq)
      expect(ours.map((x) => x.distance)).toEqual(vec0.map((x) => x.distance));
    } finally {
      s.close();
    }
  });
});

describe("chunk targeting (vn4rz.100)", () => {
  it("reads only the chunks that hold the scope", () => {
    resetVecScopeScanStats();
    const totalChunks = (
      writer.db.prepare(`SELECT count(*) AS c FROM vectors_vec_chunks`).get() as { c: number }
    ).c;
    expect(totalChunks).toBe(3);
    const sv = loadScopedVectors(reader.db, fragmentsOf(reader.db, "small"));
    expect(sv).not.toBeNull();
    expect(sv!.hashSeqs.length).toBe(300);
    expect(sv!.chunksRead).toBe(1);
    expect(sv!.bytesRead).toBe(1024 * DIM * 4);
    expect(vecScopeScanStats().loads).toBe(1);
  });

  it("declines a scope spanning more than half the chunks (vec0 serves it)", () => {
    resetVecScopeScanStats();
    expect(planScopedScan(reader.db, fragmentsOf(reader.db, "big"))).toBeNull();
    expect(vecScopeScanStats().declined).toBe(1);
  });

  it("refuses a slot whose rowid does not match the enumeration", () => {
    resetVecScopeScanStats();
    const frags = fragmentsOf(reader.db, "small");
    frags[5] = { ...frags[5]!, rowid: frags[5]!.rowid + 1_000_000 };
    expect(loadScopedVectors(reader.db, frags)).toBeNull();
    expect(vecScopeScanStats().verifyFailures).toBe(1);
  });

  it("the parallel loader assembles exactly what the in-thread loader reads", async () => {
    const frags = fragmentsOf(reader.db, "small");
    const one = loadScopedVectors(reader.db, frags)!;
    const plan = planScopedScan(reader.db, frags)!;
    const par = await loadScopedVectorsParallel(dbPath, frags, plan, 3, 0);
    expect(par).not.toBeNull();
    expect(Array.from(par!.vectors)).toEqual(Array.from(one.vectors));
    expect(Array.from(par!.mags)).toEqual(Array.from(one.mags));
    expect(par!.hashSeqs).toEqual(one.hashSeqs);
  });
});

describe("scoped vector search through the store (vn4rz.100)", () => {
  function paths(res: {
    results: { displayPath: string; score: number; body?: string }[];
  }): string[] {
    return res.results.map((r) => `${r.displayPath}@${r.score}`);
  }

  it("returns the same docs and scores as the vec0 path, and serves them from the scan", () => {
    resetVecScopeScanStats();
    const scanned = searchVecDetailedWithVector(reader.db, qv(), 20, { collections: ["small"] });
    expect(vecScopeScanStats().scans).toBe(1);
    process.env.CLAWMEM_QUERY_VEC_SCOPED_SCAN = "0";
    try {
      expect(scopedScanEnabled()).toBe(false);
      const plain = searchVecDetailedWithVector(reader.db, qv(), 20, { collections: ["small"] });
      expect(vecScopeScanStats().scans).toBe(1); // the knob routed this one to vec0
      expect(paths(scanned)).toEqual(paths(plain));
      expect(scanned.results.length).toBe(20);
      expect(scanned.degraded).toBe(false);
    } finally {
      delete process.env.CLAWMEM_QUERY_VEC_SCOPED_SCAN;
    }
  });

  it("hydrates each result with its own document body", () => {
    const res = searchVecDetailedWithVector(reader.db, qv(), 10, { collections: ["small"] });
    for (const r of res.results) {
      expect(r.body).toBe(`body of ${r.displayPath}`);
      expect(r.bodyLength).toBe(r.body!.length);
    }
  });

  it("matches vec0 with an exclusion inside the scope", () => {
    const opts = {
      collections: ["small", "big"],
      excludeCollections: ["big"],
      prefilterMaxScope: 50_000,
      scopedScanMaxChunkFraction: 1,
    };
    resetVecScopeScanStats();
    const scanned = searchVecDetailedWithVector(reader.db, qv(), 15, opts);
    expect(vecScopeScanStats().scans).toBe(1);
    process.env.CLAWMEM_QUERY_VEC_SCOPED_SCAN = "0";
    try {
      const plain = searchVecDetailedWithVector(reader.db, qv(), 15, opts);
      expect(paths(scanned)).toEqual(paths(plain));
      expect(scanned.excludedDocsSeen).toBe(plain.excludedDocsSeen);
      expect(scanned.results.every((r) => r.collectionName === "small")).toBe(true);
    } finally {
      delete process.env.CLAWMEM_QUERY_VEC_SCOPED_SCAN;
    }
  });

  it("reuses the loaded scope across legs on a read-only connection", async () => {
    resetVecScopeScanStats();
    expect(prepareScopedVecScan(reader.db, { collections: ["small"] }, true)).toBe(true);
    const loadsAfterPrep = vecScopeScanStats().loads;
    await searchVecDetailed(reader.db, "q1", MODEL, 10, { collections: ["small"] });
    await searchVecDetailed(reader.db, "q2", MODEL, 10, { collections: ["small"] });
    expect(vecScopeScanStats().loads).toBe(loadsAfterPrep);
    expect(vecScopeScanStats().scans).toBe(2);
  });

  it("caches nothing on a connection that did not opt in (MCP workers, writers)", async () => {
    expect(enableVecScopeCache(writer.db)).toBe(false); // a writer cannot opt in
    const ro = createStore(dbPath, { readonly: true });
    try {
      resetVecScopeScanStats();
      await searchVecDetailed(ro.db, "q1", MODEL, 10, { collections: ["small"] });
      await searchVecDetailed(ro.db, "q2", MODEL, 10, { collections: ["small"] });
      expect(vecScopeScanStats().loads).toBe(2);
      expect(vecScopeScanStats().scans).toBe(2);
    } finally {
      ro.close();
    }
  });

  it("the parallel prewarm feeds the legs on a read-only connection", async () => {
    const ro = createStore(dbPath, { readonly: true });
    enableVecScopeCache(ro.db);
    try {
      resetVecScopeScanStats();
      expect(await prewarmScopedVecScan(ro.db, dbPath, { collections: ["small"] }, 3)).toBe(true);
      expect(vecScopeScanStats().parallelLoads).toBe(1);
      const res = await searchVecDetailed(ro.db, "q", MODEL, 10, { collections: ["small"] });
      expect(vecScopeScanStats().loads).toBe(1); // no second, in-thread load
      expect(paths(res)).toEqual(
        paths(searchVecDetailedWithVector(writer.db, qv(), 10, { collections: ["small"] })),
      );
    } finally {
      ro.close();
    }
  });

  it("never serves a stale scope after another connection writes", () => {
    const before = searchVecDetailedWithVector(reader.db, qv(), 5, { collections: ["small"] });
    // A new small doc whose single fragment IS the query vector: it must rank first at once.
    const now = "2026-01-15T00:00:00.000Z";
    const body = "body of small/zz-new.md";
    const hash = hashContent(body);
    writer.insertContent(hash, body, now);
    writer.insertDocument("small", "zz-new.md", "zz-new.md", hash, now, now);
    writer.insertEmbedding(hash, 0, 0, QUERY, MODEL, now, "section");
    const after = searchVecDetailedWithVector(reader.db, qv(), 5, { collections: ["small"] });
    expect(before.results[0]!.displayPath).not.toBe("small/zz-new.md");
    expect(after.results[0]!.displayPath).toBe("small/zz-new.md");
    expect(after.results[0]!.score).toBeCloseTo(1, 6);
  });
});
