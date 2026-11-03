import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from "bun:test";
import { copyFileSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

/**
 * master-harness-uvnhf.17 — resident scoped vectors for `clawmem serve`.
 *
 * Pinned here:
 *  - off (the default) retains nothing and re-reads as before;
 *  - on, a new connection reuses the loaded scope: no chunk read, no worker, identical distances;
 *  - every writer path that changes bytes misses, including the two where the fragment list comes
 *    back identical (a dropped table restarts its rowids; a rebuilt file repeats its schema_version);
 *  - scope changes are honoured, the cap and LRU hold, the age bound holds, workers stay out.
 */

import {
  clearAllEmbeddings,
  createStore,
  enableVecScopeCache,
  prewarmScopedVecScan,
  searchVecDetailedWithVector,
  type Store,
} from "../../src/store.ts";
import {
  RESIDENT_SCOPE_MAX_AGE_S,
  clearResidentScopes,
  residentScopeCapBytes,
  residentScopeMaxAgeMs,
  residentScopeUsage,
  resetVecScopeScanStats,
  vecScopeScanStats,
} from "../../src/vec-scope-scan.ts";
import { compactVectors } from "../../src/vec-compact.ts";

const MODEL = "resident-test-model";
const NOW = "2026-01-15T00:00:00.000Z";
const DIM = 16;
const MB = "CLAWMEM_QUERY_RESIDENT_SCOPE_MB";
const MAX_AGE = "CLAWMEM_QUERY_RESIDENT_SCOPE_MAX_AGE_S";

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

const QUERY = randomVec(rng(7), DIM);
const hashOf = (col: string, i: number) => `${col}${String(i).padStart(4, "0")}`;
const pathOf = (col: string, i: number) => `${hashOf(col, i)}.md`;

/** `docs` one-fragment documents in `col`, each with a seeded random vector. */
function seed(store: Store, col: string, docs: number, dim = DIM, seedValue = 100): void {
  store.ensureVecTable(dim);
  const r = rng(seedValue);
  store.db.transaction(() => {
    for (let i = 0; i < docs; i++) {
      const hash = hashOf(col, i);
      store.insertContent(hash, `body ${hash}`, NOW);
      store.insertDocument(col, pathOf(col, i), pathOf(col, i), hash, NOW, NOW);
      store.insertEmbedding(hash, 0, 0, randomVec(r, dim), MODEL, NOW, "section");
    }
  })();
}

type Hit = [string, number];
const scopeOpts = (col: string) => ({ collections: [col], scopedScanMaxChunkFraction: 1 });

function readOnly(path: string): Store {
  const ro = createStore(path, { readonly: true });
  expect(enableVecScopeCache(ro.db)).toBe(true);
  return ro;
}

/** One query on a new opted-in connection, the way cmdQuery opens one per request. */
function query(path: string, col: string, q: Float32Array = QUERY, limit = 50): Hit[] {
  const ro = readOnly(path);
  try {
    return searchVecDetailedWithVector(
      ro.db,
      { embedding: q, endpointModel: MODEL },
      limit,
      scopeOpts(col),
    ).results.map((r) => [r.displayPath, r.score]);
  } finally {
    ro.close();
  }
}

/** The same scope answered by vec0 MATCH, with the score searchVecDetailed derives. */
function vec0Scores(store: Store, col: string, q: Float32Array, n: number): Map<string, number> {
  const rows = store.db
    .prepare(`SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ?`)
    .all(q, n) as { hash_seq: string; distance: number }[];
  const out = new Map<string, number>();
  for (const row of rows) {
    if (row.hash_seq.startsWith(col))
      out.set(`${col}/${row.hash_seq.split("_")[0]}.md`, 1 - row.distance);
  }
  return out;
}

function fragments(db: Store["db"], col: string): unknown[] {
  return db
    .prepare(
      `SELECT cv.hash || '_' || cv.seq AS hashSeq, r.rowid AS rowid, r.chunk_id AS chunkId, r.chunk_offset AS offset
       FROM documents d JOIN content_vectors cv ON cv.hash = d.hash
       JOIN vectors_vec_rowids r ON r.id = cv.hash || '_' || cv.seq
       WHERE d.active = 1 AND d.collection = ? ORDER BY cv.hash`,
    )
    .all(col);
}

const schemaVersion = (db: Store["db"]) =>
  (db.prepare("PRAGMA schema_version").get() as { schema_version: number }).schema_version;

let templateDir: string;
let template: string;
let dir: string;
let dbPath: string;
let writer: Store;

// A new database costs ~1 s of migrations, so each test copies one migrated, empty file.
beforeAll(() => {
  templateDir = mkdtempSync(join(tmpdir(), "clawmem-resident-template-"));
  template = join(templateDir, "index.sqlite");
  createStore(template).close();
});

afterAll(() => {
  rmSync(templateDir, { recursive: true, force: true });
});

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "clawmem-resident-"));
  dbPath = join(dir, "index.sqlite");
  copyFileSync(template, dbPath);
  writer = createStore(dbPath);
  clearResidentScopes();
  resetVecScopeScanStats();
  delete process.env[MB];
  delete process.env[MAX_AGE];
});

afterEach(() => {
  delete process.env[MB];
  delete process.env[MAX_AGE];
  clearResidentScopes();
  try {
    writer.close();
  } catch {
    // A test may have closed it to rebuild the file.
  }
  rmSync(dir, { recursive: true, force: true });
});

describe("resident scope off (A1)", () => {
  it("re-reads the chunks and retains nothing when unset", () => {
    seed(writer, "c", 20);
    const first = query(dbPath, "c");
    const afterFirst = vecScopeScanStats();
    expect(afterFirst.chunksRead).toBeGreaterThan(0);
    const second = query(dbPath, "c");
    const st = vecScopeScanStats();
    expect(st.chunksRead).toBe(afterFirst.chunksRead * 2);
    expect(st.residentHits).toBe(0);
    expect(st.residentMisses).toBe(0);
    expect(residentScopeUsage()).toEqual({ entries: 0, bytes: 0 });
    expect(second).toEqual(first);
  });

  it("treats empty, zero and invalid values as off", () => {
    const on = (v: string) => residentScopeCapBytes({ [MB]: v }, true);
    for (const v of [
      "",
      " ",
      "0",
      "00",
      "abc",
      "-5",
      "1e3",
      "1.5",
      "0x10",
      "256MB",
      "9".repeat(400),
    ]) {
      expect(on(v)).toBe(0);
    }
    expect(on("256")).toBe(256 * 1024 * 1024);
    expect(on(" 1 ")).toBe(1024 * 1024);
    expect(residentScopeCapBytes({}, true)).toBe(0);
  });

  it("releases what it held once the setting is off", () => {
    seed(writer, "c", 20);
    process.env[MB] = "16";
    query(dbPath, "c");
    expect(residentScopeUsage().entries).toBe(1);
    delete process.env[MB];
    query(dbPath, "c");
    expect(residentScopeUsage()).toEqual({ entries: 0, bytes: 0 });
  });
});

describe("resident scope hit (A2)", () => {
  it("serves a new connection without a chunk read, bit-identical to a fresh load and vec0", () => {
    seed(writer, "c", 40);
    const fresh = query(dbPath, "c"); // off: a plain load
    resetVecScopeScanStats();

    process.env[MB] = "16";
    const populate = query(dbPath, "c");
    const afterPopulate = vecScopeScanStats();
    expect(afterPopulate.residentMisses).toBe(1);
    expect(afterPopulate.residentHits).toBe(0);
    expect(afterPopulate.chunksRead).toBeGreaterThan(0);

    const served = query(dbPath, "c");
    const st = vecScopeScanStats();
    expect(st.chunksRead).toBe(afterPopulate.chunksRead);
    expect(st.loads).toBe(afterPopulate.loads);
    expect(st.residentHits).toBe(1);
    expect(st.residentMisses).toBe(1);

    expect(served).toEqual(fresh);
    expect(populate).toEqual(fresh);
    const vec0 = vec0Scores(writer, "c", QUERY, 40);
    expect(served.length).toBe(40);
    for (const [path, score] of served) expect(score).toBe(vec0.get(path)!);
  });

  it("gives every query vector its own distances from the shared arrays, and leaves them unchanged", () => {
    seed(writer, "c", 40);
    process.env[MB] = "16";
    const before = query(dbPath, "c");
    const other = randomVec(rng(11), DIM);
    const otherHits = query(dbPath, "c", other);
    const vec0 = vec0Scores(writer, "c", other, 40);
    for (const [path, score] of otherHits) expect(score).toBe(vec0.get(path)!);
    expect(query(dbPath, "c")).toEqual(before);
    expect(vecScopeScanStats().residentHits).toBe(2);
  });

  it("survives an unrelated commit in the middle of a query's legs, on one connection", () => {
    seed(writer, "c", 20);
    process.env[MB] = "16";
    const ro = readOnly(dbPath);
    try {
      const leg = () =>
        searchVecDetailedWithVector(
          ro.db,
          { embedding: QUERY, endpointModel: MODEL },
          5,
          scopeOpts("c"),
        ).results.map((r): Hit => [r.displayPath, r.score]);
      const first = leg();
      const reads = vecScopeScanStats().chunksRead;
      // What a query's own llm_cache write does: data_version moves, the scope does not.
      writer.insertContent("unrelated", "unrelated", NOW);
      expect(leg()).toEqual(first);
      expect(vecScopeScanStats().chunksRead).toBe(reads);
      expect(vecScopeScanStats().residentHits).toBe(1);
    } finally {
      ro.close();
    }
  });

  it("keys on the scope: another collection on the same database does not share an entry", () => {
    seed(writer, "aa", 10);
    seed(writer, "bb", 10, DIM, 200);
    process.env[MB] = "16";
    const a = query(dbPath, "aa");
    const b = query(dbPath, "bb");
    expect(residentScopeUsage().entries).toBe(2);
    expect(vecScopeScanStats().residentHits).toBe(0);
    expect(a.every(([p]) => p.startsWith("aa/"))).toBe(true);
    expect(b.every(([p]) => p.startsWith("bb/"))).toBe(true);
  });

  it("takes no part on a connection that did not opt in, so no uncommitted vector is retained", () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    expect(enableVecScopeCache(writer.db)).toBe(false);
    searchVecDetailedWithVector(
      writer.db,
      { embedding: QUERY, endpointModel: MODEL },
      10,
      scopeOpts("c"),
    );
    const st = vecScopeScanStats();
    expect(st.residentHits + st.residentMisses).toBe(0);
    expect(residentScopeUsage().entries).toBe(0);
  });
});

describe("a re-embed through another connection (A3)", () => {
  it("misses and returns the new vector", () => {
    seed(writer, "c", 20);
    process.env[MB] = "16";
    query(dbPath, "c");
    query(dbPath, "c");
    expect(vecScopeScanStats().residentHits).toBe(1);
    const target = "c/c0007.md";
    const stale = query(dbPath, "c").find(([p]) => p === target)!;
    expect(stale[1]).toBeLessThan(0.99);

    const before = vecScopeScanStats();
    writer.insertEmbedding(hashOf("c", 7), 0, 0, QUERY, MODEL, NOW, "section"); // DELETE + INSERT
    const after = query(dbPath, "c");
    const st = vecScopeScanStats();

    expect(after[0]![0]).toBe(target);
    expect(after[0]![1]).toBe(vec0Scores(writer, "c", QUERY, 20).get(target)!);
    expect(after[0]![1]).toBeCloseTo(1, 5);
    expect(st.residentMisses).toBe(before.residentMisses + 1);
    expect(st.residentHits).toBe(before.residentHits);
    expect(st.chunksRead).toBeGreaterThan(before.chunksRead);
  });
});

describe("the same fragment list holding new bytes (A4)", () => {
  /** Assert the premise (identical list), then that the next query serves the NEW vector. */
  function expectNewVector(change: () => void): void {
    seed(writer, "c", 6);
    process.env[MB] = "16";
    query(dbPath, "c");
    query(dbPath, "c");
    expect(vecScopeScanStats().residentHits).toBe(1);
    const target = "c/c0003.md";
    const listBefore = fragments(writer.db, "c");
    const versionBefore = schemaVersion(writer.db);

    change();

    expect(fragments(writer.db, "c")).toEqual(listBefore);
    expect(schemaVersion(writer.db)).toBeGreaterThan(versionBefore);
    const before = vecScopeScanStats();
    const after = query(dbPath, "c");
    const st = vecScopeScanStats();
    expect(after[0]![0]).toBe(target);
    expect(after[0]![1]).toBeCloseTo(1, 5);
    expect(after[0]![1]).toBe(vec0Scores(writer, "c", QUERY, 6).get(target)!);
    expect(st.residentMisses).toBe(before.residentMisses + 1);
    expect(st.residentHits).toBe(before.residentHits);
  }

  it("re-embed, then the real vec-compact", () => {
    expectNewVector(() => {
      writer.insertEmbedding(hashOf("c", 3), 0, 0, QUERY, MODEL, NOW, "section");
      compactVectors(writer.db);
    });
  });

  it("clearAllEmbeddings, then a re-embed in the same order", () => {
    expectNewVector(() => {
      clearAllEmbeddings(writer.db);
      writer.ensureVecTable(DIM);
      const r = rng(100);
      for (let i = 0; i < 6; i++) {
        const original = randomVec(r, DIM);
        writer.insertEmbedding(
          hashOf("c", i),
          0,
          0,
          i === 3 ? QUERY : original,
          MODEL,
          NOW,
          "section",
        );
      }
    });
  });

  it("a database rebuilt at the same path, whose schema_version also repeats", () => {
    seed(writer, "c", 6);
    process.env[MB] = "16";
    query(dbPath, "c");
    query(dbPath, "c");
    expect(vecScopeScanStats().residentHits).toBe(1);
    const listBefore = fragments(writer.db, "c");
    const versionBefore = schemaVersion(writer.db);

    writer.close();
    for (const suffix of ["", "-wal", "-shm"]) {
      try {
        unlinkSync(dbPath + suffix);
      } catch {
        // WAL files are gone once the last connection closes cleanly.
      }
    }
    writer = createStore(dbPath);
    writer.ensureVecTable(DIM);
    const r = rng(100);
    for (let i = 0; i < 6; i++) {
      const hash = hashOf("c", i);
      const original = randomVec(r, DIM);
      writer.insertContent(hash, `body ${hash}`, NOW);
      writer.insertDocument("c", pathOf("c", i), pathOf("c", i), hash, NOW, NOW);
      writer.insertEmbedding(hash, 0, 0, i === 3 ? QUERY : original, MODEL, NOW, "section");
    }
    expect(fragments(writer.db, "c")).toEqual(listBefore);
    expect(schemaVersion(writer.db)).toBe(versionBefore);

    const after = query(dbPath, "c");
    expect(after[0]![0]).toBe("c/c0003.md");
    expect(after[0]![1]).toBeCloseTo(1, 5);
  });
});

describe("a changed scope (A5)", () => {
  it("drops a deactivated document", () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    const all = query(dbPath, "c");
    writer.deactivateDocument("c", pathOf("c", 4), "test" as never);
    const less = query(dbPath, "c");
    expect(less.length).toBe(all.length - 1);
    expect(less.some(([p]) => p === "c/c0004.md")).toBe(false);
    expect(less).toEqual(all.filter(([p]) => p !== "c/c0004.md"));
    expect(vecScopeScanStats().residentHits).toBe(0);
    expect(vecScopeScanStats().residentMisses).toBe(2);
  });

  it("serves a document added between queries", () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    const before = query(dbPath, "c");
    const hash = hashOf("c", 99);
    writer.insertContent(hash, `body ${hash}`, NOW);
    writer.insertDocument("c", pathOf("c", 99), pathOf("c", 99), hash, NOW, NOW);
    writer.insertEmbedding(hash, 0, 0, QUERY, MODEL, NOW, "section");
    const after = query(dbPath, "c");
    expect(after.length).toBe(before.length + 1);
    expect(after[0]![0]).toBe("c/c0099.md");
    expect(vecScopeScanStats().residentHits).toBe(0);
  });

  it("honours an invalidated document, which changes no vector", () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    const all = query(dbPath, "c");
    writer.db
      .prepare(`UPDATE documents SET invalidated_at = ? WHERE path = ?`)
      .run(NOW, pathOf("c", 2));
    const less = query(dbPath, "c");
    expect(less.length).toBe(all.length - 1);
    expect(less.some(([p]) => p === "c/c0002.md")).toBe(false);
  });
});

describe("byte cap (A6)", () => {
  // 768-d: 3,076 bytes per fragment with its magnitude, so 1 MiB holds 340 fragments.
  const BIG = 768;
  const bigQuery = randomVec(rng(5), BIG);

  it("does not retain an entry larger than the cap", () => {
    seed(writer, "big", 400, BIG);
    process.env[MB] = "1";
    query(dbPath, "big", bigQuery);
    expect(residentScopeUsage()).toEqual({ entries: 0, bytes: 0 });
    const reads = vecScopeScanStats().chunksRead;
    query(dbPath, "big", bigQuery);
    expect(vecScopeScanStats().chunksRead).toBe(reads * 2);
    expect(vecScopeScanStats().residentHits).toBe(0);
  });

  it("evicts the least recently used scope, not the oldest", () => {
    seed(writer, "aa", 150, BIG, 1);
    seed(writer, "bb", 150, BIG, 2);
    seed(writer, "cc", 150, BIG, 3);
    process.env[MB] = "1";
    query(dbPath, "aa", bigQuery);
    query(dbPath, "bb", bigQuery);
    expect(residentScopeUsage().entries).toBe(2);
    expect(residentScopeUsage().bytes).toBe(2 * 150 * (BIG * 4 + 4));
    query(dbPath, "aa", bigQuery); // aa is now warmer than bb, though loaded first
    expect(vecScopeScanStats().residentHits).toBe(1);

    query(dbPath, "cc", bigQuery); // over the cap: bb goes
    const usage = residentScopeUsage();
    expect(usage.entries).toBe(2);
    expect(usage.bytes).toBeLessThanOrEqual(1024 * 1024);

    query(dbPath, "aa", bigQuery);
    expect(vecScopeScanStats().residentHits).toBe(2);
    const before = vecScopeScanStats();
    query(dbPath, "bb", bigQuery);
    expect(vecScopeScanStats().residentHits).toBe(2);
    expect(vecScopeScanStats().residentMisses).toBe(before.residentMisses + 1);
  });
});

describe("max age (A7)", () => {
  it("defaults to 6 hours, and an invalid override keeps the default", () => {
    expect(RESIDENT_SCOPE_MAX_AGE_S).toBe(6 * 60 * 60);
    expect(residentScopeMaxAgeMs({})).toBe(6 * 60 * 60 * 1000);
    expect(residentScopeMaxAgeMs({ [MAX_AGE]: "abc" })).toBe(6 * 60 * 60 * 1000);
    expect(residentScopeMaxAgeMs({ [MAX_AGE]: "-1" })).toBe(6 * 60 * 60 * 1000);
    expect(residentScopeMaxAgeMs({ [MAX_AGE]: "0.5" })).toBe(500);
  });

  it("misses and re-reads once the entry is past the age", async () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    process.env[MAX_AGE] = "0.001";
    query(dbPath, "c");
    const before = vecScopeScanStats();
    await Bun.sleep(5);
    query(dbPath, "c");
    const st = vecScopeScanStats();
    expect(st.residentHits).toBe(0);
    expect(st.residentMisses).toBe(before.residentMisses + 1);
    expect(st.chunksRead).toBeGreaterThan(before.chunksRead);
  });

  it("does not extend an entry's life on a hit", async () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    process.env[MAX_AGE] = "0.2";
    query(dbPath, "c");
    query(dbPath, "c");
    expect(vecScopeScanStats().residentHits).toBe(1);
    await Bun.sleep(120);
    query(dbPath, "c"); // still young: a hit, which must not restart the clock
    expect(vecScopeScanStats().residentHits).toBe(2);
    await Bun.sleep(120);
    query(dbPath, "c");
    expect(vecScopeScanStats().residentHits).toBe(2);
  });
});

describe("worker threads (A8)", () => {
  it("never use the cache, with the setting on", async () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    const storeUrl = resolve(import.meta.dir, "../../src/store.ts");
    const scanUrl = resolve(import.meta.dir, "../../src/vec-scope-scan.ts");
    const script = join(dir, "resident-worker.ts");
    writeFileSync(
      script,
      `import { isMainThread } from "node:worker_threads";
       import { createStore, enableVecScopeCache, searchVecDetailedWithVector } from ${JSON.stringify(storeUrl)};
       import { residentScopeCapBytes, residentScopeUsage, vecScopeScanStats } from ${JSON.stringify(scanUrl)};
       declare var self: Worker;
       self.onmessage = (ev: MessageEvent) => {
         process.env.${MB} = "16";
         const run = () => {
           const ro = createStore(ev.data.dbPath, { readonly: true });
           if (!enableVecScopeCache(ro.db)) throw new Error("no opt-in");
           searchVecDetailedWithVector(ro.db, { embedding: new Float32Array(ev.data.query), endpointModel: ev.data.model }, 10,
             { collections: ["c"], scopedScanMaxChunkFraction: 1 });
           ro.close();
         };
         run();
         const first = vecScopeScanStats().chunksRead;
         run();
         self.postMessage({ isMainThread, cap: residentScopeCapBytes(), usage: residentScopeUsage(), first, stats: vecScopeScanStats() });
       };`,
    );
    const worker = new Worker(script);
    try {
      const reply = await new Promise<any>((res, rej) => {
        worker.onmessage = (ev: MessageEvent) => res(ev.data);
        worker.onerror = (ev: ErrorEvent) => rej(new Error(ev.message));
        worker.postMessage({ dbPath, query: Array.from(QUERY), model: MODEL });
      });
      expect(reply.isMainThread).toBe(false);
      expect(reply.cap).toBe(0);
      expect(reply.usage).toEqual({ entries: 0, bytes: 0 });
      expect(reply.first).toBeGreaterThan(0);
      expect(reply.stats.chunksRead).toBe(reply.first * 2);
      expect(reply.stats.residentHits).toBe(0);
      expect(reply.stats.residentMisses).toBe(0);
    } finally {
      worker.terminate();
    }
    // The guard itself, with the setting present.
    expect(residentScopeCapBytes({ [MB]: "256" }, false)).toBe(0);
    expect(residentScopeCapBytes({ [MB]: "256" }, true)).toBe(256 * 1024 * 1024);
  });
});

describe("the parallel prewarm (A9)", () => {
  const prewarm = (path: string, loadPath = path) => {
    const ro = readOnly(path);
    return prewarmScopedVecScan(ro.db, loadPath, scopeOpts("c"), 3).finally(() => ro.close());
  };

  it("starts no worker on a hit, and what it loaded is what the next query serves", async () => {
    seed(writer, "c", 40);
    process.env[MB] = "16";

    let started = 0;
    const Real = globalThis.Worker;
    globalThis.Worker = class extends Real {
      constructor(...args: ConstructorParameters<typeof Worker>) {
        super(...args);
        started++;
      }
    } as typeof Worker;
    try {
      expect(await prewarm(dbPath)).toBe(true);
      expect(started).toBeGreaterThan(0);
      const first = vecScopeScanStats();
      expect(first.parallelLoads).toBe(1);
      expect(first.residentMisses).toBe(1);
      expect(residentScopeUsage().entries).toBe(1);

      started = 0;
      expect(await prewarm(dbPath)).toBe(true);
      const st = vecScopeScanStats();
      expect(started).toBe(0);
      expect(st.parallelLoads).toBe(1);
      expect(st.chunksRead).toBe(first.chunksRead);
      expect(st.residentHits).toBe(1);

      const served = query(dbPath, "c");
      const vec0 = vec0Scores(writer, "c", QUERY, 40);
      for (const [path, score] of served) expect(score).toBe(vec0.get(path)!);
      expect(vecScopeScanStats().parallelLoads).toBe(1);
    } finally {
      globalThis.Worker = Real;
    }
  });

  it("does not retain a parallel load whose table was dropped and refilled meanwhile", async () => {
    seed(writer, "c", 6);
    process.env[MB] = "16";
    const versionBefore = schemaVersion(writer.db);

    const ro = readOnly(dbPath);
    const loading = prewarmScopedVecScan(ro.db, dbPath, scopeOpts("c"), 3);
    // Synchronously after the enumeration: same fragment list, new bytes, before any worker reply.
    clearAllEmbeddings(writer.db);
    writer.ensureVecTable(DIM);
    const r = rng(100);
    for (let i = 0; i < 6; i++) {
      const original = randomVec(r, DIM);
      writer.insertEmbedding(
        hashOf("c", i),
        0,
        0,
        i === 3 ? QUERY : original,
        MODEL,
        NOW,
        "section",
      );
    }
    expect(schemaVersion(writer.db)).toBeGreaterThan(versionBefore);
    await loading;
    ro.close();
    expect(residentScopeUsage().entries).toBe(0);

    const after = query(dbPath, "c");
    expect(after[0]![0]).toBe("c/c0003.md");
    expect(after[0]![1]).toBeCloseTo(1, 5);
  });

  it("never retains a failed or declined load", async () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";

    // Every worker fails to open this database, so the load resolves null.
    expect(await prewarm(dbPath, join(dir, "missing", "index.sqlite"))).toBe(false);
    expect(vecScopeScanStats().verifyFailures).toBe(1);
    expect(residentScopeUsage().entries).toBe(0);

    // A scope the plan declines leaves nothing to retain on either load path.
    const ro = readOnly(dbPath);
    try {
      searchVecDetailedWithVector(ro.db, { embedding: QUERY, endpointModel: MODEL }, 10, {
        collections: ["c"],
        scopedScanMaxChunkFraction: 0,
      });
      expect(residentScopeUsage().entries).toBe(0);
    } finally {
      ro.close();
    }
  });

  it("does not retain an empty scope, so unknown collections cannot grow the cache", () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    expect(query(dbPath, "nothing-here")).toEqual([]);
    expect(residentScopeUsage().entries).toBe(0);
  });

  it("does not spoil the in-flight query when the connection closes under the load", async () => {
    seed(writer, "c", 10);
    process.env[MB] = "16";
    const ro = readOnly(dbPath);
    const loading = prewarmScopedVecScan(ro.db, dbPath, scopeOpts("c"), 3);
    ro.close();
    await loading; // never rejects
    expect(residentScopeUsage().entries).toBe(0);
  });
});
