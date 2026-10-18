import { describe, it, expect, afterAll, beforeAll } from "bun:test";
import { mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * master-harness-vn4rz.110 — `clawmem vec-compact`: rewrite vectors_vec in collection-clustered
 * order. Every vault here is a temp file; the live ~/.cache/clawmem index is never opened.
 *
 * Pinned: clustering (span == ceil(n/chunk_size)), bit-identical pairs, canonical DDL kept,
 * identical KNN / scoped search results, rollback at every stage, refusal of missing and
 * non-canonical tables, and a dry run that writes nothing and reports the same numbers.
 */

import {
  createStore,
  getVecTableDim,
  searchVecDetailedWithVector,
  type Store,
} from "../../src/store.ts";
import {
  canonicalVecTableDim,
  resetVecScopeScanStats,
  vecScopeScanStats,
} from "../../src/vec-scope-scan.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";
import { hashContent } from "../../src/indexer.ts";
import {
  compactVectors,
  readCanonicalVecShape,
  formatVecCompactReport,
  VecCompactRefusal,
  type VecCompactStage,
} from "../../src/vec-compact.ts";

const MODEL = "vec-compact-test-model";
const NOW = "2026-01-15T00:00:00.000Z";
const CLI = join(import.meta.dir, "../../src/clawmem.ts");

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

const dirs: string[] = [];
function tmpVault(): { dir: string; dbPath: string } {
  const dir = mkdtempSync(join(tmpdir(), "clawmem-vec-compact-"));
  dirs.push(dir);
  return { dir, dbPath: join(dir, "index.sqlite") };
}

afterAll(() => {
  setDefaultLlamaCpp(null);
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

const ddl = (dim: number, chunk?: number) =>
  `CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${dim}] distance_metric=cosine${chunk ? `, chunk_size=${chunk}` : ""})`;

/**
 * Seed `cols` with `docs` documents of `frags` fragments each, inserting the fragments
 * INTERLEAVED (seq-major, then doc, then collection) so every collection's vectors scatter over
 * every chunk. Returns hash lists per collection.
 */
function seedInterleaved(
  store: Store,
  cols: string[],
  docs: number,
  frags: number,
  dim: number,
  r: () => number,
): Map<string, string[]> {
  store.ensureVecTable(dim);
  const hashes = new Map<string, string[]>();
  store.db.transaction(() => {
    for (const col of cols) {
      const hs: string[] = [];
      for (let i = 0; i < docs; i++) {
        const path = `${col}-${String(i).padStart(4, "0")}.md`;
        const body = `body of ${col}/${path}`;
        const hash = hashContent(body);
        store.insertContent(hash, body, NOW);
        store.insertDocument(col, path, path, hash, NOW, NOW);
        hs.push(hash);
      }
      hashes.set(col, hs);
    }
    for (let seq = 0; seq < frags; seq++) {
      for (let i = 0; i < docs; i++) {
        for (const col of cols) {
          store.insertEmbedding(
            hashes.get(col)![i]!,
            seq,
            seq,
            randomVec(r, dim),
            MODEL,
            NOW,
            "section",
          );
        }
      }
    }
  })();
  return hashes;
}

interface Snapshot {
  ddl: string;
  pairs: Map<string, string>;
  rowids: string;
  chunks: number;
}

function snapshot(db: Store["db"]): Snapshot {
  const ddlRow = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as {
    sql: string;
  };
  const pairs = new Map<string, string>();
  for (const row of db.prepare(`SELECT hash_seq, embedding FROM vectors_vec`).all() as {
    hash_seq: string;
    embedding: Uint8Array;
  }[]) {
    pairs.set(row.hash_seq, Buffer.from(row.embedding).toString("hex"));
  }
  const rowids = JSON.stringify(
    db
      .prepare(`SELECT rowid, id, chunk_id, chunk_offset FROM vectors_vec_rowids ORDER BY rowid`)
      .all(),
  );
  const chunks = (db.prepare(`SELECT count(*) AS c FROM vectors_vec_chunks`).get() as { c: number })
    .c;
  return { ddl: ddlRow.sql, pairs, rowids, chunks };
}

/** Distinct vec0 chunks holding the active vectors of `col`, via vectors_vec_rowids. */
function span(db: Store["db"], col: string): number {
  return (
    db
      .prepare(
        `SELECT count(DISTINCT r.chunk_id) AS c FROM documents d
         JOIN content_vectors cv ON cv.hash = d.hash
         JOIN vectors_vec_rowids r ON r.id = cv.hash || '_' || cv.seq
         WHERE d.active = 1 AND d.collection = ?`,
      )
      .get(col) as { c: number }
  ).c;
}

function count(db: Store["db"], col: string): number {
  return (
    db
      .prepare(
        `SELECT count(*) AS c FROM documents d JOIN content_vectors cv ON cv.hash = d.hash WHERE d.active = 1 AND d.collection = ?`,
      )
      .get(col) as { c: number }
  ).c;
}

const CHUNK = 8;
const DIM = 8;
const COLS = ["alpha", "beta", "gamma"];
const DOCS = 6;
const FRAGS = 4; // 24 vectors per collection -> 3 chunks of 8 once clustered

function seededSmallChunkStore(path: string): Store {
  const s = createStore(path);
  s.db.exec(ddl(DIM, CHUNK)); // custom chunk_size must be preserved
  seedInterleaved(s, COLS, DOCS, FRAGS, DIM, rng(110));
  // An orphan: a vector whose only document was deactivated (must be placed last).
  const body = "orphan body";
  const hash = hashContent(body);
  s.insertContent(hash, body, NOW);
  s.insertDocument("alpha", "orphan.md", "orphan.md", hash, NOW, NOW);
  s.insertEmbedding(hash, 0, 0, randomVec(rng(5), DIM), MODEL, NOW, "section");
  s.db.prepare(`UPDATE documents SET active = 0 WHERE hash = ?`).run(hash);
  return s;
}

describe("clustering (A1)", () => {
  it("collapses each collection to ceil(n/chunk_size) chunks", () => {
    const { dbPath } = tmpVault();
    const s = seededSmallChunkStore(dbPath);
    try {
      const n = DOCS * FRAGS;
      const before = COLS.map((c) => span(s.db, c));
      // Interleaved insertion: every collection touches (almost) every chunk of the table.
      for (const b of before) expect(b).toBeGreaterThan(Math.ceil(n / CHUNK) * 2);
      const report = compactVectors(s.db);
      for (const col of COLS) {
        expect(count(s.db, col)).toBe(n);
        const after = span(s.db, col);
        expect(after).toBeLessThanOrEqual(Math.ceil(n / CHUNK) + 1);
        expect(after).toBe(report.collections.find((c) => c.name === col)!.afterSpan!);
      }
      expect(span(s.db, "alpha")).toBe(3); // exact for the first-placed collection
      expect(report.chunkSize).toBe(CHUNK);
      expect(report.totalVectors).toBe(COLS.length * n + 1);
      expect(report.unownedVectors).toBe(1);
    } finally {
      s.close();
    }
  });

  it("places collection names ascending, vectors without an active owner last", () => {
    const { dbPath } = tmpVault();
    const s = seededSmallChunkStore(dbPath);
    try {
      compactVectors(s.db);
      const rows = s.db.prepare(`SELECT id FROM vectors_vec_rowids ORDER BY rowid`).all() as {
        id: string;
      }[];
      const owner = new Map<string, string>();
      for (const d of s.db
        .prepare(`SELECT hash, collection FROM documents WHERE active = 1`)
        .all() as { hash: string; collection: string }[]) {
        owner.set(d.hash, d.collection);
      }
      const seq = rows.map((r) => owner.get(r.id.split("_")[0]!) ?? "~orphan");
      expect(seq).toEqual([...seq].sort());
      expect(seq[seq.length - 1]).toBe("~orphan");
      // within a collection: hash, then seq numerically
      const alpha = rows
        .filter((r) => owner.get(r.id.split("_")[0]!) === "alpha")
        .map((r) => r.id.split("_"));
      const sorted = [...alpha].sort((a, b) =>
        a[0]! < b[0]! ? -1 : a[0]! > b[0]! ? 1 : Number(a[1]) - Number(b[1]),
      );
      expect(alpha).toEqual(sorted);
    } finally {
      s.close();
    }
  });
});

describe("preservation (A2)", () => {
  it("keeps every embedding bit-identical, the row count, and a canonical DDL", () => {
    const { dbPath } = tmpVault();
    const s = seededSmallChunkStore(dbPath);
    try {
      const before = snapshot(s.db);
      compactVectors(s.db);
      const after = snapshot(s.db);
      expect(after.pairs.size).toBe(before.pairs.size);
      expect(after.pairs.size).toBe(COLS.length * DOCS * FRAGS + 1);
      expect(after.pairs).toEqual(before.pairs);
      expect(after.ddl).toBe(before.ddl);
      expect(getVecTableDim(s.db)).toBe(DIM);
      const shape = readCanonicalVecShape(s.db);
      expect(shape).toMatchObject({ dim: DIM, chunkSize: CHUNK });
      // rewriting really changed the layout, not just a no-op
      expect(after.rowids).not.toBe(before.rowids);
      // temp scratch is gone
      const temps = s.db
        .prepare(`SELECT name FROM sqlite_temp_master WHERE name LIKE 'vc_%'`)
        .all();
      expect(temps).toEqual([]);
    } finally {
      s.close();
    }
  });

  it("a default-chunk_size table stays passing the strict canonical validator", () => {
    const { dbPath } = tmpVault();
    const s = createStore(dbPath);
    try {
      seedInterleaved(s, COLS, 3, 2, 4, rng(3)); // ensureVecTable -> DDL without chunk_size
      expect(canonicalVecTableDim(s.db)).toBe(4);
      compactVectors(s.db);
      expect(canonicalVecTableDim(s.db)).toBe(4);
      expect(getVecTableDim(s.db)).toBe(4);
      expect(readCanonicalVecShape(s.db).chunkSize).toBe(1024);
    } finally {
      s.close();
    }
  });
});

describe("search equivalence (A3)", () => {
  const QDIM = 8;
  const Q = randomVec(rng(77), QDIM);
  const qv = () => ({ embedding: Q, endpointModel: MODEL });
  const knn = (db: Store["db"]) =>
    (
      db
        .prepare(`SELECT hash_seq, distance FROM vectors_vec WHERE embedding MATCH ? AND k = ?`)
        .all(Q, 1000) as { hash_seq: string; distance: number }[]
    ).map((r) => `${r.hash_seq}@${r.distance}`);
  const scoped = (db: Store["db"], col: string) =>
    searchVecDetailedWithVector(db, qv(), 20, { collections: [col] }).results.map(
      (r) => `${r.displayPath}@${r.score}`,
    );

  it("unscoped KNN and collection-scoped searchVec return identical order and distances", () => {
    const { dbPath } = tmpVault();
    const s = seededSmallChunkStore(dbPath);
    try {
      const knnBefore = knn(s.db);
      const scopedBefore = COLS.map((c) => scoped(s.db, c));
      compactVectors(s.db);
      expect(knn(s.db)).toEqual(knnBefore);
      expect(COLS.map((c) => scoped(s.db, c))).toEqual(scopedBefore);
      expect(scopedBefore[0]!.length).toBe(DOCS); // one result per document
    } finally {
      s.close();
    }
  });

  it("the chunk-targeted scan serves the scoped leg after compaction, with the same results", () => {
    const { dbPath } = tmpVault();
    const s = createStore(dbPath);
    try {
      // default chunk_size (1024): 3 x 600 interleaved vectors = 2 chunks; alpha fits chunk 0 after.
      seedInterleaved(s, COLS, 150, 4, QDIM, rng(8));
      resetVecScopeScanStats();
      const before = scoped(s.db, "alpha");
      expect(vecScopeScanStats().scans).toBe(0); // scattered over every chunk -> vec0 MATCH
      const spanBefore = span(s.db, "alpha");
      compactVectors(s.db);
      resetVecScopeScanStats();
      const after = scoped(s.db, "alpha");
      expect(vecScopeScanStats().scans).toBe(1); // 1 of 2 chunks -> served by the scan
      expect(after).toEqual(before);
      expect(span(s.db, "alpha")).toBeLessThan(spanBefore);
      expect(span(s.db, "alpha")).toBe(1);
    } finally {
      s.close();
    }
  });
});

describe("atomicity (A4)", () => {
  const stages: VecCompactStage[] = [
    "planned",
    "copied",
    "dropped",
    "recreated",
    "inserted",
    "verified",
  ];
  for (const failAt of stages) {
    it(`rolls back to the untouched table when the rebuild fails after '${failAt}'`, () => {
      const { dbPath } = tmpVault();
      const s = seededSmallChunkStore(dbPath);
      try {
        const before = snapshot(s.db);
        expect(() =>
          compactVectors(s.db, {
            onStage: (st) => {
              if (st === failAt) throw new Error(`injected failure after ${st}`);
            },
          }),
        ).toThrow(`injected failure after ${failAt}`);
        const after = snapshot(s.db);
        expect(after.ddl).toBe(before.ddl);
        expect(after.pairs).toEqual(before.pairs);
        expect(after.rowids).toBe(before.rowids); // same rowids and chunk placement: untouched
        expect(after.chunks).toBe(before.chunks);
        expect(
          s.db.prepare(`SELECT name FROM sqlite_temp_master WHERE name LIKE 'vc_%'`).all(),
        ).toEqual([]);
        // the connection is usable and a retry succeeds
        compactVectors(s.db);
        expect(snapshot(s.db).pairs).toEqual(before.pairs);
      } finally {
        s.close();
      }
    });
  }
});

describe("refusal (A5)", () => {
  it("refuses a vault without vectors_vec and writes nothing", () => {
    const { dbPath } = tmpVault();
    const s = createStore(dbPath);
    try {
      expect(() => compactVectors(s.db)).toThrow(VecCompactRefusal);
      expect(() => compactVectors(s.db, { dryRun: true })).toThrow(VecCompactRefusal);
      expect(
        s.db.prepare(`SELECT name FROM sqlite_master WHERE name LIKE 'vectors_vec%'`).all(),
      ).toEqual([]);
    } finally {
      s.close();
    }
  });

  for (const [label, sql] of [
    [
      "l2 metric",
      `CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[8])`,
    ],
    [
      "extra metadata column",
      `CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[8] distance_metric=cosine, collection TEXT)`,
    ],
    [
      "int8 element type",
      `CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding int8[8] distance_metric=cosine)`,
    ],
  ] as const) {
    it(`refuses a non-canonical schema (${label}) and leaves it untouched`, () => {
      const { dbPath } = tmpVault();
      const s = createStore(dbPath);
      try {
        s.db.exec(sql);
        const before = s.db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get();
        expect(() => compactVectors(s.db)).toThrow(VecCompactRefusal);
        expect(
          s.db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get(),
        ).toEqual(before);
        expect(
          s.db.prepare(`SELECT name FROM sqlite_temp_master WHERE name LIKE 'vc_%'`).all(),
        ).toEqual([]);
      } finally {
        s.close();
      }
    });
  }
});

describe("dry run (A6)", () => {
  it("writes nothing and reports the spans the real run then achieves", () => {
    const { dbPath } = tmpVault();
    const w = seededSmallChunkStore(dbPath);
    w.db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
    const files = () =>
      [dbPath, `${dbPath}-wal`].map((f) => {
        try {
          const st = statSync(f);
          return `${st.size}:${st.mtimeMs}`;
        } catch {
          return "absent";
        }
      });
    const ro = createStore(dbPath, { readonly: true });
    try {
      const snapBefore = snapshot(w.db);
      const filesBefore = files();
      const dvBefore = (ro.db.prepare(`PRAGMA data_version`).get() as { data_version: number })
        .data_version;

      const dry = compactVectors(ro.db, { dryRun: true, collections: ["beta"] });

      expect(files()).toEqual(filesBefore);
      expect(
        (ro.db.prepare(`PRAGMA data_version`).get() as { data_version: number }).data_version,
      ).toBe(dvBefore);
      expect(snapshot(w.db)).toEqual(snapBefore);
      expect(dry.dryRun).toBe(true);
      expect(dry.totalVectors).toBe(COLS.length * DOCS * FRAGS + 1);
      expect(dry.totalChunksBefore).toBe(snapBefore.chunks);
      expect(dry.totalChunksAfter).toBe(Math.ceil(dry.totalVectors / CHUNK));
      const n = DOCS * FRAGS;
      for (const col of COLS) {
        const row = dry.collections.find((c) => c.name === col)!;
        expect(row.vectors).toBe(n);
        expect(row.currentSpan).toBe(span(w.db, col));
        expect(row.projectedSpan).toBeLessThanOrEqual(Math.ceil(n / CHUNK) + 1);
        expect(row.afterSpan).toBeUndefined();
      }
      // read-only handle really cannot rewrite
      expect(() => compactVectors(ro.db)).toThrow();
      expect(snapshot(w.db)).toEqual(snapBefore);

      const real = compactVectors(w.db);
      for (const col of COLS) {
        expect(real.collections.find((c) => c.name === col)!.currentSpan).toBe(
          dry.collections.find((c) => c.name === col)!.currentSpan,
        );
        expect(real.collections.find((c) => c.name === col)!.afterSpan).toBe(
          dry.collections.find((c) => c.name === col)!.projectedSpan,
        );
      }
      expect(real.totalChunksAfter).toBe(dry.totalChunksAfter);
      expect(formatVecCompactReport(real)).toContain("elapsed");
    } finally {
      ro.close();
      w.close();
    }
  });

  it("reports the top-N collections by vector count plus any requested name", () => {
    const { dbPath } = tmpVault();
    const s = seededSmallChunkStore(dbPath);
    try {
      const r = compactVectors(s.db, { dryRun: true, top: 1, collections: ["gamma", "nope"] });
      expect(r.collections.map((c) => c.name)).toEqual(["alpha", "gamma", "nope"]);
      expect(r.collections[2]).toMatchObject({ vectors: 0, currentSpan: 0, projectedSpan: 0 });
      expect(r.collectionCount).toBe(3);
    } finally {
      s.close();
    }
  });
});

describe("CLI", () => {
  function run(dbPath: string, home: string, args: string[]) {
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined && !k.startsWith("CLAWMEM_") && k !== "INDEX_PATH") env[k] = v;
    }
    const p = Bun.spawnSync([process.execPath, CLI, "vec-compact", ...args], {
      env: {
        ...env,
        INDEX_PATH: dbPath,
        HOME: home,
        XDG_CACHE_HOME: join(home, "cache"),
        CLAWMEM_CONFIG_DIR: join(home, "cfg"),
        CLAWMEM_NO_LOCAL_MODELS: "true",
      },
    });
    return { code: p.exitCode, out: p.stdout.toString(), err: p.stderr.toString() };
  }

  it("--dry-run --json emits the machine-readable object; the real run compacts; refusal exits 1", () => {
    const { dir, dbPath } = tmpVault();
    const s = seededSmallChunkStore(dbPath);
    const snapBefore = snapshot(s.db);
    s.close();

    const dry = run(dbPath, dir, ["--dry-run", "--json", "-c", "beta"]);
    expect(dry.code).toBe(0);
    const j = JSON.parse(dry.out);
    expect(j).toMatchObject({
      dryRun: true,
      dim: DIM,
      chunkSize: CHUNK,
      totalVectors: COLS.length * DOCS * FRAGS + 1,
    });
    expect(j.collections.map((c: { name: string }) => c.name)).toContain("beta");
    expect(typeof j.collections[0].currentSpan).toBe("number");
    expect(typeof j.elapsedMs).toBe("number");

    const check = createStore(dbPath, { readonly: true });
    expect(snapshot(check.db)).toEqual(snapBefore);
    check.close();

    const real = run(dbPath, dir, []);
    expect(real.code).toBe(0);
    expect(real.out).toContain("collection");
    expect(real.out).toContain("elapsed");
    const after = createStore(dbPath, { readonly: true });
    expect(snapshot(after.db).pairs).toEqual(snapBefore.pairs);
    expect(span(after.db, "alpha")).toBe(3);
    after.close();

    // no vectors_vec -> refusal, non-zero exit
    const empty = tmpVault();
    createStore(empty.dbPath).close();
    const refused = run(empty.dbPath, empty.dir, []);
    expect(refused.code).toBe(1);
    expect(refused.err).toContain("vec-compact refused");
    // Three CLI subprocesses: bun's 5 s default timed out under host load.
  }, 60_000);
});

beforeAll(() => {
  setDefaultLlamaCpp({
    embed: async () => ({ embedding: [], model: MODEL }),
    query: async () => null,
    generate: async () => null,
    expandQuery: async () => [],
  } as any);
});
