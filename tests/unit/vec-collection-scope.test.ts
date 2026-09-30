import { describe, it, expect, beforeAll, afterAll } from "bun:test";

/**
 * vn4rz.69 — include-scoped vector search must fill `limit` in-scope docs.
 *
 * Regression: searchVec()/searchVecDetailed() post-filtered the GLOBAL top limit*3 fragments by
 * collection, and the escalation loop ignored include scopes. A collection that is a small
 * minority of the vault (memory-topics = 0.7% of the live vault) collapsed to ~1 doc because
 * almost none of its fragments sit in the global top window (Recall@10 0.89 -> 0.11).
 *
 * Fixture: a large "big" collection sits nearest the query; every "small" doc sits beyond the
 * global top limit*3. Expected results are computed by brute force over the scope, so the test
 * pins both COUNT (fills limit) and ORDER (exact nearest in-scope docs, total order
 * (bestDist, filepath)).
 */

import {
  createStore,
  searchVec,
  searchVecDetailed,
  searchVecDetailedWithVector,
  type Store,
} from "../../src/store.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";
import { hashContent } from "../../src/indexer.ts";

const MODEL = "scope-test-model";
const DIM = 4;
const OLD = "2020-01-15T00:00:00.000Z";
const NEW = "2026-01-15T00:00:00.000Z";
const OLD_WINDOW = { start: "2020-01-01T00:00:00.000Z", end: "2020-12-31T23:59:59.999Z" };

function vecAt(angleRad: number): Float32Array {
  return new Float32Array([Math.cos(angleRad), Math.sin(angleRad), 0, 0]);
}
const QUERY_VEC = vecAt(0);
const qv = { embedding: QUERY_VEC, endpointModel: MODEL };

// Every query embeds to QUERY_VEC, so searchVec/searchVecDetailed run the real code path
// (embed -> guard -> MATCH -> hydrate) without an embed server.
const fakeLlm = {
  embed: async () => ({ embedding: Array.from(QUERY_VEC), model: MODEL }),
  query: async () => null,
  generate: async () => null,
  expandQuery: async () => [],
} as any;

/** filepath -> best (smallest) angle among its fragments; cosine distance is monotone in angle. */
const bestAngle = new Map<string, number>();

function seed(store: Store, col: string, path: string, angles: number[], modifiedAt = NEW): void {
  const body = `body of ${col}/${path}`;
  const hash = hashContent(body + col + path);
  store.insertContent(hash, body, modifiedAt);
  store.insertDocument(col, path, path, hash, modifiedAt, modifiedAt);
  store.ensureVecTable(DIM);
  angles.forEach((a, seq) => {
    store.insertEmbedding(
      hash,
      seq,
      seq,
      vecAt(a),
      MODEL,
      modifiedAt,
      "section",
      undefined,
      `${col}/${path}`,
    );
  });
  bestAngle.set(`${col}/${path}`, Math.min(...angles));
}

/** Brute-force expected order over a scope: nearest best fragment first, displayPath tiebreak. */
function expected(pred: (displayPath: string) => boolean, limit: number): string[] {
  return [...bestAngle.entries()]
    .filter(([p]) => pred(p))
    .sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, limit)
    .map(([p]) => p);
}

const isSmall = (p: string) => p.startsWith("small/");
const isOld = (p: string) => p.startsWith("small/") || p.startsWith("archive/");

let store: Store;

beforeAll(() => {
  setDefaultLlamaCpp(fakeLlm);
  store = createStore(":memory:");
  // 80 "big" docs nearest the query (angles 0.01..0.80) — they own the global top 60.
  for (let i = 0; i < 80; i++)
    seed(store, "big", `b${String(i).padStart(2, "0")}.md`, [0.01 + i * 0.01]);
  // 12 "small" docs, all beyond the big block, each with 3 fragments (dedupe matters), seeded
  // in a scrambled order so insertion order can't masquerade as distance order. Two docs share
  // an exact best angle to exercise the (bestDist, filepath) tiebreak.
  const smallBase = [1.3, 1.02, 1.18, 1.1, 1.26, 1.06, 1.22, 1.14, 1.34, 1.06, 1.38, 1.42];
  smallBase.forEach((base, i) => {
    seed(
      store,
      "small",
      `s${String(i).padStart(2, "0")}.md`,
      [base + 0.02, base, base + 0.01],
      OLD,
    );
  });
  // An old doc OUTSIDE the "small" collection, for dateRange scope and include+exclude combos.
  seed(store, "archive", "a00.md", [1.08], OLD);
});

afterAll(() => {
  setDefaultLlamaCpp(null);
  store?.close();
});

describe("include-scoped vector search fills limit (vn4rz.69)", () => {
  it("fixture sanity: no in-scope fragment is in the global top limit*3", () => {
    const unscoped = searchVecDetailedWithVector(store.db, qv, 20, {});
    expect(unscoped.results.every((r) => r.collectionName === "big")).toBe(true);
  });

  it("searchVec(collections) returns `limit` in-scope docs, exact nearest-first", async () => {
    const res = await searchVec(store.db, "q", MODEL, 5, undefined, ["small"]);
    expect(res.map((r) => r.displayPath)).toEqual(expected(isSmall, 5));
    expect(res.every((r) => r.collectionName === "small")).toBe(true);
  });

  it("searchVec(collections) returns ALL in-scope docs when fewer than limit", async () => {
    const res = await searchVec(store.db, "q", MODEL, 20, undefined, ["small"]);
    expect(res.length).toBe(12);
    expect(res.map((r) => r.displayPath)).toEqual(expected(isSmall, 20));
  });

  it("searchVecDetailed({collections}) fills limit, not degraded", async () => {
    const det = await searchVecDetailed(store.db, "q", MODEL, 5, { collections: ["small"] });
    expect(det.results.map((r) => r.displayPath)).toEqual(expected(isSmall, 5));
    expect(det.degraded).toBe(false);
    const scores = det.results.map((r) => r.score);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
  });

  it("searchVec(dateRange) scope fills limit across collections", async () => {
    const res = await searchVec(store.db, "q", MODEL, 6, undefined, undefined, OLD_WINDOW);
    expect(res.map((r) => r.displayPath)).toEqual(expected(isOld, 6));
  });

  it("include + exclude combine: excluded docs never appear, limit still fills", () => {
    const det = searchVecDetailedWithVector(store.db, qv, 4, {
      collections: ["small", "archive"],
      excludeCollections: ["archive"],
    });
    expect(det.results.map((r) => r.displayPath)).toEqual(expected(isSmall, 4));
    expect(det.excludedDocsSeen).toBe(1);
  });

  it("scope too large for the pre-filter falls back to escalation, which now engages for include scopes", () => {
    const det = searchVecDetailedWithVector(store.db, qv, 5, {
      collections: ["small"],
      prefilterMaxScope: 0,
    });
    expect(det.results.map((r) => r.displayPath)).toEqual(expected(isSmall, 5));
    expect(det.scannedFragments).toBeGreaterThan(15); // escalated past the initial k = limit*3
  });

  it("unscoped searchVec is unchanged: global nearest docs", async () => {
    const res = await searchVec(store.db, "q", MODEL, 5);
    expect(res.map((r) => r.displayPath)).toEqual([
      "big/b00.md",
      "big/b01.md",
      "big/b02.md",
      "big/b03.md",
      "big/b04.md",
    ]);
  });
});
