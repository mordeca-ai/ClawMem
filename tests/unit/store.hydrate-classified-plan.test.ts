import { describe, it, expect } from "bun:test";

/**
 * master-harness-apktc — hydrateVecResultsClassified must be index-served.
 *
 * hxa17 removed the computed-expression predicate `WHERE cv.hash || '_' || cv.seq IN (...)`
 * from hydrateVecResults(); the identical predicate survived in the classified hydrate used
 * by searchVecDetailedWithVector(). No index can serve a computed expression, so the planner
 * inverted the join and read `SCAN d USING INDEX idx_documents_effective_time` — a full scan
 * of `documents`. The rewrite filters on the indexed `cv.hash`, so the plan must SEARCH.
 *
 * The SQL under test comes from buildClassifiedHydrateQuery(), the exact builder
 * hydrateVecResultsClassified() executes, across every optional predicate it can append.
 */

import { createStore, buildClassifiedHydrateQuery, searchVecDetailedWithVector, type Store, type VecSearchDetailedOpts } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";

const MODEL = "test-model";
const DIM = 4;

function seed(store: Store, n: number): string[] {
  const hashSeqs: string[] = [];
  const now = new Date().toISOString();
  store.ensureVecTable(DIM);
  for (let i = 0; i < n; i++) {
    const col = i % 2 === 0 ? "user" : "_clawmem";
    const path = `doc${i}.md`;
    const body = `body ${col}/${path}`;
    const hash = hashContent(body);
    store.insertContent(hash, body, now);
    store.insertDocument(col, path, path, hash, now, now);
    for (let seq = 0; seq < 3; seq++) {
      store.insertEmbedding(hash, seq, seq, new Float32Array([1, i / n, seq / 3, 0]), MODEL, now, "section", undefined, `${col}/${path}`);
      hashSeqs.push(`${hash}_${seq}`);
    }
  }
  return hashSeqs;
}

function plan(store: Store, opts: VecSearchDetailedOpts, hashSeqs: string[]): string[] {
  const { sql, params } = buildClassifiedHydrateQuery(hashSeqs, opts);
  return (store.db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...params) as { detail: string }[]).map(r => r.detail);
}

const VARIANTS: [string, VecSearchDetailedOpts][] = [
  ["no filters", {}],
  ["collections filter", { collections: ["user"] }],
  ["dateRange filter", { dateRange: { start: "2000-01-01T00:00:00Z", end: "2100-01-01T00:00:00Z" } }],
  ["observationsOnly", { observationsOnly: true }],
];

describe("hydrateVecResultsClassified query plan (master-harness-apktc)", () => {
  const store = createStore(":memory:");
  const hashSeqs = seed(store, 60);
  const requested = hashSeqs.filter((_, i) => i % 2 === 0); // a subset, like a real MATCH window

  for (const [name, opts] of VARIANTS) {
    it(`${name}: SEARCHes documents via an index, never SCANs documents`, () => {
      const detail = plan(store, opts, requested);
      // Unfiltered / dateRange / observationsOnly: `SEARCH d USING INDEX idx_documents_hash (hash=?)`
      // (the old predicate SCANned d here). With a collection filter the planner may prefer
      // idx_documents_collection — still a SEARCH, under both the old and new predicate.
      expect(detail.some(d => /^SEARCH d USING (COVERING )?INDEX /.test(d))).toBe(true);
      if (!opts.collections) expect(detail).toContain("SEARCH d USING INDEX idx_documents_hash (hash=?)");
      expect(detail.filter(d => /^SCAN d\b/.test(d))).toEqual([]);
    });
  }

  it("the predicate filters on the indexed column, with one parameter per distinct hash", () => {
    const { sql, params } = buildClassifiedHydrateQuery(requested, {});
    expect(sql).not.toContain("cv.hash || '_' || cv.seq IN");
    expect(sql).toMatch(/WHERE cv\.hash IN \(/);
    expect(params.length).toBe(new Set(requested.map(hs => hs.slice(0, hs.lastIndexOf("_")))).size);
  });

  it("classified search still returns deduped, excluded-accounted results", () => {
    const out = searchVecDetailedWithVector(store.db, { embedding: new Float32Array([1, 0, 0, 0]), endpointModel: MODEL }, 5, { excludeCollections: ["_clawmem"] });
    expect(out.results.length).toBe(5);
    expect(out.results.every(r => r.collectionName === "user")).toBe(true);
    expect(new Set(out.results.map(r => r.filepath)).size).toBe(5);
    expect(out.excludedDocsSeen).toBeGreaterThan(0);
  });
});
