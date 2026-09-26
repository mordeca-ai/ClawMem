/**
 * master-harness-b1q42.82 — searchFTS join-order pin.
 *
 * With a plain `FROM documents_fts f JOIN documents d`, bun's bundled SQLite 3.51.x planner
 * chooses `documents` as the OUTER loop whenever a predicate on `d` looks selective (a
 * collection scope via idx_documents_collection, or post-ANALYZE a full scan) and then
 * re-runs the FTS MATCH once per document row (`SCAN f VIRTUAL TABLE INDEX 0:=M3`). On a
 * 21k-doc collection that took ~38s per query. `CROSS JOIN` pins the FTS scan outer.
 *
 * These tests EXPLAIN QUERY PLAN the exact statement searchFTS runs (via the exported
 * buildSearchFTSSql) on a fixture built through the store's own schema path, and assert the
 * FTS virtual-table scan is the outer loop without the per-row `=` constraint. A sensitivity
 * guard proves the fixture still provokes the bad plan for the un-pinned SQL, so the plan
 * assertion is not vacuous.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import {
  createStore,
  insertContent,
  insertDocument,
  searchFTS,
  buildSearchFTSSql,
  buildFTS5Query,
  type SearchFTSFilters,
  type Store,
} from "../../src/store.ts";

const BIG = "bigcoll";
const SMALL = "smallcoll";
const DOCS = 300;

function seed(store: Store): void {
  const now = new Date().toISOString();
  store.db.exec("BEGIN");
  for (let i = 0; i < DOCS; i++) {
    const collection = i % 10 === 0 ? SMALL : BIG;
    const hash = `joinorder_${i}`;
    const body = `document ${i} alpha beta${i % 7 === 0 ? " integrator lease guard" : ""}`;
    insertContent(store.db, hash, body, now);
    insertDocument(store.db, collection, `notes/doc-${i}.md`, `Doc ${i}`, hash, now, now);
  }
  store.db.exec("COMMIT");
}

function planRows(store: Store, sql: string, params: (string | number)[]): string[] {
  return (store.db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...params) as { detail: string }[]).map((r) => r.detail);
}

/** The FTS table is the outer loop and is driven by MATCH alone (no per-row `=` rowid constraint). */
function ftsIsOuter(plan: string[]): boolean {
  const first = plan[0] ?? "";
  return first.startsWith("SCAN f VIRTUAL TABLE") && !first.includes(":=");
}

const SCOPES: { name: string; filters: SearchFTSFilters }[] = [
  { name: "large collection scope", filters: { collections: [BIG] } },
  { name: "small collection scope", filters: { collections: [SMALL] } },
  { name: "multi-collection scope", filters: { collections: [BIG, SMALL] } },
  { name: "unscoped", filters: {} },
  { name: "excluded collections", filters: { excludeCollections: [SMALL] } },
  { name: "date range + scope", filters: { collections: [BIG], dateRange: { start: "2000-01-01T00:00:00Z", end: "2999-01-01T00:00:00Z" } } },
];

for (const analyzed of [false, true]) {
  describe(`searchFTS join order (${analyzed ? "after ANALYZE" : "no stats"})`, () => {
    let store: Store;
    const ftsQuery = buildFTS5Query("integrator lease guard")!;

    beforeAll(() => {
      store = createStore(":memory:");
      seed(store);
      if (analyzed) store.db.exec("ANALYZE");
    });
    afterAll(() => store.close());

    // The live vault carries no sqlite_stat1 (never ANALYZEd), so the no-stats fixture is the
    // faithful reproduction; post-ANALYZE the un-pinned plan depends on the stats, so the guard
    // only runs without them. The pinned-plan assertions run in both modes.
    it.skipIf(analyzed)("sensitivity guard: the un-pinned JOIN gets the documents-outer plan on this fixture", () => {
      const { sql, params } = buildSearchFTSSql(ftsQuery, 20, { collections: [BIG] });
      const unpinned = sql.replace(/CROSS JOIN documents d/, "JOIN documents d");
      const plan = planRows(store, unpinned, params);
      expect(ftsIsOuter(plan)).toBe(false);
    });

    for (const { name, filters } of SCOPES) {
      it(`pins the FTS scan as the outer loop — ${name}`, () => {
        const { sql, params } = buildSearchFTSSql(ftsQuery, 20, filters);
        const plan = planRows(store, sql, params);
        expect({ first: plan[0], ftsOuter: ftsIsOuter(plan) }).toEqual({ first: plan[0], ftsOuter: true });
      });
    }

    it("returns the same rows as the un-pinned SQL (no behaviour change)", () => {
      for (const { filters } of SCOPES) {
        const { sql, params } = buildSearchFTSSql(ftsQuery, 50, filters);
        const unpinned = sql.replace(/CROSS JOIN documents d/, "JOIN documents d");
        const pinnedRows = store.db.prepare(sql).all(...params) as { filepath: string; bm25_score: number }[];
        const oldRows = store.db.prepare(unpinned).all(...params) as { filepath: string; bm25_score: number }[];
        const key = (r: { filepath: string; bm25_score: number }) => `${r.bm25_score}\u0000${r.filepath}`;
        expect(pinnedRows.map(key).sort()).toEqual(oldRows.map(key).sort());
      }
    });

    it("searchFTS scoped to a collection still returns only that collection's matches", () => {
      const results = searchFTS(store.db, "integrator lease guard", 100, undefined, [SMALL]);
      // docs where i % 70 === 0 (i % 7 === 0 AND i % 10 === 0) → 0,70,...,280 = 5
      expect(results.length).toBe(5);
      for (const r of results) expect(r.collectionName).toBe(SMALL);
    });
  });
}
