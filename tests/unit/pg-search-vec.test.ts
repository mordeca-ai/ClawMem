/**
 * PG vector read path — the cheap tier (master-harness-2wx75).
 *
 * NO DATABASE. Every case drives the REAL exported functions from
 * src/pg/search.ts; the DB is replaced by a RECORDING FAKE that captures the
 * SQL text and the bind values, which is what makes the query CONTRACT (the
 * `<=>` order-by that is the only shape the HNSW index can serve, the
 * `d.active = true` fence, the collection filter, the limit) assertable at all —
 * a live plan proves it for one corpus on one day; these assertions hold for
 * every caller.
 *
 * WHAT THIS TIER CANNOT PROVE, stated plainly rather than implied: that
 * PostgreSQL honours those predicates. `active = true` excluding an inactive row
 * is the engine's behaviour, not this module's, and it is proven by the live
 * smoke + (when it lands) the integration tier — here it is proven only that the
 * predicate is IN the query the engine is asked to run, which is the half that
 * can regress in this file.
 */

import { describe, it, expect } from "bun:test";
import {
  assertQueryModelMatchesStored,
  buildVecSearchQuery,
  dedupeToSearchResults,
  getStoredVecModels,
  normalizeCollections,
  type PgQueryable,
  type PgVecRow,
} from "../../src/pg/search.ts";
import { PgVecReadModelMismatchError } from "../../src/pg/errors.ts";

/** A `pg` client stand-in that records what it was asked and replays fixtures. */
function fakeClient(rowsFor: (sql: string, values: unknown[]) => unknown[]): PgQueryable & {
  calls: { text: string; values: unknown[] }[];
} {
  const calls: { text: string; values: unknown[] }[] = [];
  return {
    calls,
    async query(text: string, values: unknown[] = []) {
      calls.push({ text, values });
      return { rows: rowsFor(text, values) as never[] };
    },
  };
}

function row(over: Partial<PgVecRow> = {}): PgVecRow {
  return {
    hash: "a".repeat(64), seq: 0, pos: 0,
    fragment_type: "section", fragment_label: "Intro",
    collection: "research", path: "a/b.md", title: "B",
    modified_at: "2026-09-01T00:00:00.000Z", body: "hello",
    distance: 0.25, ...over,
  };
}

describe("buildVecSearchQuery — the SQL contract", () => {
  it("orders by the pgvector cosine operator on the raw column (the only HNSW-servable shape)", () => {
    const q = buildVecSearchQuery("[0.1,0.2]", null, 64);
    // Not `ORDER BY distance` (the projection alias) and not a wrapped column:
    // either one silently becomes a seq scan over every fragment in the vault.
    expect(q.text).toMatch(/ORDER BY\s+cv\.embedding <=> \$1::vector/);
  });

  it("fences to active, non-invalidated documents", () => {
    const q = buildVecSearchQuery("[0.1]", null, 64);
    expect(q.text).toContain("d.active = true");
    expect(q.text).toContain("d.invalidated_at IS NULL");
  });

  it("omits the collection filter when none is requested, and binds only vector + limit", () => {
    const q = buildVecSearchQuery("[0.1]", null, 64);
    // `d.collection` is in the SELECT list either way; what must be absent is
    // the FILTER.
    expect(q.text).not.toContain("ANY(");
    expect(q.values).toEqual(["[0.1]", 64]);
    expect(q.text).toContain("LIMIT $2");
  });

  it("adds a parameterized collection filter when collections are requested", () => {
    const q = buildVecSearchQuery("[0.1]", ["research", "decisions"], 40);
    expect(q.text).toContain("d.collection = ANY($2::text[])");
    expect(q.values).toEqual(["[0.1]", ["research", "decisions"], 40]);
    expect(q.text).toContain("LIMIT $3");
    // Nothing is ever interpolated into SQL text.
    expect(q.text).not.toContain("research");
  });

  it("applies the fragment limit as a bind parameter, not literal text", () => {
    expect(buildVecSearchQuery("[0.1]", null, 7).values.at(-1)).toBe(7);
  });
});

describe("normalizeCollections", () => {
  it("treats undefined as no filter", () => expect(normalizeCollections(undefined)).toBeNull());
  it("wraps a single name", () => expect(normalizeCollections("research")).toEqual(["research"]));
  it("trims and drops empties, and an all-empty list is no filter", () => {
    expect(normalizeCollections([" research ", ""])).toEqual(["research"]);
    expect(normalizeCollections([" "])).toBeNull();
  });
});

describe("getStoredVecModels — scoped to the rows the search will read", () => {
  it("scopes the DISTINCT to active documents in the requested collections", async () => {
    const c = fakeClient(() => [{ model: "embeddinggemma" }]);
    expect(await getStoredVecModels(c, ["research"])).toEqual(["embeddinggemma"]);
    const call = c.calls[0]!;
    expect(call.text).toContain("SELECT DISTINCT cv.model");
    expect(call.text).toContain("d.active = true");
    expect(call.text).toContain("d.collection = ANY($1::text[])");
    expect(call.values).toEqual([["research"]]);
  });

  it("drops the filter when no collection is requested", async () => {
    const c = fakeClient(() => []);
    await getStoredVecModels(c, null);
    expect(c.calls[0]!.text).not.toContain("ANY(");
    expect(c.calls[0]!.values).toEqual([]);
  });
});

describe("assertQueryModelMatchesStored — THE FENCE, and it must be able to go RED", () => {
  it("THROWS on a model mismatch, naming BOTH models", () => {
    // The negative test the guard exists for. A guard whose test cannot fail is
    // not a guard: delete the throw in assertQueryModelMatchesStored and this
    // case goes red, which is exactly the property being bought here.
    let caught: unknown;
    try {
      assertQueryModelMatchesStored(["embeddinggemma"], "nomic-embed-text", "research");
    } catch (e) { caught = e; }
    expect(caught).toBeInstanceOf(PgVecReadModelMismatchError);
    const err = caught as PgVecReadModelMismatchError;
    expect(err.message).toContain("embeddinggemma");
    expect(err.message).toContain("nomic-embed-text");
    expect(err.message).toContain("research");
    expect(err.storedModels).toEqual(["embeddinggemma"]);
    expect(err.queryModel).toBe("nomic-embed-text");
  });

  it("THROWS on a heterogeneous stored set even when the query model is one of them", () => {
    // There is no "mostly comparable". Half the corpus in another vector space
    // is a silently wrong ranking, not a partial one.
    expect(() => assertQueryModelMatchesStored(["a", "embeddinggemma"], "embeddinggemma", "all"))
      .toThrow(PgVecReadModelMismatchError);
  });

  it("passes when the single stored model is the query model", () => {
    expect(() => assertQueryModelMatchesStored(["embeddinggemma"], "embeddinggemma", "research"))
      .not.toThrow();
  });

  it("no-ops on an empty stored set — nothing embedded yet cannot disagree", () => {
    expect(() => assertQueryModelMatchesStored([], "embeddinggemma", "research")).not.toThrow();
  });
});

describe("dedupeToSearchResults — the SearchResult contract", () => {
  it("returns the sqlite shape, with score = 1 - cosine distance", () => {
    const [r] = dedupeToSearchResults([row({ distance: 0.25 })], 10);
    expect(r).toMatchObject({
      filepath: "clawmem://research/a/b.md",
      displayPath: "research/a/b.md",
      collectionName: "research",
      source: "vec",
      score: 0.75,
      chunkPos: 0,
      fragmentType: "section",
      fragmentLabel: "Intro",
      context: null,
    });
    expect(r!.docid).toBe("aaaaaa");
    expect(r!.bodyLength).toBe("hello".length);
  });

  it("keeps the best-scoring fragment per document", () => {
    const h = "b".repeat(64);
    const out = dedupeToSearchResults(
      [row({ hash: h, seq: 0, distance: 0.6 }), row({ hash: h, seq: 3, pos: 99, distance: 0.1 })],
      10,
    );
    expect(out).toHaveLength(1);
    expect(out[0]!.chunkPos).toBe(99);
    expect(out[0]!.score).toBeCloseTo(0.9, 10);
  });

  it("breaks exact distance ties by filepath, so recall does not track row order", () => {
    const out = dedupeToSearchResults([
      row({ path: "z.md", distance: 0.5 }),
      row({ path: "a.md", distance: 0.5 }),
    ], 10);
    expect(out.map(r => r.displayPath)).toEqual(["research/a.md", "research/z.md"]);
  });

  it("honours the document limit after dedup", () => {
    const rows = [1, 2, 3, 4].map(i => row({ path: `${i}.md`, distance: i / 10 }));
    expect(dedupeToSearchResults(rows, 2).map(r => r.displayPath))
      .toEqual(["research/1.md", "research/2.md"]);
  });

  it("parses a numeric distance delivered as a string by the pg driver", () => {
    expect(dedupeToSearchResults([row({ distance: "0.25" })], 1)[0]!.score).toBe(0.75);
  });
});

describe("inactive documents are excluded", () => {
  /**
   * Honest about what layer this is. The fake stands in for the ENGINE: it
   * refuses to apply the `active` filter unless the query actually asked for it,
   * then applies it to a fixture. So the case fails if the predicate is dropped
   * from buildVecSearchQuery (the regression this file can catch), and it does
   * not pretend to re-prove that PostgreSQL honours a WHERE clause.
   */
  it("never surfaces a row whose document is inactive", () => {
    const fixture = [
      { ...row({ path: "live.md", distance: 0.2 }), active: true },
      { ...row({ path: "retired.md", distance: 0.1 }), active: false },
    ];
    const q = buildVecSearchQuery("[0.1]", ["research"], 64);
    expect(q.text).toContain("d.active = true"); // else the engine would return both
    const served = fixture.filter(r => (q.text.includes("d.active = true") ? r.active : true));
    const out = dedupeToSearchResults(served as PgVecRow[], 10);
    expect(out.map(r => r.displayPath)).toEqual(["research/live.md"]);
  });
});
