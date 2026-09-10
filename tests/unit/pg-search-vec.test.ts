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
  DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS,
  assertQueryModelMatchesStored,
  buildVecSearchQuery,
  dedupeToSearchResults,
  getStoredVecModels,
  normalizeCollections,
  pgSearchVec,
  pgSearchVecDetailed,
  type PgQueryable,
  type PgVecEmbedder,
  type PgVecRow,
} from "../../src/pg/search.ts";
import { PgVecReadModelMismatchError, PgVecSearchTimeoutError } from "../../src/pg/errors.ts";

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

// ===========================================================================
// pgSearchVec — the WIRING (master-harness-2wx75 slice 2, GAP 1 + GAP 4/8)
//
// Everything above tests pure functions. That left the orchestration itself —
// which of those functions pgSearchVec actually CALLS, and in what transaction
// shape — proven by reading. A refactor that dropped the fence call, or that
// dropped the statement_timeout, would leave every case above green. These
// cases fail on exactly that, and the embedder is an injected fake so no GPU
// endpoint is involved.
// ===========================================================================

/** An embedder that answers with a fixed vector + model, and records its calls. */
function fakeEmbedder(model: string, embedding = [0.1, 0.2]): PgVecEmbedder & {
  calls: { text: string; isQuery?: boolean }[];
} {
  const calls: { text: string; isQuery?: boolean }[] = [];
  return {
    calls,
    async embed(text, options) {
      calls.push({ text, isQuery: options?.isQuery });
      return { embedding, model };
    },
  };
}

/**
 * A recording fake that answers the whole pgSearchVec conversation: the
 * transaction verbs, the DISTINCT model fence, and the ANN scan.
 */
function searchClient(opts: { storedModels: string[]; rows?: PgVecRow[] }) {
  const calls: { text: string; values: unknown[] }[] = [];
  const client: PgQueryable & { calls: typeof calls } = {
    calls,
    async query(text: string, values: unknown[] = []) {
      calls.push({ text, values });
      if (text.includes("SELECT DISTINCT cv.model")) {
        return { rows: opts.storedModels.map(m => ({ model: m })) as never[] };
      }
      if (text.includes("<=>")) return { rows: (opts.rows ?? []) as never[] };
      return { rows: [] as never[] };
    },
  };
  return client;
}

const sqlOf = (c: { calls: { text: string }[] }) => c.calls.map(x => x.text.trim());

describe("pgSearchVec wires the fence — the call site, not just the function", () => {
  it("REFUSES a search whose query model differs from the stored model", async () => {
    // THE GAP-1 CASE. assertQueryModelMatchesStored has its own unit cases, but
    // they all pass if pgSearchVec never calls it. Delete the call site in
    // src/pg/search.ts and this case goes red (verified by doing exactly that).
    const c = searchClient({ storedModels: ["embeddinggemma"] });
    let caught: unknown;
    try {
      await pgSearchVec(c, "hello", { embedder: fakeEmbedder("nomic-embed-text") });
    } catch (e) { caught = e; }
    expect(caught).toBeInstanceOf(PgVecReadModelMismatchError);
    expect((caught as Error).message).toContain("embeddinggemma");
    expect((caught as Error).message).toContain("nomic-embed-text");
    // And it must REFUSE, not degrade: the ANN scan never ran.
    expect(sqlOf(c).some(t => t.includes("<=>"))).toBe(false);
  });

  it("REFUSES a heterogeneous stored set through the same call site", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma", "other-model"] });
    await expect(pgSearchVec(c, "hello", { embedder: fakeEmbedder("embeddinggemma") }))
      .rejects.toBeInstanceOf(PgVecReadModelMismatchError);
  });

  it("runs the ANN scan when the models agree, embedding the QUERY-side format", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma"], rows: [row({ path: "hit.md" })] });
    const emb = fakeEmbedder("embeddinggemma");
    const out = await pgSearchVec(c, "hello", { embedder: emb });
    expect(out.map(r => r.displayPath)).toEqual(["research/hit.md"]);
    expect(emb.calls).toEqual([{ text: "task: search result | query: hello", isQuery: true }]);
  });

  it("returns empty without embedding anything when nothing is stored in scope", async () => {
    const c = searchClient({ storedModels: [] });
    const emb = fakeEmbedder("embeddinggemma");
    expect(await pgSearchVec(c, "hello", { embedder: emb })).toEqual([]);
    expect(emb.calls).toHaveLength(0);
  });
});

describe("pgSearchVec bounds both SQL legs with transaction-local GUCs", () => {
  it("wraps each leg in BEGIN/COMMIT with SET LOCAL — never a session SET + RESET", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma"], rows: [row()] });
    await pgSearchVec(c, "hello", { embedder: fakeEmbedder("embeddinggemma") });
    const sql = sqlOf(c);
    // GAP 8: a session-level SET with a finally-RESET leaks onto the next
    // checkout of a POOLED connection. Neither verb may appear at all.
    expect(sql.some(t => /^SET (?!LOCAL)/.test(t))).toBe(false);
    expect(sql.some(t => t.startsWith("RESET"))).toBe(false);
    expect(sql.filter(t => t === "BEGIN")).toHaveLength(2);
    expect(sql.filter(t => t === "COMMIT")).toHaveLength(2);
    // GAP 4: the bound is present on BOTH legs, at the documented default.
    expect(sql.filter(t => t === `SET LOCAL statement_timeout = ${DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS}`))
      .toHaveLength(2);
    expect(sql.filter(t => t === "SET LOCAL hnsw.iterative_scan = strict_order")).toHaveLength(2);
    // The default itself must stay inside the p95 <= 1800 ms hook budget.
    expect(DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS).toBeLessThan(1800);
  });

  it("honours an explicit statementTimeoutMs, and 0 means PostgreSQL's no-bound", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma"], rows: [] });
    await pgSearchVec(c, "hello", { embedder: fakeEmbedder("embeddinggemma"), statementTimeoutMs: 250 });
    expect(sqlOf(c).filter(t => t === "SET LOCAL statement_timeout = 250")).toHaveLength(2);

    const c0 = searchClient({ storedModels: ["embeddinggemma"], rows: [] });
    await pgSearchVec(c0, "hello", { embedder: fakeEmbedder("embeddinggemma"), statementTimeoutMs: 0 });
    expect(sqlOf(c0).filter(t => t === "SET LOCAL statement_timeout = 0")).toHaveLength(2);
  });

  it("refuses a non-integer / negative timeout instead of interpolating it into SQL", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma"] });
    await expect(pgSearchVec(c, "hello", {
      embedder: fakeEmbedder("embeddinggemma"), statementTimeoutMs: -1,
    })).rejects.toThrow(/non-negative integer/);
    // Nothing was smuggled into a SET LOCAL.
    expect(sqlOf(c).some(t => t.includes("statement_timeout"))).toBe(false);
  });

  it("survives a pgvector too old to know hnsw.iterative_scan (savepoint, not fatal)", async () => {
    const calls: string[] = [];
    const c: PgQueryable = {
      async query(text: string, values: unknown[] = []) {
        calls.push(text.trim());
        if (text.includes("hnsw.iterative_scan")) {
          const e = Object.assign(new Error('unrecognized configuration parameter'), { code: "42704" });
          throw e;
        }
        if (text.includes("SELECT DISTINCT cv.model")) return { rows: [{ model: "embeddinggemma" }] as never[] };
        if (text.includes("<=>")) return { rows: [row({ path: "old.md" })] as never[] };
        return { rows: [] as never[] };
      },
    };
    const out = await pgSearchVec(c, "hello", { embedder: fakeEmbedder("embeddinggemma") });
    expect(out.map(r => r.displayPath)).toEqual(["research/old.md"]);
    expect(calls.filter(t => t === "ROLLBACK TO SAVEPOINT clawmem_hnsw_guc")).toHaveLength(2);
    expect(calls.filter(t => t === "COMMIT")).toHaveLength(2);
  });

  it("surfaces a cancelled statement as PgVecSearchTimeoutError, not a raw driver throw", async () => {
    const c: PgQueryable = {
      async query(text: string) {
        if (text.includes("<=>")) {
          throw Object.assign(new Error("canceling statement due to statement timeout"), { code: "57014" });
        }
        if (text.includes("SELECT DISTINCT cv.model")) return { rows: [{ model: "embeddinggemma" }] as never[] };
        return { rows: [] as never[] };
      },
    };
    let caught: unknown;
    try {
      await pgSearchVec(c, "hello", { embedder: fakeEmbedder("embeddinggemma"), statementTimeoutMs: 5 });
    } catch (e) { caught = e; }
    expect(caught).toBeInstanceOf(PgVecSearchTimeoutError);
    expect((caught as PgVecSearchTimeoutError).stage).toBe("ann-scan");
    expect((caught as PgVecSearchTimeoutError).timeoutMs).toBe(5);
  });
});

// ===========================================================================
// THE DEGRADED CHANNEL (master-harness-2wx75 GAP 7).
//
// pgSearchVec collapses five distinct outcomes into the same `[]`. These cases
// pin the typed channel that separates them, and the two CONTROLS are what make
// the rest mean anything: `no-stored-vectors` (we could not look) and
// genuine-empty (we looked, nothing matched) both return zero results, and they
// MUST NOT be the same value.
// ===========================================================================

/** An embedder that reports being down: no embedding at all. */
const deadEmbedder: PgVecEmbedder = { async embed() { return null; } };

/** An embedder that takes `ms` to answer — used to burn the wall-clock budget. */
function slowEmbedder(model: string, ms: number): PgVecEmbedder {
  return {
    async embed() {
      await new Promise(r => setTimeout(r, ms));
      return { embedding: [0.1, 0.2], model };
    },
  };
}

describe("pgSearchVecDetailed — degraded reasons are distinguishable", () => {
  it("no-stored-vectors: the fence found nothing embedded in scope", async () => {
    const c = searchClient({ storedModels: [] });
    const out = await pgSearchVecDetailed(c, "hello", { embedder: fakeEmbedder("embeddinggemma") });
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("no-stored-vectors");
    expect(out.results).toEqual([]);
    expect(out.scannedFragments).toBe(0);
    expect(out.storedModels).toBe(0);
    // Nothing was embedded and no ANN scan ran.
    expect(out.embedModel).toBeUndefined();
    expect(sqlOf(c).some(t => t.includes("<=>"))).toBe(false);
  });

  it("budget-exhausted-pre-embed: timeoutMs was spent before the embed leg", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma"], rows: [row()] });
    const emb = fakeEmbedder("embeddinggemma");
    const out = await pgSearchVecDetailed(c, "hello", { embedder: emb, timeoutMs: 0 });
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("budget-exhausted-pre-embed");
    expect(out.results).toEqual([]);
    expect(out.scannedFragments).toBe(0);
    // The fence DID run and DID find a model — that is what makes this reason
    // different from no-stored-vectors.
    expect(out.storedModels).toBe(1);
    expect(out.embedModel).toBeUndefined();
    expect(emb.calls).toHaveLength(0);
    expect(sqlOf(c).some(t => t.includes("<=>"))).toBe(false);
  });

  it("embed-unavailable: the endpoint returned no embedding", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma"], rows: [row()] });
    const out = await pgSearchVecDetailed(c, "hello", { embedder: deadEmbedder });
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("embed-unavailable");
    expect(out.results).toEqual([]);
    expect(out.scannedFragments).toBe(0);
    expect(out.storedModels).toBe(1);
    expect(out.embedModel).toBeUndefined();
    expect(sqlOf(c).some(t => t.includes("<=>"))).toBe(false);
  });

  it("budget-exhausted-pre-sql: the budget went to the embed round trip", async () => {
    const c = searchClient({ storedModels: ["embeddinggemma"], rows: [row()] });
    const out = await pgSearchVecDetailed(c, "hello", {
      embedder: slowEmbedder("embeddinggemma", 25),
      timeoutMs: 5,
    });
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("budget-exhausted-pre-sql");
    expect(out.results).toEqual([]);
    expect(out.scannedFragments).toBe(0);
    expect(out.storedModels).toBe(1);
    // The embed leg DID run and DID report a model — this is the one degraded
    // path that knows it.
    expect(out.embedModel).toBe("embeddinggemma");
    expect(sqlOf(c).some(t => t.includes("<=>"))).toBe(false);
  });

  it("CONTROL — healthy: rows present is degraded:false with the pre-dedup count", async () => {
    const c = searchClient({
      storedModels: ["embeddinggemma"],
      rows: [row({ path: "a.md", distance: 0.1 }), row({ path: "b.md", distance: 0.3 })],
    });
    const out = await pgSearchVecDetailed(c, "hello", { embedder: fakeEmbedder("embeddinggemma") });
    expect(out.degraded).toBe(false);
    expect(out.degradedReason).toBeUndefined();
    expect(out.results.length).toBeGreaterThan(0);
    expect(out.results.map(r => r.displayPath)).toEqual(["research/a.md", "research/b.md"]);
    expect(out.scannedFragments).toBe(2);
    expect(out.storedModels).toBe(1);
    expect(out.embedModel).toBe("embeddinggemma");
  });

  it("CONTROL — genuine empty: we searched and nothing matched, so NOT degraded", async () => {
    // THE POINT OF THE WHOLE SLICE. Same zero results as no-stored-vectors
    // above, and it must NOT carry the same meaning: the fence passed, the
    // embed leg produced a vector, the ANN scan ran, zero rows came back.
    const c = searchClient({ storedModels: ["embeddinggemma"], rows: [] });
    const out = await pgSearchVecDetailed(c, "hello", { embedder: fakeEmbedder("embeddinggemma") });
    expect(out.degraded).toBe(false);
    expect(out.degradedReason).toBeUndefined();
    expect(out.results).toEqual([]);
    expect(out.scannedFragments).toBe(0);
    expect(out.storedModels).toBe(1);
    expect(out.embedModel).toBe("embeddinggemma");
    // And the ANN scan really ran — that is the difference, not the result.
    expect(sqlOf(c).some(t => t.includes("<=>"))).toBe(true);
  });

  it("the two zero-result outcomes are not the same value", async () => {
    const cannotLook = await pgSearchVecDetailed(searchClient({ storedModels: [] }), "hello", {
      embedder: fakeEmbedder("embeddinggemma"),
    });
    const nothingMatched = await pgSearchVecDetailed(
      searchClient({ storedModels: ["embeddinggemma"], rows: [] }),
      "hello",
      { embedder: fakeEmbedder("embeddinggemma") },
    );
    expect(cannotLook.results).toEqual(nothingMatched.results);
    expect(cannotLook.degraded).not.toBe(nothingMatched.degraded);
  });
});

describe("pgSearchVec stays a byte-identical back-compat wrapper", () => {
  it("degraded input: the bare array equals detailed().results", async () => {
    const mk = () => searchClient({ storedModels: [] });
    const opts = { embedder: fakeEmbedder("embeddinggemma") };
    expect(await pgSearchVec(mk(), "hello", opts))
      .toEqual((await pgSearchVecDetailed(mk(), "hello", opts)).results);
  });

  it("healthy input: the bare array equals detailed().results", async () => {
    const mk = () => searchClient({
      storedModels: ["embeddinggemma"],
      rows: [row({ path: "a.md", distance: 0.1 }), row({ path: "b.md", distance: 0.3 })],
    });
    const opts = { embedder: fakeEmbedder("embeddinggemma") };
    const bare = await pgSearchVec(mk(), "hello", opts);
    expect(bare).toEqual((await pgSearchVecDetailed(mk(), "hello", opts)).results);
    expect(bare).toHaveLength(2);
  });
});
