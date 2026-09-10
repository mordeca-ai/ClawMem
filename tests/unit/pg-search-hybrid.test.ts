/**
 * PostgreSQL HYBRID (RRF fusion) READ-PATH unit tests
 * (master-harness-2wx75 slice 5).
 *
 * No database and no embed endpoint: a fake `PgQueryable` routes on SQL text,
 * exactly as tests/unit/pg-search-vec.test.ts and tests/unit/pg-search-fts.test.ts
 * do, so the FUSION POLICY and the DEGRADED-ARM SEMANTICS are assertable
 * without either arm's infrastructure.
 *
 * WHAT THIS TIER OWES, AND WHY. src/pg/search-hybrid.ts encodes three rulings
 * (ranks-not-scores, equal-and-unconfigurable weights, never-silently-two-arm).
 * A ruling with no case that goes red when it is undone is a comment. So:
 *
 *  - "ranks, not scores" is proven by a case where one arm's NATIVE score is a
 *    million times the other's and loses anyway. Introduce any score
 *    normalization or blend and that case goes red.
 *  - "never silently two-arm" is proven by the arms-matrix case, which asserts
 *    the three-way equivalence (degraded === false ⇔ arms === "vec+fts" ⇔ no
 *    armFailures) over every combination of arm health.
 *  - the k constant is proven WIRED by a case that changes it and observes the
 *    fused scores move. A k nothing can move is not a parameter.
 *
 * NEGATIVE CASES ARE THE POINT HERE. Every degraded reason of both arms, both
 * throw paths, the both-degraded floor, and the both-threw rethrow each have a
 * case; the happy path has two.
 */

import { describe, it, expect } from "bun:test";
import {
  fuseRankedArms,
  pgSearchHybrid,
  pgSearchHybridDetailed,
  DEFAULT_PG_HYBRID_RRF_K,
  type PgHybridArms,
} from "../../src/pg/search-hybrid.ts";
import type { PgQueryable, PgVecEmbedder, PgVecRow } from "../../src/pg/search.ts";
import type { PgFtsRow } from "../../src/pg/search-fts.ts";
import type { SearchResult } from "../../src/store.ts";

const MODEL = "embeddinggemma";

/** A SearchResult stand-in: only `filepath` (the fusion key) and `score` matter. */
function res(path: string, score: number, source: "vec" | "fts" = "vec"): SearchResult {
  return {
    filepath: `clawmem://research/${path}`,
    displayPath: `research/${path}`,
    title: path,
    context: null,
    hash: "a".repeat(64),
    docid: "aaaaaa",
    collectionName: "research",
    modifiedAt: "2026-09-01T00:00:00.000Z",
    bodyLength: 0,
    body: "",
    score,
    source,
  } as SearchResult;
}

const pathsOf = (rs: SearchResult[]) => rs.map(r => r.displayPath);

// ===========================================================================
// fuseRankedArms — the fusion policy, pure
// ===========================================================================

describe("fuseRankedArms — rank order", () => {
  it("ranks a document BOTH arms returned above one only a single arm returned", () => {
    const vec = [res("a.md", 0.9), res("shared.md", 0.5)];
    const fts = [res("b.md", 4.0, "fts"), res("shared.md", 1.0, "fts")];
    // shared.md sits at rank 1 in both; a.md and b.md at rank 0 in one each.
    // Two rank-1 contributions must beat one rank-0 contribution — that IS
    // the reason to fuse at all.
    expect(pathsOf(fuseRankedArms([vec, fts], 10))[0]).toBe("research/shared.md");
  });

  it("CONSUMES RANKS, NOT SCORES: a huge native score at a worse rank still loses", () => {
    // The FTS arm's ts_rank_cd is unbounded above; the vec arm's is 1 - cosine
    // distance, i.e. <= 1. If ANY score normalization or blending leaked in,
    // the 1e6 would dominate. It must not: rank 0 beats rank 1, full stop.
    const vec = [res("tiny-score.md", 0.0001)];
    const fts = [res("other.md", 0.0, "fts"), res("huge-score.md", 1e6, "fts")];
    const out = pathsOf(fuseRankedArms([vec, fts], 10));
    expect(out.indexOf("research/tiny-score.md")).toBeLessThan(out.indexOf("research/huge-score.md"));
  });

  it("returns FUSED scores, on neither arm's native scale", () => {
    const out = fuseRankedArms([[res("a.md", 0.9)]], 10);
    // 1/(60+0+1) + 0.05 rank-0 bonus — not the 0.9 that went in.
    expect(out[0]!.score).not.toBe(0.9);
    expect(out[0]!.score).toBeCloseTo(1 / (DEFAULT_PG_HYBRID_RRF_K + 1) + 0.05, 10);
  });

  it("SINGLE-LIST PASSTHROUGH preserves that arm's own order", () => {
    const fts = [res("first.md", 9, "fts"), res("second.md", 8, "fts"), res("third.md", 7, "fts")];
    expect(pathsOf(fuseRankedArms([fts], 10)))
      .toEqual(["research/first.md", "research/second.md", "research/third.md"]);
  });

  it("an EMPTY arm list does not perturb the other arm's order", () => {
    const vec = [res("first.md", 0.9), res("second.md", 0.8)];
    expect(pathsOf(fuseRankedArms([vec, []], 10)))
      .toEqual(pathsOf(fuseRankedArms([vec], 10)));
  });

  it("NO arms at all fuses to nothing (not a throw)", () => {
    expect(fuseRankedArms([], 10)).toEqual([]);
    expect(fuseRankedArms([[], []], 10)).toEqual([]);
  });

  it("TIES: symmetric ranks across arms produce equal fused scores", () => {
    // a.md is rank 0 in vec / rank 1 in fts; b.md is the mirror image. Equal
    // fused score is the CORRECT answer; the assertion is that fusion does not
    // secretly break the tie with a score (which would make them differ).
    const vec = [res("a.md", 0.9), res("b.md", 0.1)];
    const fts = [res("b.md", 500, "fts"), res("a.md", 0.001, "fts")];
    const out = fuseRankedArms([vec, fts], 10);
    expect(out).toHaveLength(2);
    expect(out[0]!.score).toBeCloseTo(out[1]!.score, 12);
  });

  it("the k CONSTANT IS WIRED: a larger k lowers every fused score", () => {
    const vec = [res("a.md", 0.9), res("b.md", 0.8)];
    const small = fuseRankedArms([vec], 10, 1);
    const large = fuseRankedArms([vec], 10, 10_000);
    expect(small[0]!.score).toBeGreaterThan(large[0]!.score);
    expect(small[1]!.score).toBeGreaterThan(large[1]!.score);
    // …and the default is what an omitted k means.
    expect(fuseRankedArms([vec], 10)[0]!.score)
      .toBe(fuseRankedArms([vec], 10, DEFAULT_PG_HYBRID_RRF_K)[0]!.score);
  });

  it("caps the UNION back to `limit`", () => {
    const vec = [res("v1.md", 0.9), res("v2.md", 0.8)];
    const fts = [res("f1.md", 9, "fts"), res("f2.md", 8, "fts")];
    expect(fuseRankedArms([vec, fts], 3)).toHaveLength(3);
    expect(fuseRankedArms([vec, fts], 10)).toHaveLength(4);
  });

  it("does not mutate the caller's lists (the weights array is rebuilt per call)", () => {
    const vec = [res("a.md", 0.9)];
    const fts = [res("b.md", 9, "fts")];
    fuseRankedArms([vec, fts], 10);
    fuseRankedArms([vec, fts], 10);
    // Two identical calls must fuse identically — a weights array sanitized in
    // place and shared across calls is exactly how that would drift.
    expect(fuseRankedArms([vec, fts], 10).map(r => r.score))
      .toEqual(fuseRankedArms([vec, fts], 10).map(r => r.score));
    expect(vec[0]!.score).toBe(0.9);
    expect(fts[0]!.score).toBe(9);
  });
});

// ===========================================================================
// pgSearchHybridDetailed — the arms, and what a degraded arm does
// ===========================================================================

type ArmPlan = {
  /** [] ⇒ the vec arm degrades "no-stored-vectors". */
  vecModels?: string[];
  vecRows?: PgVecRow[];
  /** 0 ⇒ the FTS arm degrades "empty-tsquery". */
  lexemes?: number;
  ftsRows?: PgFtsRow[];
  /** Throw from the vec ANN scan / the FTS ranked scan. */
  vecThrows?: Error;
  ftsThrows?: Error;
  embedder?: PgVecEmbedder;
};

function vecRow(path: string, distance: number): PgVecRow {
  return {
    hash: path.padEnd(64, "x").slice(0, 64), seq: 0, pos: 0,
    fragment_type: "section", fragment_label: null,
    collection: "research", path, title: path,
    modified_at: "2026-09-01T00:00:00.000Z", body: "b", distance,
  } as PgVecRow;
}

function ftsRow(path: string, rank: number): PgFtsRow {
  return {
    hash: path.padEnd(64, "y").slice(0, 64),
    collection: "research", path, title: path,
    modified_at: "2026-09-01T00:00:00.000Z", body: "b", rank,
  } as PgFtsRow;
}

/** One fake client serving BOTH arms, routed on SQL text. */
function hybridClient(plan: ArmPlan = {}): PgQueryable & { sql: string[] } {
  const sql: string[] = [];
  return {
    sql,
    async query(text: string) {
      sql.push(text);
      if (text.includes("SELECT DISTINCT cv.model")) {
        return { rows: (plan.vecModels ?? [MODEL]).map(model => ({ model })) as never[] };
      }
      if (text.includes("<=>")) {
        if (plan.vecThrows) throw plan.vecThrows;
        return { rows: (plan.vecRows ?? []) as never[] };
      }
      if (text.includes("numnode")) return { rows: [{ n: plan.lexemes ?? 2 }] as never[] };
      if (text.includes("ts_rank_cd")) {
        if (plan.ftsThrows) throw plan.ftsThrows;
        return { rows: (plan.ftsRows ?? []) as never[] };
      }
      return { rows: [] as never[] }; // BEGIN / COMMIT / SET LOCAL
    },
  };
}

const embedder: PgVecEmbedder = { async embed() { return { embedding: [0.1, 0.2], model: MODEL }; } };

function run(plan: ArmPlan = {}, opts = {}) {
  return pgSearchHybridDetailed(hybridClient(plan), "zebrafish", {
    collections: "research", embedder: plan.embedder ?? embedder, ...opts,
  });
}

describe("pgSearchHybridDetailed — both arms healthy", () => {
  it("reports arms: vec+fts, NOT degraded, no failures, and fuses both orderings", async () => {
    const out = await run({
      vecRows: [vecRow("v.md", 0.1), vecRow("shared.md", 0.4)],
      ftsRows: [ftsRow("f.md", 0.9), ftsRow("shared.md", 0.2)],
    });
    expect(out.arms).toBe("vec+fts");
    expect(out.degraded).toBe(false);
    expect(out.armFailures).toEqual([]);
    expect(pathsOf(out.results)[0]).toBe("research/shared.md"); // in both arms
    expect(pathsOf(out.results).sort())
      .toEqual(["research/f.md", "research/shared.md", "research/v.md"]);
    // Each arm's own detailed result is carried verbatim.
    expect(out.vec!.degraded).toBe(false);
    expect(out.fts!.degraded).toBe(false);
    expect(out.rrfK).toBe(DEFAULT_PG_HYBRID_RRF_K);
  });

  it("GENUINE-EMPTY survives fusion: two healthy arms, zero matches, NOT degraded", async () => {
    // The whole reason both arms have a degraded channel is that this value
    // must stay distinguishable from "we could not look". Fusion must not
    // collapse it into the degraded case.
    const out = await run({ vecRows: [], ftsRows: [] });
    expect(out.results).toEqual([]);
    expect(out.arms).toBe("vec+fts");
    expect(out.degraded).toBe(false);
    expect(out.armFailures).toEqual([]);
  });

  it("RUNS THE ARMS IN SEQUENCE: no vec SQL after the first fts SQL", async () => {
    // THE 25P01 REGRESSION GUARD. `PgQueryable` is one connection and a
    // connection holds one transaction; the first draft of this module ran both
    // arms under Promise.allSettled and PostgreSQL rejected the interleaved
    // transaction blocks with "ROLLBACK TO SAVEPOINT can only be used in
    // transaction blocks". It failed INTERMITTENTLY, on microtask ordering, so
    // the guard has to be an explicit ordering assertion rather than a live
    // green run. Reintroduce concurrency and this goes red deterministically.
    const c = hybridClient({ vecRows: [], ftsRows: [] });
    await pgSearchHybridDetailed(c, "q", { collections: "research", embedder });
    const isVec = (t: string) => t.includes("<=>") || t.includes("SELECT DISTINCT cv.model");
    const isFts = (t: string) => t.includes("numnode") || t.includes("ts_rank_cd");
    const firstFts = c.sql.findIndex(isFts);
    const lastVec = c.sql.reduce((acc, t, i) => (isVec(t) ? i : acc), -1);
    expect(firstFts).toBeGreaterThan(-1);
    expect(lastVec).toBeGreaterThan(-1);
    expect(lastVec).toBeLessThan(firstFts);
  });

  it("forwards ONE scope and ONE budget to BOTH arms", async () => {
    const c = hybridClient({ vecRows: [], ftsRows: [] });
    await pgSearchHybridDetailed(c, "q", {
      collections: "research", embedder, statementTimeoutMs: 250,
    });
    // Both arms bound their SQL legs with the SAME statement_timeout: a hybrid
    // whose legs disagreed about the budget would be two searches, not one.
    expect(c.sql.filter(t => t === "SET LOCAL statement_timeout = 250").length)
      .toBeGreaterThanOrEqual(3); // vec: fence + ann; fts: ranked scan
    expect(c.sql.some(t => t.includes("<=>"))).toBe(true);
    expect(c.sql.some(t => t.includes("ts_rank_cd"))).toBe(true);
  });
});

describe("pgSearchHybridDetailed — ONE arm degraded", () => {
  it("VEC degraded (no-stored-vectors) ⇒ fts-only, degraded, the FTS order verbatim", async () => {
    const out = await run({
      vecModels: [],
      ftsRows: [ftsRow("first.md", 9), ftsRow("second.md", 8)],
    });
    expect(out.arms).toBe("fts-only");
    expect(out.degraded).toBe(true);
    expect(out.armFailures).toEqual([{ arm: "vec", kind: "degraded", reason: "no-stored-vectors" }]);
    expect(pathsOf(out.results)).toEqual(["research/first.md", "research/second.md"]);
  });

  it("VEC degraded (embed-unavailable) ⇒ fts-only, and the reason is the ARM'S OWN", async () => {
    const out = await run({
      embedder: { async embed() { return null as never; } },
      ftsRows: [ftsRow("f.md", 9)],
    });
    expect(out.arms).toBe("fts-only");
    expect(out.armFailures).toEqual([{ arm: "vec", kind: "degraded", reason: "embed-unavailable" }]);
  });

  it("FTS degraded (empty-tsquery) ⇒ vec-only, degraded, the vec order verbatim", async () => {
    const out = await run({
      lexemes: 0,
      vecRows: [vecRow("near.md", 0.1), vecRow("far.md", 0.9)],
    });
    expect(out.arms).toBe("vec-only");
    expect(out.degraded).toBe(true);
    expect(out.armFailures).toEqual([{ arm: "fts", kind: "degraded", reason: "empty-tsquery" }]);
    expect(pathsOf(out.results)).toEqual(["research/near.md", "research/far.md"]);
  });

  it("a single-arm answer NEVER reports arms: vec+fts, even with rows to show", async () => {
    // THE ANTI-REGRESSION. Delete the `arms` discriminator, or compute it from
    // `results.length`, and this is the case that goes red: a caller rendering
    // "hybrid" over a one-arm ranking is lying to its user.
    for (const plan of [{ vecModels: [], ftsRows: [ftsRow("f.md", 9)] },
                        { lexemes: 0, vecRows: [vecRow("v.md", 0.1)] }]) {
      const out = await run(plan);
      expect(out.results.length).toBeGreaterThan(0);
      expect(out.arms).not.toBe("vec+fts");
      expect(out.degraded).toBe(true);
    }
  });
});

describe("pgSearchHybridDetailed — BOTH arms degraded", () => {
  it("reports arms: none, degraded, empty results, and BOTH causes", async () => {
    const out = await run({ vecModels: [], lexemes: 0 });
    expect(out.arms).toBe("none");
    expect(out.degraded).toBe(true);
    expect(out.results).toEqual([]);
    expect(out.armFailures).toEqual([
      { arm: "vec", kind: "degraded", reason: "no-stored-vectors" },
      { arm: "fts", kind: "degraded", reason: "empty-tsquery" },
    ]);
  });
});

describe("pgSearchHybridDetailed — a THROWING arm is demoted, not fatal", () => {
  it("vec throws, fts healthy ⇒ fts-only, the error CARRIED not swallowed", async () => {
    const boom = new Error("canceling statement due to statement timeout");
    const out = await run({ vecThrows: boom, ftsRows: [ftsRow("f.md", 9)] });
    expect(out.arms).toBe("fts-only");
    expect(out.degraded).toBe(true);
    expect(out.armFailures).toEqual([{ arm: "vec", kind: "threw", error: boom }]);
    expect(out.vec).toBeUndefined();     // there is no detailed result to carry
    expect(out.fts!.degraded).toBe(false);
    expect(pathsOf(out.results)).toEqual(["research/f.md"]);
  });

  it("fts throws, vec healthy ⇒ vec-only, the error CARRIED not swallowed", async () => {
    const boom = new Error("fts leg cancelled");
    const out = await run({ ftsThrows: boom, vecRows: [vecRow("v.md", 0.1)] });
    expect(out.arms).toBe("vec-only");
    expect(out.armFailures).toEqual([{ arm: "fts", kind: "threw", error: boom }]);
    expect(out.fts).toBeUndefined();
    expect(pathsOf(out.results)).toEqual(["research/v.md"]);
  });

  it("BOTH arms throw ⇒ REJECTS (never an empty answer), carrying both errors", async () => {
    const v = new Error("vec down");
    const f = new Error("fts down");
    const p = run({ vecThrows: v, ftsThrows: f });
    await expect(p).rejects.toThrow("vec down");
    const err = (await p.catch(e => e)) as Error;
    expect(err.cause).toBe(f);
  });
});

describe("pgSearchHybridDetailed — the three-way invariant", () => {
  it("degraded === false ⇔ arms === 'vec+fts' ⇔ armFailures is empty, over every combination",
    async () => {
      const matrix: { plan: ArmPlan; arms: PgHybridArms }[] = [
        { plan: { vecRows: [vecRow("v.md", 0.1)], ftsRows: [ftsRow("f.md", 9)] }, arms: "vec+fts" },
        { plan: { vecRows: [], ftsRows: [] }, arms: "vec+fts" },
        { plan: { vecModels: [] }, arms: "fts-only" },
        { plan: { vecThrows: new Error("x") }, arms: "fts-only" },
        { plan: { lexemes: 0 }, arms: "vec-only" },
        { plan: { ftsThrows: new Error("x") }, arms: "vec-only" },
        { plan: { vecModels: [], lexemes: 0 }, arms: "none" },
        { plan: { vecModels: [], ftsThrows: new Error("x") }, arms: "none" },
      ];
      for (const { plan, arms } of matrix) {
        const out = await run(plan);
        expect(out.arms).toBe(arms);
        expect(out.degraded).toBe(arms !== "vec+fts");
        expect(out.armFailures.length === 0).toBe(arms === "vec+fts");
      }
    });
});

describe("pgSearchHybrid — the bare wrapper", () => {
  it("returns exactly the detailed call's results", async () => {
    const plan: ArmPlan = {
      vecRows: [vecRow("v.md", 0.1), vecRow("shared.md", 0.4)],
      ftsRows: [ftsRow("f.md", 0.9), ftsRow("shared.md", 0.2)],
    };
    const bare = await pgSearchHybrid(hybridClient(plan), "q", { collections: "research", embedder });
    const detailed = await run(plan);
    expect(pathsOf(bare)).toEqual(pathsOf(detailed.results));
    // …and it CANNOT express the arms. That is why the detailed one exists.
    expect((bare as unknown as { arms?: string }).arms).toBeUndefined();
  });

  it("a degraded-to-nothing hybrid is an empty array here — indistinguishable from no matches",
    async () => {
      // Recorded as a KNOWN LIMIT of this signature, not a bug: the caller that
      // needs the difference must use pgSearchHybridDetailed.
      const bare = await pgSearchHybrid(hybridClient({ vecModels: [], lexemes: 0 }), "q",
        { collections: "research", embedder });
      expect(bare).toEqual([]);
    });
});
