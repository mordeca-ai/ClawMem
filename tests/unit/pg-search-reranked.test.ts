/**
 * PostgreSQL RERANKED READ-PATH unit tests (master-harness-2wx75 slice 6).
 *
 * No database, no embed endpoint and no reranker service: a fake `PgQueryable`
 * routes on SQL text (exactly as tests/unit/pg-search-hybrid.test.ts does) and
 * the reranker is INJECTED, so the whole rerank policy is assertable without
 * any of its infrastructure.
 *
 * WHAT THIS TIER OWES. src/pg/search-reranked.ts encodes six rulings, and a
 * ruling with no case that goes red when it is undone is a comment. So:
 *
 *  - "a rerank problem degrades to FUSION" is proven by the STATUS MATRIX
 *    below: one case per PgRerankStatus member plus a partial-coverage case,
 *    and every non-"applied" case asserts the returned order is the hybrid's
 *    fused `filepath` sequence BYTE-IDENTICALLY. Let a rerank failure
 *    propagate, or reorder on a degenerate response, and the matrix goes red.
 *  - "composition did not disturb slice 5" is proven by re-asserting slice 5's
 *    own three-way invariant (degraded === false ⇔ arms === "vec+fts" ⇔ no
 *    armFailures) over the EMBEDDED hybrid object, over every arm combination.
 *  - "the budget is one deadline that DECREMENTS" is proven by (a) a hybrid
 *    that eats the whole deadline, which must leave the reranker's call count
 *    at ZERO (the count, not just the status), (b) the timeoutMs the reranker
 *    actually receives being the REMAINDER and strictly less than the
 *    deadline, and (c) totalMs staying inside deadlineMs + slack even when the
 *    reranker overruns. Hand the reranker the full deadline instead of the
 *    remainder and (b) and (c) both go red.
 *  - "blendRerank, not blendFusionAndRerank" is proven by the partial-coverage
 *    case (a reranker returning a subset must not DROP the unscored
 *    documents — blendFusionAndRerank maps over the rerank output and would)
 *    and by the degenerate case (the floor + onFallback hook, which
 *    blendFusionAndRerank has no equivalent of).
 *
 * The timing cases use deliberately coarse sleeps and a generous slack so they
 * assert the ARITHMETIC rather than the scheduler.
 */

import { describe, it, expect } from "bun:test";
import {
  pgSearchReranked,
  pgSearchRerankedDetailed,
  PG_RERANK_MIN_BUDGET_MS,
  DEFAULT_PG_RERANK_DEADLINE_MS,
  PG_RERANK_TEXT_CHARS,
  type PgReranker,
  type PgRerankStatus,
} from "../../src/pg/search-reranked.ts";
import type { PgQueryable, PgVecEmbedder, PgVecRow } from "../../src/pg/search.ts";
import type { PgFtsRow } from "../../src/pg/search-fts.ts";
import type { PgHybridArms } from "../../src/pg/search-hybrid.ts";
import type { SearchResult } from "../../src/store.ts";

const MODEL = "embeddinggemma";
/** Timing slack, ms. Coarse on purpose: these cases assert arithmetic, not the scheduler. */
const SLACK = 300;

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

// ===========================================================================
// Fakes: one client serving both arms, plus a recording reranker
// ===========================================================================

type ArmPlan = {
  /** [] ⇒ the vec arm degrades "no-stored-vectors". */
  vecModels?: string[];
  vecRows?: PgVecRow[];
  /** 0 ⇒ the FTS arm degrades "empty-tsquery". */
  lexemes?: number;
  ftsRows?: PgFtsRow[];
  vecThrows?: Error;
  ftsThrows?: Error;
  embedder?: PgVecEmbedder;
  /** Wall clock the vec ANN leg burns — how the hybrid is made to eat the deadline. */
  vecDelayMs?: number;
};

function vecRow(path: string, distance: number, body = `body of ${path}`): PgVecRow {
  return {
    hash: path.padEnd(64, "x").slice(0, 64), seq: 0, pos: 0,
    fragment_type: "section", fragment_label: null,
    collection: "research", path, title: path,
    modified_at: "2026-09-01T00:00:00.000Z", body, distance,
  } as PgVecRow;
}

function ftsRow(path: string, rank: number, body = `body of ${path}`): PgFtsRow {
  return {
    hash: path.padEnd(64, "y").slice(0, 64),
    collection: "research", path, title: path,
    modified_at: "2026-09-01T00:00:00.000Z", body, rank,
  } as PgFtsRow;
}

function hybridClient(plan: ArmPlan = {}): PgQueryable {
  return {
    async query(text: string) {
      if (text.includes("SELECT DISTINCT cv.model")) {
        return { rows: (plan.vecModels ?? [MODEL]).map(model => ({ model })) as never[] };
      }
      if (text.includes("<=>")) {
        if (plan.vecDelayMs) await sleep(plan.vecDelayMs);
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

type Recorder = {
  reranker: PgReranker;
  calls: { query: string; documents: { file: string; text: string }[]; timeoutMs?: number }[];
};

/** A reranker that records every call, then answers via `score`. */
function recording(
  score: (doc: { file: string; text: string }, i: number) => { file: string; score: number }[] | number | undefined,
  behaviour: { sleepMs?: number; throws?: Error; subsetOf?: (files: string[]) => string[] } = {},
): Recorder {
  const calls: Recorder["calls"] = [];
  const reranker: PgReranker = async (query, documents, opts) => {
    calls.push({ query, documents, ...(opts?.timeoutMs === undefined ? {} : { timeoutMs: opts.timeoutMs }) });
    if (behaviour.sleepMs) await sleep(behaviour.sleepMs);
    if (behaviour.throws) throw behaviour.throws;
    const files = behaviour.subsetOf ? behaviour.subsetOf(documents.map(d => d.file)) : documents.map(d => d.file);
    return files.map((file, i) => {
      const d = documents.find(x => x.file === file)!;
      const s = score(d, i);
      return { file, score: typeof s === "number" ? s : 0 };
    });
  };
  return { reranker, calls };
}

/** Reranker that scores in REVERSE document order — a visible, unambiguous reordering. */
function reversing(): Recorder {
  const calls: Recorder["calls"] = [];
  const reranker: PgReranker = async (query, documents, opts) => {
    calls.push({ query, documents, ...(opts?.timeoutMs === undefined ? {} : { timeoutMs: opts.timeoutMs }) });
    return documents.map((d, i) => ({ file: d.file, score: (i + 1) / documents.length }));
  };
  return { reranker, calls };
}

const pathsOf = (rs: SearchResult[]) => rs.map(r => r.displayPath);

/** Three fused documents: shared.md is in both arms, so fusion puts it first. */
const THREE: ArmPlan = {
  vecRows: [vecRow("v.md", 0.1), vecRow("shared.md", 0.4)],
  ftsRows: [ftsRow("f.md", 0.9), ftsRow("shared.md", 0.2)],
};

function run(plan: ArmPlan = THREE, opts: Record<string, unknown> = {}) {
  return pgSearchRerankedDetailed(hybridClient(plan), "zebrafish", {
    collections: "research", embedder: plan.embedder ?? embedder, ...opts,
  });
}

// ===========================================================================
// applied — the rerank actually re-orders
// ===========================================================================

describe("pgSearchRerankedDetailed — applied", () => {
  it("REORDERS the fused list and reports rerank: 'applied'", async () => {
    const rr = reversing();
    const out = await run(THREE, { reranker: rr.reranker });

    expect(out.rerank).toBe("applied");
    expect(out.candidateCount).toBe(3);
    expect(out.rerankedCount).toBe(3);
    // The reranker scored the LAST fused document highest, so the final order
    // is the fusion reversed. If the blend were ignoring rerank scores this
    // would still read as the fused order — which is exactly the silent
    // collapse the degenerate floor exists to make visible.
    const fused = pathsOf(out.hybrid.results);
    expect(pathsOf(out.results)).toEqual([...fused].reverse());
    expect(pathsOf(out.results)).not.toEqual(fused);
    // Blended scores are on blendRerank's [0,1] scale, not the RRF scale.
    expect(out.results[0]!.score).toBeLessThanOrEqual(1);
    expect(out.results[0]!.score).not.toBe(out.hybrid.results[0]!.score);
  });

  it("hands the reranker the QUERY and the candidate bodies, truncated", async () => {
    const long = "z".repeat(PG_RERANK_TEXT_CHARS + 500);
    const rr = reversing();
    await run({ vecRows: [vecRow("v.md", 0.1, long)], ftsRows: [] }, { reranker: rr.reranker });
    expect(rr.calls).toHaveLength(1);
    expect(rr.calls[0]!.query).toBe("zebrafish");
    expect(rr.calls[0]!.documents[0]!.file).toBe("clawmem://research/v.md");
    expect(rr.calls[0]!.documents[0]!.text.length).toBe(PG_RERANK_TEXT_CHARS);
  });

  it("PARTIAL COVERAGE never drops a document (blendRerank maps over CANDIDATES)", async () => {
    // THE blendFusionAndRerank REGRESSION GUARD. A reranker that returns only
    // one of three rows must not lose the other two: blendFusionAndRerank maps
    // over the RERANK OUTPUT and would return a single document.
    const rr = recording(() => 0.99, { subsetOf: files => [files[2]!] });
    const out = await run(THREE, { reranker: rr.reranker });

    expect(out.rerank).toBe("applied");
    expect(out.rerankedCount).toBe(1);
    expect(out.results).toHaveLength(3);
    expect(pathsOf(out.results).sort()).toEqual(pathsOf(out.hybrid.results).sort());
    // The one scored document is promoted; the unscored two keep their
    // relative fused order behind it.
    const fused = pathsOf(out.hybrid.results);
    expect(pathsOf(out.results)[0]).toBe(fused[2]);
    expect(pathsOf(out.results).slice(1)).toEqual([fused[0]!, fused[1]!]);
  });

  it("caps at `limit` while still reranking a WIDER pool", async () => {
    const rr = reversing();
    const out = await run(THREE, { reranker: rr.reranker, limit: 2 });
    // limit 2 ⇒ the hybrid returns 2, so the pool is 2. rerankCap floors at 30
    // and never truncates below the fused list.
    expect(out.results).toHaveLength(2);
    expect(out.candidateCount).toBe(2);
  });

  it("a rerankCap SMALLER than the fused list reranks the head and keeps the tail", async () => {
    const rr = reversing();
    const out = await run(THREE, { reranker: rr.reranker, rerankCap: 2 });
    expect(out.candidateCount).toBe(2);
    const fused = pathsOf(out.hybrid.results);
    // Head reranked (reversed), tail appended in fused order — nothing dropped.
    expect(pathsOf(out.results)).toEqual([fused[1]!, fused[0]!, fused[2]!]);
  });
});

// ===========================================================================
// THE STATUS MATRIX — every non-"applied" status degrades to the fused order
// ===========================================================================

describe("pgSearchRerankedDetailed — the status matrix", () => {
  it("skipped-no-reranker: a caller choice, not a failure", async () => {
    const out = await run(THREE, {});
    expect(out.rerank).toBe("skipped-no-reranker");
    expect(out.candidateCount).toBe(0);
    expect(out.timings.rerankMs).toBe(0);
  });

  it("skipped-no-text: every candidate body empty ⇒ the reranker is NOT called", async () => {
    const rr = reversing();
    const out = await run(
      { vecRows: [vecRow("v.md", 0.1, "")], ftsRows: [ftsRow("f.md", 0.9, "")] },
      { reranker: rr.reranker },
    );
    expect(out.rerank).toBe("skipped-no-text");
    expect(rr.calls).toHaveLength(0);   // the COUNT, not just the status
    expect(out.results.length).toBeGreaterThan(0); // the documents still come back
  });

  it("skipped-no-text: zero fused candidates ⇒ the reranker is NOT called", async () => {
    const rr = reversing();
    const out = await run({ vecRows: [], ftsRows: [] }, { reranker: rr.reranker });
    expect(out.rerank).toBe("skipped-no-text");
    expect(out.hybrid.arms).toBe("vec+fts");   // genuine-empty, NOT degraded
    expect(out.hybrid.degraded).toBe(false);
    expect(rr.calls).toHaveLength(0);
  });

  it("skipped-budget: the hybrid ate the deadline ⇒ reranker CALL COUNT IS ZERO", async () => {
    const rr = reversing();
    const out = await run(
      { ...THREE, vecDelayMs: 120 },
      { reranker: rr.reranker, deadlineMs: 100 },
    );
    expect(out.rerank).toBe("skipped-budget");
    expect(rr.calls).toHaveLength(0);
    expect(out.candidateCount).toBe(3);        // there WERE candidates
    expect(out.rerankReason).toContain(`${PG_RERANK_MIN_BUDGET_MS}ms floor`);
    // …and the call did not run long past its budget waiting to discover that.
    expect(out.timings.totalMs).toBeLessThanOrEqual(100 + SLACK);
    expect(out.timings.hybridMs).toBeGreaterThanOrEqual(100);
  });

  it("degenerate: every score at/below the floor ⇒ onFallback becomes the STATUS", async () => {
    // The broken-reranker shape d0hz measured: ~1e-11 scores that contribute
    // nothing at weight 0.9 and silently collapse the blend back to RRF order.
    const rr = recording(() => 1e-11);
    const out = await run(THREE, { reranker: rr.reranker });
    expect(out.rerank).toBe("degenerate");
    expect(rr.calls).toHaveLength(1);          // it DID respond
    expect(out.rerankedCount).toBe(3);
    expect(out.rerankReason).toContain("degenerate floor");
    expect(out.timings.rerankMs).toBeGreaterThanOrEqual(0);
  });

  it("degenerate: an EMPTY response folds into the same status, with its own reason", async () => {
    const rr = recording(() => 0, { subsetOf: () => [] });
    const out = await run(THREE, { reranker: rr.reranker });
    expect(out.rerank).toBe("degenerate");
    expect(out.rerankedCount).toBe(0);
    expect(out.rerankReason).toBe("reranker returned no scores");
  });

  it("failed: the reranker THREW ⇒ the search does not, and the error is CARRIED", async () => {
    const boom = new Error("rerank endpoint refused the connection");
    const rr = recording(() => 0.9, { throws: boom });
    const out = await run(THREE, { reranker: rr.reranker });
    expect(out.rerank).toBe("failed");
    expect(out.rerankError).toBe(boom);
    expect(out.rerankReason).toBe("rerank endpoint refused the connection");
  });

  it("EVERY non-'applied' status returns the fused order BYTE-IDENTICALLY", async () => {
    // THE LOAD-BEARING ASSERTION of this slice: a rerank problem degrades to
    // fusion — not to garbage, not to empty, not to a reordering nobody chose.
    const cases: { status: Exclude<PgRerankStatus, "applied">; plan: ArmPlan; opts: Record<string, unknown> }[] = [
      { status: "skipped-no-reranker", plan: THREE, opts: {} },
      { status: "skipped-no-text", plan: { vecRows: [vecRow("v.md", 0.1, "")], ftsRows: [] },
        opts: { reranker: reversing().reranker } },
      { status: "skipped-budget", plan: { ...THREE, vecDelayMs: 120 },
        opts: { reranker: reversing().reranker, deadlineMs: 100 } },
      { status: "degenerate", plan: THREE, opts: { reranker: recording(() => 1e-11).reranker } },
      { status: "failed", plan: THREE, opts: { reranker: recording(() => 1, { throws: new Error("x") }).reranker } },
    ];
    for (const { status, plan, opts } of cases) {
      const out = await run(plan, opts);
      expect(out.rerank).toBe(status);
      // Byte-identical filepath sequence, AND the very same array — a copy that
      // happened to sort the same way would be luck, not a contract.
      expect(out.results.map(r => r.filepath)).toEqual(out.hybrid.results.map(r => r.filepath));
      expect(out.results).toBe(out.hybrid.results);
    }
    // …and the matrix covered every member of the union except "applied".
    expect(cases.map(c => c.status).sort()).toEqual(
      ["degenerate", "failed", "skipped-budget", "skipped-no-reranker", "skipped-no-text"],
    );
  });
});

// ===========================================================================
// THE BUDGET — one deadline that decrements (ruling 4)
// ===========================================================================

describe("pgSearchRerankedDetailed — the decrementing deadline", () => {
  it("hands the reranker the REMAINDER, not the whole deadline", async () => {
    const rr = reversing();
    const deadlineMs = 1000;
    const out = await run({ ...THREE, vecDelayMs: 300 }, { reranker: rr.reranker, deadlineMs });
    expect(rr.calls).toHaveLength(1);
    const handed = rr.calls[0]!.timeoutMs!;
    // Strictly less than the deadline by roughly the hybrid's own cost. Pass
    // `deadlineMs` here instead of `remaining` and this goes red.
    expect(handed).toBeLessThan(deadlineMs);
    expect(handed).toBeLessThanOrEqual(deadlineMs - out.timings.hybridMs + 1);
    expect(handed).toBeGreaterThan(deadlineMs - out.timings.hybridMs - SLACK);
  });

  it("an OVERRUNNING reranker fails inside the deadline rather than blowing it", async () => {
    // The reranker ignores its timeoutMs and sleeps past the remainder. The
    // call must still land inside deadlineMs: hand it the full deadline and
    // both assertions here go red (status would be "applied", totalMs > budget).
    const rr = recording(() => 0.9, { sleepMs: 600 });
    const deadlineMs = 700;
    const out = await run({ ...THREE, vecDelayMs: 300 }, { reranker: rr.reranker, deadlineMs });
    expect(out.rerank).toBe("failed");
    expect(out.rerankReason).toContain("remaining budget");
    expect(out.timings.totalMs).toBeLessThanOrEqual(deadlineMs + SLACK);
  });

  it("reports measured stage timings that ADD UP", async () => {
    const rr = recording(() => 0.9, { sleepMs: 40 });
    const out = await run({ ...THREE, vecDelayMs: 60 }, { reranker: rr.reranker, deadlineMs: 2000 });
    expect(out.rerank).toBe("applied");
    expect(out.timings.hybridMs).toBeGreaterThanOrEqual(60);
    expect(out.timings.rerankMs).toBeGreaterThanOrEqual(40);
    // The whole point of returning these: the budget is ADDITIVE and readable,
    // not inferred.
    expect(out.timings.totalMs).toBeGreaterThanOrEqual(out.timings.hybridMs + out.timings.rerankMs);
    expect(out.timings.totalMs).toBeLessThanOrEqual(out.timings.hybridMs + out.timings.rerankMs + SLACK);
  });

  it("the DEFAULT deadline is what an omitted deadlineMs means", async () => {
    const out = await run(THREE, { reranker: reversing().reranker });
    expect(out.deadlineMs).toBe(DEFAULT_PG_RERANK_DEADLINE_MS);
    expect(DEFAULT_PG_RERANK_DEADLINE_MS).toBeLessThan(1800); // the bead's hook p95 clause
  });

  it("a deadline BELOW the floor skips before the hybrid's cost even matters", async () => {
    const rr = reversing();
    const out = await run(THREE, { reranker: rr.reranker, deadlineMs: PG_RERANK_MIN_BUDGET_MS - 1 });
    expect(out.rerank).toBe("skipped-budget");
    expect(rr.calls).toHaveLength(0);
  });
});

// ===========================================================================
// COMPOSITION — slice 5's invariant must survive being embedded
// ===========================================================================

describe("pgSearchRerankedDetailed — composition did not disturb slice 5", () => {
  it("re-asserts degraded === false ⇔ arms === 'vec+fts' ⇔ no armFailures, over the EMBEDDED hybrid",
    async () => {
      const matrix: { plan: ArmPlan; arms: PgHybridArms }[] = [
        { plan: THREE, arms: "vec+fts" },
        { plan: { vecRows: [], ftsRows: [] }, arms: "vec+fts" },
        { plan: { vecModels: [], ftsRows: [ftsRow("f.md", 9)] }, arms: "fts-only" },
        { plan: { vecThrows: new Error("x"), ftsRows: [ftsRow("f.md", 9)] }, arms: "fts-only" },
        { plan: { lexemes: 0, vecRows: [vecRow("v.md", 0.1)] }, arms: "vec-only" },
        { plan: { ftsThrows: new Error("x"), vecRows: [vecRow("v.md", 0.1)] }, arms: "vec-only" },
        { plan: { vecModels: [], lexemes: 0 }, arms: "none" },
        { plan: { vecModels: [], ftsThrows: new Error("x") }, arms: "none" },
      ];
      for (const { plan, arms } of matrix) {
        const out = await run(plan, { reranker: reversing().reranker });
        expect(out.hybrid.arms).toBe(arms);
        expect(out.hybrid.degraded).toBe(arms !== "vec+fts");
        expect(out.hybrid.armFailures.length === 0).toBe(arms === "vec+fts");
        // …and the rerank status is an INDEPENDENT axis: a degraded-arm answer
        // is still rerankable, and a healthy two-arm answer can still have a
        // dead reranker. Neither field may leak into the other.
        expect(out.rerank).not.toBe(undefined);
      }
    });

  it("a DEGRADED single arm is still reranked (rerank state is orthogonal to arm state)", async () => {
    const rr = reversing();
    const out = await run(
      { vecModels: [], ftsRows: [ftsRow("first.md", 9), ftsRow("second.md", 8)] },
      { reranker: rr.reranker },
    );
    expect(out.hybrid.arms).toBe("fts-only");
    expect(out.hybrid.degraded).toBe(true);
    expect(out.rerank).toBe("applied");
    expect(pathsOf(out.results)).toEqual(["research/second.md", "research/first.md"]);
  });

  it("BOTH arms failing still REJECTS — retrieval failure is not a rerank degrade", async () => {
    // Slice 5's ruling, which this module must not soften: "we could not look"
    // stays distinguishable from "nothing matched". Wrapping the hybrid call in
    // a try/catch here would make that unreachable again.
    const p = run(
      { vecThrows: new Error("vec down"), ftsThrows: new Error("fts down") },
      { reranker: reversing().reranker },
    );
    await expect(p).rejects.toThrow("vec down");
  });
});

// ===========================================================================
// The bare wrapper
// ===========================================================================

describe("pgSearchReranked — the bare wrapper", () => {
  it("returns exactly the detailed call's results", async () => {
    const bare = await pgSearchReranked(hybridClient(THREE), "zebrafish", {
      collections: "research", embedder, reranker: reversing().reranker,
    });
    const detailed = await run(THREE, { reranker: reversing().reranker });
    expect(pathsOf(bare)).toEqual(pathsOf(detailed.results));
    // …and it CANNOT express whether the rerank ran. That is why the detailed
    // one exists.
    expect((bare as unknown as { rerank?: string }).rerank).toBeUndefined();
  });
});
