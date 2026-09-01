// Reranker-health guard — bug-first tests. These assert CORRECT behavior; the failures they would
// catch are the exact ones that shipped before: a reranker returning HTTP 200 + finite positive
// ~1e-11 scores that passed liveness yet silently collapsed ranking to RRF, and partial endpoint
// output that zero-fills into a false-pass. See RERANKER-HEALTH-GUARD-DESIGN.md.
import { test, expect, describe, afterEach, beforeEach } from "bun:test";
import { createStore, RerankCoverageError, RerankMalformedResponseError } from "../../src/store.ts";
import { blendRerank, RERANK_DEGENERATE_FLOOR } from "../../src/search-utils.ts";
import {
  probeRerankHealth,
  rerankFailureAdvice,
  toLogit,
  RERANK_LOGIT_CLAMP_EPS,
  type GoldenTriple,
} from "../../src/health/rerank-health.ts";

// ---------------------------------------------------------------------------
// blendRerank — degenerate-floor trip, onFallback emit, options overload
// ---------------------------------------------------------------------------
describe("blendRerank", () => {
  const candidates = [
    { file: "a", score: 3 },
    { file: "b", score: 2 },
    { file: "c", score: 1 },
  ]; // RRF order: a, b, c

  test("degenerate reranker (~1e-11) falls back to RRF order AND fires onFallback", () => {
    // The historical bug: these finite positive scores passed the old `> 0` check, contributed
    // ~nothing at weight 0.9, and silently produced RRF order with NO signal that the reranker died.
    const reranked = [
      { file: "c", score: 1e-11 },
      { file: "b", score: 5e-12 },
      { file: "a", score: 2e-11 },
    ];
    let reason = "";
    const out = blendRerank(candidates, reranked, { onFallback: (r) => (reason = r) });
    expect(out.map((o) => o.file)).toEqual(["a", "b", "c"]); // RRF order preserved
    expect(reason).toContain("degenerate floor"); // the degrade is now VISIBLE
  });

  test("healthy reranker can promote a doc over RRF #1, and does NOT fire onFallback", () => {
    const reranked = [
      { file: "c", score: 0.95 }, // c is RRF-last but reranker-best
      { file: "a", score: 0.2 },
      { file: "b", score: 0.1 },
    ];
    let fired = false;
    const out = blendRerank(candidates, reranked, { onFallback: () => (fired = true) });
    expect(out[0]!.file).toBe("c"); // reranker promoted c over RRF #1 (a)
    expect(fired).toBe(false);
  });

  test("empty rerank output falls back and reports 'no scores'", () => {
    let reason = "";
    const out = blendRerank(candidates, [], { onFallback: (r) => (reason = r) });
    expect(out.map((o) => o.file)).toEqual(["a", "b", "c"]);
    expect(reason).toContain("no scores");
  });

  test("numeric 3rd arg (back-compat) still sets rerankWeight", () => {
    const reranked = [
      { file: "b", score: 0.9 },
      { file: "a", score: 0.1 },
    ];
    const out = blendRerank([{ file: "a", score: 3 }, { file: "b", score: 1 }], reranked, 0.9);
    expect(out[0]!.file).toBe("b"); // reranker-dominant at weight 0.9
  });

  test("2-arg call still works (default weight, no options)", () => {
    const out = blendRerank(
      [{ file: "a", score: 2 }, { file: "b", score: 1 }],
      [{ file: "b", score: 0.9 }, { file: "a", score: 0.1 }],
    );
    expect(out[0]!.file).toBe("b");
  });

  test("custom degenerateFloor is respected", () => {
    // 0.001 scores are above the default 1e-4 floor but below a custom 0.01 floor → fallback.
    let fired = false;
    const out = blendRerank(
      [{ file: "a", score: 2 }, { file: "b", score: 1 }],
      [{ file: "b", score: 0.001 }, { file: "a", score: 0.001 }],
      { degenerateFloor: 0.01, onFallback: () => (fired = true) },
    );
    expect(fired).toBe(true);
    expect(out.map((o) => o.file)).toEqual(["a", "b"]); // RRF order
  });

  test("the default degenerate floor sits above the broken regime and below working scores", () => {
    expect(RERANK_DEGENERATE_FLOOR).toBeGreaterThan(8.03e-7); // broken zerank-2 GGUF max-ever
    expect(RERANK_DEGENERATE_FLOOR).toBeLessThan(0.1); // weakest working score observed
  });
});

// ---------------------------------------------------------------------------
// store.rerank seam — coverage-before-zero-fill + noCache (the H1/M4/H2 mechanics)
// ---------------------------------------------------------------------------
describe("store.rerank probe seam", () => {
  const originalFetch = globalThis.fetch;
  const originalUrl = process.env.CLAWMEM_RERANK_URL;
  const originalNoLocal = process.env.CLAWMEM_NO_LOCAL_MODELS;
  const originalApiKey = process.env.CLAWMEM_RERANK_API_KEY;

  beforeEach(() => {
    process.env.CLAWMEM_RERANK_URL = "http://rerank.test:8090";
    process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    if (originalUrl === undefined) delete process.env.CLAWMEM_RERANK_URL;
    else process.env.CLAWMEM_RERANK_URL = originalUrl;
    if (originalNoLocal === undefined) delete process.env.CLAWMEM_NO_LOCAL_MODELS;
    else process.env.CLAWMEM_NO_LOCAL_MODELS = originalNoLocal;
    if (originalApiKey === undefined) delete process.env.CLAWMEM_RERANK_API_KEY;
    else process.env.CLAWMEM_RERANK_API_KEY = originalApiKey;
  });

  function mockRerank(results: { index: number; relevance_score: number }[]): () => number {
    let calls = 0;
    globalThis.fetch = (async () => {
      calls++;
      return new Response(JSON.stringify({ results }), { status: 200 });
    }) as unknown as typeof fetch;
    return () => calls;
  }

  const docs = [
    { file: "a", text: "alpha document about one topic" },
    { file: "b", text: "beta document about another topic" },
  ];

  test("requireLiveCoverage THROWS (malformed) when the endpoint returns fewer results than the batch", async () => {
    mockRerank([{ index: 0, relevance_score: 0.7 }]); // 1 result for a 2-doc batch — wrong count
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankMalformedResponseError);
  });

  test("requireLiveCoverage THROWS (coverage) when a later batch fails after an earlier one scored", async () => {
    // 6 docs → 2 batches (4 + 2). Batch 1 returns 4 valid results (scored=true → local skipped);
    // batch 2 returns HTTP 500 → break → docs 4,5 never scored → end-of-fn coverage error.
    const docs6 = Array.from({ length: 6 }, (_, i) => ({ file: `d${i}`, text: `text number ${i}` }));
    let call = 0;
    globalThis.fetch = (async () => {
      call++;
      if (call === 1) {
        return new Response(
          JSON.stringify({ results: [0, 1, 2, 3].map((index) => ({ index, relevance_score: 0.5 })) }),
          { status: 200 },
        );
      }
      return new Response("err", { status: 500 }); // batch 2 fails
    }) as unknown as typeof fetch;
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs6, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankCoverageError);
  });

  test("full coverage returns real scores sorted descending, no throw", async () => {
    mockRerank([
      { index: 0, relevance_score: 0.7 },
      { index: 1, relevance_score: 0.2 },
    ]);
    const store = createStore(":memory:");
    const out = await store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true });
    expect(out[0]!.file).toBe("a");
    expect(out[0]!.score).toBe(0.7);
  });

  test("WITHOUT requireLiveCoverage, partial output silently zero-fills (documents the bug coverage defends against)", async () => {
    mockRerank([{ index: 0, relevance_score: 0.7 }]); // b omitted
    const store = createStore(":memory:");
    const out = await store.rerank("q", docs, "m", undefined, { noCache: true });
    const b = out.find((r) => r.file === "b");
    expect(b!.score).toBe(0); // omitted score is indistinguishable from a true 0 after the map
  });

  test("noCache forces a live call every time (no cache read)", async () => {
    const calls = mockRerank([
      { index: 0, relevance_score: 0.7 },
      { index: 1, relevance_score: 0.2 },
    ]);
    const store = createStore(":memory:");
    await store.rerank("q", docs, "m", undefined, { noCache: true });
    await store.rerank("q", docs, "m", undefined, { noCache: true });
    expect(calls()).toBe(2); // both calls hit the endpoint
  });

  test("without noCache, an identical second call is served from cache (no second fetch)", async () => {
    const calls = mockRerank([
      { index: 0, relevance_score: 0.7 },
      { index: 1, relevance_score: 0.2 },
    ]);
    const store = createStore(":memory:");
    await store.rerank("q", docs, "m"); // populates cache
    await store.rerank("q", docs, "m"); // cache hit
    expect(calls()).toBe(1);
  });

  // Malformed-response contract under requireLiveCoverage (impl-review High) — a responding-but-
  // garbage reranker must surface, not false-pass or crash.
  test("requireLiveCoverage throws on a duplicate index", async () => {
    mockRerank([
      { index: 0, relevance_score: 0.9 },
      { index: 0, relevance_score: 0.8 }, // duplicate; doc b never scored
    ]);
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankMalformedResponseError);
  });

  test("requireLiveCoverage throws on an out-of-range index", async () => {
    mockRerank([
      { index: 0, relevance_score: 0.9 },
      { index: 5, relevance_score: 0.1 }, // out of range for a 2-doc batch
    ]);
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankMalformedResponseError);
  });

  test("requireLiveCoverage throws on a wrong result count", async () => {
    mockRerank([
      { index: 0, relevance_score: 0.9 },
      { index: 1, relevance_score: 0.2 },
      { index: 0, relevance_score: 0.5 }, // 3 results for a 2-doc batch
    ]);
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankMalformedResponseError);
  });

  test("requireLiveCoverage throws on a non-numeric (string) score", async () => {
    mockRerank([
      { index: 0, relevance_score: 0.9 },
      { index: 1, relevance_score: "0.2" as unknown as number }, // string survives JSON
    ]);
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankMalformedResponseError);
  });

  test("requireLiveCoverage throws on an invalid-JSON body from a 200 response", async () => {
    globalThis.fetch = (async () => new Response("not json at all", { status: 200 })) as unknown as typeof fetch;
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankMalformedResponseError);
  });

  test("requireLiveCoverage throws on a null/primitive JSON body", async () => {
    globalThis.fetch = (async () => new Response("null", { status: 200 })) as unknown as typeof fetch;
    const store = createStore(":memory:");
    await expect(
      store.rerank("q", docs, "m", undefined, { noCache: true, requireLiveCoverage: true }),
    ).rejects.toBeInstanceOf(RerankMalformedResponseError);
  });

  test("non-probe path skips an out-of-range entry instead of crashing (defensive)", async () => {
    mockRerank([
      { index: 0, relevance_score: 0.9 },
      { index: 5, relevance_score: 0.1 }, // out of range — must not crash
    ]);
    const store = createStore(":memory:");
    const out = await store.rerank("q", docs, "m", undefined, { noCache: true }); // no requireLiveCoverage
    expect(out.find((r) => r.file === "a")!.score).toBe(0.9);
    expect(out.find((r) => r.file === "b")!.score).toBe(0); // b skipped → zero-filled, no crash
  });

  // W3 remote-reranker auth — the remote GPU reranker may sit behind an authenticated gateway.
  test("sends Authorization: Bearer to the remote reranker when CLAWMEM_RERANK_API_KEY is set", async () => {
    process.env.CLAWMEM_RERANK_API_KEY = "test-rerank-key";
    let seenHeaders: Record<string, string> | undefined;
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      seenHeaders = init?.headers as Record<string, string>;
      return new Response(
        JSON.stringify({ results: [{ index: 0, relevance_score: 0.7 }, { index: 1, relevance_score: 0.2 }] }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;
    const store = createStore(":memory:");
    await store.rerank("q", docs, "m", undefined, { noCache: true });
    expect(seenHeaders?.["Authorization"]).toBe("Bearer test-rerank-key");
    expect(seenHeaders?.["Content-Type"]).toBe("application/json");
  });

  test("omits Authorization to the remote reranker when CLAWMEM_RERANK_API_KEY is unset (backward compatible)", async () => {
    delete process.env.CLAWMEM_RERANK_API_KEY;
    let seenHeaders: Record<string, string> | undefined;
    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      seenHeaders = init?.headers as Record<string, string>;
      return new Response(
        JSON.stringify({ results: [{ index: 0, relevance_score: 0.7 }, { index: 1, relevance_score: 0.2 }] }),
        { status: 200 },
      );
    }) as unknown as typeof fetch;
    const store = createStore(":memory:");
    await store.rerank("q", docs, "m", undefined, { noCache: true });
    expect(seenHeaders?.["Authorization"]).toBeUndefined();
    expect(seenHeaders?.["Content-Type"]).toBe("application/json");
  });
});

// ---------------------------------------------------------------------------
// probeRerankHealth — calibration band + per-pair discrimination + coverage
// ---------------------------------------------------------------------------
describe("probeRerankHealth", () => {
  const triples: GoldenTriple[] = [
    { query: "q1", relevant: "r1", hardNegative: "n1" },
    { query: "q2", relevant: "r2", hardNegative: "n2" },
  ];

  // Fake store whose rerank scores each doc by a supplied function — no network, deterministic.
  function fakeStore(scoreOf: (file: string) => number) {
    return {
      rerank: async (_q: string, d: { file: string; text: string }[]) =>
        d.map((x) => ({ file: x.file, score: scoreOf(x.file) })).sort((a, b) => b.score - a.score),
    } as unknown as Parameters<typeof probeRerankHealth>[0];
  }

  test("healthy reranker (rel high, neg low) → ok", async () => {
    const res = await probeRerankHealth(fakeStore((f) => (f.endsWith("-rel") ? 0.9 : 0.1)), { triples });
    expect(res.ok).toBe(true);
    expect(res.coverageOk).toBe(true);
    expect(res.failures).toEqual([]);
  });

  test("degenerate reranker (~1e-11 everywhere) → fails the calibration band", async () => {
    const res = await probeRerankHealth(fakeStore(() => 1e-11), { triples });
    expect(res.ok).toBe(false);
    expect(res.failures.some((f) => f.includes("calibration"))).toBe(true);
  });

  test("constant-output reranker (0.5 everywhere) → band passes but per-pair margin fails", async () => {
    const res = await probeRerankHealth(fakeStore(() => 0.5), { triples });
    expect(res.ok).toBe(false);
    expect(res.maxScore).toBe(0.5); // calibration band is satisfied...
    expect(res.failures.some((f) => f.includes("margin"))).toBe(true); // ...but discrimination is not
  });

  test("coverage failure (RerankCoverageError) surfaces as a probe failure", async () => {
    const throwing = {
      rerank: async () => {
        throw new RerankCoverageError(["x"]);
      },
    } as unknown as Parameters<typeof probeRerankHealth>[0];
    const res = await probeRerankHealth(throwing, { triples });
    expect(res.ok).toBe(false);
    expect(res.coverageOk).toBe(false);
    expect(res.failures.some((f) => f.includes("coverage"))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// master-harness-1nvlz — logit-space discrimination + honest, arm-routed advice
//
// The bug these lock down: the margin was computed in SIGMOID space and compared to a fixed
// additive threshold. The sigmoid saturates, so a pair the live model separated by 4.06 LOGITS
// compressed to 0.035 and doctor declared a correctly-ordering reranker "degenerate" — then
// prescribed re-deploying the sidecar while printing "max score 1.0e+0" one line above.
// ---------------------------------------------------------------------------
describe("toLogit (sigmoid inverse, clamped)", () => {
  test("inverts the proxy's sigmoid exactly — reproduces the raw endpoint logits", () => {
    // Measured pairs (proxy 8091 score → raw 8090 logit), 2026-09-01 live capture. Asserted as a
    // round trip because the captured scores are printed to 4dp, which is coarser than the logit.
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
    for (const rawLogit of [-0.6606, -1.966, 7.3493, 3.1873]) {
      expect(toLogit(sigmoid(rawLogit))).toBeCloseTo(rawLogit, 6);
    }
    // and the 4dp-rounded live captures land in the right neighbourhood
    expect(toLogit(0.3406)).toBeCloseTo(-0.66, 2);
    expect(toLogit(0.9994)).toBeGreaterThan(7);
  });

  test("s = 1.0 clamps instead of returning Infinity (the NaN-poisoning trap)", () => {
    const l = toLogit(1);
    expect(Number.isFinite(l)).toBe(true);
    expect(l).toBeCloseTo(Math.log(1 / RERANK_LOGIT_CLAMP_EPS - 1), 3);
    // and the margin arithmetic stays finite/decidable rather than NaN
    expect(Number.isNaN(toLogit(1) - toLogit(1))).toBe(false);
  });

  test("s = 0 clamps symmetrically", () => {
    expect(Number.isFinite(toLogit(0))).toBe(true);
    expect(toLogit(0)).toBeCloseTo(-toLogit(1), 6);
  });

  test("scores outside [0,1] are already logit space and pass through unchanged", () => {
    expect(toLogit(7.3493)).toBe(7.3493);
    expect(toLogit(-1.966)).toBe(-1.966);
  });

  test("non-finite input does not throw", () => {
    expect(Number.isNaN(toLogit(NaN))).toBe(true);
  });
});

describe("probeRerankHealth — logit-space margin (1nvlz)", () => {
  const triples: GoldenTriple[] = [{ query: "q", relevant: "r", hardNegative: "n" }];
  function pairStore(rel: number, neg: number) {
    return {
      rerank: async (_q: string, d: { file: string; text: string }[]) =>
        d.map((x) => ({ file: x.file, score: x.file.endsWith("-rel") ? rel : neg })),
    } as unknown as Parameters<typeof probeRerankHealth>[0];
  }

  test("REGRESSION: saturated-tail pair that the OLD sigmoid margin failed now PASSES", async () => {
    // Live golden pair 6 ("is the http put method idempotent"): sigmoid margin 0.039 (< the old
    // 0.25 threshold → false RED), logit margin 4.06 (→ correctly green).
    const res = await probeRerankHealth(pairStore(0.9994, 0.9604), { triples });
    expect(res.ok).toBe(true);
    expect(res.inversions).toBe(0);
    expect(res.minLogitMargin).toBeGreaterThan(4); // ~4.2 from the 4dp-rounded capture
    expect(res.minLogitMargin).toBeGreaterThan(res.thresholds.discrimLogitMargin);
  });

  test("ordering is correct but the separation is genuinely thin → margin failure, NOT an inversion", async () => {
    // logit(0.52) - logit(0.50) = 0.080 < 0.5
    const res = await probeRerankHealth(pairStore(0.52, 0.5), { triples });
    expect(res.ok).toBe(false);
    expect(res.inversions).toBe(0);
    expect(res.calibrationFailed).toBe(false);
    expect(res.failures.some((f) => f.includes("logit margin"))).toBe(true);
    expect(res.failures.some((f) => f.includes("INVERTED"))).toBe(false);
  });

  test("genuine inversion is caught by the scale-free ordering assertion", async () => {
    const res = await probeRerankHealth(pairStore(0.1, 0.9), { triples });
    expect(res.ok).toBe(false);
    expect(res.inversions).toBe(1);
    expect(res.calibrationFailed).toBe(false);
    expect(res.failures.some((f) => f.includes("INVERTED"))).toBe(true);
    expect(res.minLogitMargin).toBeLessThan(0);
  });

  test("collapse (~1e-11 constant) trips the calibration arm and flags calibrationFailed", async () => {
    const res = await probeRerankHealth(pairStore(1e-11, 1e-11), { triples });
    expect(res.ok).toBe(false);
    expect(res.calibrationFailed).toBe(true);
  });

  test("clamp keeps a both-saturated pair decidable (margin 0, not NaN)", async () => {
    const res = await probeRerankHealth(pairStore(1, 1), { triples });
    expect(res.ok).toBe(false);
    expect(Number.isNaN(res.minLogitMargin)).toBe(false);
    expect(res.minLogitMargin).toBe(0);
    expect(res.inversions).toBe(1); // rel > neg is false
  });
});

describe("rerankFailureAdvice — the zerank-2 prescription is gated to the calibration arm", () => {
  const base = { calibrationFailed: false, inversions: 0, pairsScored: 8, pairsTotal: 8 };

  test("calibration collapse → zerank-2 / seq-cls-sidecar prescription", () => {
    const advice = rerankFailureAdvice({ ...base, calibrationFailed: true });
    expect(advice).toContain("zerank-2");
    expect(advice).toContain("seq-cls sidecar");
  });

  test("thin-margin failure → honest message, NO re-deploy prescription", () => {
    const advice = rerankFailureAdvice(base);
    expect(advice).not.toContain("zerank-2");
    expect(advice.toLowerCase()).toContain("do not re-deploy");
  });

  test("inversion failure → ordering message, NO re-deploy prescription", () => {
    const advice = rerankFailureAdvice({ ...base, inversions: 3 });
    expect(advice).not.toContain("zerank-2");
    expect(advice).toContain("hard negative");
  });

  test("coverage failure → coverage message, NO re-deploy prescription", () => {
    const advice = rerankFailureAdvice({ ...base, pairsScored: 5 });
    expect(advice).not.toContain("zerank-2");
    expect(advice).toContain("did not score every probe doc");
  });
});
