/**
 * master-harness-b1q42.83 — query-anchored relevance gate on query expansion.
 *
 * Specimens are the LIVE qmd-query-expansion-1.7B outputs measured on 2026-09-26
 * (ollama on the GPU host), not invented noise.
 */
import { describe, it, expect, beforeEach, afterEach, spyOn } from "bun:test";
import {
  anchorExpansions,
  contentTokens,
  lightStem,
  sanitizeExpandedQueries,
  setDefaultLlamaCpp,
  type Queryable,
} from "../../src/llm.ts";
import {
  createStore,
  expandQueryCacheKey,
  getCacheKey,
  setCachedResult,
  DEFAULT_QUERY_MODEL,
  type Store,
} from "../../src/store.ts";

const JARGON_QUERY = "integrator lease guard";
const JARGON_LIVE: Queryable[] = [
  { type: "lex", text: "combine rent watch" },
  { type: "lex", text: "multi lease care" },
  { type: "vec", text: "combine rent watch" },
  { type: "vec", text: "multi lease care" },
  {
    type: "hyde",
    text: "The topic of integrator lease guard covers combined rental guard. Proper implementation follows established patterns and best practices.",
  },
];

const PRUNER_QUERY = "how does the worktree pruner decide a branch is stale";
const PRUNER_LIVE: Queryable[] = [
  { type: "lex", text: "what causes a" },
  { type: "lex", text: "how does the" },
  { type: "vec", text: "how does the worktree pruner mark a branch as stale" },
];

const GOOD_QUERY = "bd dolt contention retry";
const GOOD_SET: Queryable[] = [
  { type: "lex", text: "dolt lock contention retries" },
  { type: "lex", text: "bd dolt retry backoff" },
  { type: "vec", text: "retrying bd commands when the dolt database is contended" },
  { type: "hyde", text: "When several bd processes write at once, dolt reports contention and bd retries the transaction with backoff." },
];

describe("lightStem / contentTokens", () => {
  it("collapses plural, past and gerund forms onto one stem", () => {
    expect(lightStem("leases")).toBe(lightStem("lease"));
    expect(lightStem("leased")).toBe(lightStem("lease"));
    expect(lightStem("retries")).toBe(lightStem("retry"));
    expect(lightStem("worktrees")).toBe(lightStem("worktree"));
    expect(lightStem("caching")).toBe(lightStem("cache"));
    expect(lightStem("classes")).toBe(lightStem("class"));
  });

  it("drops stopwords, question words and <3-char tokens", () => {
    expect([...contentTokens("how does the worktree pruner decide a branch is stale")].sort())
      .toEqual(["branch", "decid", "pruner", "stal", "worktre"].sort());
    expect(contentTokens("what is it").size).toBe(0);
    expect([...contentTokens("bd dolt contention retry")].sort()).toEqual(["contention", "dolt", "retry"]);
  });
});

describe("anchorExpansions (pure gate)", () => {
  it("drops the jargon specimen's lex+vec legs ('combine rent watch', 'multi lease care')", () => {
    const kept = anchorExpansions(JARGON_LIVE, JARGON_QUERY);
    expect(kept.filter(k => k.type !== "hyde")).toEqual([]);
  });

  it("drops stopword-fragment legs but keeps an anchored paraphrase", () => {
    expect(anchorExpansions(PRUNER_LIVE, PRUNER_QUERY)).toEqual([
      { type: "vec", text: "how does the worktree pruner mark a branch as stale" },
    ]);
  });

  it("passes a good expansion set through untouched", () => {
    expect(anchorExpansions(GOOD_SET, GOOD_QUERY)).toEqual(GOOD_SET);
  });

  it("uses a 1-token threshold for queries with <= 2 content tokens", () => {
    const items: Queryable[] = [
      { type: "lex", text: "lease renewal" },
      { type: "vec", text: "rental agreement" },
    ];
    expect(anchorExpansions(items, "integrator leases")).toEqual([{ type: "lex", text: "lease renewal" }]);
  });

  it("exempts hyde legs from the overlap rule", () => {
    const hyde: Queryable = { type: "hyde", text: "A passage with no shared vocabulary whatsoever." };
    expect(anchorExpansions([hyde], JARGON_QUERY)).toEqual([hyde]);
  });

  it("does not filter when the query has no content tokens", () => {
    const items: Queryable[] = [{ type: "lex", text: "anything goes" }];
    expect(anchorExpansions(items, "what is it")).toEqual(items);
  });
});

describe("hyde boilerplate junk pattern", () => {
  it("sanitizeExpandedQueries drops the zero-information hyde boilerplate", () => {
    expect(sanitizeExpandedQueries([JARGON_LIVE[4]!])).toEqual([]);
  });

  it("keeps a real passage that merely mentions best practices", () => {
    const real: Queryable = { type: "hyde", text: "The integrator lease guard follows best practices for lease expiry." };
    expect(sanitizeExpandedQueries([real])).toEqual([real]);
  });
});

describe("store.expandQuery with the anchor gate", () => {
  let store: Store;
  let calls = 0;
  let next: Queryable[] = [];
  let errSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    calls = 0;
    next = [];
    setDefaultLlamaCpp({
      expandQuery: async () => { calls++; return next; },
    } as never);
    store = createStore(":memory:");
    errSpy = spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    errSpy.mockRestore();
    store.close();
    setDefaultLlamaCpp(null);
  });

  it("returns [] (not garbage, not the fallback) when every leg is dropped, and caches it", async () => {
    next = JARGON_LIVE;
    expect(await store.expandQuery(JARGON_QUERY)).toEqual([]);
    expect(calls).toBe(1);
    // Cached empty decision is honoured — no regeneration.
    expect(await store.expandQuery(JARGON_QUERY)).toEqual([]);
    expect(calls).toBe(1);
  });

  it("logs the dropped-leg count without the query text", async () => {
    next = PRUNER_LIVE;
    await store.expandQuery(PRUNER_QUERY);
    const lines = errSpy.mock.calls.map((c: unknown[]) => String(c[0]));
    const gate = lines.filter((l: string) => l.startsWith("[expandQuery] anchor gate dropped"));
    expect(gate).toEqual(["[expandQuery] anchor gate dropped 2/3 expansion legs (kept 1)"]);
    expect(gate[0]).not.toContain("pruner");
  });

  it("passes a good set through and emits no gate log", async () => {
    next = GOOD_SET;
    const out = await store.expandQuery(GOOD_QUERY);
    expect(out).toEqual(GOOD_SET.map(g => ({ type: g.type, query: g.text })));
    expect(errSpy.mock.calls.some((c: unknown[]) => String(c[0]).includes("anchor gate"))).toBe(false);
  });

  it("the cache version bump means a pre-seeded v3 (unfiltered) row is never returned", async () => {
    const oldKey = getCacheKey("expandQuery:v3-qmd-terse-typed", {
      query: JARGON_QUERY,
      model: DEFAULT_QUERY_MODEL,
      provider: "qmd-terse",
    });
    expect(expandQueryCacheKey(JARGON_QUERY)).not.toBe(oldKey);
    setCachedResult(store.db, oldKey, JSON.stringify([
      { type: "lex", query: "combine rent watch" },
      { type: "vec", query: "multi lease care" },
    ]));
    next = JARGON_LIVE;
    expect(await store.expandQuery(JARGON_QUERY)).toEqual([]);
    expect(calls).toBe(1);
  });
});
