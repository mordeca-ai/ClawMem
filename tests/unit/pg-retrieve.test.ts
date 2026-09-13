/**
 * `pg retrieve` unit tests (master-harness-2wx75 slice 7).
 *
 * No database, no embedder, no reranker service: every dependency of
 * runRetrieve/retrieveCli is injected, so this tier drives the REAL flag
 * parsing, mode dispatch, contract mapping and exit-code policy.
 *
 * What goes red if undone:
 *  - the pinned wire shape (exact key set, schema tag) — a renamed field fails
 *    the key-set assertion, not just a value check;
 *  - thrown => ONE JSON object + exit 2, for a thrown search AND for a flag
 *    that did not parse;
 *  - degraded-empty => exit 0 with degraded true;
 *  - rerankStatus passthrough, and null for search/vsearch;
 *  - --no-rerank never constructs the reranker.
 */

import { describe, it, expect } from "bun:test";
import {
  EXIT_OK,
  EXIT_THREW,
  RETRIEVE_DEFAULT_LIMIT,
  RETRIEVE_SCHEMA,
  parseRetrieveArgs,
  retrieveCli,
  runRetrieve,
  type RetrieveArgs,
  type RetrieveDeps,
  type RetrieveOutput,
} from "../../src/pg/retrieve.ts";
import type { PgQueryable } from "../../src/pg/search.ts";
import type { PgReranker, PgRerankStatus, PgRerankedSearchResult } from "../../src/pg/search-reranked.ts";
import type { PgHybridSearchResult } from "../../src/pg/search-hybrid.ts";
import type { SearchResult } from "../../src/store.ts";

const CONTRACT_KEYS = [
  "schema", "mode", "query", "limit", "results", "degraded", "degradedReason",
  "rerankStatus", "timings", "error",
].sort();
const TIMING_KEYS = ["totalMs", "hybridMs", "rerankMs", "embedMs"].sort();

function sr(collection: string, path: string, score: number): SearchResult {
  return {
    filepath: `clawmem://${collection}/${path}`,
    displayPath: `${collection}/${path}`,
    title: path,
    context: null,
    hash: "a".repeat(64),
    docid: "aaaaaa",
    collectionName: collection,
    modifiedAt: "",
    bodyLength: 4,
    body: "body",
    score,
    source: "fts",
  };
}

const ROWS = [sr("research", "a.md", 0.9), sr("research", "sub/b.md", 0.5)];

function hybrid(over: Partial<PgHybridSearchResult> = {}): PgHybridSearchResult {
  return { results: ROWS, arms: "vec+fts", degraded: false, armFailures: [], rrfK: 60, ...over };
}

function reranked(status: PgRerankStatus, over: Partial<PgRerankedSearchResult> = {}): PgRerankedSearchResult {
  return {
    results: ROWS,
    hybrid: hybrid(),
    rerank: status,
    candidateCount: 2,
    timings: { hybridMs: 12.34, rerankMs: 5.67, totalMs: 20 },
    deadlineMs: 1500,
    ...over,
  };
}

type Calls = { fts: unknown[]; vec: unknown[]; reranked: unknown[]; rerankerBuilt: number; vaults: string[]; disposed: number };

function fakeDeps(over: Partial<RetrieveDeps> = {}): { deps: RetrieveDeps; calls: Calls } {
  const calls: Calls = { fts: [], vec: [], reranked: [], rerankerBuilt: 0, vaults: [], disposed: 0 };
  const client = {} as PgQueryable;
  const deps: RetrieveDeps = {
    async withClient(vault, fn) { calls.vaults.push(vault); return fn(client); },
    embedder: () => ({ async embed() { return { embedding: [1], model: "m" }; } }),
    async reranker() {
      calls.rerankerBuilt++;
      const r: PgReranker = async (_q, docs) => docs.map(d => ({ file: d.file, score: 1 }));
      return r;
    },
    searchFts: (async (_c, _q, o) => { calls.fts.push(o); return { results: ROWS, degraded: false, scannedRows: 2 }; }) as RetrieveDeps["searchFts"],
    searchVec: (async (_c, _q, o) => {
      calls.vec.push(o);
      // Exercise the embed-timing wrapper the way the real vec arm does.
      await o?.embedder?.embed("q", { isQuery: true });
      return { results: ROWS, degraded: false, storedModels: 1, scannedFragments: 2 };
    }) as RetrieveDeps["searchVec"],
    searchReranked: (async (_c, _q, o) => {
      calls.reranked.push(o);
      await o?.embedder?.embed("q", { isQuery: true });
      return reranked(o?.reranker ? "applied" : "skipped-no-reranker");
    }) as RetrieveDeps["searchReranked"],
    async dispose() { calls.disposed++; },
    ...over,
  };
  return { deps, calls };
}

const baseArgs = (over: Partial<RetrieveArgs> = {}): RetrieveArgs => ({
  mode: "query", query: "q", limit: 10, collections: undefined, vault: "sfw",
  deadlineMs: undefined, noRerank: false, ...over,
});

function assertContractShape(o: RetrieveOutput) {
  expect(Object.keys(o).sort()).toEqual(CONTRACT_KEYS);
  expect(Object.keys(o.timings).sort()).toEqual(TIMING_KEYS);
  expect(o.schema).toBe(RETRIEVE_SCHEMA);
  expect(typeof o.timings.totalMs).toBe("number");
  for (const r of o.results) expect(Object.keys(r).sort()).toEqual(["file", "score"]);
}

async function cli(argv: string[], deps: RetrieveDeps) {
  const lines: string[] = [];
  const code = await retrieveCli(argv, deps, { stdout: s => lines.push(s) });
  return { code, lines, out: JSON.parse(lines.join("")) as RetrieveOutput };
}

// ===========================================================================
// Flag parsing
// ===========================================================================

describe("parseRetrieveArgs", () => {
  it("parses every flag", () => {
    const a = parseRetrieveArgs([
      "--mode", "vsearch", "--query", "hello world", "--limit", "5",
      "--collection", "a, b", "--vault", "nsfw", "--deadline-ms", "900", "--no-rerank",
    ]);
    expect(a).toEqual({
      mode: "vsearch", query: "hello world", limit: 5, collections: ["a", "b"],
      vault: "nsfw", deadlineMs: 900, noRerank: true,
    });
  });

  it("defaults: limit 10, vault public => sfw, no collections, no deadline, rerank on", () => {
    const a = parseRetrieveArgs(["--mode", "query", "--query", "x"]);
    expect(a.limit).toBe(RETRIEVE_DEFAULT_LIMIT);
    expect(RETRIEVE_DEFAULT_LIMIT).toBe(10);
    expect(a.vault).toBe("sfw");
    expect(a.collections).toBeUndefined();
    expect(a.deadlineMs).toBeUndefined();
    expect(a.noRerank).toBe(false);
  });

  it("accepts --vault public and --vault sfw as the same vault", () => {
    expect(parseRetrieveArgs(["--mode", "search", "--query", "x", "--vault", "public"]).vault).toBe("sfw");
    expect(parseRetrieveArgs(["--mode", "search", "--query", "x", "--vault", "sfw"]).vault).toBe("sfw");
  });

  it.each([
    [[], /--mode/],
    [["--mode", "bogus", "--query", "x"], /--mode must be one of/],
    [["--mode", "query"], /--query/],
    [["--mode", "query", "--query", "   "], /--query/],
    [["--mode", "query", "--query", "x", "--limit", "0"], /--limit/],
    [["--mode", "query", "--query", "x", "--limit", "abc"], /--limit/],
    [["--mode", "query", "--query", "x", "--deadline-ms", "-5"], /--deadline-ms/],
    [["--mode", "query", "--query", "x", "--vault", "private"], /--vault/],
    [["--mode", "query", "--query", "x", "--colection", "a"], /unknown argument/],
    [["--mode", "query", "--query"], /requires a value/],
    [["--mode", "query", "--query", "x", "--collection", ","], /no collection/],
  ] as [string[], RegExp][])("refuses %j", (argv, re) => {
    expect(() => parseRetrieveArgs(argv)).toThrow(re);
  });
});

// ===========================================================================
// Mode dispatch
// ===========================================================================

describe("runRetrieve dispatch", () => {
  it("search -> FTS arm only, with limit/collections/deadline forwarded as timeoutMs", async () => {
    const { deps, calls } = fakeDeps();
    const { output, exitCode } = await runRetrieve(
      baseArgs({ mode: "search", limit: 3, collections: ["research"], deadlineMs: 700 }), deps);
    expect(exitCode).toBe(EXIT_OK);
    expect(calls.fts).toEqual([{ limit: 3, collections: ["research"], timeoutMs: 700 }]);
    expect(calls.vec).toHaveLength(0);
    expect(calls.reranked).toHaveLength(0);
    expect(calls.rerankerBuilt).toBe(0);
    expect(output.rerankStatus).toBeNull();
    expect(output.timings.embedMs).toBeNull();
    expect(output.timings.hybridMs).toBeNull();
  });

  it("vsearch -> vector arm only; embedMs is measured; rerankStatus null", async () => {
    const { deps, calls } = fakeDeps();
    const { output } = await runRetrieve(baseArgs({ mode: "vsearch" }), deps);
    expect(calls.vec).toHaveLength(1);
    expect(calls.fts).toHaveLength(0);
    expect(calls.reranked).toHaveLength(0);
    expect(output.rerankStatus).toBeNull();
    expect(typeof output.timings.embedMs).toBe("number");
  });

  it("query -> reranked path with the reranker injected and deadline forwarded as deadlineMs", async () => {
    const { deps, calls } = fakeDeps();
    const { output } = await runRetrieve(baseArgs({ mode: "query", deadlineMs: 4000, vault: "nsfw" }), deps);
    expect(calls.reranked).toHaveLength(1);
    const o = calls.reranked[0] as { reranker?: unknown; deadlineMs?: number; timeoutMs?: number };
    expect(typeof o.reranker).toBe("function");
    expect(o.deadlineMs).toBe(4000);
    expect(o.timeoutMs).toBeUndefined();
    expect(calls.rerankerBuilt).toBe(1);
    expect(calls.vaults).toEqual(["nsfw"]);
    expect(output.rerankStatus).toBe("applied");
    expect(output.timings.hybridMs).toBe(12.3);
    expect(output.timings.rerankMs).toBe(5.7);
  });

  it("--no-rerank: the reranker is never BUILT and never injected", async () => {
    const { deps, calls } = fakeDeps();
    const { output } = await runRetrieve(baseArgs({ mode: "query", noRerank: true }), deps);
    expect(calls.rerankerBuilt).toBe(0);
    expect((calls.reranked[0] as { reranker?: unknown }).reranker).toBeUndefined();
    expect(output.rerankStatus).toBe("skipped-no-reranker");
  });
});

// ===========================================================================
// Contract mapping
// ===========================================================================

describe("contract shape", () => {
  it("success: pinned keys, file = <collection>/<path>, score passthrough, exit 0", async () => {
    const { deps } = fakeDeps();
    const { code, lines, out } = await cli(["--mode", "search", "--query", "hello"], deps);
    expect(code).toBe(EXIT_OK);
    expect(lines).toHaveLength(1);
    expect(lines[0]!.endsWith("\n")).toBe(true);
    assertContractShape(out);
    expect(out).toMatchObject({ mode: "search", query: "hello", limit: 10, degraded: false, degradedReason: null, error: null });
    expect(out.results).toEqual([
      { file: "research/a.md", score: 0.9 },
      { file: "research/sub/b.md", score: 0.5 },
    ]);
    expect(out.results.some(r => r.file.startsWith("clawmem://"))).toBe(false);
  });

  it("degraded-empty: exit 0, degraded true, reason carried", async () => {
    const { deps } = fakeDeps({
      searchVec: (async () => ({
        results: [], degraded: true, degradedReason: "embed-unavailable", storedModels: 1, scannedFragments: 0,
      })) as RetrieveDeps["searchVec"],
    });
    const { code, out } = await cli(["--mode", "vsearch", "--query", "x"], deps);
    expect(code).toBe(EXIT_OK);
    assertContractShape(out);
    expect(out.results).toEqual([]);
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("embed-unavailable");
    expect(out.error).toBeNull();
  });

  it("query single-arm degrade: degraded true with the arm-native reason(s)", async () => {
    const { deps } = fakeDeps({
      searchReranked: (async () => reranked("applied", {
        hybrid: hybrid({
          arms: "fts-only", degraded: true,
          armFailures: [{ arm: "vec", kind: "threw", error: new TypeError("boom") }],
        }),
      })) as RetrieveDeps["searchReranked"],
    });
    const { code, out } = await cli(["--mode", "query", "--query", "x"], deps);
    expect(code).toBe(EXIT_OK);
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("vec:threw:TypeError: boom");
  });

  it("THROWN search: ONE JSON object, results [], error '<type>: <message>', exit 2", async () => {
    class PgVecSearchTimeoutError extends Error { override name = "PgVecSearchTimeoutError"; }
    const { deps, calls } = fakeDeps({
      searchFts: (async () => { throw new PgVecSearchTimeoutError("statement timeout"); }) as RetrieveDeps["searchFts"],
    });
    const { code, lines, out } = await cli(["--mode", "search", "--query", "x"], deps);
    expect(code).toBe(EXIT_THREW);
    expect(EXIT_THREW).toBe(2);
    expect(lines).toHaveLength(1);
    assertContractShape(out);
    expect(out.results).toEqual([]);
    expect(out.error).toBe("PgVecSearchTimeoutError: statement timeout");
    expect(out.mode).toBe("search");
    expect(calls.disposed).toBe(1);
  });

  it("THROWN connection (withClient rejects, e.g. PG unreachable): exit 2", async () => {
    const { deps } = fakeDeps({
      withClient: (async () => { throw new Error("connect ECONNREFUSED"); }) as RetrieveDeps["withClient"],
    });
    const { code, out } = await cli(["--mode", "query", "--query", "x"], deps);
    expect(code).toBe(EXIT_THREW);
    expect(out.error).toBe("Error: connect ECONNREFUSED");
    expect(out.rerankStatus).toBeNull();
  });

  it("usage error: still ONE JSON object, error 'UsageError: …', exit 2, nothing searched", async () => {
    const { deps, calls } = fakeDeps();
    const { code, lines, out } = await cli(["--mode", "nope", "--query", "x"], deps);
    expect(code).toBe(EXIT_THREW);
    expect(lines).toHaveLength(1);
    assertContractShape(out);
    expect(out.error).toMatch(/^UsageError: --mode must be one of/);
    expect(out.mode).toBeNull();
    expect(calls.fts.length + calls.vec.length + calls.reranked.length).toBe(0);
    expect(calls.disposed).toBe(1);
  });

  it.each([
    "applied", "skipped-no-reranker", "skipped-budget", "skipped-no-text", "degenerate", "failed",
  ] as PgRerankStatus[])("rerankStatus %s passes through verbatim, exit 0", async status => {
    const { deps } = fakeDeps({
      searchReranked: (async () => reranked(status, status === "applied" ? {} : { rerankReason: "why" })) as RetrieveDeps["searchReranked"],
    });
    const { code, out } = await cli(["--mode", "query", "--query", "x"], deps);
    expect(code).toBe(EXIT_OK);
    expect(out.rerankStatus).toBe(status);
    expect(out.error).toBeNull();
  });

  it("stdout discipline: a console.log inside the call chain goes to stderr, not stdout", async () => {
    const written: string[] = [];
    const origWrite = process.stdout.write.bind(process.stdout);
    const { deps } = fakeDeps({
      searchFts: (async () => {
        console.log("noise that would corrupt the JSON");
        return { results: [], degraded: false, scannedRows: 0 };
      }) as RetrieveDeps["searchFts"],
    });
    (process.stdout as { write: unknown }).write = (s: string) => { written.push(String(s)); return true; };
    const origErr = console.error;
    console.error = () => {};
    let lines: string[] = [];
    try {
      lines = [];
      await retrieveCli(["--mode", "search", "--query", "x"], deps, { stdout: s => lines.push(s) });
    } finally {
      (process.stdout as { write: unknown }).write = origWrite;
      console.error = origErr;
    }
    expect(written.join("")).not.toContain("noise");
    expect(lines).toHaveLength(1);
    expect(() => JSON.parse(lines[0]!)).not.toThrow();
  });
});
