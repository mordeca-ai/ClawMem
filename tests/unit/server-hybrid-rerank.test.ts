/**
 * master-harness-h06j — REST hybrid modes (/search mode=hybrid, /retrieve hybrid) must RRF-fuse
 * the BM25 + vector legs and rerank the candidates through the store reranker, like CLI
 * `clawmem query`. Previously they returned an un-reranked max-score merge (ADR-0059-rejected).
 *
 * The retrieval legs and the reranker are stubbed on the Store object so the test controls the
 * fused order exactly and proves the FINAL order follows rerank scores; the real handler,
 * fusion, blend, enrichment, composite scoring and MMR all run.
 */

import { describe, test, expect, beforeAll, afterAll, afterEach } from "bun:test";
import { unlinkSync } from "fs";
import { createStore, type Store, type SearchResult } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { startServer } from "../../src/server.ts";
import { fuseAndRerank, type RankedResult } from "../../src/search-utils.ts";

const TEST_DB = "/tmp/clawmem-server-hybrid-rerank-test.sqlite";
const PORT = 17461;
const BASE = `http://127.0.0.1:${PORT}`;
const QUERY = "kitchen kiosk casting strategy for cast receivers";

let store: Store;
let server: ReturnType<typeof startServer>;
const byTitle = new Map<string, SearchResult>();
const original: Partial<Pick<Store, "searchFTS" | "searchVec" | "rerank">> = {};
let rerankCalls: { query: string; files: string[] }[] = [];

function rmDb() {
  for (const suffix of ["", "-wal", "-shm"]) { try { unlinkSync(TEST_DB + suffix); } catch {} }
}

// Distinct vocabularies so MMR's bigram diversity never collapses them.
const DOCS: [string, string][] = [
  ["Alpha", "alpha tokyo lantern sequence orchid"],
  ["Bravo", "bravo granite harbor violin meadow"],
  ["Charlie", "charlie copper falcon ribbon quartz"],
  ["Delta", "delta saffron glacier walnut compass"],
  ["Echo", "echo marble thunder pepper canyon"],
  ["Foxtrot", "foxtrot velvet anchor cobalt willow"],
  ["Target", "target kiosk receiver casting chromecast"],
];

beforeAll(() => {
  rmDb();
  process.env.INDEX_PATH = TEST_DB;
  delete process.env.CLAWMEM_API_TOKEN;
  store = createStore(TEST_DB);
  const now = new Date().toISOString();
  DOCS.forEach(([title, words]) => {
    const body = `# ${title}\n\n${words}.`;
    const hash = hashContent(body);
    store.insertContent(hash, body, now);
    store.insertDocument("test", `notes/${title.toLowerCase()}.md`, title, hash, now, now);
  });
  for (const [title, words] of DOCS) {
    const hit = store.searchFTS(words.split(" ")[1]!, 5).find(r => r.title === title);
    if (!hit) throw new Error(`seed lookup failed for ${title}`);
    byTitle.set(title, hit);
  }
  original.searchFTS = store.searchFTS;
  original.searchVec = store.searchVec;
  original.rerank = store.rerank;
  server = startServer(store, PORT);
});

afterEach(() => {
  store.searchFTS = original.searchFTS!;
  store.searchVec = original.searchVec!;
  store.rerank = original.rerank!;
  rerankCalls = [];
});

afterAll(() => {
  server.stop();
  store.close();
  rmDb();
});

const r = (t: string) => byTitle.get(t)!;

/**
 * Fused order: Alpha (BM25 #1) and Bravo (vec #1) tie on top; Target is fusion rank 3
 * (BM25 #2 + vec #6). The stub reranker scores Target 1.0 and everything else 0.01.
 */
function stubLegs(rerankImpl?: Store["rerank"]) {
  store.searchFTS = (() => [r("Alpha"), r("Target")]) as Store["searchFTS"];
  store.searchVec = (async () =>
    [r("Bravo"), r("Charlie"), r("Delta"), r("Echo"), r("Foxtrot"), r("Target")]) as Store["searchVec"];
  store.rerank = rerankImpl ?? (async (query, docs) => {
    rerankCalls.push({ query, files: docs.map(d => d.file) });
    return docs
      .map(d => ({ file: d.file, score: d.file === r("Target").filepath ? 1.0 : 0.01 }))
      .sort((a, b) => b.score - a.score);
  });
}

async function post(path: string, body: object) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, data: await res.json() as any };
}

describe("POST /search mode=hybrid reranks fused candidates (h06j)", () => {
  test("rerank order wins over fusion order; reranked:true", async () => {
    stubLegs();
    const { status, data } = await post("/search", { query: QUERY, mode: "hybrid" });
    expect(status).toBe(200);
    expect(data.reranked).toBe(true);
    expect(rerankCalls.length).toBe(1);
    expect(rerankCalls[0]!.query).toBe(QUERY);
    expect(rerankCalls[0]!.files).toContain(r("Target").filepath);
    // Fusion alone ranks Target 3rd; the reranker lifts it to first.
    expect(data.results[0].title).toBe("Target");
  });

  test("reranker throws → 200, fused order, reranked:false with reason", async () => {
    stubLegs(async () => { throw new Error("rerank endpoint unreachable"); });
    const { status, data } = await post("/search", { query: QUERY, mode: "hybrid" });
    expect(status).toBe(200);
    expect(data.reranked).toBe(false);
    expect(data.rerankFallback).toContain("rerank endpoint unreachable");
    expect(["Alpha", "Bravo"]).toContain(data.results[0].title);
    expect(data.results.map((x: any) => x.title)).toContain("Target");
    expect(data.results[0].title).not.toBe("Target");
  });

  test("degenerate (all ~0) reranker scores → fused order, reranked:false", async () => {
    stubLegs(async (_q, docs) => docs.map(d => ({ file: d.file, score: 1e-11 })));
    const { status, data } = await post("/search", { query: QUERY, mode: "hybrid" });
    expect(status).toBe(200);
    expect(data.reranked).toBe(false);
    expect(data.rerankFallback).toBe("degenerate");
    expect(data.results[0].title).not.toBe("Target");
  });
});

describe("POST /search non-hybrid modes do not rerank (h06j A3)", () => {
  test("mode=keyword never calls the reranker and has no reranked field", async () => {
    stubLegs();
    const { status, data } = await post("/search", { query: QUERY, mode: "keyword" });
    expect(status).toBe(200);
    expect(rerankCalls.length).toBe(0);
    expect(data).not.toHaveProperty("reranked");
  });

  test("mode=semantic never calls the reranker", async () => {
    stubLegs();
    const { status, data } = await post("/search", { query: QUERY, mode: "semantic" });
    expect(status).toBe(200);
    expect(rerankCalls.length).toBe(0);
    expect(data).not.toHaveProperty("reranked");
  });

  test("mode=auto with a short query stays keyword-only", async () => {
    stubLegs();
    const { status, data } = await post("/search", { query: "kiosk casting", mode: "auto" });
    expect(status).toBe(200);
    expect(rerankCalls.length).toBe(0);
    expect(data).not.toHaveProperty("reranked");
  });
});

describe("POST /retrieve hybrid reranks fused candidates (h06j)", () => {
  test("mode=hybrid: rerank order wins; reranked:true", async () => {
    stubLegs();
    const { status, data } = await post("/retrieve", { query: QUERY, mode: "hybrid" });
    expect(status).toBe(200);
    expect(data.mode).toBe("hybrid");
    expect(data.reranked).toBe(true);
    expect(rerankCalls.length).toBe(1);
    expect(data.results[0].title).toBe("Target");
  });

  test("auto-classified hybrid (full response): reranker throws → 200, reranked:false", async () => {
    stubLegs(async () => { throw new Error("boom"); });
    const { status, data } = await post("/retrieve", { query: QUERY, compact: false });
    expect(status).toBe(200);
    expect(data.mode).toBe("hybrid");
    expect(data.reranked).toBe(false);
    expect(data.results[0].title).not.toBe("Target");
  });

  test("mode=keyword never calls the reranker", async () => {
    stubLegs();
    const { status, data } = await post("/retrieve", { query: QUERY, mode: "keyword" });
    expect(status).toBe(200);
    expect(rerankCalls.length).toBe(0);
    expect(data).not.toHaveProperty("reranked");
  });
});

describe("fuseAndRerank (shared CLI/REST pipeline)", () => {
  const ranked = (file: string): RankedResult => ({ file, displayPath: file, title: file, body: file, score: 1 });

  test("no candidates → no rerank call, reranked:false", async () => {
    let called = false;
    const out = await fuseAndRerank("q", [{ results: [], weight: 2 }], async () => { called = true; return []; }, { rerankCap: 30 });
    expect(called).toBe(false);
    expect(out.reranked).toBe(false);
    expect(out.fallbackReason).toBe("no-candidates");
    expect(out.blended).toEqual([]);
  });

  test("caps the rerank pool at rerankCap", async () => {
    let seen = 0;
    const list = Array.from({ length: 50 }, (_, i) => ranked(`f${i}`));
    const out = await fuseAndRerank("q", [{ results: list, weight: 2 }], async (_q, docs) => {
      seen = docs.length;
      return docs.map(d => ({ file: d.file, score: 0.5 }));
    }, { rerankCap: 30 });
    expect(seen).toBe(30);
    expect(out.reranked).toBe(true);
    expect(out.blended.length).toBe(30);
  });

  test("without degenerateFloor, near-zero scores still count as reranked (CLI behavior preserved)", async () => {
    const out = await fuseAndRerank("q", [{ results: [ranked("a"), ranked("b")], weight: 2 }],
      async (_q, docs) => docs.map(d => ({ file: d.file, score: 0 })), { rerankCap: 30 });
    expect(out.reranked).toBe(true);
  });
});
