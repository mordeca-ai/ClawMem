import { describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, expandQueryCacheKey } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { fakeEmbedVector, goldenRerankScore } from "../helpers/fake-embed-server.ts";

/**
 * master-harness-vn4rz.96 — the MCP `query` tool runs its vector legs (original + each
 * vec/hyde expansion) through the same worker-pool runner as the CLI `query`.
 *
 * The MCP server is driven in a subprocess against a file-backed vault and a fake embed
 * endpoint that holds every embed request for DELAY_MS and records how many were in flight at
 * once. That is the only signal that tells a parallel run from a sequential one, since both
 * return the same ranked lists:
 *   - CLAWMEM_QUERY_VEC_PARALLEL=1  → max 1 embed in flight (the old sequential loop)
 *   - default (4)                   → >= 2 in flight (legs overlap)
 * and the ordered result paths must be identical in both arms, for the default call, a
 * collection-scoped call and an includeInternal call (so collections / excludeCollections
 * reach the workers), and for a second call that reuses the warm pool.
 */

const DRIVER = new URL("./mcp-query-vec-legs.driver.ts", import.meta.url).pathname;
const DIM = 512;
const MODEL = "fake-embed-a";
const DELAY_MS = 250;
const QUERY = "orbital ferry schedule";
const LEGS = [
  { type: "lex", query: "orbital ferry timetable" },
  { type: "vec", query: "orbital ferry departure times" },
  { type: "hyde", query: "The orbital ferry departs the station every six hours on a fixed schedule." },
  { type: "vec", query: "ferry schedule for orbital transfers" },
];

function startDelayedEmbedServer() {
  let inFlight = 0;
  let maxInFlight = 0;
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname.endsWith("/v1/rerank")) {
        const rr = (await req.json()) as { documents: string[] };
        return Response.json({
          results: rr.documents.map((text, index) => ({ index, relevance_score: goldenRerankScore(text) })),
        });
      }
      if (!url.pathname.endsWith("/v1/embeddings")) return new Response("not found", { status: 404 });
      const body = (await req.json()) as { input: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input];
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise(r => setTimeout(r, DELAY_MS));
      inFlight--;
      return Response.json({
        data: inputs.map((t, index) => ({ embedding: fakeEmbedVector(t, DIM), index })),
        model: MODEL,
      });
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}`,
    maxInFlight: () => maxInFlight,
    reset: () => { maxInFlight = 0; },
    stop: () => server.stop(true),
  };
}

function seedVault(db: string) {
  const store = createStore(db);
  try {
    const now = new Date().toISOString();
    store.ensureVecTable(DIM);
    const add = (col: string, path: string, body: string) => {
      const hash = hashContent(body + col + path);
      store.insertContent(hash, body, now);
      store.insertDocument(col, path, path, hash, now, now);
      store.insertEmbedding(hash, 0, 1, new Float32Array(fakeEmbedVector(body, DIM)), MODEL, now, "full", undefined, `${col}/${path}`);
    };
    add("colA", "ferry.md", "The orbital ferry departs every six hours on a fixed schedule.");
    add("colA", "times.md", "Orbital ferry departure times and station transfers.");
    add("colB", "other.md", "Ferry schedule for orbital transfers between stations.");
    add("colB", "galley.md", "Cooking notes about bread and galley stoves.");
    // Visibility exclusion: nearest to the query, but an _clawmem internal doc.
    add("_clawmem", "deductions/ferry.md", "orbital ferry schedule orbital ferry schedule deduction");
    store.db
      .prepare(`INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)`)
      .run(expandQueryCacheKey(QUERY), JSON.stringify(LEGS), now);
  } finally {
    store.db.close();
  }
}

async function runArm(db: string, dir: string, embedUrl: string, parallel?: string) {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined && !k.startsWith("CLAWMEM_") && k !== "INDEX_PATH") env[k] = v;
  }
  Object.assign(env, {
    INDEX_PATH: db,
    CLAWMEM_CONFIG_DIR: dir,
    CLAWMEM_EMBED_URL: embedUrl,
    CLAWMEM_LLM_URL: embedUrl,
    CLAWMEM_RERANK_URL: embedUrl,
    CLAWMEM_NO_LOCAL_MODELS: "true",
    CLAWMEM_DISABLE_FTS_BYPASS: "true",
    DRIVER_QUERY: QUERY,
  });
  if (parallel !== undefined) env.CLAWMEM_QUERY_VEC_PARALLEL = parallel;
  const proc = Bun.spawn([process.execPath, DRIVER], { env, stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  expect(code, `driver failed: ${stderr}`).toBe(0);
  const line = stdout.split("\n").find(l => l.startsWith("DRIVER_RESULT "));
  expect(line, `no result line: ${stdout}\n${stderr}`).toBeTruthy();
  return JSON.parse(line!.slice("DRIVER_RESULT ".length)) as Record<string, string[]>;
}

describe("MCP query tool — vector legs run in parallel workers", () => {
  it("overlaps the legs by default, stays sequential at PARALLEL=1, and ranks identically", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-mcp-vec-legs-"));
    const db = join(dir, "index.sqlite");
    seedVault(db);
    const server = startDelayedEmbedServer();
    try {
      const seq = await runArm(db, dir, server.url, "1");
      const seqMax = server.maxInFlight();
      server.reset();
      const par = await runArm(db, dir, server.url);
      const parMax = server.maxInFlight();

      // Sequential arm: the pre-vn4rz.96 loop, one embed request at a time.
      expect(seqMax).toBe(1);
      // Parallel arm: four vector legs (original + vec + hyde + vec) overlap, bounded by the pool.
      expect(parMax).toBeGreaterThanOrEqual(2);
      expect(parMax).toBeLessThanOrEqual(4);

      // Exactness, including the options that have to reach the workers.
      expect(par).toEqual(seq);
      expect(seq.default!.length).toBeGreaterThan(0);
      expect(seq.default!.some(p => p.includes("deductions/"))).toBe(false); // excludeCollections reached the legs
      expect(seq.includeInternal!.some(p => p.includes("deductions/"))).toBe(true);
      expect(seq.scoped!.length).toBeGreaterThan(0);
      expect(seq.scoped!.every(p => p.includes("colA"))).toBe(true); // collections reached the legs
      expect(par["default-again"]).toEqual(par.default!); // warm pool reuse is stable
    } finally {
      server.stop();
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);

  it("PARALLEL=2 bounds concurrent embeds to 2", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-mcp-vec-legs-"));
    const db = join(dir, "index.sqlite");
    seedVault(db);
    const server = startDelayedEmbedServer();
    try {
      await runArm(db, dir, server.url, "2");
      expect(server.maxInFlight()).toBe(2);
    } finally {
      server.stop();
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);
});
