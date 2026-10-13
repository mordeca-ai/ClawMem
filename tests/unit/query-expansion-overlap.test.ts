import { describe, expect, it } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { runVecLegsOverlapped, settlesImmediately } from "../../src/vec-legs.ts";
import { fakeEmbedVector, goldenRerankScore } from "../helpers/fake-embed-server.ts";

/**
 * master-harness-vn4rz.97 — the CLI hybrid `query` starts the original query's vector leg
 * while query expansion is still being generated, instead of after it. Covered here:
 *   1. runVecLegsOverlapped: the original leg starts before the expansion resolves, workers
 *      are warmed, and results keep the sequential leg order
 *   2. settlesImmediately: a resolved promise (llm_cache hit) vs one still in flight
 *   3. CLI wiring (subprocess, live LLM delayed 1.5s): the original query is embedded before
 *      the LLM answers, and the output matches the sequential run (CLAWMEM_QUERY_VEC_PARALLEL=1)
 */

const CLI = new URL("../../src/clawmem.ts", import.meta.url).pathname;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("runVecLegsOverlapped", () => {
  it("starts the original leg and warms workers before the expansion lands; keeps leg order", async () => {
    const events: string[] = [];
    let resolveExpansion!: (legs: string[]) => void;
    const expansion = new Promise<string[]>((r) => { resolveExpansion = r; });
    const pool = { size: 4, warm: async (n: number) => { events.push(`warm:${n}`); } };
    const runLeg = async (q: string) => {
      events.push(`start:${q}`);
      await sleep(q === "orig" ? 30 : 5); // the original leg finishes last
      return `result:${q}`;
    };

    const pending = runVecLegsOverlapped({ original: "orig", expansion, legsOf: (l) => l, runLeg, pool, warm: 3 });
    await sleep(10);
    expect(events).toEqual(["start:orig", "warm:3"]);

    resolveExpansion(["vec a", "hyde b"]);
    const { expansion: got, results } = await pending;
    expect(got).toEqual(["vec a", "hyde b"]);
    expect(results).toEqual(["result:orig", "result:vec a", "result:hyde b"]);
  });

  it("an original-leg failure surfaces after the expansion, never as an unhandled rejection", async () => {
    const pool = { size: 2, warm: async () => {} };
    const runLeg = async (q: string) => {
      if (q === "orig") throw new Error("vector store broken");
      return q;
    };
    await expect(
      runVecLegsOverlapped({ original: "orig", expansion: sleep(20).then(() => ["x"]), legsOf: (l) => l, runLeg, pool, warm: 1 }),
    ).rejects.toThrow("vector store broken");
  });

  it("an empty expansion still returns the original leg alone", async () => {
    const pool = { size: 4, warm: async () => {} };
    const { results } = await runVecLegsOverlapped({
      original: "orig", expansion: Promise.resolve([] as string[]), legsOf: (l) => l,
      runLeg: async (q: string) => q, pool, warm: 0,
    });
    expect(results).toEqual(["orig"]);
  });
});

describe("settlesImmediately", () => {
  it("true for a promise already settled (cache hit), false for one still waiting on IO", async () => {
    expect(await settlesImmediately(Promise.resolve(1))).toBe(true);
    expect(await settlesImmediately(Promise.reject(new Error("x")).catch(() => 0))).toBe(true);
    expect(await settlesImmediately((async () => 1)())).toBe(true);
    expect(await settlesImmediately(sleep(50))).toBe(false);
  });
});

describe("CLI query — the original vector leg overlaps a live expansion (subprocess)", () => {
  it("embeds the original query before the LLM answers, with output identical to the sequential run", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-expansion-overlap-"));
    const docs = join(dir, "docs");
    mkdirSync(docs);
    writeFileSync(join(dir, "config.yaml"), `collections:\n  testcol:\n    path: ${docs}\n    pattern: "**/*.md"\n`);

    const query = "orbital ferry schedule";
    const LLM_DELAY_MS = 1500;
    const db = join(dir, "index.sqlite");
    const store = createStore(db);
    try {
      const now = new Date().toISOString();
      const bodies = {
        "ferry.md": "# Orbital ferry schedule\n\nThe orbital ferry departs every six hours.\n",
        "dock.md": "# Docking bay rota\n\nFerry crews rotate through the docking bays weekly.\n",
        "cargo.md": "# Cargo manifest rules\n\nCargo on the orbital ferry is weighed before departure.\n",
      };
      store.ensureVecTable(512);
      for (const [path, body] of Object.entries(bodies)) {
        const hash = hashContent(body + path);
        store.insertContent(hash, body, now);
        store.insertDocument("testcol", path, path, hash, now, now);
        store.insertEmbedding(hash, 0, 1, new Float32Array(fakeEmbedVector(body, 512)), "fake-embed-a", now, "full", undefined, `testcol/${path}`);
      }
    } finally {
      store.db.close();
    }

    // One fake endpoint for embed + LLM + rerank. The LLM answers after LLM_DELAY_MS; every
    // embed input is stamped so the test can see whether it arrived before that answer.
    const embedded: { text: string; at: number }[] = [];
    const llm: { startedAt: number; answeredAt: number }[] = [];
    const server = Bun.serve({
      port: 0,
      async fetch(req) {
        const url = new URL(req.url);
        if (url.pathname.endsWith("/v1/rerank")) {
          const rr = (await req.json()) as { documents: string[] };
          return Response.json({ results: rr.documents.map((t, index) => ({ index, relevance_score: goldenRerankScore(t) })) });
        }
        if (url.pathname.endsWith("/chat/completions")) {
          const startedAt = Date.now();
          await sleep(LLM_DELAY_MS);
          llm.push({ startedAt, answeredAt: Date.now() });
          const content = [
            "lex: orbital ferry timetable",
            "vec: orbital ferry departure times",
            "hyde: The orbital ferry leaves the station every six hours on a fixed schedule.",
          ].join("\n");
          return Response.json({ choices: [{ message: { content } }] });
        }
        if (url.pathname.endsWith("/v1/embeddings")) {
          const body = (await req.json()) as { input: string | string[] };
          const inputs = Array.isArray(body.input) ? body.input : [body.input];
          for (const text of inputs) embedded.push({ text, at: Date.now() });
          return Response.json({
            data: inputs.map((t, index) => ({ embedding: fakeEmbedVector(t, 512), index })),
            model: "fake-embed-a",
          });
        }
        return new Response("not found", { status: 404 });
      },
    });

    async function run(extra: Record<string, string>): Promise<string> {
      const env: Record<string, string> = {};
      for (const [key, value] of Object.entries(process.env)) {
        if (value !== undefined && !key.startsWith("CLAWMEM_") && key !== "INDEX_PATH") env[key] = value;
      }
      Object.assign(env, {
        INDEX_PATH: db,
        CLAWMEM_CONFIG_DIR: dir,
        CLAWMEM_EMBED_URL: `http://127.0.0.1:${server.port}`,
        CLAWMEM_LLM_URL: `http://127.0.0.1:${server.port}`,
        CLAWMEM_RERANK_URL: `http://127.0.0.1:${server.port}`,
        CLAWMEM_NO_LOCAL_MODELS: "true",
        CLAWMEM_DISABLE_FTS_BYPASS: "true",
        ...extra,
      });
      const proc = Bun.spawn([process.execPath, CLI, "query", query, "--json"], { env, stdout: "pipe", stderr: "pipe" });
      const [stdout, stderr, code] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
      ]);
      expect(code, `query failed: ${stderr}`).toBe(0);
      return stdout;
    }

    try {
      // Fresh vault: llm_cache is empty, so the expansion is a live (delayed) LLM call.
      const overlapped = await run({});
      expect(llm).toHaveLength(1);
      const answeredAt = llm[0]!.answeredAt;
      const original = embedded.filter((e) => e.text.includes(query));
      expect(original.length).toBeGreaterThan(0);
      // The original leg's embed happened while the LLM was still generating.
      expect(original[0]!.at).toBeLessThan(answeredAt);
      // The expansion legs can only start once the expansion is known.
      const hydeLeg = embedded.filter((e) => e.text.includes("leaves the station every six hours"));
      expect(hydeLeg.length).toBeGreaterThan(0);
      expect(hydeLeg[0]!.at).toBeGreaterThanOrEqual(answeredAt);

      // Same expansion (now an llm_cache hit), sequential in-process legs: identical output.
      const sequential = await run({ CLAWMEM_QUERY_VEC_PARALLEL: "1" });
      expect(llm).toHaveLength(1);
      // Compare rank order and the fused/reranked search score; recency moves with the clock.
      const ranked = (out: string) =>
        (JSON.parse(out) as { file: string; searchScore: number }[]).map((r) => [r.file, r.searchScore]);
      expect(ranked(overlapped).length).toBeGreaterThan(0);
      expect(ranked(sequential)).toEqual(ranked(overlapped));
    } finally {
      server.stop(true);
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);
});
