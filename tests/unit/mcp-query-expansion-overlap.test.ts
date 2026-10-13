import { describe, expect, it } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildMcpServer } from "../../src/mcp.ts";
import { createStore } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { fakeEmbedVector, goldenRerankScore } from "../helpers/fake-embed-server.ts";

/**
 * master-harness-vn4rz.101 — the MCP `query` tool starts the original query's vector leg
 * while query expansion is still being generated, matching CLI query (vn4rz.97).
 *
 * Asserting that on the MCP path the original leg STARTS before expansion RESOLVES:
 *   - The original query's vector leg embeds before the LLM generates the expansion.
 *   - Expansion vector legs (e.g. hyde) only embed after the LLM answers.
 *   - The overlapped run produces the identical ranked list to the sequential run
 *     (CLAWMEM_QUERY_VEC_PARALLEL=1).
 */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

if (process.env.DRIVER_RUN === "1") {
  const query = process.env.DRIVER_QUERY!;
  const built = buildMcpServer();
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await built.server.connect(serverTransport);
  const client = new Client({ name: "mcp-overlap-driver", version: "0.0.0" });
  await client.connect(clientTransport);
  const res = await client.callTool({
    name: "query",
    arguments: { query, compact: true, limit: 10 },
  });
  built.closeAllStores();
  const results = (res.structuredContent?.results ?? []).map((x: any) => ({
    file: x.path ?? x.file ?? x.displayPath ?? "",
    score: x.score,
  }));
  console.log("DRIVER_RESULT " + JSON.stringify(results));
  process.exit(0);
}

describe("MCP query — the original vector leg overlaps a live expansion (subprocess)", () => {
  it("embeds the original query before the LLM answers, with output identical to the sequential run", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-mcp-overlap-"));
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

    async function run(extra: Record<string, string>): Promise<{ file: string; score: number }[]> {
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
        DRIVER_RUN: "1",
        DRIVER_QUERY: query,
        ...extra,
      });
      const proc = Bun.spawn([process.execPath, import.meta.filename], { env, stdout: "pipe", stderr: "pipe" });
      const [stdout, stderr, code] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
      ]);
      expect(code, `driver failed: ${stderr}`).toBe(0);
      const line = stdout.split("\n").find((l) => l.startsWith("DRIVER_RESULT "));
      expect(line, `no result line: ${stdout}\n${stderr}`).toBeTruthy();
      return JSON.parse(line!.slice("DRIVER_RESULT ".length));
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
      const ranked = (results: { file: string; score: number }[]) => results.map((r) => [r.file, r.score]);
      expect(ranked(overlapped).length).toBeGreaterThan(0);
      expect(ranked(sequential)).toEqual(ranked(overlapped));
    } finally {
      server.stop(true);
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);
});
