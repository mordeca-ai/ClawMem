import { describe, expect, it } from "bun:test";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { fakeEmbedVector, startFakeEmbedServer } from "../helpers/fake-embed-server.ts";

const CLI = new URL("../../src/clawmem.ts", import.meta.url).pathname;

describe("CLI vsearch minimum score", () => {
  it("keeps a relevant old hit when composite ranking falls below the raw cosine floor", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-vsearch-floor-"));
    const docs = join(dir, "docs");
    mkdirSync(docs);
    writeFileSync(join(dir, "config.yaml"), `collections:\n  testcol:\n    path: ${docs}\n    pattern: "**/*.md"\n`);

    const query = "archive deployment manifest";
    const body = "# Archive deployment manifest\n\nAn old but exact match.\n";
    const hash = hashContent(body + "old.md");
    const otherBody = "# Bananas\n\nA fresh unrelated note.\n";
    const otherHash = hashContent(otherBody + "new.md");
    const db = join(dir, "index.sqlite");
    const store = createStore(db);
    try {
      const old = "2000-01-01T00:00:00.000Z";
      store.insertContent(hash, body, old);
      store.insertDocument("testcol", "old.md", "Old match", hash, old, old);
      store.db.exec("UPDATE documents SET content_type = 'daily_log', confidence = 0.1, quality_score = 0 WHERE path = 'old.md'");
      const now = new Date().toISOString();
      store.insertContent(otherHash, otherBody, now);
      store.insertDocument("testcol", "new.md", "Unrelated fresh note", otherHash, now, now);
      store.db.exec("UPDATE documents SET pinned = 1 WHERE path = 'new.md'");
      store.ensureVecTable(512);
      store.insertEmbedding(hash, 0, 1, new Float32Array(fakeEmbedVector(query, 512)), "fake-embed-a", old, "full", undefined, "testcol/old.md");
      store.insertEmbedding(otherHash, 0, 1, new Float32Array(fakeEmbedVector("banana custard", 512)), "fake-embed-a", now, "full", undefined, "testcol/new.md");
    } finally {
      store.db.close();
    }

    const server = startFakeEmbedServer({ dim: 512, model: "fake-embed-a" });
    async function run(floor?: string) {
      const env: Record<string, string> = {};
      for (const [key, value] of Object.entries(process.env)) {
        if (value !== undefined && !key.startsWith("CLAWMEM_") && key !== "INDEX_PATH") env[key] = value;
      }
      const proc = Bun.spawn(["bun", CLI, "vsearch", query, "-c", "testcol", "--json", ...(floor ? ["--min-score", floor] : [])], {
        env: { ...env, INDEX_PATH: db, CLAWMEM_CONFIG_DIR: dir, CLAWMEM_EMBED_URL: server.url, CLAWMEM_NO_LOCAL_MODELS: "true" },
        stdout: "pipe",
        stderr: "pipe",
      });
      const [stdout, stderr, code] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
      expect(code).toBe(0);
      expect(stderr).not.toContain("Error");
      return JSON.parse(stdout) as Array<{ file: string; score: number; searchScore: number }>;
    }

    try {
      const defaultResults = await run();
      expect(defaultResults).toHaveLength(1);
      expect(defaultResults[0]!.file).toContain("old.md");
      expect(defaultResults[0]!.searchScore).toBeGreaterThan(0.3);
      expect(defaultResults[0]!.score).toBeLessThan(0.3);
      const unfiltered = await run("0");
      const fresh = unfiltered.find(r => r.file.includes("new.md"));
      expect(fresh).toBeDefined();
      expect(fresh!.searchScore).toBeLessThan(0.3);
      expect(fresh!.score).toBeGreaterThan(0.3);
      expect(await run("1.01")).toEqual([]);
    } finally {
      server.stop();
    }
  }, 120_000);
});
