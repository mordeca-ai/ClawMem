/**
 * master-harness-vn4rz.84 — a frontmatter tombstone takes a document out of
 * EVERY sqlite search path, end to end through the real CLI:
 *   `search` (BM25), `vsearch` (vector), `query` (hybrid: BM25 + vector fused,
 *   then reranked), and removing the tombstone puts it back on all three.
 *
 * Real subprocesses against a fake OpenAI-compatible embed/rerank server
 * (tests/helpers/fake-embed-server.ts), the same rig as
 * exit-code-contract.test.ts: `clawmem update` indexes through the collection's
 * frontmatter_map, `clawmem embed` writes vectors, and the three search verbs
 * read them. Nothing is stubbed between the index write and the search read.
 */
import { describe, it, expect, afterAll } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startFakeEmbedServer, type FakeEmbedServer } from "../helpers/fake-embed-server.ts";

const CLI = new URL("../../src/clawmem.ts", import.meta.url).pathname;
const DIM = 512;
const MODEL = "fake-embed-a";
const QUERY = "wombat harbour ferry timetable";

const dir = mkdtempSync(join(tmpdir(), "clawmem-vn4rz84-search-"));
const docs = join(dir, "docs");
mkdirSync(docs, { recursive: true });
writeFileSync(
  join(dir, "config.yaml"),
  [
    "collections:",
    "  memtopics:",
    `    path: ${docs}`,
    '    pattern: "**/*.md"',
    "    frontmatter_map:",
    "      title: name",
    "      invalidated_at: invalidated_at",
    "      superseded_by: superseded_by",
    "",
  ].join("\n"),
);
const db = join(dir, "index.sqlite");

// Same body with and without the tombstone, so removing it changes the file
// (re-parsed) but not the body hash (vectors stay valid, no re-embed needed).
const OLD_BODY = "The wombat harbour ferry timetable, winter edition, departs at dawn.";
const old = (tombstone: boolean) =>
  [
    "---",
    "name: Wombat ferry timetable (winter)",
    ...(tombstone ? ["superseded_by: new.md"] : []),
    "---",
    "",
    OLD_BODY,
    "",
  ].join("\n");
writeFileSync(join(docs, "old.md"), old(true));
writeFileSync(
  join(docs, "new.md"),
  [
    "---",
    "name: Wombat ferry timetable (summer)",
    "---",
    "",
    "The wombat harbour ferry timetable, summer edition, departs at noon.",
    "",
  ].join("\n"),
);

let server: FakeEmbedServer | undefined;
afterAll(() => {
  server?.stop();
  rmSync(dir, { recursive: true, force: true });
});

async function cli(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined && !k.startsWith("CLAWMEM_") && k !== "INDEX_PATH") env[k] = v;
  }
  const proc = Bun.spawn(["bun", CLI, ...args], {
    env: {
      ...env,
      INDEX_PATH: db,
      CLAWMEM_CONFIG_DIR: dir,
      CLAWMEM_NO_LOCAL_MODELS: "true",
      CLAWMEM_EMBED_URL: server!.url,
      CLAWMEM_RERANK_URL: server!.url,
    },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { code, stdout, stderr };
}

/** Files returned by one search verb, as JSON. */
async function files(verb: "search" | "vsearch" | "query"): Promise<string[]> {
  const r = await cli([verb, QUERY, "-c", "memtopics", "--json", "--min-score", "0"]);
  expect({ verb, code: r.code, stderr: r.code === 0 ? "" : r.stderr }).toEqual({
    verb,
    code: 0,
    stderr: "",
  });
  const out = r.stdout.trim();
  if (out === "" || out.startsWith("No results")) return [];
  return (JSON.parse(out) as { file: string }[]).map((x) => x.file);
}

describe("vn4rz.84: a frontmatter tombstone leaves search, vsearch and query; removing it restores all three", () => {
  it("tombstoned -> absent everywhere; tombstone removed -> present everywhere", async () => {
    server = startFakeEmbedServer({ dim: DIM, model: MODEL });

    const up1 = await cli(["update"]);
    expect(up1.code).toBe(0);
    expect(up1.stdout).toContain(
      "supersession 1 tombstoned, superseded_by 1 resolved / 0 unresolved",
    );
    const emb = await cli(["embed"]);
    expect(emb.code).toBe(0);

    for (const verb of ["search", "vsearch", "query"] as const) {
      const hits = await files(verb);
      expect({ verb, newFound: hits.some((f) => f.endsWith("new.md")) }).toEqual({
        verb,
        newFound: true,
      });
      expect({ verb, oldFound: hits.some((f) => f.endsWith("old.md")) }).toEqual({
        verb,
        oldFound: false,
      });
    }

    writeFileSync(join(docs, "old.md"), old(false));
    const up2 = await cli(["update"]);
    expect(up2.code).toBe(0);
    expect(up2.stdout).toContain("~1");

    for (const verb of ["search", "vsearch", "query"] as const) {
      const hits = await files(verb);
      expect({ verb, oldFound: hits.some((f) => f.endsWith("old.md")) }).toEqual({
        verb,
        oldFound: true,
      });
    }
  }, 240_000);
});
