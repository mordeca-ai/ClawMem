/**
 * PG documents-tier EMBED-INCREMENTAL — integration tests (master-harness-vn4rz.86).
 *
 * Live clawmem-pg cluster, throwaway schema per run (tests/integration/
 * pg-test-schema.ts). SKIPS when CLAWMEM_PG_URL is unset. Run through
 * `clawmem-lander pg-itest -- bun test tests/integration/pg-reindex-embed-incremental.test.ts`
 * so the URL points at a throwaway database.
 *
 * THE DEFECT. reindexCollection re-embedded every fragment of every document
 * on every run: the only dedupe was an in-process Set, never the vault's
 * content_vectors. Measured live: every agents-skills tick wrote ~16.4k
 * fragments for a one-file delta. These cases drive the REAL reindexCollection
 * against fixture files with a stub embedder (no network, no yoshiee) and
 * assert what reached the embedder and what the vault holds afterwards.
 *
 * Every case uses its own collection AND its own file content: content_vectors
 * is keyed on the content hash alone, so two cases sharing bytes would share
 * vectors and contaminate each other's counts.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { createPgTestSchema, PG_TEST_SETUP_TIMEOUT_MS, type PgTestSchema } from "./pg-test-schema.ts";
import { closePool } from "../../src/pg/client.ts";
import { embedDim, setPgSchema } from "../../src/pg/config.ts";
import { reindexCollection, type ReindexOptions } from "../../src/pg/reindex.ts";
import { resetVaultCache } from "../../src/pg/vaults.ts";
import { setDefaultLlamaCpp, type EmbeddingResult, type LlamaCpp } from "../../src/llm.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const d = URL_ ? describe : describe.skip;
const MODEL = "stub-embed-vn4rz86";

/** Every text the stub embedder was handed, in order. */
const embedded: string[] = [];

/** Deterministic, never-zero vector derived from the text. */
function vecFor(text: string, dim: number): number[] {
  const v = new Array(dim).fill(0);
  for (let i = 0; i < text.length; i++) v[i % dim] += text.charCodeAt(i) / 1000;
  v[0] += 1;
  return v;
}

const stubEmbedder = {
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    embedded.push(...texts);
    const dim = embedDim();
    return texts.map(t => ({ embedding: vecFor(t, dim), model: MODEL, endpoint: "stub://vn4rz86", input: t }));
  },
} as unknown as LlamaCpp;

const section = (tag: string, h: string) =>
  `## ${h}\n\n` +
  `The ${tag} section about ${h} carries enough prose to stand on its own as a fragment. `.repeat(12) +
  "\n\n";

/** Four documents; multi.md splits into several fragments. */
function writeFixture(root: string, tag: string): void {
  writeFileSync(
    join(root, "multi.md"),
    `# Multi ${tag}\n\nIntro for ${tag}.\n\n` + section(tag, "Alpha") + section(tag, "Beta") + section(tag, "Gamma"),
  );
  for (const n of ["a", "b", "c"]) {
    writeFileSync(join(root, `${n}.md`), `# Doc ${n} ${tag}\n\nBody of ${n} for ${tag}.\n`);
  }
}

d("PG reindex EMBED-INCREMENTAL (vn4rz.86)", () => {
  let harness: PgTestSchema | undefined;
  let configDir: string;
  const roots: string[] = [];
  const savedConfigDir = process.env.CLAWMEM_CONFIG_DIR;

  beforeAll(async () => {
    harness = createPgTestSchema({ url: URL_!, prefix: "clawmem_test_embedinc", dim: embedDim() });
    await harness.setup();
    setPgSchema(harness.schema);
    // Hermetic routing: an EMPTY config dir, so the fixture collections resolve
    // to the sfw vault (the throwaway database) and never read the real config.
    configDir = mkdtempSync(join(tmpdir(), "clawmem-embedinc-cfg-"));
    process.env.CLAWMEM_CONFIG_DIR = configDir;
    resetVaultCache();
    setDefaultLlamaCpp(stubEmbedder);
  }, PG_TEST_SETUP_TIMEOUT_MS);

  afterAll(async () => {
    setDefaultLlamaCpp(null);
    await harness?.settled();
    setPgSchema(null);
    await closePool();
    await harness?.teardown();
    if (savedConfigDir === undefined) delete process.env.CLAWMEM_CONFIG_DIR;
    else process.env.CLAWMEM_CONFIG_DIR = savedConfigDir;
    resetVaultCache();
    if (configDir) rmSync(configDir, { recursive: true, force: true });
    for (const r of roots) rmSync(r, { recursive: true, force: true });
  });

  function fixture(tag: string): { name: string; root: string } {
    const root = mkdtempSync(join(tmpdir(), `clawmem-embedinc-${tag}-`));
    roots.push(root);
    writeFixture(root, tag);
    return { name: `__vn4rz86_${tag}`, root };
  }

  /** One reindex pass; returns the stats plus how many texts reached the embedder. */
  async function run(name: string, root: string, extra: ReindexOptions = {}) {
    const before = embedded.length;
    const stats = await reindexCollection(name, root, "**/*.md", {
      sweep: false, embedBatchSize: 3, ...extra,
    });
    return { stats, sentToEmbedder: embedded.length - before };
  }

  async function hashOf(name: string, path: string): Promise<string> {
    return harness!.withSchema(async c => {
      const { rows } = await c.query<{ hash: string }>(
        "SELECT hash FROM documents WHERE collection = $1 AND path = $2", [name, path]);
      return rows[0]!.hash;
    });
  }

  async function vectorRows(hash: string): Promise<number> {
    return harness!.withSchema(async c => {
      const { rows } = await c.query<{ n: string }>(
        "SELECT count(*)::text AS n FROM content_vectors WHERE hash = $1", [hash]);
      return Number(rows[0]!.n);
    });
  }

  /** Byte-level snapshot of every stored vector row for a collection's hashes. */
  async function snapshot(name: string): Promise<string[]> {
    return harness!.withSchema(async c => {
      const { rows } = await c.query<{ r: string }>(
        `SELECT concat_ws('|', cv.hash, cv.seq, cv.pos, cv.model, cv.embedding::text,
                          cv.embedded_at::text, cv.embed_input_fp, cv.fragment_type,
                          cv.fragment_label, cv.canonical_id) AS r
           FROM content_vectors cv
          WHERE cv.hash IN (SELECT hash FROM documents WHERE collection = $1)
          ORDER BY cv.hash, cv.seq`, [name]);
      return rows.map(x => x.r);
    });
  }

  it("(a) a second run over unchanged files embeds NOTHING and leaves stored rows byte-identical", async () => {
    const { name, root } = fixture("a");
    const first = await run(name, root);
    expect(first.stats.documentsWritten).toBe(4);
    expect(first.stats.fragmentsEmbedded).toBeGreaterThan(4); // multi.md splits
    expect(first.stats.hashesAlreadyEmbedded).toBe(0);
    const before = await snapshot(name);
    expect(before.length).toBe(first.stats.fragmentsEmbedded);

    const second = await run(name, root);
    expect(second.stats.documentsWritten).toBe(4);
    expect(second.stats.fragmentsEmbedded).toBe(0);
    expect(second.sentToEmbedder).toBe(0);
    expect(second.stats.hashesAlreadyEmbedded).toBe(4);
    expect(await snapshot(name)).toEqual(before);
  });

  it("(b) changing ONE file embeds only that file's fragments", async () => {
    const { name, root } = fixture("b");
    await run(name, root);
    writeFileSync(join(root, "a.md"), "# Doc a b\n\nBody of a, EDITED for b.\n\nA second paragraph.\n");

    const next = await run(name, root);
    const newHash = await hashOf(name, "a.md");
    const rows = await vectorRows(newHash);
    expect(rows).toBeGreaterThan(0);
    expect(next.stats.fragmentsEmbedded).toBe(rows);
    expect(next.sentToEmbedder).toBe(rows);
    expect(next.stats.hashesAlreadyEmbedded).toBe(3);
  });

  it("(c) a PARTIAL stored set (and a zero-row hash) re-embeds; complete ones do not", async () => {
    const { name, root } = fixture("c");
    const first = await run(name, root);
    const multi = await hashOf(name, "multi.md");
    const a = await hashOf(name, "a.md");
    const multiFull = await vectorRows(multi);
    const aFull = await vectorRows(a);
    expect(multiFull).toBeGreaterThan(1);
    await harness!.withSchema(c =>
      c.query("DELETE FROM content_vectors WHERE hash = $1 AND seq >= 1", [multi]));
    await harness!.withSchema(c => c.query("DELETE FROM content_vectors WHERE hash = $1", [a]));
    expect(await vectorRows(multi)).toBe(1);
    expect(await vectorRows(a)).toBe(0);

    const next = await run(name, root);
    expect(next.stats.fragmentsEmbedded).toBe(multiFull + aFull);
    expect(next.stats.hashesAlreadyEmbedded).toBe(2);
    expect(await vectorRows(multi)).toBe(multiFull);
    expect(await vectorRows(a)).toBe(aFull);
    expect(first.stats.fragmentsEmbedded).toBeGreaterThanOrEqual(multiFull + aFull);
  });

  it("(d) reembed: true re-embeds every fragment", async () => {
    const { name, root } = fixture("d");
    const first = await run(name, root);
    const again = await run(name, root, { reembed: true });
    expect(again.stats.fragmentsEmbedded).toBe(first.stats.fragmentsEmbedded);
    expect(again.sentToEmbedder).toBe(first.stats.fragmentsEmbedded);
    expect(again.stats.hashesAlreadyEmbedded).toBe(0);
  });

  it("skipEmbed still embeds nothing and reports no skips", async () => {
    const { name, root } = fixture("e");
    const r = await run(name, root, { skipEmbed: true });
    expect(r.stats.documentsWritten).toBe(4);
    expect(r.stats.fragmentsEmbedded).toBe(0);
    expect(r.sentToEmbedder).toBe(0);
    expect(r.stats.hashesAlreadyEmbedded).toBe(0);
  });
});
