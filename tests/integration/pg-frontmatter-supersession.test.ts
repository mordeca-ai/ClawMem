/**
 * PG frontmatter supersession — integration tests (master-harness-vn4rz.84).
 *
 * Live clawmem-pg cluster, throwaway schema per run (tests/integration/
 * pg-test-schema.ts). SKIPS when CLAWMEM_PG_URL is unset. Run through
 * `clawmem-lander pg-itest -- bun test tests/integration/pg-frontmatter-supersession.test.ts`
 * so the URL points at a throwaway database.
 *
 * Drives the REAL reindexCollection (skipEmbed: no embedder involved) against
 * fixture files and asserts what the vault holds and what pgSearchFts returns:
 *   P1  a tombstone sets invalidated_at + invalidation_source, resolves
 *       superseded_by (any file order), and leaves FTS;
 *   P2  removing it and re-indexing restores the document;
 *   P3  an invalidation from another source survives a re-index;
 *   P4  a collection without the map keys writes no invalidation;
 *   P5  without migration 014's column, a collection that declares the keys
 *       refuses loudly and one that does not indexes as before.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import {
  createPgTestSchema,
  PG_TEST_SETUP_TIMEOUT_MS,
  type PgTestSchema,
} from "./pg-test-schema.ts";
import { closePool } from "../../src/pg/client.ts";
import { embedDim, setPgSchema } from "../../src/pg/config.ts";
import { reindexCollection } from "../../src/pg/reindex.ts";
import { pgSearchFts } from "../../src/pg/search-fts.ts";
import { resetVaultCache } from "../../src/pg/vaults.ts";
import type { FrontmatterMap } from "../../src/frontmatter-map.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const d = URL_ ? describe : describe.skip;

const MAP: FrontmatterMap = {
  title: "name",
  invalidated_at: "invalidated_at",
  superseded_by: "superseded_by",
};
const fm = (lines: string[], body: string) => `---\n${lines.join("\n")}\n---\n\n${body}\n`;

type Row = {
  id: string;
  invalidated_at: Date | null;
  invalidation_source: string | null;
  superseded_by: string | null;
};

d("PG frontmatter supersession (vn4rz.84)", () => {
  let harness: PgTestSchema | undefined;
  let configDir: string;
  const roots: string[] = [];
  const savedConfigDir = process.env.CLAWMEM_CONFIG_DIR;

  beforeAll(async () => {
    harness = createPgTestSchema({ url: URL_!, prefix: "clawmem_test_supersede", dim: embedDim() });
    await harness.setup();
    setPgSchema(harness.schema);
    configDir = mkdtempSync(join(tmpdir(), "clawmem-supersede-cfg-"));
    process.env.CLAWMEM_CONFIG_DIR = configDir;
    resetVaultCache();
  }, PG_TEST_SETUP_TIMEOUT_MS);

  afterAll(async () => {
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

  function root(tag: string): string {
    const r = mkdtempSync(join(tmpdir(), `clawmem-supersede-${tag}-`));
    mkdirSync(join(r, "topics"), { recursive: true });
    roots.push(r);
    return r;
  }

  const run = (name: string, r: string, map?: FrontmatterMap) =>
    reindexCollection(name, r, "**/*.md", { skipEmbed: true, sweep: false }, map);

  async function row(name: string, path: string): Promise<Row> {
    return harness!.withSchema(async (c) => {
      const { rows } = await c.query<Row>(
        `SELECT id::text AS id, invalidated_at, invalidation_source, superseded_by::text AS superseded_by
           FROM documents WHERE collection = $1 AND path = $2`,
        [name, path],
      );
      return rows[0]!;
    });
  }

  async function ftsPaths(name: string, q: string): Promise<string[]> {
    return harness!.withSchema(async (c) =>
      (await pgSearchFts(c, q, { collections: name, statementTimeoutMs: 0 })).map(
        (r) => r.filepath,
      ),
    );
  }

  it("P1 + P2: tombstone applies, resolves, leaves FTS; removal restores", async () => {
    const name = "__vn4rz84_p1";
    const r = root("p1");
    // a-old sorts before its replacement: resolution must not depend on walk order.
    writeFileSync(
      join(r, "a-old.md"),
      fm(["name: Old numbat", "superseded_by: topics/z-new.md"], "numbat survey first edition"),
    );
    writeFileSync(
      join(r, "topics/z-new.md"),
      fm(["name: New numbat"], "numbat survey second edition"),
    );
    writeFileSync(
      join(r, "b-dated.md"),
      fm(["name: Dated numbat", "invalidated_at: 2026-09-15"], "numbat census dated"),
    );
    const stats = await run(name, r, MAP);
    expect(stats.frontmatterVocab.invalidatedByFrontmatter).toBe(2);
    expect(stats.frontmatterVocab.supersededByResolved).toBe(1);

    const old = await row(name, "a-old.md");
    const neu = await row(name, "topics/z-new.md");
    expect(old.invalidated_at).not.toBeNull();
    expect(old.invalidation_source).toBe("frontmatter");
    expect(old.superseded_by).toBe(neu.id);
    expect(neu.invalidated_at).toBeNull();
    expect((await row(name, "b-dated.md")).invalidated_at?.toISOString()).toBe(
      "2026-09-15T00:00:00.000Z",
    );

    const hits = await ftsPaths(name, "numbat survey");
    expect(hits.some((p) => p.endsWith("topics/z-new.md"))).toBe(true);
    expect(hits.some((p) => p.endsWith("a-old.md"))).toBe(false);

    // A second pass keeps the index-time stamp (no bump on re-apply).
    await run(name, r, MAP);
    expect((await row(name, "a-old.md")).invalidated_at?.getTime()).toBe(
      old.invalidated_at!.getTime(),
    );

    writeFileSync(join(r, "a-old.md"), fm(["name: Old numbat"], "numbat survey first edition"));
    await run(name, r, MAP);
    expect(await row(name, "a-old.md")).toMatchObject({
      invalidated_at: null,
      invalidation_source: null,
      superseded_by: null,
    });
    expect((await ftsPaths(name, "numbat survey")).some((p) => p.endsWith("a-old.md"))).toBe(true);
  });

  it("P3: an invalidation from another source survives re-index, with or without a tombstone", async () => {
    const name = "__vn4rz84_p3";
    const r = root("p3");
    writeFileSync(join(r, "obs.md"), fm(["name: Obs bilby"], "bilby observation"));
    await run(name, r, MAP);
    await harness!.withSchema((c) =>
      c.query(
        `UPDATE documents SET invalidated_at = '2026-05-05T05:05:05Z', invalidated_by = 9,
                invalidation_source = 'contradiction' WHERE collection = $1 AND path = 'obs.md'`,
        [name],
      ),
    );
    await run(name, r, MAP);
    writeFileSync(
      join(r, "obs.md"),
      fm(["name: Obs bilby", "invalidated_at: 2026-09-15"], "bilby observation"),
    );
    await run(name, r, MAP);
    const after = await row(name, "obs.md");
    expect(after.invalidation_source).toBe("contradiction");
    expect(after.invalidated_at?.toISOString()).toBe("2026-05-05T05:05:05.000Z");
  });

  it("P4: no supersession map keys = the bare keys write nothing", async () => {
    const name = "__vn4rz84_p4";
    const r = root("p4");
    writeFileSync(
      join(r, "x.md"),
      fm(["title: X", "superseded_by: y.md", "invalidated_at: 2026-01-01"], "quoll x"),
    );
    const stats = await run(name, r, { title: "name" });
    expect(stats.frontmatterVocab.invalidatedByFrontmatter).toBe(0);
    expect(await row(name, "x.md")).toMatchObject({
      invalidated_at: null,
      invalidation_source: null,
    });
  });

  // LAST: drops the column from the throwaway schema.
  it("P5: without migration 014, a declaring collection refuses; a plain one indexes as before", async () => {
    await harness!.withSchema((c) =>
      c.query("ALTER TABLE documents DROP COLUMN invalidation_source"),
    );
    const r = root("p5");
    writeFileSync(join(r, "x.md"), fm(["name: X", "superseded_by: y.md"], "dunnart x"));
    await expect(run("__vn4rz84_p5a", r, MAP)).rejects.toThrow(
      /migration 014_documents_invalidation_source not applied/,
    );
    expect(
      await harness!.withSchema(
        async (c) =>
          (await c.query("SELECT 1 FROM documents WHERE collection = '__vn4rz84_p5a'")).rowCount,
      ),
    ).toBe(0);
    const stats = await run("__vn4rz84_p5b", r, { title: "name" });
    expect(stats.documentsWritten).toBe(1);
  });
});
