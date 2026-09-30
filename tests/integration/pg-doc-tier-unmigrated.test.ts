/**
 * A vault whose content_vectors PREDATES migration 010 (no doc_tier column) —
 * the live nsfw vault until it is migrated (master-harness-vn4rz.77 skeptic fix).
 *
 * Live clawmem-pg cluster, throwaway schema built from migrations 001-009 only.
 * SKIPS when CLAWMEM_PG_URL is unset; run via `clawmem-lander pg-itest`.
 *
 * Without the per-vault column detection, the literal `cv.doc_tier` in the ANN
 * query and the model fence is `column cv.doc_tier does not exist` on every
 * search. Force hasDocTierColumn to "present" and these cases go RED.
 */
import { describe, it, expect, beforeAll, afterAll, beforeEach } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { createPgTestSchema, PG_TEST_SETUP_TIMEOUT_MS, type PgTestSchema } from "./pg-test-schema.ts";
import { closePool, toVectorLiteral } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { MIGRATIONS_DIR } from "../../src/pg/migrate.ts";
import {
  hasDocTierColumn,
  pgSearchVec,
  resetDocTierColumnCache,
  type PgVecEmbedder,
} from "../../src/pg/search.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 768;
const MODEL = "embeddinggemma";
const d = URL_ ? describe : describe.skip;

function atAngle(deg: number): number[] {
  const t = (deg * Math.PI) / 180;
  const v = new Array(DIM).fill(0);
  v[0] = Math.cos(t);
  v[1] = Math.sin(t);
  return v;
}
const embedder: PgVecEmbedder = { async embed() { return { embedding: atAngle(0), model: MODEL }; } };

d("PG search on a pre-010 vault (no content_vectors.doc_tier)", () => {
  let harness: PgTestSchema | undefined;
  let dir = "";
  let schema = "";

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "clawmem-pre010-"));
    mkdirSync(dir, { recursive: true });
    for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql") && f < "010")) {
      copyFileSync(join(MIGRATIONS_DIR, f), join(dir, f));
    }
    harness = createPgTestSchema({
      url: URL_!, prefix: "clawmem_test_pre010", dim: DIM, migrationsDir: dir,
      seed: async c => {
        await c.query(`INSERT INTO content (hash, doc) VALUES ('h_pre', 'pre-010 doc')`);
        await c.query(
          `INSERT INTO documents (collection, path, title, hash) VALUES ('pre', 'a.md', 'A', 'h_pre')`,
        );
        await c.query(
          `INSERT INTO content_vectors (hash, seq, pos, model, embedding) VALUES ('h_pre', 0, 0, $1, $2::vector)`,
          [MODEL, toVectorLiteral(atAngle(10))],
        );
      },
    });
    ({ schema } = harness);
    await harness.setup();
    setPgSchema(schema);
  }, PG_TEST_SETUP_TIMEOUT_MS);

  afterAll(async () => {
    await harness?.settled();
    setPgSchema(null);
    await closePool();
    await harness?.teardown();
    if (dir) rmSync(dir, { recursive: true, force: true });
  });

  beforeEach(() => resetDocTierColumnCache());

  it("the fixture really is pre-010: content_vectors has no doc_tier column", async () => {
    const r = await harness!.withSchema(c =>
      c.query(
        `SELECT 1 FROM pg_attribute WHERE attrelid = to_regclass('content_vectors')
          AND attname = 'doc_tier' AND NOT attisdropped`,
      ));
    expect(r.rowCount).toBe(0);
    expect(await harness!.withSchema(c => hasDocTierColumn(c))).toBe(false);
  });

  it("pgSearchVec returns vec results, without error, on the unmigrated vault", async () => {
    const out = await harness!.withSchema(c => pgSearchVec(c, "anything", { embedder, collections: "pre", limit: 5 }));
    expect(out.map(r => r.hash)).toEqual(["h_pre"]);
  });

  it("the unscoped search (model fence with no collection filter) also works", async () => {
    const out = await harness!.withSchema(c => pgSearchVec(c, "anything", { embedder, limit: 5 }));
    expect(out.map(r => r.hash)).toContain("h_pre");
  });
});
