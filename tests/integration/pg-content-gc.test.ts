/**
 * Content GC integration tests (master-harness-vn4rz.49).
 *
 * Live clawmem-pg cluster, throwaway schema per run (same harness shape as
 * pg-write-path.test.ts). SKIPS when CLAWMEM_PG_URL is unset.
 *
 * Proves: (a) a superseded hash's vectors are removed; (b) a hash referenced
 * only by an active=false document, or only by origin_documents, is RETAINED;
 * (c) a second pass deletes nothing; (d) the batch cap is honoured; plus the
 * race guards — a young row inside the grace window is kept, and a row an
 * in-flight (uncommitted) writer references is skipped, never cascade-deleted.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool, toVectorLiteral } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import {
  deactivateAbsentDocuments, gcOrphanedContent, insertEmbeddingsBatch, upsertDocument,
} from "../../src/pg/write.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 768;
const MODEL = "embeddinggemma";
const COLLECTION = "__vn4rz49_gc_test";
const d = URL_ ? describe : describe.skip;

d("PG content GC", () => {
  let pool: pg.Pool;
  let schema: string;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: URL_ });
    schema = `clawmem_test_gc_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const c = await pool.connect();
    try {
      await c.query(`CREATE SCHEMA ${schema}`);
      await c.query(`SET search_path TO ${schema}, public`);
      for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
        await c.query(substituteMigrationParams(readFileSync(join(MIGRATIONS_DIR, f), "utf-8"), schema, DIM));
      }
    } finally {
      c.release();
    }
    setPgSchema(schema);
  });

  afterAll(async () => {
    setPgSchema(null);
    await closePool();
    if (schema) await pool.query(`DROP SCHEMA IF EXISTS ${schema} CASCADE`);
    await pool.end();
  });

  async function q<T extends pg.QueryResultRow>(sql: string, params: unknown[] = []): Promise<T[]> {
    const c = await pool.connect();
    try {
      await c.query(`SET search_path TO ${schema}, public`);
      return (await c.query<T>(sql, params)).rows;
    } finally {
      c.release();
    }
  }

  async function writeDoc(path: string, hash: string, fragments = 3): Promise<void> {
    await upsertDocument({ collection: COLLECTION, path, title: path, hash, body: `# ${hash}` });
    await insertEmbeddingsBatch(
      Array.from({ length: fragments }, (_, seq) => ({
        collection: COLLECTION, path, hash, seq, pos: seq, model: MODEL,
        embedding: new Array(DIM).fill(0.01 * (seq + 1)),
      })),
    );
  }

  /** Content + vectors with no referencing row at all (a pure orphan). */
  async function seedOrphan(hash: string, fragments = 2): Promise<void> {
    await q(`INSERT INTO content (hash, doc) VALUES ($1, 'orphan')`, [hash]);
    for (let seq = 0; seq < fragments; seq++) {
      await q(
        `INSERT INTO content_vectors (hash, seq, pos, model, embedding) VALUES ($1, $2, 0, $3, $4::vector)`,
        [hash, seq, MODEL, toVectorLiteral(new Array(DIM).fill(0.5))],
      );
    }
  }

  async function vectorsFor(hash: string): Promise<number> {
    const r = await q<{ n: number }>(`SELECT count(*)::int n FROM content_vectors WHERE hash = $1`, [hash]);
    return r[0]!.n;
  }
  async function contentExists(hash: string): Promise<boolean> {
    return (await q(`SELECT 1 FROM content WHERE hash = $1`, [hash])).length === 1;
  }
  async function orphanCount(): Promise<number> {
    const r = await q<{ n: number }>(
      `SELECT count(*)::int n FROM content c
        WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.hash = c.hash)
          AND NOT EXISTS (SELECT 1 FROM origin_documents o WHERE o.hash = c.hash)`,
    );
    return r[0]!.n;
  }

  it("(a) REMOVES a superseded hash's content and vectors after a pass", async () => {
    await writeDoc("doc.md", "h_v1");
    await writeDoc("doc.md", "h_v2"); // same path, new content: h_v1 is superseded
    expect(await vectorsFor("h_v1")).toBe(3);

    const r = await gcOrphanedContent({ graceSeconds: 0 });

    expect(await contentExists("h_v1")).toBe(false);
    expect(await vectorsFor("h_v1")).toBe(0);
    expect(r.contentDeleted).toBeGreaterThanOrEqual(1);
    expect(r.vectorsDeleted).toBeGreaterThanOrEqual(3);
    // The live hash is untouched.
    expect(await vectorsFor("h_v2")).toBe(3);
    expect(await orphanCount()).toBe(0);
  });

  it("(b) RETAINS a hash referenced only by an active=false document", async () => {
    await writeDoc("keep.md", "h_inactive");
    await writeDoc("other.md", "h_other");
    await deactivateAbsentDocuments(COLLECTION, ["doc.md", "other.md"], "gc test");
    const inactive = await q<{ active: boolean }>(
      `SELECT active FROM documents WHERE collection = $1 AND path = 'keep.md'`, [COLLECTION],
    );
    expect(inactive[0]!.active).toBe(false);

    await gcOrphanedContent({ graceSeconds: 0 });

    expect(await contentExists("h_inactive")).toBe(true);
    expect(await vectorsFor("h_inactive")).toBe(3);
  });

  it("(b) RETAINS a hash referenced only by origin_documents", async () => {
    await seedOrphan("h_origin", 2);
    await q(
      `INSERT INTO origin_documents (collection, path, title, hash) VALUES ('__origin_gc', 'o.md', 'o', 'h_origin')`,
    );

    await gcOrphanedContent({ graceSeconds: 0 });

    expect(await contentExists("h_origin")).toBe(true);
    expect(await vectorsFor("h_origin")).toBe(2);
  });

  it("(c) is IDEMPOTENT — a second pass deletes nothing", async () => {
    await seedOrphan("h_idem", 2);
    const first = await gcOrphanedContent({ graceSeconds: 0 });
    expect(first.contentDeleted).toBe(1);
    const second = await gcOrphanedContent({ graceSeconds: 0 });
    expect(second.contentDeleted).toBe(0);
    expect(second.vectorsDeleted).toBe(0);
    expect(second.eligibleContent).toBe(0);
    expect(second.capped).toBe(false);
  });

  it("(d) HONOURS the batch cap, reports capped, and converges on the next pass", async () => {
    for (let i = 0; i < 5; i++) await seedOrphan(`h_cap_${i}`, 1);

    const dry = await gcOrphanedContent({ graceSeconds: 0, dryRun: true });
    expect(dry.eligibleContent).toBe(5);
    expect(dry.eligibleVectors).toBe(5);
    expect(dry.contentDeleted).toBe(0);
    expect(await orphanCount()).toBe(5);

    const capped = await gcOrphanedContent({ graceSeconds: 0, batchSize: 2, maxBatches: 1 });
    expect(capped.batches).toBe(1);
    expect(capped.contentDeleted).toBe(2);
    expect(capped.vectorsDeleted).toBe(2);
    expect(capped.capped).toBe(true);
    expect(await orphanCount()).toBe(3);

    const rest = await gcOrphanedContent({ graceSeconds: 0, batchSize: 2, maxBatches: 10 });
    expect(rest.contentDeleted).toBe(3);
    expect(rest.capped).toBe(false);
    expect(await orphanCount()).toBe(0);
  });

  it("RACE: an orphan younger than the grace window is KEPT", async () => {
    await seedOrphan("h_young", 1);
    const r = await gcOrphanedContent({ graceSeconds: 3600 });
    expect(r.withinGraceContent).toBeGreaterThanOrEqual(1);
    expect(await contentExists("h_young")).toBe(true);
    await gcOrphanedContent({ graceSeconds: 0 });
    expect(await contentExists("h_young")).toBe(false);
  });

  it("RACE: a row an IN-FLIGHT writer references is skipped, never cascade-deleted", async () => {
    await seedOrphan("h_inflight", 2);
    const writer = await pool.connect();
    try {
      await writer.query(`SET search_path TO ${schema}, public`);
      await writer.query("BEGIN");
      // Uncommitted: the FK check holds KEY SHARE on content(h_inflight).
      await writer.query(
        `INSERT INTO documents (collection, path, title, hash) VALUES ($1, 'inflight.md', 'x', 'h_inflight')`,
        [COLLECTION],
      );
      const r = await gcOrphanedContent({ graceSeconds: 0 });
      expect(r.contentDeleted).toBe(0);
      await writer.query("COMMIT");
    } catch (e) {
      await writer.query("ROLLBACK");
      throw e;
    } finally {
      writer.release();
    }
    const doc = await q(`SELECT 1 FROM documents WHERE collection = $1 AND path = 'inflight.md'`, [COLLECTION]);
    expect(doc.length).toBe(1);
    expect(await vectorsFor("h_inflight")).toBe(2);
  });
});
