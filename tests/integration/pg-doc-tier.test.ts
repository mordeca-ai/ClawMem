/**
 * content_vectors.doc_tier + the partial HNSW index — integration tests
 * (master-harness-vn4rz.77).
 *
 * Live clawmem-pg cluster, throwaway schema per run (tests/integration/
 * pg-test-schema.ts). SKIPS when CLAWMEM_PG_URL is unset. Run through
 * `clawmem-lander pg-itest -- bun test tests/integration/pg-doc-tier.test.ts`
 * so the URL points at a throwaway database.
 *
 * WHAT IS UNDER TEST. Migration 010's column + triggers, 011's backfill,
 * 012's partial index, src/pg/write.ts reconcileDocTier, and src/pg/search.ts's use
 * of the flag. Writes go through the REAL write path (upsertDocument,
 * insertEmbeddingsBatch, dropLegacyDocumentRows) wherever one exists, because
 * the thing that must hold is "the writers this repo ships keep the flag
 * right", not "a hand-rolled INSERT does".
 *
 * WHAT `cv.doc_tier` IN THE SEARCH DOES AND DOES NOT DO, stated so no test
 * below overclaims: the documents JOIN already drops origin-only vectors, so
 * "an origin vector is not returned" holds with or without the predicate. The
 * predicate's job is to let the planner use the partial index — proven by the
 * plan test, which is the behavioural case that goes RED without it.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync } from "fs";
import { join } from "path";
import { createPgTestSchema, PG_TEST_SETUP_TIMEOUT_MS, type PgTestSchema } from "./pg-test-schema.ts";
import { closePool, toVectorLiteral } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { dropLegacyDocumentRows } from "../../src/pg/origin.ts";
import { buildVecSearchQuery, pgSearchVec, type PgVecEmbedder } from "../../src/pg/search.ts";
import {
  insertEmbeddingsBatch,
  reconcileDocTierForVault,
  upsertDocument,
} from "../../src/pg/write.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 768;
const MODEL = "embeddinggemma";
const COLLECTION = "__vn4rz77_doc_tier";
const PARTIAL_IDX = "content_vectors_embedding_doc_hnsw_idx";
const FULL_IDX = "content_vectors_embedding_hnsw_idx";
const d = URL_ ? describe : describe.skip;

/** Unit vector at angle `deg` in the (x, y) plane, padded to DIM. */
function atAngle(deg: number): number[] {
  const t = (deg * Math.PI) / 180;
  const v = new Array(DIM).fill(0);
  v[0] = Math.cos(t);
  v[1] = Math.sin(t);
  return v;
}
const QUERY_VEC = atAngle(0);
const embedder: PgVecEmbedder = { async embed() { return { embedding: QUERY_VEC, model: MODEL }; } };

d("PG content_vectors.doc_tier (vn4rz.77)", () => {
  let pool: pg.Pool;
  let schema: string;
  let harness: PgTestSchema | undefined;

  beforeAll(async () => {
    harness = createPgTestSchema({ url: URL_!, prefix: "clawmem_test_doctier", dim: DIM });
    ({ pool, schema } = harness);
    await harness.setup();
    setPgSchema(schema);
  }, PG_TEST_SETUP_TIMEOUT_MS);

  afterAll(async () => {
    await harness?.settled();
    setPgSchema(null);
    await closePool();
    await harness?.teardown();
  });

  async function withSchema<T>(fn: (c: pg.PoolClient) => Promise<T>): Promise<T> {
    const c = await pool.connect();
    try {
      await c.query(`SET search_path TO ${schema}, public`);
      return await fn(c);
    } finally {
      c.release();
    }
  }
  async function q<T extends pg.QueryResultRow>(sql: string, params: unknown[] = []): Promise<T[]> {
    return withSchema(async c => (await c.query<T>(sql, params)).rows);
  }

  /** A curated document + its fragment vectors, through the real write path. */
  async function writeDoc(path: string, hash: string, angles: number[] = [45], collection = COLLECTION) {
    await upsertDocument({ collection, path, title: path, hash, body: `# ${path} ${hash}` });
    await insertEmbeddingsBatch(angles.map((deg, seq) => ({
      collection, path, hash, seq, pos: seq, model: MODEL, embedding: atAngle(deg),
    })));
  }

  /** An origin-tier record (origin_documents, DEFAULT partition) + its vectors. */
  async function writeOrigin(path: string, hash: string, angles: number[] = [0], collection = COLLECTION) {
    await q(`INSERT INTO content (hash, doc) VALUES ($1, $2) ON CONFLICT DO NOTHING`, [hash, `origin ${path}`]);
    await q(
      `INSERT INTO origin_documents (collection, path, title, hash) VALUES ($1, $2, $3, $4)`,
      [collection, path, path, hash],
    );
    await insertEmbeddingsBatch(angles.map((deg, seq) => ({
      collection, path, hash, seq, pos: seq, model: MODEL, embedding: atAngle(deg),
    })));
  }

  async function flags(hash: string): Promise<boolean[]> {
    const r = await q<{ doc_tier: boolean }>(
      `SELECT doc_tier FROM content_vectors WHERE hash = $1 ORDER BY seq`, [hash],
    );
    return r.map(x => x.doc_tier);
  }

  async function search(collections?: string) {
    return withSchema(c => pgSearchVec(c, "anything", { embedder, collections, limit: 10 }));
  }

  it("a vector written after its documents row is born doc_tier = true (the reindex order)", async () => {
    await writeDoc("born.md", "h_born", [45, 50]);
    expect(await flags("h_born")).toEqual([true, true]);
  });

  it("an origin-only vector is doc_tier = false and is not returned, even as the NEAREST vector", async () => {
    await writeOrigin("o_near.md", "h_origin_near", [0]);
    expect(await flags("h_origin_near")).toEqual([false]);

    // CONTROL: it IS the nearest vector in the table, so its absence below is
    // about tiering, not distance.
    const nearest = await q<{ hash: string }>(
      `SELECT hash FROM content_vectors ORDER BY embedding <=> $1::vector LIMIT 1`,
      [toVectorLiteral(QUERY_VEC)],
    );
    expect(nearest[0]!.hash).toBe("h_origin_near");

    const out = await search(COLLECTION);
    expect(out.map(r => r.hash)).not.toContain("h_origin_near");
    expect(out.map(r => r.hash)).toContain("h_born");
  });

  it("tier move origin -> documents flips the SAME vectors to true, with no re-embed", async () => {
    await writeOrigin("o_move.md", "h_move", [1, 2]);
    const before = await q<{ seq: number; embedded_at: Date; e: string }>(
      `SELECT seq, embedded_at, embedding::text e FROM content_vectors WHERE hash = 'h_move' ORDER BY seq`,
    );
    expect(await flags("h_move")).toEqual([false, false]);

    // The move: a documents row now references the already-embedded hash.
    await upsertDocument({ collection: COLLECTION, path: "moved.md", title: "moved", hash: "h_move", body: "x" });

    expect(await flags("h_move")).toEqual([true, true]);
    const after = await q<{ seq: number; embedded_at: Date; e: string }>(
      `SELECT seq, embedded_at, embedding::text e FROM content_vectors WHERE hash = 'h_move' ORDER BY seq`,
    );
    expect(after).toEqual(before); // same rows, same embedded_at, same vectors
    expect((await search(COLLECTION)).map(r => r.hash)).toContain("h_move");
  });

  it("tier move documents -> origin (dropLegacyDocumentRows) flips to false and keeps the vectors", async () => {
    const coll = `${COLLECTION}_legacy`;
    await writeDoc("legacy.md", "h_legacy", [10, 11], coll);
    expect(await flags("h_legacy")).toEqual([true, true]);
    await q(
      `INSERT INTO origin_documents (collection, path, title, hash) VALUES ($1, 'legacy.md', 'legacy', 'h_legacy')`,
      [coll],
    );

    const r = await dropLegacyDocumentRows(coll, "sfw", { apply: true });

    expect(r.deleted).toBe(1);
    expect(await flags("h_legacy")).toEqual([false, false]);
  });

  it("deleting one of two documents rows sharing a hash keeps true; deleting the last flips false", async () => {
    await writeDoc("twin_a.md", "h_twin", [20]);
    await upsertDocument({ collection: COLLECTION, path: "twin_b.md", title: "b", hash: "h_twin", body: "x" });
    await q(`DELETE FROM documents WHERE collection = $1 AND path = 'twin_a.md'`, [COLLECTION]);
    expect(await flags("h_twin")).toEqual([true]);
    await q(`DELETE FROM documents WHERE collection = $1 AND path = 'twin_b.md'`, [COLLECTION]);
    expect(await flags("h_twin")).toEqual([false]);
  });

  it("an INACTIVE documents row still marks true (the GC protection rule; reactivation flips nothing)", async () => {
    await writeDoc("sleepy.md", "h_sleepy", [30]);
    await q(`UPDATE documents SET active = false WHERE collection = $1 AND path = 'sleepy.md'`, [COLLECTION]);
    expect(await flags("h_sleepy")).toEqual([true]);
    // ...and the reader's active fence, not the flag, keeps it out of results.
    expect((await search(COLLECTION)).map(r => r.hash)).not.toContain("h_sleepy");
  });

  it("a content change re-pointing documents.hash flips the old hash false and the new one true", async () => {
    await writeDoc("edit.md", "h_edit_v1", [40]);
    await writeDoc("edit.md", "h_edit_v2", [41]); // same path, new content
    expect(await flags("h_edit_v1")).toEqual([false]);
    expect(await flags("h_edit_v2")).toEqual([true]);
  });

  it("deleting a content row cascades through documents + vectors without a trigger error", async () => {
    await writeDoc("doomed.md", "h_doomed", [60]);
    await q(`DELETE FROM content WHERE hash = 'h_doomed'`);
    expect(await flags("h_doomed")).toEqual([]);
    expect((await q(`SELECT 1 FROM documents WHERE hash = 'h_doomed'`)).length).toBe(0);
  });

  it("the triggers kept every flag right so far: a reconcile pass on this schema repairs NOTHING", async () => {
    const r = await reconcileDocTierForVault("sfw");
    expect(r).toMatchObject({ vault: "sfw", markedTrue: 0, markedFalse: 0 });
  });

  it("reconcileDocTier repairs a hand-corrupted flag in BOTH directions, then converges", async () => {
    await q(`UPDATE content_vectors SET doc_tier = false WHERE hash = 'h_born' AND seq = 0`);
    await q(`UPDATE content_vectors SET doc_tier = true WHERE hash = 'h_origin_near' AND seq = 0`);

    const r = await reconcileDocTierForVault("sfw");

    expect(r).toMatchObject({ markedTrue: 1, markedFalse: 1 });
    expect(await flags("h_born")).toEqual([true, true]);
    expect(await flags("h_origin_near")).toEqual([false]);
    expect(await reconcileDocTierForVault("sfw")).toMatchObject({ markedTrue: 0, markedFalse: 0 });
  });

  it("the concurrent-insert race the triggers cannot see leaves a stale false, and reconcile heals it", async () => {
    await q(`INSERT INTO content (hash, doc) VALUES ('h_race', 'race')`);
    const a = await pool.connect();
    try {
      await a.query(`SET search_path TO ${schema}, public`);
      await a.query("BEGIN");
      await a.query(
        `INSERT INTO documents (collection, path, title, hash) VALUES ($1, 'race.md', 'race', 'h_race')`,
        [COLLECTION],
      );
      // B commits a vector while A's documents row is still invisible to it.
      await q(
        `INSERT INTO content_vectors (hash, seq, pos, model, embedding) VALUES ('h_race', 0, 0, $1, $2::vector)`,
        [MODEL, toVectorLiteral(atAngle(5))],
      );
      await a.query("COMMIT");
    } finally {
      a.release();
    }
    expect(await flags("h_race")).toEqual([false]); // the documented race window

    const r = await reconcileDocTierForVault("sfw");
    expect(r).toMatchObject({ markedTrue: 1, markedFalse: 0 });
    expect(await flags("h_race")).toEqual([true]);
  });

  /** One migration file, substituted for this schema, as the runner would send it. */
  function migrationSql(file: string): string {
    return substituteMigrationParams(readFileSync(join(MIGRATIONS_DIR, file), "utf-8"), schema, DIM);
  }

  /** Can another session read content_vectors while `holder` keeps its transaction open? */
  async function readerBlockedBy(holderSql: string): Promise<boolean> {
    const holder = await pool.connect();
    try {
      await holder.query(`SET search_path TO ${schema}, public`);
      await holder.query("BEGIN");
      await holder.query(holderSql);
      try {
        await withSchema(async r => {
          await r.query("BEGIN");
          try {
            await r.query("SET LOCAL lock_timeout = '300ms'");
            await r.query("SELECT count(*) FROM content_vectors");
          } finally {
            await r.query("ROLLBACK");
          }
        });
        return false;
      } catch (e) {
        if ((e as { code?: string }).code === "55P03") return true; // lock_not_available
        throw e;
      }
    } finally {
      await holder.query("ROLLBACK").catch(() => {});
      holder.release();
    }
  }

  it("backfill: re-applying migration 011 onto an all-false column re-derives exactly the documents set", async () => {
    // The state right after 010's ADD COLUMN ... DEFAULT false on a populated table.
    await q(`UPDATE content_vectors SET doc_tier = false`);
    await withSchema(async c => {
      await c.query("BEGIN");
      try {
        await c.query(migrationSql("011_content_vectors_doc_tier_backfill.sql"));
        await c.query("COMMIT");
      } catch (e) {
        await c.query("ROLLBACK");
        throw e;
      }
    });
    const mismatches = await q<{ n: number }>(
      `SELECT count(*)::int n FROM content_vectors cv
        WHERE cv.doc_tier IS DISTINCT FROM EXISTS (SELECT 1 FROM documents d WHERE d.hash = cv.hash)`,
    );
    expect(mismatches[0]!.n).toBe(0);
    const trues = await q<{ n: number }>(`SELECT count(*)::int n FROM content_vectors WHERE doc_tier`);
    expect(trues[0]!.n).toBeGreaterThan(0);
    expect(await flags("h_origin_near")).toEqual([false]);
    expect(await flags("h_born")).toEqual([true, true]);
  });

  it("the 011 backfill does NOT block readers while it runs; 010's ADD COLUMN does (why they are split)", async () => {
    // CONTROL first: 010's DDL (ADD COLUMN, DROP TRIGGER — re-applied
    // idempotently here) takes ACCESS EXCLUSIVE, so a reader with a
    // lock_timeout is refused. That is the lock the 2 min 55 s backfill would
    // have run under had it stayed in 010 (measured on the scratch copy of the
    // live vault); 010 alone holds it for milliseconds.
    expect(await readerBlockedBy(migrationSql("010_content_vectors_doc_tier.sql"))).toBe(true);
    // 011 holds only ROW EXCLUSIVE: the same reader proceeds.
    await q(`UPDATE content_vectors SET doc_tier = false WHERE hash = 'h_born'`);
    expect(await readerBlockedBy(migrationSql("011_content_vectors_doc_tier_backfill.sql"))).toBe(false);
    // Both holders rolled back; put the flag right for the tests below.
    await reconcileDocTierForVault("sfw");
    expect(await flags("h_born")).toEqual([true, true]);
  });

  it("migration 012 built a VALID partial index whose predicate is doc_tier", async () => {
    const r = await q<{ def: string; valid: boolean }>(
      `SELECT pg_get_indexdef(i.indexrelid) def, i.indisvalid valid
         FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
         JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = $2`,
      [schema, PARTIAL_IDX],
    );
    expect(r.length).toBe(1);
    expect(r[0]!.valid).toBe(true);
    expect(r[0]!.def).toMatch(/USING hnsw \(embedding vector_cosine_ops\)/);
    expect(r[0]!.def).toMatch(/WHERE doc_tier/);
  });

  it("the planner serves buildVecSearchQuery from the PARTIAL index, not the full one", async () => {
    // A fixture table is tiny, so seq scans and sorts are disabled to make the
    // planner choose between the ordered index paths it would take at scale.
    // Of the two HNSW indexes, only the partial one proves the `cv.doc_tier`
    // predicate — take that predicate out of the query and this goes RED
    // (the plan falls back to content_vectors_embedding_hnsw_idx).
    const { text, values } = buildVecSearchQuery(toVectorLiteral(QUERY_VEC), null, 64);
    const plan = await withSchema(async c => {
      await c.query("BEGIN");
      try {
        await c.query("SET LOCAL enable_seqscan = off");
        await c.query("SET LOCAL enable_sort = off");
        const r = await c.query<{ "QUERY PLAN": string }>(`EXPLAIN (COSTS OFF) ${text}`, values);
        return r.rows.map(x => x["QUERY PLAN"]).join("\n");
      } finally {
        await c.query("ROLLBACK");
      }
    });
    expect(plan).toContain(PARTIAL_IDX);
    expect(plan).not.toMatch(new RegExp(`\\b${FULL_IDX}\\b`));
  });
});
