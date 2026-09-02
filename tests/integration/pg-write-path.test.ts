/**
 * PostgreSQL write-path integration tests (master-harness-vn4rz.7, ADR-0162).
 *
 * These run against the LIVE clawmem-pg cluster (127.0.0.1:5433), inside a
 * throwaway schema per run — "don't mock what you can run" (testing-strategy §3).
 * A mocked pg client would prove nothing about the two things that actually
 * matter here: that ON CONFLICT hits the constraint we named, and that a refusal
 * inside the write transaction really rolls back.
 *
 * They SKIP (not fail) when CLAWMEM_PG_URL is unset, so `bun test` on a machine
 * without the cluster stays green.
 *
 * The suite's central obligation is the one the brief calls non-negotiable:
 * prove the instrument can go RED. Every guard here is exercised in both
 * directions — the refusal fires AND the matching write is untouched — and the
 * negative cases assert that NOTHING was written, not merely that a throw
 * happened.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR } from "../../src/pg/migrate.ts";
import { toVectorLiteral } from "../../src/pg/client.ts";
import {
  PgVecBatchModelMismatchError,
  PgVecDimensionMismatchError,
  PgVecWriteModelMismatchError,
} from "../../src/pg/errors.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 768;
const VAULT_MODEL = "embeddinggemma";
const FOREIGN_MODEL = "ggml-org/embeddinggemma-300M-GGUF-Q8_0"; // the 2026-08-09 poisoner

const d = URL_ ? describe : describe.skip;

function vec(fill: number, n = DIM): number[] {
  return new Array(n).fill(fill);
}

d("PG write path", () => {
  let pool: pg.Pool;
  let schema: string;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: URL_ });
    schema = `clawmem_test_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const c = await pool.connect();
    try {
      await c.query(`CREATE SCHEMA ${schema}`);
      await c.query(`SET search_path TO ${schema}, public`);
      for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
        const sql = readFileSync(join(MIGRATIONS_DIR, f), "utf-8").replaceAll(":EMBED_DIM", String(DIM));
        await c.query(sql);
      }
    } finally {
      c.release();
    }
  });

  afterAll(async () => {
    if (schema) await pool.query(`DROP SCHEMA IF EXISTS ${schema} CASCADE`);
    await pool.end();
  });

  /** Run fn on a client pinned to the test schema. */
  async function withSchema<T>(fn: (c: pg.PoolClient) => Promise<T>): Promise<T> {
    const c = await pool.connect();
    try {
      await c.query(`SET search_path TO ${schema}, public`);
      return await fn(c);
    } finally {
      c.release();
    }
  }

  /**
   * The write transaction under test, re-expressed against the test schema.
   * Mirrors src/pg/write.ts::insertEmbeddingsBatch exactly — advisory lock,
   * geometry, intra-batch heterogeneity, per-row dimension, per-row model fence,
   * then the INSERTs — so what these tests prove is the real ordering.
   */
  async function insertVectors(
    writes: { hash: string; seq: number; model: string; embedding: number[] }[],
    opts: { disablePreflight?: boolean } = {},
  ): Promise<void> {
    await withSchema(async c => {
      await c.query("BEGIN");
      try {
        await c.query("SELECT pg_advisory_xact_lock($1)", [0x1a2b3c4d]);

        const { rows: dimRows } = await c.query<{ dims: number }>(
          `SELECT atttypmod AS dims FROM pg_attribute
            WHERE attrelid = '${schema}.content_vectors'::regclass
              AND attname = 'embedding' AND NOT attisdropped`,
        );
        const dim = dimRows[0]!.dims;

        const batchModels = [...new Set(writes.map(w => w.model))].sort();
        if (batchModels.length > 1) {
          throw new PgVecBatchModelMismatchError(batchModels[0]!, batchModels[1]!);
        }
        for (const w of writes) {
          if (w.embedding.length !== dim) {
            throw new PgVecDimensionMismatchError(dim, w.embedding.length, `hash=${w.hash}`);
          }
        }
        if (!opts.disablePreflight) {
          const { rows } = await c.query<{ model: string }>(
            "SELECT DISTINCT model FROM content_vectors ORDER BY model",
          );
          const stored = rows.map(r => r.model);
          for (const w of writes) {
            if (stored.length > 0 && !(stored.length === 1 && stored[0] === w.model)) {
              throw new PgVecWriteModelMismatchError(stored, w.model, "the test endpoint");
            }
          }
        }
        for (const w of writes) {
          await c.query(
            `INSERT INTO content_vectors (hash, seq, pos, model, embedding)
             VALUES ($1, $2, 0, $3, $4::vector)
             ON CONFLICT ON CONSTRAINT content_vectors_pkey DO UPDATE SET
               model = EXCLUDED.model, embedding = EXCLUDED.embedding`,
            [w.hash, w.seq, w.model, toVectorLiteral(w.embedding)],
          );
        }
        await c.query("COMMIT");
      } catch (e) {
        await c.query("ROLLBACK");
        throw e;
      }
    });
  }

  async function seedContent(hash: string, doc = "hello world"): Promise<void> {
    await withSchema(c =>
      c.query(
        `INSERT INTO content (hash, doc) VALUES ($1, $2)
         ON CONFLICT ON CONSTRAINT content_pkey DO UPDATE SET doc = EXCLUDED.doc`,
        [hash, doc],
      ),
    );
  }

  async function vectorCount(): Promise<number> {
    const { rows } = await withSchema(c => c.query<{ n: string }>("SELECT count(*)::text n FROM content_vectors"));
    return Number(rows[0]!.n);
  }

  // =========================================================================
  // Migrations
  // =========================================================================

  it("applies twice with no error and no catalog drift", async () => {
    const snapshot = async () => {
      const { rows } = await withSchema(c =>
        c.query<{ s: string }>(
          `SELECT table_name || '.' || column_name || ':' || data_type AS s
             FROM information_schema.columns WHERE table_schema = $1 ORDER BY 1`,
          [schema],
        ),
      );
      return rows.map(r => r.s).join("\n");
    };
    const before = await snapshot();
    await withSchema(async c => {
      await c.query(`SET search_path TO ${schema}, public`);
      for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
        await c.query(readFileSync(join(MIGRATIONS_DIR, f), "utf-8").replaceAll(":EMBED_DIM", String(DIM)));
      }
    });
    expect(await snapshot()).toBe(before);
  });

  it("built the HNSW index VALID, not INVALID", async () => {
    const { rows } = await withSchema(c =>
      c.query<{ relname: string; indisvalid: boolean }>(
        `SELECT ci.relname, i.indisvalid FROM pg_index i
           JOIN pg_class ci ON ci.oid = i.indexrelid
           JOIN pg_namespace n ON n.oid = ci.relnamespace
          WHERE n.nspname = $1 AND ci.relname LIKE '%hnsw%'`,
        [schema],
      ),
    );
    expect(rows.length).toBe(1);
    expect(rows[0]!.indisvalid).toBe(true);
  });

  it("has NO vault column anywhere (ADR-0162 §1: database-per-vault)", async () => {
    const { rows } = await withSchema(c =>
      c.query<{ table_name: string }>(
        `SELECT table_name FROM information_schema.columns
          WHERE table_schema = $1 AND column_name = 'vault'`,
        [schema],
      ),
    );
    expect(rows).toEqual([]);
  });

  // =========================================================================
  // The geometry preflight — BOTH directions
  // =========================================================================

  it("ACCEPTS a matching-model write (the guard is not just an off switch)", async () => {
    await seedContent("h_ok");
    await insertVectors([{ hash: "h_ok", seq: 0, model: VAULT_MODEL, embedding: vec(0.1) }]);
    expect(await vectorCount()).toBe(1);
  });

  it("REFUSES a foreign-model write and rolls back with NOTHING written", async () => {
    await seedContent("h_foreign");
    const before = await vectorCount();
    expect(before).toBeGreaterThan(0); // the vault must be non-empty or the guard no-ops

    await expect(
      insertVectors([{ hash: "h_foreign", seq: 0, model: FOREIGN_MODEL, embedding: vec(0.2) }]),
    ).rejects.toThrow(PgVecWriteModelMismatchError);

    // The load-bearing assertion: the transaction rolled back. A guard that
    // throws AFTER the insert would pass a bare rejects.toThrow and still poison
    // the vault.
    expect(await vectorCount()).toBe(before);
    const { rows } = await withSchema(c =>
      c.query("SELECT 1 FROM content_vectors WHERE hash = $1", ["h_foreign"]),
    );
    expect(rows).toEqual([]);
  });

  it("PROVES THE INSTRUMENT CAN GO RED: with the preflight disabled, the same foreign write LANDS", async () => {
    // break -> RED. This is the control that makes the test above meaningful:
    // if the foreign write were rejected by something OTHER than the preflight
    // (a constraint, a type error), disabling the preflight would change
    // nothing and the passing test would prove nothing.
    await seedContent("h_break");
    const before = await vectorCount();
    await insertVectors(
      [{ hash: "h_break", seq: 0, model: FOREIGN_MODEL, embedding: vec(0.3) }],
      { disablePreflight: true },
    );
    expect(await vectorCount()).toBe(before + 1);

    const { rows } = await withSchema(c =>
      c.query<{ model: string }>("SELECT DISTINCT model FROM content_vectors ORDER BY model"),
    );
    expect(rows.map(r => r.model).sort()).toEqual([VAULT_MODEL, FOREIGN_MODEL].sort());

    // restore -> GREEN. Clean the poison back out and confirm the guard is
    // armed again on a now-homogeneous vault.
    await withSchema(c => c.query("DELETE FROM content_vectors WHERE hash = $1", ["h_break"]));
    await expect(
      insertVectors([{ hash: "h_break", seq: 0, model: FOREIGN_MODEL, embedding: vec(0.3) }]),
    ).rejects.toThrow(PgVecWriteModelMismatchError);
  });

  it("REFUSES a batch carrying two models, writing none of it", async () => {
    await seedContent("h_a");
    await seedContent("h_b");
    const before = await vectorCount();
    await expect(
      insertVectors([
        { hash: "h_a", seq: 0, model: VAULT_MODEL, embedding: vec(0.4) },
        { hash: "h_b", seq: 0, model: FOREIGN_MODEL, embedding: vec(0.5) },
      ]),
    ).rejects.toThrow(PgVecBatchModelMismatchError);
    expect(await vectorCount()).toBe(before);
  });

  // =========================================================================
  // Dimension safety — BOTH directions
  // =========================================================================

  it("REFUSES a short embedding rather than padding it", async () => {
    await seedContent("h_short");
    const before = await vectorCount();
    await expect(
      insertVectors([{ hash: "h_short", seq: 0, model: VAULT_MODEL, embedding: vec(0.6, 384) }]),
    ).rejects.toThrow(PgVecDimensionMismatchError);
    expect(await vectorCount()).toBe(before);
  });

  it("REFUSES a long embedding rather than truncating it", async () => {
    await seedContent("h_long");
    const before = await vectorCount();
    await expect(
      insertVectors([{ hash: "h_long", seq: 0, model: VAULT_MODEL, embedding: vec(0.7, 1536) }]),
    ).rejects.toThrow(PgVecDimensionMismatchError);
    expect(await vectorCount()).toBe(before);
  });

  it("PROVES THE DIMENSION INSTRUMENT CAN GO RED: Postgres itself refuses a wrong-width vector", async () => {
    // Even with the app-level check bypassed, the vector(768) column must reject
    // it — so the app check is defence in depth, not the only thing standing
    // between us and a silently reshaped vector. Assert the ENGINE's refusal too.
    await seedContent("h_engine");
    await expect(
      withSchema(c =>
        c.query(
          `INSERT INTO content_vectors (hash, seq, pos, model, embedding)
           VALUES ($1, 0, 0, $2, $3::vector)`,
          ["h_engine", VAULT_MODEL, toVectorLiteral(vec(0.8, 384))],
        ),
      ),
    ).rejects.toThrow(/expected 768 dimensions, not 384/);
  });

  // =========================================================================
  // Facets + ON CONFLICT targeting
  // =========================================================================

  it("defaults every facet to an explicit 'unknown' (ADR-0162 §3)", async () => {
    await seedContent("h_facets", "# Facet doc\n\nbody text");
    await withSchema(c =>
      c.query(
        `INSERT INTO documents (collection, path, title, hash) VALUES ($1,$2,$3,$4)`,
        ["c1", "p1.md", "Facet doc", "h_facets"],
      ),
    );
    const { rows } = await withSchema(c =>
      c.query(`SELECT domain, audience, trust_tier, sensitivity, content_type, source_ref
                 FROM documents WHERE collection='c1' AND path='p1.md'`),
    );
    expect(rows[0]).toEqual({
      domain: "unknown", audience: "unknown", trust_tier: "unknown",
      sensitivity: "unknown", content_type: "unknown", source_ref: null,
    });
  });

  it("REFUSES an out-of-enum facet value at the database, not in application code", async () => {
    await seedContent("h_bad");
    await expect(
      withSchema(c =>
        c.query(
          `INSERT INTO documents (collection, path, title, hash, audience) VALUES ($1,$2,$3,$4,$5)`,
          ["c1", "bad.md", "t", "h_bad", "marketing"],
        ),
      ),
    ).rejects.toThrow(/documents_audience_check/);
  });

  it("upserts documents against the NAMED (collection, path) constraint, not the pkey", async () => {
    await seedContent("h_v1", "# v1");
    await seedContent("h_v2", "# v2");
    const ins = (hash: string, title: string) =>
      withSchema(c =>
        c.query<{ id: string }>(
          `INSERT INTO documents (collection, path, title, hash) VALUES ($1,$2,$3,$4)
           ON CONFLICT ON CONSTRAINT documents_collection_path_key DO UPDATE SET
             title = EXCLUDED.title, hash = EXCLUDED.hash,
             revision_count = documents.revision_count + 1
           RETURNING id`,
          ["c2", "same.md", title, hash],
        ),
      );
    const a = (await ins("h_v1", "first")).rows[0]!.id;
    const b = (await ins("h_v2", "second")).rows[0]!.id;
    expect(b).toBe(a); // same row, updated in place — not a duplicate

    const { rows } = await withSchema(c =>
      c.query<{ n: string; title: string; revision_count: string }>(
        "SELECT count(*)::text n, max(title) title, max(revision_count)::text revision_count FROM documents WHERE collection='c2'",
      ),
    );
    expect(rows[0]!.n).toBe("1");
    expect(rows[0]!.title).toBe("second");
    expect(rows[0]!.revision_count).toBe("2");
  });

  // =========================================================================
  // FTS
  // =========================================================================

  it("maintains the tsvector by trigger, weighted, and matches an explicitly-configured query", async () => {
    await seedContent("h_fts", "The quick brown foxes were jumping over lazy dogs repeatedly.");
    await withSchema(c =>
      c.query(
        `INSERT INTO documents (collection, path, title, hash) VALUES ($1,$2,$3,$4)`,
        ["c3", "fts.md", "Marmalade Chronicles", "h_fts"],
      ),
    );
    const { rows } = await withSchema(c =>
      c.query<{ fts: string }>("SELECT fts::text AS fts FROM documents WHERE collection='c3'"),
    );
    // Title terms carry weight A, body terms weight D — the setweight() call is
    // observable in the stored vector, not merely intended.
    expect(rows[0]!.fts).toMatch(/marmalad':\d+A/);
    expect(rows[0]!.fts).toMatch(/fox/);

    // The FTS configuration is named EXPLICITLY at query time. ADR-0162 §2: a
    // bare call at one end and 'english' at the other fails SILENTLY.
    const hit = await withSchema(c =>
      c.query("SELECT 1 FROM documents WHERE collection='c3' AND fts @@ to_tsquery('english', 'jumping')"),
    );
    expect(hit.rows.length).toBe(1); // 'jumping' stems to 'jump' under english
  });

  it("PROVES THE FTS TRIGGER CAN GO RED: a body change with a stale trigger would not match", async () => {
    // Control for the test above: drop the trigger, insert, and confirm the fts
    // column stays NULL. If fts were populated by something else (a default, a
    // generated column), the trigger test would be proving nothing.
    await withSchema(c => c.query(`DROP TRIGGER documents_fts_trg ON ${schema}.documents`));
    await seedContent("h_notrg", "unindexed body");
    await withSchema(c =>
      c.query(`INSERT INTO documents (collection, path, title, hash) VALUES ($1,$2,$3,$4)`,
        ["c4", "notrg.md", "No Trigger", "h_notrg"]),
    );
    const { rows } = await withSchema(c =>
      c.query<{ fts: string | null }>("SELECT fts::text AS fts FROM documents WHERE collection='c4'"),
    );
    expect(rows[0]!.fts).toBeNull();

    // restore -> GREEN
    await withSchema(c =>
      c.query(`CREATE TRIGGER documents_fts_trg
                 BEFORE INSERT OR UPDATE OF title, description, tags, hash ON ${schema}.documents
                 FOR EACH ROW EXECUTE FUNCTION documents_fts_refresh()`),
    );
    await withSchema(c => c.query(`UPDATE documents SET title = title WHERE collection='c4'`));
    const after = await withSchema(c =>
      c.query<{ fts: string | null }>("SELECT fts::text AS fts FROM documents WHERE collection='c4'"),
    );
    expect(after.rows[0]!.fts).not.toBeNull();
  });

  // =========================================================================
  // entity_nodes + memory_evolution
  // =========================================================================

  it("upserts entity_nodes accumulating mention_count", async () => {
    const up = (name: string) =>
      withSchema(c =>
        c.query(
          `INSERT INTO entity_nodes (entity_id, name, mention_count, last_seen)
           VALUES ($1,$2,1,now())
           ON CONFLICT ON CONSTRAINT entity_nodes_pkey DO UPDATE SET
             name = COALESCE(EXCLUDED.name, entity_nodes.name),
             mention_count = entity_nodes.mention_count + 1, last_seen = now()`,
          ["e1", name],
        ),
      );
    await up("Yoshiee");
    await up("Yoshiee");
    const { rows } = await withSchema(c =>
      c.query<{ mention_count: string }>("SELECT mention_count::text FROM entity_nodes WHERE entity_id='e1'"),
    );
    expect(rows[0]!.mention_count).toBe("2");
  });

  it("cascades memory_evolution when its document is deleted", async () => {
    await seedContent("h_me", "# me");
    const { rows: dr } = await withSchema(c =>
      c.query<{ id: string }>(
        `INSERT INTO documents (collection, path, title, hash) VALUES ('c5','me.md','me',$1) RETURNING id`,
        ["h_me"],
      ),
    );
    const id = dr[0]!.id;
    await withSchema(c =>
      c.query(`INSERT INTO memory_evolution (memory_id, triggered_by, reasoning) VALUES ($1,$1,'because')`, [id]),
    );
    await withSchema(c => c.query("DELETE FROM documents WHERE id = $1", [id]));
    const { rows } = await withSchema(c => c.query("SELECT 1 FROM memory_evolution WHERE memory_id = $1", [id]));
    expect(rows).toEqual([]);
  });
});
