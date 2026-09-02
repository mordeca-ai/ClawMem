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
import { closePool, toVectorLiteral } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import {
  insertEmbeddingsBatch,
  insertMemoryEvolution,
  upsertDocument,
  upsertEntityNode,
  type EmbeddingWrite,
} from "../../src/pg/write.ts";
import {
  PgVecBatchModelMismatchError,
  PgVecDimensionMismatchError,
  PgVecWriteModelMismatchError,
} from "../../src/pg/errors.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 768;
const VAULT_MODEL = "embeddinggemma";
const FOREIGN_MODEL = "ggml-org/embeddinggemma-300M-GGUF-Q8_0"; // the 2026-08-09 poisoner
// EmbeddingWrite.collection became REQUIRED at master-harness-0ynkd (it is what
// routes a fragment's vectors to the same vault as its document). "docs" is an
// ordinary sfw collection — deliberately NOT a PRIVATE_ROOTS member, so these
// pre-existing model/geometry assertions keep exercising the sfw path unchanged.
const SFW_COLLECTION = "docs";

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
    // THE POINT OF THE SCHEMA KNOB: from here on the REAL exported write
    // functions resolve their unqualified tables inside this throwaway schema,
    // so the suite exercises src/pg/write.ts itself rather than a copy of it.
    setPgSchema(schema);
  });

  afterAll(async () => {
    setPgSchema(null);
    await closePool();
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
   * THE FUNCTION UNDER TEST — the real exported one (vn4rz.7 pass C, Task 3.1).
   *
   * Pass B could not call this directly: write.ts resolves its tables through
   * the connection's search_path and there was no way to point that at a test
   * schema, so the suite re-expressed the whole transaction and any future
   * drift between write.ts and that copy would have gone uncaught. setPgSchema()
   * in beforeAll closes that gap; this wrapper only widens the argument type.
   */
  async function insertVectors(
    writes: { hash: string; seq: number; model: string; embedding: number[]; pos?: number }[],
  ): Promise<void> {
    await insertEmbeddingsBatch(
      writes.map<EmbeddingWrite>(w => ({
        collection: SFW_COLLECTION,
        hash: w.hash, seq: w.seq, pos: w.pos ?? 0,
        model: w.model, embedding: w.embedding,
      })),
    );
  }

  /**
   * THE CONTROL, and ONLY the control. A deliberate re-expression of the write
   * transaction with the model preflight REMOVED, used to prove the preflight
   * is what refuses a foreign-model write — nothing else in the schema does.
   * It is never used to assert positive behaviour of the production path.
   */
  async function insertVectorsNoPreflight(
    writes: { hash: string; seq: number; model: string; embedding: number[] }[],
  ): Promise<void> {
    await withSchema(async c => {
      await c.query("BEGIN");
      try {
        await c.query("SELECT pg_advisory_xact_lock($1)", [0x1a2b3c4d]);
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
    await insertVectorsNoPreflight(
      [{ hash: "h_break", seq: 0, model: FOREIGN_MODEL, embedding: vec(0.3) }],
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
  // Fragment-level embedding (delta amendment 1)
  // =========================================================================

  it("stores MANY fragments per document under one hash, keyed (hash, seq)", async () => {
    // The regression this locks down: the first cut of the reindexer wrote ONE
    // vector per document. Document-count parity stayed green across that gap
    // because a doc-count check structurally cannot see the vector layer, while
    // within-document retrieval granularity was destroyed.
    await seedContent("h_frag", "# Doc\n\nbody");
    const frags = [0, 1, 2, 3].map(seq => ({
      hash: "h_frag", seq, model: VAULT_MODEL, embedding: vec(0.1 + seq / 100),
    }));
    await insertVectors(frags);

    const { rows } = await withSchema(c =>
      c.query<{ n: string; maxseq: string }>(
        "SELECT count(*)::text n, max(seq)::text maxseq FROM content_vectors WHERE hash='h_frag'",
      ),
    );
    expect(rows[0]!.n).toBe("4");
    expect(rows[0]!.maxseq).toBe("3");
  });

  it("re-embedding a fragment REPLACES it rather than duplicating (composite-key upsert)", async () => {
    await seedContent("h_reembed", "# Doc\n\nbody");
    await insertVectors([{ hash: "h_reembed", seq: 2, model: VAULT_MODEL, embedding: vec(0.5) }]);
    await insertVectors([{ hash: "h_reembed", seq: 2, model: VAULT_MODEL, embedding: vec(0.9) }]);
    const { rows } = await withSchema(c =>
      c.query<{ n: string }>("SELECT count(*)::text n FROM content_vectors WHERE hash='h_reembed'"),
    );
    expect(rows[0]!.n).toBe("1");
  });

  it("carries the fragment provenance columns rather than narrowing them away", async () => {
    await seedContent("h_meta", "# Doc\n\nbody");
    await withSchema(c =>
      c.query(
        `INSERT INTO content_vectors (hash, seq, pos, model, embedding, fragment_type, fragment_label, canonical_id, embed_input_fp)
         VALUES ($1, 0, 42, $2, $3::vector, 'section', 'The principle', 'abc123', 'fp0')`,
        ["h_meta", VAULT_MODEL, toVectorLiteral(vec(0.1))],
      ),
    );
    const { rows } = await withSchema(c =>
      c.query(`SELECT pos, fragment_type, fragment_label, canonical_id, embed_input_fp
                 FROM content_vectors WHERE hash='h_meta'`),
    );
    expect(rows[0]).toEqual({
      pos: 42, fragment_type: "section", fragment_label: "The principle",
      canonical_id: "abc123", embed_input_fp: "fp0",
    });
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

  // =========================================================================
  // Fragment-level shape (vn4rz.7 pass C, Task 1) — through the REAL function
  // =========================================================================

  it("round-trips the FULL content_vectors shape: (hash,seq) key, pos, and every fragment column", async () => {
    await seedContent("h_fragshape", "# frag doc");
    const writes: EmbeddingWrite[] = [
      { collection: SFW_COLLECTION, hash: "h_fragshape", seq: 0, pos: 1, embedding: vec(0.11), model: VAULT_MODEL,
        fragmentType: "full", fragmentLabel: null, canonicalId: "cid_frag", embedInputFp: "fp0" },
      { collection: SFW_COLLECTION, hash: "h_fragshape", seq: 1, pos: 14, embedding: vec(0.12), model: VAULT_MODEL,
        fragmentType: "section", fragmentLabel: "Why this exists", canonicalId: "cid_frag", embedInputFp: "fp1" },
      { collection: SFW_COLLECTION, hash: "h_fragshape", seq: 2, pos: 40, embedding: vec(0.13), model: VAULT_MODEL,
        fragmentType: "frontmatter", fragmentLabel: "title", canonicalId: "cid_frag", embedInputFp: "fp2" },
    ];
    await insertEmbeddingsBatch(writes);

    const { rows } = await withSchema(c =>
      c.query<{ seq: number; pos: number; fragment_type: string; fragment_label: string | null;
                canonical_id: string; embed_input_fp: string; dims: number }>(
        `SELECT seq, pos, fragment_type, fragment_label, canonical_id, embed_input_fp,
                vector_dims(embedding) AS dims
           FROM content_vectors WHERE hash = 'h_fragshape' ORDER BY seq`,
      ),
    );
    expect(rows.map(r => r.seq)).toEqual([0, 1, 2]);
    expect(rows.map(r => r.pos)).toEqual([1, 14, 40]);
    expect(rows.map(r => r.fragment_type)).toEqual(["full", "section", "frontmatter"]);
    expect(rows.map(r => r.fragment_label)).toEqual([null, "Why this exists", "title"]);
    expect(rows.map(r => r.canonical_id)).toEqual(["cid_frag", "cid_frag", "cid_frag"]);
    expect(rows.map(r => r.embed_input_fp)).toEqual(["fp0", "fp1", "fp2"]);
    expect(rows.map(r => r.dims)).toEqual([DIM, DIM, DIM]);

    // (hash, seq) is a COMPOSITE key: re-writing seq=1 updates in place and
    // does not collapse the other fragments of the same document.
    await insertEmbeddingsBatch([{ ...writes[1]!, pos: 99, embedInputFp: "fp1b" }]);
    const { rows: after } = await withSchema(c =>
      c.query<{ n: string; pos: number }>(
        `SELECT count(*) OVER ()::text AS n, pos FROM content_vectors WHERE hash='h_fragshape' AND seq=1`,
      ),
    );
    expect(after[0]!.n).toBe("1");
    expect(after[0]!.pos).toBe(99);
    const { rows: all } = await withSchema(c =>
      c.query<{ n: string }>("SELECT count(*)::text n FROM content_vectors WHERE hash='h_fragshape'"),
    );
    expect(all[0]!.n).toBe("3");
  });

  it("REFUSES two models inside ONE MULTI-FRAGMENT batch on an EMPTY vault", async () => {
    // The highest-value invariant in the bead: on an empty content_vectors the
    // database-level fence no-ops (nothing to be inconsistent with), so the
    // intra-batch check is the ONLY thing standing between a flapping endpoint
    // and a permanently heterogeneous vector space. Pass B only ever exercised
    // it with ONE vector per batch; the fragment refactor made batches large,
    // which is exactly when a per-row-only check would slip.
    await withSchema(c => c.query("TRUNCATE content_vectors"));
    await seedContent("h_multi", "# multi");
    expect(await vectorCount()).toBe(0);

    const batch: EmbeddingWrite[] = [];
    for (let seq = 0; seq < 17; seq++) {
      batch.push({ collection: SFW_COLLECTION, hash: "h_multi", seq, pos: seq * 10, embedding: vec(0.2), model: VAULT_MODEL });
    }
    // one poisoned fragment, buried in the middle of a realistic-size batch
    batch[9] = { collection: SFW_COLLECTION, hash: "h_multi", seq: 9, pos: 90, embedding: vec(0.2), model: FOREIGN_MODEL };

    await expect(insertEmbeddingsBatch(batch)).rejects.toThrow(PgVecBatchModelMismatchError);
    expect(await vectorCount()).toBe(0); // the WHOLE batch rolled back, not 16 of 17
  });

  it("ACCEPTS the same multi-fragment batch once it is homogeneous (the guard is not an off switch)", async () => {
    const batch: EmbeddingWrite[] = [];
    for (let seq = 0; seq < 17; seq++) {
      batch.push({ collection: SFW_COLLECTION, hash: "h_multi", seq, pos: seq * 10, embedding: vec(0.2), model: VAULT_MODEL });
    }
    await insertEmbeddingsBatch(batch);
    expect(await vectorCount()).toBe(17);
  });

  // =========================================================================
  // Two-writer race on an EMPTY table (vn4rz.7 pass C, Task 3.3)
  // =========================================================================

  /**
   * A genuine two-connection race, forced by explicit interleaving rather than
   * hoped for from timing.
   *
   * Writer A: BEGIN, (lock), read the DISTINCT model set, then PAUSE.
   * Writer B: BEGIN, (lock), read, INSERT the foreign model, COMMIT.
   * Writer A: resumes, INSERTs the vault model, COMMIT.
   *
   * WITHOUT the advisory lock both readers see an empty table under READ
   * COMMITTED, both conclude "nothing to be inconsistent with", and the vault
   * ends up holding two models. WITH the lock, B blocks at acquisition until A
   * commits, then reads a non-empty table and refuses.
   *
   * This one has to be a re-expression: the invariant lives in the MIDDLE of
   * insertEmbeddingsBatch's transaction and the function exposes no injection
   * point between its read and its write. The real-function test below covers
   * the production path unbarriered; this pair is what proves the LOCK — and
   * not luck — is the discriminator.
   */
  async function racePair(useLock: boolean): Promise<{ aOk: boolean; bErr: unknown }> {
    const a = await pool.connect();
    const b = await pool.connect();
    const preflight = async (c: pg.PoolClient, model: string) => {
      const { rows } = await c.query<{ model: string }>(
        "SELECT DISTINCT model FROM content_vectors ORDER BY model",
      );
      const stored = rows.map(r => r.model);
      if (stored.length > 0 && !(stored.length === 1 && stored[0] === model)) {
        throw new PgVecWriteModelMismatchError(stored, model, "the race harness");
      }
    };
    const ins = (c: pg.PoolClient, hash: string, model: string, fill: number) =>
      c.query(
        `INSERT INTO content_vectors (hash, seq, pos, model, embedding)
         VALUES ($1, 0, 0, $2, $3::vector)
         ON CONFLICT ON CONSTRAINT content_vectors_pkey DO NOTHING`,
        [hash, model, toVectorLiteral(vec(fill))],
      );
    try {
      await a.query(`SET search_path TO ${schema}, public`);
      await b.query(`SET search_path TO ${schema}, public`);
      await a.query("BEGIN");
      await b.query("BEGIN");

      if (useLock) await a.query("SELECT pg_advisory_xact_lock($1)", [0x1a2b3c4d]);
      await preflight(a, VAULT_MODEL);

      // B runs to completion (or blocks on the lock) while A is paused.
      const bRun = (async () => {
        if (useLock) await b.query("SELECT pg_advisory_xact_lock($1)", [0x1a2b3c4d]);
        await preflight(b, FOREIGN_MODEL);
        await ins(b, "race_b", FOREIGN_MODEL, 0.9);
        await b.query("COMMIT");
      })();
      const bSettled = bRun.then(() => null).catch(e => e);
      // Give B a real chance to get all the way through when nothing blocks it.
      await new Promise(r => setTimeout(r, 250));

      await ins(a, "race_a", VAULT_MODEL, 0.1);
      await a.query("COMMIT");

      const bErr = await bSettled;
      if (bErr) await b.query("ROLLBACK").catch(() => {});
      return { aOk: true, bErr };
    } finally {
      a.release();
      b.release();
    }
  }

  it("PROVES THE RACE HARNESS CAN GO RED: without the advisory lock, two writers BOTH land, on an empty table", async () => {
    await withSchema(c => c.query("TRUNCATE content_vectors"));
    await seedContent("race_a", "a");
    await seedContent("race_b", "b");

    const { bErr } = await racePair(false);
    expect(bErr).toBeNull(); // B's preflight saw an empty table and let it through

    const { rows } = await withSchema(c =>
      c.query<{ model: string }>("SELECT DISTINCT model FROM content_vectors ORDER BY model"),
    );
    // The anomaly, reproduced: a heterogeneous vector space from two writers
    // that each individually passed their own check.
    expect(rows.map(r => r.model).sort()).toEqual([VAULT_MODEL, FOREIGN_MODEL].sort());
  });

  it("CLOSES THE RACE: with the advisory lock, the SAME interleaving leaves exactly one model", async () => {
    await withSchema(c => c.query("TRUNCATE content_vectors"));

    const { bErr } = await racePair(true);
    expect(bErr).toBeInstanceOf(PgVecWriteModelMismatchError);

    const { rows } = await withSchema(c =>
      c.query<{ model: string }>("SELECT DISTINCT model FROM content_vectors ORDER BY model"),
    );
    expect(rows.map(r => r.model)).toEqual([VAULT_MODEL]);
  });

  it("the REAL insertEmbeddingsBatch survives a 24-way concurrent mixed-model race on an empty table", async () => {
    await withSchema(c => c.query("TRUNCATE content_vectors"));
    for (let i = 0; i < 24; i++) await seedContent(`conc_${i}`, `doc ${i}`);

    const attempts = Array.from({ length: 24 }, (_, i) =>
      insertEmbeddingsBatch([{
        collection: SFW_COLLECTION,
        hash: `conc_${i}`, seq: 0, pos: 0, embedding: vec(0.5),
        model: i % 2 === 0 ? VAULT_MODEL : FOREIGN_MODEL,
      }]),
    );
    const settled = await Promise.allSettled(attempts);
    const rejected = settled.filter(r => r.status === "rejected");

    const { rows } = await withSchema(c =>
      c.query<{ model: string }>("SELECT DISTINCT model FROM content_vectors ORDER BY model"),
    );
    // The invariant, not the arithmetic: whichever model got there first, the
    // database holds exactly ONE vector space afterwards.
    //
    // MEASURED DISCRIMINATION (vn4rz.7 pass C): with pg_advisory_xact_lock
    // deleted from insertEmbeddingsBatch, this test goes RED in 1 of 3 runs at
    // 24 writers (it went red 0 of 3 at 8). It is therefore a real stress test
    // of the production path but a FLAKY red-prover — the deterministic proof
    // that the lock is what closes the race is the barriered racePair() pair
    // above, which goes red 3 of 3.
    expect(rows.length).toBe(1);
    expect(rejected.length).toBeGreaterThan(0);
    for (const r of rejected) {
      expect((r as PromiseRejectedResult).reason).toBeInstanceOf(PgVecWriteModelMismatchError);
    }
  });

  // =========================================================================
  // upsertEntityNode / insertMemoryEvolution — through the REAL functions
  // (vn4rz.7 pass C, Task 3.2: these were only ever exercised as raw SQL, so a
  // parameter-ordering bug in either would have been invisible.)
  // =========================================================================

  it("upsertEntityNode maps every field to its own column and accumulates by mentionDelta", async () => {
    // Distinct sentinel per field: a swapped parameter shows up as a swapped
    // column, which a same-shaped-value test could never detect.
    await upsertEntityNode({
      entityId: "ent_1", entityType: "TYPE_SENTINEL", name: "NAME_SENTINEL",
      description: "DESC_SENTINEL", canonicalId: "CANON_SENTINEL", mentionDelta: 3,
    });
    const read = () =>
      withSchema(c =>
        c.query<{ entity_type: string; name: string; description: string;
                  canonical_id: string; mention_count: string }>(
          `SELECT entity_type, name, description, canonical_id, mention_count::text
             FROM entity_nodes WHERE entity_id = 'ent_1'`,
        ),
      );
    expect((await read()).rows[0]).toEqual({
      entity_type: "TYPE_SENTINEL", name: "NAME_SENTINEL", description: "DESC_SENTINEL",
      canonical_id: "CANON_SENTINEL", mention_count: "3",
    });

    // Second sighting: mention_count accumulates by the SAME delta parameter
    // used in the VALUES list ($6 is referenced twice — an ordering slip there
    // would increment by the wrong value).
    await upsertEntityNode({ entityId: "ent_1", mentionDelta: 4 });
    const again = (await read()).rows[0]!;
    expect(again.mention_count).toBe("7");
    // COALESCE semantics: a sighting that carries no name must not erase one.
    expect(again.name).toBe("NAME_SENTINEL");
    expect(again.description).toBe("DESC_SENTINEL");
  });

  it("upsertEntityNode REFUSES a null entity_id instead of writing a nameless row", async () => {
    await expect(
      upsertEntityNode({ entityId: null as unknown as string, name: "orphan" }),
    ).rejects.toThrow(/entity_id/);
    const { rows } = await withSchema(c =>
      c.query("SELECT 1 FROM entity_nodes WHERE name = 'orphan'"),
    );
    expect(rows).toEqual([]);
  });

  it("insertMemoryEvolution maps previous_* and new_* to the right columns and returns the new id", async () => {
    await seedContent("h_me2", "# me2");
    const docId = await upsertDocument({
      collection: "c_me", path: "me2.md", title: "me2", hash: "h_me2", body: "# me2",
    });
    const id = await insertMemoryEvolution({
      memoryId: docId, triggeredBy: docId, version: 7,
      previousKeywords: "PREV_KW", newKeywords: "NEW_KW",
      previousContext: "PREV_CTX", newContext: "NEW_CTX", reasoning: "REASON",
    });
    expect(Number.isInteger(id)).toBe(true);
    const { rows } = await withSchema(c =>
      c.query(
        `SELECT version, previous_keywords, new_keywords, previous_context, new_context, reasoning
           FROM memory_evolution WHERE id = $1`,
        [id],
      ),
    );
    // previous_* and new_* are adjacent same-typed parameters — precisely the
    // pair a re-order would swap without any error.
    expect(rows[0]).toEqual({
      version: 7, previous_keywords: "PREV_KW", new_keywords: "NEW_KW",
      previous_context: "PREV_CTX", new_context: "NEW_CTX", reasoning: "REASON",
    });
  });

  it("insertMemoryEvolution REFUSES a memory_id that references no document", async () => {
    const before = await withSchema(c =>
      c.query<{ n: string }>("SELECT count(*)::text n FROM memory_evolution"),
    );
    await expect(
      insertMemoryEvolution({ memoryId: 2147483000, triggeredBy: 2147483000, reasoning: "dangling" }),
    ).rejects.toThrow(/memory_evolution|foreign key/i);
    const after = await withSchema(c =>
      c.query<{ n: string }>("SELECT count(*)::text n FROM memory_evolution"),
    );
    expect(after.rows[0]!.n).toBe(before.rows[0]!.n);
  });
});
