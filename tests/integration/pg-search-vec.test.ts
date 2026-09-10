/**
 * PostgreSQL VECTOR READ-PATH integration tests
 * (master-harness-2wx75 slice 2, GAP 2 + GAP 4).
 *
 * These run against the LIVE clawmem-pg cluster (127.0.0.1:5433), inside a
 * throwaway schema per run, following tests/integration/pg-write-path.test.ts
 * exactly — same setPgSchema knob, same skip-when-CLAWMEM_PG_URL-is-unset, same
 * central obligation: prove the instrument can go RED.
 *
 * WHY THIS TIER HAD TO EXIST. tests/unit/pg-search-vec.test.ts proves the
 * predicates are IN the SQL. It cannot prove PostgreSQL honours them — its
 * "inactive document excluded" case asserts that the string `d.active = true`
 * appears in the emitted query, which would still pass if the predicate did
 * nothing. Everything below is BEHAVIOURAL: rows go in, a search runs, and the
 * assertion is about which documents came back and in what order. No case here
 * string-matches SQL.
 *
 * THE EMBEDDER IS INJECTED, THE DATABASE IS NOT. `pgSearchVec` takes an optional
 * `embedder`; these tests pass a deterministic one so the query vector is a
 * KNOWN vector and distance ORDER is assertable. Depending on the remote GPU
 * embed endpoint instead would make this suite a report on that endpoint's
 * uptime. The database — the thing whose behaviour is actually in question — is
 * the real one.
 *
 * GEOMETRY. EMBED_DIM is overridden to 8 for the fixture schema and every
 * vector is [cos t, sin t, 0...] against a query of [1, 0, 0...], so cosine
 * distance is exactly 1 - cos t and the expected ORDER is arithmetic, not
 * folklore.
 *
 * NOT ASSERTED HERE, deliberately: anything touching `documents.fts`. The FTS
 * trigger hard-qualifies public.content and is not schema-portable, so inside a
 * throwaway schema the column comes back title-only. That is master-harness-
 * vn4rz.46, filed separately, and it is why pg-write-path.test.ts has one
 * pre-existing failure. This suite's fixtures are seeded by direct SQL rather
 * than through src/pg/write.ts — seeding is fixture setup, not the code under
 * test; the code under test is src/pg/search.ts and it is called for real.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool, toVectorLiteral } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { pgSearchVec, type PgVecEmbedder } from "../../src/pg/search.ts";
import { PgVecReadModelMismatchError, PgVecSearchTimeoutError } from "../../src/pg/errors.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 8;
const VAULT_MODEL = "embeddinggemma";
const FOREIGN_MODEL = "nomic-embed-text";

const d = URL_ ? describe : describe.skip;

/** Unit vector at angle `deg` in the (x, y) plane, padded to DIM. */
function atAngle(deg: number): number[] {
  const t = (deg * Math.PI) / 180;
  const v = new Array(DIM).fill(0);
  v[0] = Math.cos(t);
  v[1] = Math.sin(t);
  return v;
}

/** The query vector: 0 degrees. Cosine distance to atAngle(t) is 1 - cos t. */
const QUERY_VEC = atAngle(0);

function embedderFor(model: string, vec: number[] = QUERY_VEC): PgVecEmbedder {
  return { async embed() { return { embedding: vec, model }; } };
}

/** One fixture document + its single fragment vector. */
type Fixture = {
  path: string;
  collection: string;
  angleDeg: number;
  active?: boolean;
  model?: string;
};

const FIXTURES: Fixture[] = [
  // research/ — the ordering ladder. 0 < 60 < 90 degrees => distance 0 < 0.5 < 1.
  { path: "near.md", collection: "research", angleDeg: 0 },
  { path: "mid.md", collection: "research", angleDeg: 60 },
  { path: "far.md", collection: "research", angleDeg: 90 },
  // The fence's victim: NEAREST of all, and inactive. If `d.active = true` did
  // nothing this would out-rank every row above and every ordering assertion
  // below would name it first.
  { path: "retired.md", collection: "research", angleDeg: 0, active: false },
  // A second collection, closer than research/mid.md, so a collection filter
  // that failed to narrow would visibly pull this into a research-only search.
  { path: "elsewhere.md", collection: "decisions", angleDeg: 30 },
];

d("PG vector read path", () => {
  let pool: pg.Pool;
  let schema: string;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: URL_ });
    schema = `clawmem_rtest_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const c = await pool.connect();
    try {
      await c.query(`CREATE SCHEMA ${schema}`);
      await c.query(`SET search_path TO ${schema}, public`);
      for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
        const sql = substituteMigrationParams(readFileSync(join(MIGRATIONS_DIR, f), "utf-8"), schema, DIM);
        await c.query(sql);
      }
      // Fixtures. Direct SQL: this is setup, not the code under test.
      for (const [i, fx] of FIXTURES.entries()) {
        const hash = String(i).repeat(64).slice(0, 64);
        await c.query(`INSERT INTO content (hash, doc) VALUES ($1, $2)`, [hash, `body of ${fx.path}`]);
        await c.query(
          `INSERT INTO documents (collection, path, title, hash, active)
           VALUES ($1, $2, $3, $4, $5)`,
          [fx.collection, fx.path, fx.path.replace(/\.md$/, ""), hash, fx.active ?? true],
        );
        await c.query(
          `INSERT INTO content_vectors (hash, seq, pos, model, embedding)
           VALUES ($1, 0, 0, $2, $3::vector)`,
          [hash, fx.model ?? VAULT_MODEL, toVectorLiteral(atAngle(fx.angleDeg))],
        );
      }
      // A collection with documents but NO vectors — "nothing embedded yet".
      const bare = "e".repeat(64);
      await c.query(`INSERT INTO content (hash, doc) VALUES ($1, $2)`, [bare, "unembedded"]);
      await c.query(
        `INSERT INTO documents (collection, path, title, hash) VALUES ('drafts', 'raw.md', 'raw', $1)`,
        [bare],
      );
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

  /** The real pgSearchVec, on a real client, in the throwaway schema. */
  async function search(opts: Parameters<typeof pgSearchVec>[2] = {}) {
    return withSchema(c => pgSearchVec(c, "anything", { embedder: embedderFor(VAULT_MODEL), ...opts }));
  }

  // =========================================================================
  // The instrument can go RED: the inactive row IS there and IS the nearest
  // =========================================================================

  it("CONTROL: without the active fence the inactive document would rank FIRST", async () => {
    // This is the case that makes every "retired.md is absent" assertion below
    // mean something. Same ANN order-by, same query vector, the ONLY difference
    // is the missing `d.active = true`. If it did not come back first here, the
    // absence of retired.md later would prove nothing about the predicate.
    const { rows } = await withSchema(c =>
      c.query<{ path: string; active: boolean }>(
        `SELECT d.path, d.active
           FROM content_vectors cv JOIN documents d ON d.hash = cv.hash
          WHERE d.collection = 'research'
          ORDER BY cv.embedding <=> $1::vector
          LIMIT 1`,
        [toVectorLiteral(QUERY_VEC)],
      ),
    );
    expect(rows[0]!.path).toBe("retired.md");
    expect(rows[0]!.active).toBe(false);
  });

  // =========================================================================
  // Behaviour
  // =========================================================================

  it("EXCLUDES an inactive document from the results", async () => {
    const out = await search({ collections: "research" });
    expect(out.map(r => r.displayPath)).not.toContain("research/retired.md");
    expect(out.map(r => r.displayPath)).toEqual([
      "research/near.md", "research/mid.md", "research/far.md",
    ]);
  });

  it("orders by ACTUAL vector distance, and the scores are 1 - cosine distance", async () => {
    const out = await search({ collections: "research" });
    expect(out.map(r => r.displayPath)).toEqual([
      "research/near.md", "research/mid.md", "research/far.md",
    ]);
    // cos 0 = 1, cos 60 = 0.5, cos 90 = 0  =>  distances 0, 0.5, 1
    expect(out[0]!.score).toBeCloseTo(1, 5);
    expect(out[1]!.score).toBeCloseTo(0.5, 5);
    expect(out[2]!.score).toBeCloseTo(0, 5);
    expect(out[0]!.score).toBeGreaterThan(out[1]!.score);
    expect(out[1]!.score).toBeGreaterThan(out[2]!.score);
  });

  it("NARROWS to the requested collection — and the excluded row is one that would have ranked", async () => {
    // decisions/elsewhere.md sits at 30 degrees: nearer than research/mid.md and
    // research/far.md, so a filter that did nothing would put it second overall.
    const all = await search();
    expect(all.map(r => r.displayPath)).toEqual([
      "research/near.md", "decisions/elsewhere.md", "research/mid.md", "research/far.md",
    ]);

    const narrowed = await search({ collections: "research" });
    expect(narrowed.map(r => r.displayPath)).not.toContain("decisions/elsewhere.md");

    const other = await search({ collections: ["decisions"] });
    expect(other.map(r => r.displayPath)).toEqual(["decisions/elsewhere.md"]);
  });

  it("honours the document limit", async () => {
    expect(await search({ limit: 2 }).then(o => o.map(r => r.displayPath)))
      .toEqual(["research/near.md", "decisions/elsewhere.md"]);
    expect(await search({ limit: 1 }).then(o => o.length)).toBe(1);
    // The control: unbounded, there are more than 2.
    expect((await search()).length).toBeGreaterThan(2);
  });

  it("returns EMPTY for a collection with documents but nothing embedded", async () => {
    expect(await search({ collections: "drafts" })).toEqual([]);
  });

  it("returns EMPTY for a collection that does not exist at all", async () => {
    expect(await search({ collections: "no-such-collection" })).toEqual([]);
  });

  // =========================================================================
  // The model-identity fence, end to end (GAP 1's live half)
  // =========================================================================

  it("REFUSES a search whose query model differs from the stored model, naming BOTH", async () => {
    let caught: unknown;
    try {
      await withSchema(c => pgSearchVec(c, "anything", {
        collections: "research",
        embedder: embedderFor(FOREIGN_MODEL),
      }));
    } catch (e) { caught = e; }
    expect(caught).toBeInstanceOf(PgVecReadModelMismatchError);
    const err = caught as PgVecReadModelMismatchError;
    expect(err.message).toContain(VAULT_MODEL);
    expect(err.message).toContain(FOREIGN_MODEL);
    expect(err.storedModels).toEqual([VAULT_MODEL]);
    expect(err.queryModel).toBe(FOREIGN_MODEL);
    // And the SAME call with the RIGHT model succeeds — the refusal is about the
    // model, not about anything else being broken.
    expect((await search({ collections: "research" })).length).toBe(3);
  });

  // =========================================================================
  // GAP 4: the statement_timeout actually fires, and is typed
  // =========================================================================

  it("surfaces a statement_timeout as PgVecSearchTimeoutError — proven by inducing one", async () => {
    // INDUCING IT FOR REAL: a second session takes ACCESS EXCLUSIVE on
    // content_vectors, which blocks the search's very first read. A lock wait is
    // exactly what statement_timeout is for, and it is deterministic — unlike
    // hoping an 8-row ANN scan overruns a millisecond.
    const locker = await pool.connect();
    let caught: unknown;
    try {
      await locker.query(`SET search_path TO ${schema}, public`);
      await locker.query("BEGIN");
      await locker.query("LOCK TABLE content_vectors IN ACCESS EXCLUSIVE MODE");

      const t0 = Date.now();
      try {
        await search({ collections: "research", statementTimeoutMs: 400 });
      } catch (e) { caught = e; }
      const elapsed = Date.now() - t0;

      expect(caught).toBeInstanceOf(PgVecSearchTimeoutError);
      const err = caught as PgVecSearchTimeoutError;
      expect(err.timeoutMs).toBe(400);
      expect(err.scope).toBe("research");
      // Not a raw driver throw, and NOT a silent empty result.
      expect(err.name).toBe("PgVecSearchTimeoutError");
      // It really was the server-side bound, not a client that gave up early or
      // one that hung: the wait is bounded and of the right order.
      expect(elapsed).toBeGreaterThanOrEqual(300);
      expect(elapsed).toBeLessThan(8000);
    } finally {
      await locker.query("ROLLBACK").catch(() => {});
      locker.release();
    }
    // RESTORE -> GREEN, in the same case: with the lock gone, the identical
    // call returns results. That is what proves the throw above came from the
    // induced condition and not from the timeout being permanently broken.
    const after = await search({ collections: "research", statementTimeoutMs: 400 });
    expect(after.map(r => r.displayPath)).toEqual([
      "research/near.md", "research/mid.md", "research/far.md",
    ]);
  }, 20_000);

  it("leaves NO session GUC behind on the pooled connection it used", async () => {
    // GAP 8. SET LOCAL unwinds with the transaction; a session SET would still
    // be visible on the same client afterwards. Asserted on the very client the
    // search ran on, which is the connection a later caller would inherit.
    await withSchema(async c => {
      await pgSearchVec(c, "anything", {
        collections: "research", embedder: embedderFor(VAULT_MODEL), statementTimeoutMs: 400,
      });
      const { rows } = await c.query<{ v: string }>(
        "SELECT current_setting('statement_timeout') AS v");
      expect(rows[0]!.v).toBe("0");
      const { rows: h } = await c.query<{ v: string }>(
        "SELECT current_setting('hnsw.iterative_scan') AS v");
      expect(h[0]!.v).toBe("off");
      // And the connection is usable — not stuck in an aborted transaction.
      const { rows: ok } = await c.query<{ n: number }>("SELECT 1 AS n");
      expect(ok[0]!.n).toBe(1);
    });
  });
});
