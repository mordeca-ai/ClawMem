/**
 * PostgreSQL HYBRID (RRF fusion) READ-PATH integration tests
 * (master-harness-2wx75 slice 5).
 *
 * Runs against the LIVE clawmem-pg cluster (127.0.0.1:5433) in a throwaway
 * schema, following tests/integration/pg-search-fts.test.ts and
 * tests/integration/pg-search-vec.test.ts exactly — same setPgSchema knob, same
 * skip-when-CLAWMEM_PG_URL-is-unset, same obligation to prove the instrument
 * can go RED.
 *
 * WHY THIS TIER HAD TO EXIST. tests/unit/pg-search-hybrid.test.ts fuses two
 * FAKE arm outputs. It proves the fusion policy and cannot prove the thing this
 * slice is actually about: that the two arms, driven from ONE call against ONE
 * real client, both really run, really return the orderings the fusion assumes,
 * and key on the SAME document identity. The two arms compute `filepath`
 * independently (`clawmem://<collection>/<path>`, in two different modules) and
 * RRF fuses on that string — if they ever disagreed by a character, every
 * document would fuse as two distinct documents and the unit tier, which builds
 * both sides from one helper, would stay green. The BOTH-ARMS case below is
 * that assertion.
 *
 * THE EXPERIMENT. Three documents, engineered so the fused order is NOT either
 * arm's order:
 *
 *   lexical-only.md : "zebrafish" in the TITLE (weight A ⇒ FTS rank 0), and NO
 *                     content_vectors row at all ⇒ absent from the vec arm.
 *   vector-only.md  : the word "zebrafish" appears nowhere ⇒ absent from the
 *                     FTS arm; vector at 0° ⇒ vec rank 0.
 *   both.md         : "zebrafish" in the BODY only (weight D ⇒ FTS rank 1) AND
 *                     a vector at 30° ⇒ vec rank 1.
 *
 * Each arm ranks `both.md` SECOND, so neither arm alone would surface it first.
 * Two rank-1 contributions out-score one rank-0 contribution under RRF, so the
 * fused answer must put `both.md` FIRST. That inversion is the proof fusion
 * happened, rather than one arm passing through.
 *
 * WHAT IS DELIBERATELY NOT ASSERTED HERE: index usage (three documents cost a
 * seq scan and an EXPLAIN assertion would test the planner, not this module),
 * p95 latency, and the yoshiee-DOWN embed leg. Those are separate probes.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool, toVectorLiteral } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { pgSearchVec, type PgVecEmbedder } from "../../src/pg/search.ts";
import { pgSearchFts } from "../../src/pg/search-fts.ts";
import { pgSearchHybrid, pgSearchHybridDetailed } from "../../src/pg/search-hybrid.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 8;
const VAULT_MODEL = "embeddinggemma";

const d = URL_ ? describe : describe.skip;

/** Unit vector at angle `deg` in the (x, y) plane, padded to DIM. */
function atAngle(deg: number): number[] {
  const t = (deg * Math.PI) / 180;
  const v = new Array(DIM).fill(0);
  v[0] = Math.cos(t);
  v[1] = Math.sin(t);
  return v;
}

/** The query vector: 0°. Cosine distance to atAngle(t) is 1 - cos t. */
const embedder: PgVecEmbedder = {
  async embed() { return { embedding: atAngle(0), model: VAULT_MODEL }; },
};

type Fixture = {
  path: string;
  collection: string;
  title: string;
  body: string;
  /** undefined ⇒ NO content_vectors row: the document is invisible to the vec arm. */
  angleDeg?: number;
};

const FIXTURES: Fixture[] = [
  { path: "lexical-only.md", collection: "research", title: "Zebrafish ledger",
    body: "Counted twice before dawn, with no other distinguishing terms." },
  { path: "vector-only.md", collection: "research", title: "Quiet notes",
    body: "Nothing here names the fish at all.", angleDeg: 0 },
  { path: "both.md", collection: "research", title: "Unremarkable log",
    body: "The zebrafish larvae were tallied and then set aside.", angleDeg: 30 },
  // A collection with a lexical match but NOTHING embedded: the live
  // no-stored-vectors degrade for the vec arm.
  { path: "unembedded.md", collection: "drafts", title: "Zebrafish draft",
    body: "A zebrafish draft nobody has embedded yet." },
];

const P = (p: string, c = "research") => `${c}/${p}`;
const pathsOf = (rs: { displayPath: string }[]) => rs.map(r => r.displayPath);

d("PG hybrid (RRF) read path", () => {
  let pool: pg.Pool;
  let schema: string;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: URL_ });
    schema = `clawmem_hybridtest_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const c = await pool.connect();
    try {
      await c.query(`CREATE SCHEMA ${schema}`);
      await c.query(`SET search_path TO ${schema}, public`);
      for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
        await c.query(substituteMigrationParams(readFileSync(join(MIGRATIONS_DIR, f), "utf-8"), schema, DIM));
      }
      // Fixtures by direct SQL: setup, not the code under test. `content` MUST
      // land first — the FTS trigger reads the body out of it by hash.
      for (const [i, fx] of FIXTURES.entries()) {
        const hash = String(i).repeat(64).slice(0, 64);
        await c.query(`INSERT INTO content (hash, doc) VALUES ($1, $2)`, [hash, fx.body]);
        await c.query(
          `INSERT INTO documents (collection, path, title, hash, active) VALUES ($1, $2, $3, $4, true)`,
          [fx.collection, fx.path, fx.title, hash],
        );
        if (fx.angleDeg !== undefined) {
          await c.query(
            `INSERT INTO content_vectors (hash, seq, pos, model, embedding)
             VALUES ($1, 0, 0, $2, $3::vector)`,
            [hash, VAULT_MODEL, toVectorLiteral(atAngle(fx.angleDeg))],
          );
        }
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

  async function withSchema<T>(fn: (c: pg.PoolClient) => Promise<T>): Promise<T> {
    const c = await pool.connect();
    try {
      await c.query(`SET search_path TO ${schema}, public`);
      return await fn(c);
    } finally {
      c.release();
    }
  }

  const hybrid = (query: string, opts: Record<string, unknown> = {}) =>
    withSchema(c => pgSearchHybridDetailed(c, query, { collections: "research", embedder, ...opts }));

  // =========================================================================
  // CONTROLS: each arm's own live ordering, so a red fusion case is attributable
  // =========================================================================

  it("CONTROL: the FTS arm alone ranks lexical-only.md FIRST and both.md SECOND", async () => {
    const out = await withSchema(c => pgSearchFts(c, "zebrafish", { collections: "research" }));
    expect(pathsOf(out)).toEqual([P("lexical-only.md"), P("both.md")]);
  });

  it("CONTROL: the vec arm alone ranks vector-only.md FIRST and both.md SECOND", async () => {
    const out = await withSchema(c => pgSearchVec(c, "zebrafish", { collections: "research", embedder }));
    expect(pathsOf(out)).toEqual([P("vector-only.md"), P("both.md")]);
  });

  // =========================================================================
  // THE LOAD-BEARING CASE
  // =========================================================================

  it("FUSES both live arms and promotes the document NEITHER arm ranked first", async () => {
    const out = await hybrid("zebrafish");

    expect(out.arms).toBe("vec+fts");
    expect(out.degraded).toBe(false);
    expect(out.armFailures).toEqual([]);

    // The union of the two arms — three distinct documents. If the two modules
    // ever computed `filepath` differently this would be FOUR (both.md twice).
    expect(pathsOf(out.results).sort())
      .toEqual([P("both.md"), P("lexical-only.md"), P("vector-only.md")].sort());

    // THE INVERSION. Rank 1 in both arms beats rank 0 in one.
    expect(pathsOf(out.results)[0]).toBe(P("both.md"));

    // …and that is not either arm's own order, which the CONTROLs above pinned.
    expect(pathsOf(out.results)[0]).not.toBe(P("lexical-only.md"));
    expect(pathsOf(out.results)[0]).not.toBe(P("vector-only.md"));

    // The score is the FUSED scale, not a live ts_rank_cd or 1 - distance. Both
    // arms give both.md a value well under 0.04 at these fixtures; two rank-1
    // RRF contributions give ~0.0723.
    expect(out.results[0]!.score).toBeCloseTo(2 * (1 / (out.rrfK + 2) + 0.02), 6);
  });

  it("respects the collection scope in BOTH arms at once", async () => {
    // drafts/unembedded.md matches "zebrafish" lexically. A research-scoped
    // hybrid must not surface it — and it must not appear via either leg.
    const out = await hybrid("zebrafish");
    expect(pathsOf(out.results)).not.toContain(P("unembedded.md", "drafts"));
  });

  it("caps the fused union at `limit`", async () => {
    const out = await hybrid("zebrafish", { limit: 2 });
    expect(out.results).toHaveLength(2);
    expect(pathsOf(out.results)[0]).toBe(P("both.md"));
  });

  // =========================================================================
  // DEGRADED ARMS, LIVE
  // =========================================================================

  it("LIVE no-stored-vectors ⇒ fts-only, and it does NOT claim to be hybrid", async () => {
    // drafts/ has a document and zero embedded vectors — the vec arm's model
    // fence finds nothing in scope and degrades. The FTS arm is fine.
    const out = await hybrid("zebrafish", { collections: "drafts" });
    expect(out.arms).toBe("fts-only");
    expect(out.degraded).toBe(true);
    expect(out.armFailures).toEqual([{ arm: "vec", kind: "degraded", reason: "no-stored-vectors" }]);
    expect(pathsOf(out.results)).toEqual([P("unembedded.md", "drafts")]);
  });

  it("LIVE empty-tsquery ⇒ vec-only, and it does NOT claim to be hybrid", async () => {
    // An all-stopword query really does normalize to zero lexemes in
    // PostgreSQL's english config, so the FTS arm degrades while the vec arm
    // (which embeds whatever it is handed) answers normally.
    const out = await hybrid("the and of");
    expect(out.arms).toBe("vec-only");
    expect(out.degraded).toBe(true);
    expect(out.armFailures).toEqual([{ arm: "fts", kind: "degraded", reason: "empty-tsquery" }]);
    // The vec arm's own live order, verbatim.
    expect(pathsOf(out.results)).toEqual([P("vector-only.md"), P("both.md")]);
  });

  it("LIVE both arms degraded ⇒ arms: none, empty, and BOTH causes", async () => {
    const out = await hybrid("the and of", { collections: "drafts" });
    expect(out.arms).toBe("none");
    expect(out.degraded).toBe(true);
    expect(out.results).toEqual([]);
    expect(out.armFailures.map(f => f.arm)).toEqual(["vec", "fts"]);
  });

  it("LIVE genuine-empty ⇒ still two-arm and NOT degraded", async () => {
    // A real term that matches no document. Both arms ran; both found nothing.
    // This must stay distinguishable from the degraded cases above.
    const out = await hybrid("pomegranate");
    expect(out.arms).toBe("vec+fts");
    expect(out.degraded).toBe(false);
    // The vec arm answers by distance, so it returns rows for any query; the
    // trustworthy part is that NOTHING was degraded.
    expect(out.fts!.results).toEqual([]);
    expect(out.fts!.degraded).toBe(false);
  });

  it("the bare wrapper returns the same live fused ordering", async () => {
    const bare = await withSchema(c =>
      pgSearchHybrid(c, "zebrafish", { collections: "research", embedder }));
    expect(pathsOf(bare)[0]).toBe(P("both.md"));
  });
});
