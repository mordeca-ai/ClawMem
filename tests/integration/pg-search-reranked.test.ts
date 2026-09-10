/**
 * PostgreSQL RERANKED READ-PATH integration tests (master-harness-2wx75 slice 6).
 *
 * Runs against the LIVE clawmem-pg cluster (127.0.0.1:5433) in a throwaway
 * schema, following tests/integration/pg-search-hybrid.test.ts exactly — same
 * `setPgSchema` knob, same skip-when-CLAWMEM_PG_URL-is-unset, same throwaway
 * schema, same obligation to prove the instrument can go RED.
 *
 * WHAT IS LIVE HERE AND WHAT IS NOT. The RETRIEVAL is live: two real arms, one
 * real client, one real fusion over real rows. The RERANKER IS STILL INJECTED
 * AND DETERMINISTIC, on purpose — src/pg/ never reads the
 * CLAWMEM_RERANK_URL / CLAWMEM_RERANK_API_KEY contract (ruling 5), so there is
 * nothing about a real cross-encoder that this module could get wrong; what
 * this tier can get wrong is the JOIN between the live fused rows and the
 * rerank stage. Specifically:
 *
 *  - the `file` key the reranker is handed is the live `filepath` the two arms
 *    computed independently, and the blend maps back onto THOSE rows. Off by a
 *    character and the unit tier — which builds both sides from one helper —
 *    stays green while every live document silently loses its rerank score.
 *  - the candidate TEXT is the live `body` column, not a fixture string. A
 *    read path that returned rows without `body` would make every live rerank
 *    a `"skipped-no-text"`; slice 5's arms do return it (src/pg/search.ts
 *    ~line 541) and this tier is what keeps that true.
 *  - the degrade-to-fusion contract has to hold over LIVE orderings, including
 *    the live degraded-arm cases.
 *
 * A live cross-encoder probe (yoshiee up / yoshiee down, real latency against
 * the 1800 ms hook clause) is a SEPARATE probe and deliberately not here: it
 * would test store.ts's rerank() and the GPU host, neither of which this
 * module contains.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool, toVectorLiteral } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { type PgVecEmbedder } from "../../src/pg/search.ts";
import { pgSearchHybridDetailed, type PgHybridArms } from "../../src/pg/search-hybrid.ts";
import {
  pgSearchReranked,
  pgSearchRerankedDetailed,
  PG_RERANK_MIN_BUDGET_MS,
  type PgReranker,
  type PgRerankStatus,
} from "../../src/pg/search-reranked.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 8;
const VAULT_MODEL = "embeddinggemma";
const SLACK = 400;

const d = URL_ ? describe : describe.skip;

function atAngle(deg: number): number[] {
  const t = (deg * Math.PI) / 180;
  const v = new Array(DIM).fill(0);
  v[0] = Math.cos(t);
  v[1] = Math.sin(t);
  return v;
}

const embedder: PgVecEmbedder = {
  async embed() { return { embedding: atAngle(0), model: VAULT_MODEL }; },
};

type Fixture = { path: string; collection: string; title: string; body: string; angleDeg?: number };

/**
 * The same three-document experiment slice 5 used (so the live FUSED order is
 * already pinned by that suite's CONTROLs), plus two documents this slice
 * needs: an unembedded one for the live degraded-arm case, and one with an
 * EMPTY body for the live "skipped-no-text" case.
 */
const FIXTURES: Fixture[] = [
  { path: "lexical-only.md", collection: "research", title: "Zebrafish ledger",
    body: "Counted twice before dawn, with no other distinguishing terms." },
  { path: "vector-only.md", collection: "research", title: "Quiet notes",
    body: "Nothing here names the fish at all.", angleDeg: 0 },
  { path: "both.md", collection: "research", title: "Unremarkable log",
    body: "The zebrafish larvae were tallied and then set aside.", angleDeg: 30 },
  { path: "unembedded.md", collection: "drafts", title: "Zebrafish draft",
    body: "A zebrafish draft nobody has embedded yet." },
  // Body is the empty string, and it IS embedded — so the vec arm returns the
  // row and the rerank stage has a candidate with no text to rank.
  { path: "bodyless.md", collection: "hollow", title: "Hollow", body: "", angleDeg: 10 },
];

const P = (p: string, c = "research") => `${c}/${p}`;
const pathsOf = (rs: { displayPath: string }[]) => rs.map(r => r.displayPath);

/** A recording reranker; `score` is applied to the documents it was handed. */
function recording(
  score: (doc: { file: string; text: string }, i: number, n: number) => number,
  behaviour: { throws?: Error; empty?: boolean } = {},
) {
  const calls: { documents: { file: string; text: string }[]; timeoutMs?: number }[] = [];
  const reranker: PgReranker = async (_q, documents, opts) => {
    calls.push({ documents, ...(opts?.timeoutMs === undefined ? {} : { timeoutMs: opts.timeoutMs }) });
    if (behaviour.throws) throw behaviour.throws;
    if (behaviour.empty) return [];
    return documents.map((doc, i) => ({ file: doc.file, score: score(doc, i, documents.length) }));
  };
  return { reranker, calls };
}

/** Scores in REVERSE of the order handed in — an unambiguous, visible reordering. */
const reversing = () => recording((_d, i, n) => (i + 1) / n);

d("PG reranked read path", () => {
  let pool: pg.Pool;
  let schema: string;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: URL_ });
    schema = `clawmem_reranktest_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const c = await pool.connect();
    try {
      await c.query(`CREATE SCHEMA ${schema}`);
      await c.query(`SET search_path TO ${schema}, public`);
      for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
        await c.query(substituteMigrationParams(readFileSync(join(MIGRATIONS_DIR, f), "utf-8"), schema, DIM));
      }
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

  const reranked = (query: string, opts: Record<string, unknown> = {}) =>
    withSchema(c => pgSearchRerankedDetailed(c, query, { collections: "research", embedder, ...opts }));

  // =========================================================================
  // CONTROL: the live fused order this slice is re-ordering
  // =========================================================================

  it("CONTROL: the live hybrid fuses to both.md, lexical-only.md, vector-only.md", async () => {
    const out = await withSchema(c =>
      pgSearchHybridDetailed(c, "zebrafish", { collections: "research", embedder }));
    expect(out.arms).toBe("vec+fts");
    expect(pathsOf(out.results)[0]).toBe(P("both.md"));
    expect(pathsOf(out.results)).toHaveLength(3);
  });

  // =========================================================================
  // THE LOAD-BEARING CASE
  // =========================================================================

  it("RERANKS the LIVE fused list, and the reranker sees live filepaths and live bodies", async () => {
    const rr2 = reversing();
    const out2 = await reranked("zebrafish", { reranker: rr2.reranker });
    expect(out2.rerank).toBe("applied");
    expect(rr2.calls).toHaveLength(1);

    // THE JOIN. The keys handed to the reranker are the arms' own live
    // `filepath` values — and the blend maps back onto exactly those rows, so
    // no document is lost and none is duplicated.
    const handed = rr2.calls[0]!.documents.map(doc => doc.file);
    expect(handed).toEqual(out2.hybrid.results.map(r => r.filepath));
    expect(new Set(handed).size).toBe(handed.length);
    expect(handed.every(f => f.startsWith("clawmem://research/"))).toBe(true);

    // LIVE BODY TEXT, not a fixture string: the read path really does carry
    // `body`, which is what makes reranking possible at all.
    const texts = rr2.calls[0]!.documents.map(doc => doc.text);
    expect(texts.every(t => t.length > 0)).toBe(true);
    expect(texts.some(t => t.includes("zebrafish"))).toBe(true);

    // …and the rerank actually moved the live answer: the reranker scored the
    // LAST fused document highest, so the final order is the fusion reversed.
    expect(pathsOf(out2.results)).toEqual([...pathsOf(out2.hybrid.results)].reverse());
    expect(pathsOf(out2.results)[0]).not.toBe(P("both.md"));
  });

  it("does not drop or duplicate any live document, and respects the collection scope", async () => {
    const out = await reranked("zebrafish", { reranker: reversing().reranker });
    expect(pathsOf(out.results).sort())
      .toEqual([P("both.md"), P("lexical-only.md"), P("vector-only.md")].sort());
    expect(pathsOf(out.results)).not.toContain(P("unembedded.md", "drafts"));
  });

  // =========================================================================
  // THE STATUS MATRIX, LIVE
  // =========================================================================

  it("EVERY non-'applied' status returns the LIVE fused order byte-identically", async () => {
    const cases: { status: Exclude<PgRerankStatus, "applied">; query: string; opts: Record<string, unknown> }[] = [
      { status: "skipped-no-reranker", query: "zebrafish", opts: {} },
      // Live empty-body document: the vec arm returns it, and there is nothing
      // to rank.
      { status: "skipped-no-text", query: "zebrafish",
        opts: { collections: "hollow", reranker: reversing().reranker } },
      { status: "skipped-budget", query: "zebrafish",
        opts: { reranker: reversing().reranker, deadlineMs: PG_RERANK_MIN_BUDGET_MS - 1 } },
      { status: "degenerate", query: "zebrafish",
        opts: { reranker: recording(() => 1e-11).reranker } },
      { status: "failed", query: "zebrafish",
        opts: { reranker: recording(() => 1, { throws: new Error("live rerank refused") }).reranker } },
    ];
    for (const { status, query, opts } of cases) {
      const out = await reranked(query, opts);
      expect(out.rerank).toBe(status);
      expect(out.results.map(r => r.filepath)).toEqual(out.hybrid.results.map(r => r.filepath));
      expect(out.results).toBe(out.hybrid.results);
    }
    expect(cases.map(c => c.status).sort()).toEqual(
      ["degenerate", "failed", "skipped-budget", "skipped-no-reranker", "skipped-no-text"],
    );
  });

  it("LIVE skipped-no-text: an empty-body candidate is NOT handed to the reranker", async () => {
    const rr = reversing();
    const out = await reranked("zebrafish", { collections: "hollow", reranker: rr.reranker });
    expect(out.rerank).toBe("skipped-no-text");
    expect(out.hybrid.results.length).toBeGreaterThan(0);   // the row IS there
    expect(out.hybrid.results[0]!.displayPath).toBe(P("bodyless.md", "hollow"));
    expect(rr.calls).toHaveLength(0);                       // the COUNT, not just the status
  });

  it("LIVE skipped-budget: the reranker's call count is ZERO and the call stays inside its budget", async () => {
    const rr = reversing();
    const deadlineMs = PG_RERANK_MIN_BUDGET_MS - 1;
    const out = await reranked("zebrafish", { reranker: rr.reranker, deadlineMs });
    expect(out.rerank).toBe("skipped-budget");
    expect(rr.calls).toHaveLength(0);
    expect(out.candidateCount).toBe(3);
    expect(out.timings.totalMs).toBeLessThanOrEqual(deadlineMs + SLACK);
  });

  it("LIVE budget DECREMENTS: the reranker receives less than the full deadline", async () => {
    const rr = reversing();
    const deadlineMs = 5000;
    const out = await reranked("zebrafish", { reranker: rr.reranker, deadlineMs });
    expect(out.rerank).toBe("applied");
    const handed = rr.calls[0]!.timeoutMs!;
    expect(handed).toBeLessThan(deadlineMs);
    expect(handed).toBeLessThanOrEqual(deadlineMs - out.timings.hybridMs + 1);
    // The live two-arm retrieval genuinely costs something, so the remainder is
    // measurably smaller than the deadline — this is the additive budget the
    // parity run needs to READ.
    expect(out.timings.hybridMs).toBeGreaterThan(0);
  });

  // =========================================================================
  // COMPOSITION, LIVE
  // =========================================================================

  it("re-asserts slice 5's invariant over the EMBEDDED live hybrid, across live arm states", async () => {
    const cases: { opts: Record<string, unknown>; query: string; arms: PgHybridArms }[] = [
      { query: "zebrafish", opts: {}, arms: "vec+fts" },                       // both live arms
      { query: "pomegranate", opts: {}, arms: "vec+fts" },                     // genuine-empty fts
      { query: "zebrafish", opts: { collections: "drafts" }, arms: "fts-only" }, // live no-stored-vectors
      { query: "the and of", opts: {}, arms: "vec-only" },                     // live empty-tsquery
      { query: "the and of", opts: { collections: "drafts" }, arms: "none" },   // both degraded
    ];
    for (const { query, opts, arms } of cases) {
      const out = await reranked(query, { reranker: reversing().reranker, ...opts });
      expect(out.hybrid.arms).toBe(arms);
      expect(out.hybrid.degraded).toBe(arms !== "vec+fts");
      expect(out.hybrid.armFailures.length === 0).toBe(arms === "vec+fts");
    }
  });

  it("a LIVE degraded single arm is still reranked (the two axes are orthogonal)", async () => {
    const out = await reranked("the and of", { reranker: reversing().reranker });
    expect(out.hybrid.arms).toBe("vec-only");
    expect(out.hybrid.degraded).toBe(true);
    expect(out.rerank).toBe("applied");
    // The vec arm's live order, reversed by the reranker.
    expect(pathsOf(out.results)).toEqual([...pathsOf(out.hybrid.results)].reverse());
  });

  it("the bare wrapper returns the same live reranked ordering", async () => {
    const bare = await withSchema(c => pgSearchReranked(c, "zebrafish", {
      collections: "research", embedder, reranker: reversing().reranker,
    }));
    const detailed = await reranked("zebrafish", { reranker: reversing().reranker });
    expect(pathsOf(bare)).toEqual(pathsOf(detailed.results));
  });
});
