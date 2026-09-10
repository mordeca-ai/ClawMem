/**
 * PostgreSQL LEXICAL (FTS) READ-PATH integration tests
 * (master-harness-2wx75 slice 4).
 *
 * These run against the LIVE clawmem-pg cluster (127.0.0.1:5433), inside a
 * throwaway schema per run, following tests/integration/pg-search-vec.test.ts
 * exactly — same setPgSchema knob, same skip-when-CLAWMEM_PG_URL-is-unset, same
 * central obligation: prove the instrument can go RED.
 *
 * WHY THIS TIER HAD TO EXIST. tests/unit/pg-search-fts.test.ts proves the
 * predicates and the weight array are IN the SQL. It cannot prove PostgreSQL
 * honours them, and — critically — it cannot prove the stored tsvector contains
 * BODY terms at all. Everything below is BEHAVIOURAL: rows go in, a search
 * runs, and the assertion is about which documents came back and in what order.
 * No case here string-matches SQL.
 *
 * THIS SUITE IS ALSO THE REGRESSION GUARD FOR master-harness-vn4rz.46.
 * Before migration 007, the FTS trigger hard-qualified `public.content`, so in
 * a THROWAWAY SCHEMA (exactly this setup) the weight-D body branch resolved to
 * nothing and the stored tsvector was silently TITLE-ONLY — present,
 * well-formed, and missing every body term. The "body-term search" case below
 * is therefore doing two jobs at once: it is this arm's core acceptance AND the
 * standing detector for that defect returning. If 007 is reverted, that case
 * goes red first.
 *
 * WHAT IS DELIBERATELY NOT ASSERTED: that `documents_fts_idx` (GIN) is actually
 * USED. At five-document scale PostgreSQL correctly costs a sequential scan
 * cheaper than an index scan, so an EXPLAIN-based assertion here would be a
 * test of the planner's cost model rather than of this module's SQL. The unit
 * tier asserts the SHAPE that keeps the index available (`@@` against the bare
 * column); index usage at real scale is a separate probe.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { pgSearchFts, pgSearchFtsDetailed } from "../../src/pg/search-fts.ts";
import { PgFtsSearchTimeoutError } from "../../src/pg/errors.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 8;

const d = URL_ ? describe : describe.skip;

type Fixture = {
  path: string;
  collection: string;
  title: string;
  body: string;
  active?: boolean;
  invalidated?: boolean;
};

/**
 * THE FIXTURES ARE THE EXPERIMENT. Each one exists to make exactly one
 * assertion below able to fail:
 *
 *  - body-only.md      : "zebrafish" appears ONLY in the body. Finding it is
 *                        the arm's core acceptance and the vn4rz.46 guard.
 *  - weight-title.md   : "marmalade" in the TITLE (weight A), body unrelated.
 *  - weight-body.md    : "marmalade" ONLY in the body (weight D). The pair is
 *                        the WEIGHT PROOF: title must out-rank body 10:1.
 *  - phrase-adjacent.md: "brown fox" adjacent.
 *  - phrase-split.md   : "brown" and "fox" present but NOT adjacent. The pair
 *                        proves websearch_to_tsquery really parses a quoted
 *                        phrase rather than being bypassed.
 *  - elsewhere.md      : the same "marmalade" term in ANOTHER collection, so a
 *                        collection filter that failed to narrow is visible.
 *  - retired.md        : "zebrafish", inactive.  } the fences' victims — both
 *  - invalid.md        : "zebrafish", invalidated.} would otherwise be returned.
 */
const FIXTURES: Fixture[] = [
  { path: "body-only.md", collection: "research", title: "Quiet ledger",
    body: "The zebrafish larvae were counted twice before dawn." },
  { path: "weight-title.md", collection: "research", title: "Marmalade chronicle",
    body: "Nothing in this body mentions the fruit preserve at all." },
  { path: "weight-body.md", collection: "research", title: "Unremarkable notes",
    body: "He spread marmalade on the toast and said nothing further." },
  { path: "phrase-adjacent.md", collection: "research", title: "Adjacent",
    body: "The quick brown fox jumps over the lazy dog." },
  { path: "phrase-split.md", collection: "research", title: "Split",
    body: "The fox was brown, elderly, and disinclined to jump." },
  { path: "elsewhere.md", collection: "decisions", title: "Marmalade elsewhere",
    body: "Another collection entirely, also about marmalade." },
  { path: "retired.md", collection: "research", title: "Retired zebrafish",
    body: "The zebrafish tally, superseded.", active: false },
  { path: "invalid.md", collection: "research", title: "Invalidated zebrafish",
    body: "The zebrafish tally, invalidated.", invalidated: true },
];

d("PG lexical (FTS) read path", () => {
  let pool: pg.Pool;
  let schema: string;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: URL_ });
    schema = `clawmem_ftstest_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    const c = await pool.connect();
    try {
      await c.query(`CREATE SCHEMA ${schema}`);
      await c.query(`SET search_path TO ${schema}, public`);
      for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
        const sql = substituteMigrationParams(readFileSync(join(MIGRATIONS_DIR, f), "utf-8"), schema, DIM);
        await c.query(sql);
      }
      // Fixtures. Direct SQL: this is setup, not the code under test. `content`
      // MUST be inserted first — the FTS trigger reads the body out of it by
      // hash when the document row lands.
      for (const [i, fx] of FIXTURES.entries()) {
        const hash = String(i).repeat(64).slice(0, 64);
        await c.query(`INSERT INTO content (hash, doc) VALUES ($1, $2)`, [hash, fx.body]);
        await c.query(
          `INSERT INTO documents (collection, path, title, hash, active, invalidated_at)
           VALUES ($1, $2, $3, $4, $5, $6)`,
          [fx.collection, fx.path, fx.title, hash, fx.active ?? true,
           fx.invalidated ? new Date() : null],
        );
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

  /** The real pgSearchFts, on a real client, in the throwaway schema. */
  async function search(query: string, opts: Parameters<typeof pgSearchFts>[2] = {}) {
    return withSchema(c => pgSearchFts(c, query, opts));
  }
  async function searchDetailed(query: string, opts: Parameters<typeof pgSearchFtsDetailed>[2] = {}) {
    return withSchema(c => pgSearchFtsDetailed(c, query, opts));
  }

  // =========================================================================
  // The instrument can go RED: the stored tsvector really does carry the body
  // =========================================================================

  it("CONTROL: the stored tsvector carries BODY lexemes at weight D (the vn4rz.46 guard)", async () => {
    // Read the column directly. Before migration 007 this came back title-only
    // in a non-public schema, with NO error — so this is the assertion that
    // tells a future reader whether a red "body-term search" case below is a
    // bug in src/pg/search-fts.ts or a regression in the trigger.
    const { rows } = await withSchema(c =>
      c.query<{ fts: string }>(
        `SELECT fts::text AS fts FROM documents WHERE path = 'body-only.md'`),
    );
    //
    // MEASURED SURPRISE, recorded rather than worked around: tsvector's text
    // output prints NO weight letter for weight D, because D is the default and
    // setweight(..., 'D') leaves no marker. So the body lexeme reads
    // `'zebrafish':4` while the title reads `'ledger':2A`. The absence of a
    // letter IS weight D — asserting /:\d+D/ would be asserting a rendering
    // PostgreSQL does not produce. What matters behaviourally is the RANK
    // separation, and that is the WEIGHT PROOF case below.
    expect(rows[0]!.fts).toContain("zebrafish"); // stems to 'zebrafish'
    expect(rows[0]!.fts).toMatch(/'zebrafish':\d+(?![A-C])/); // body: no A/B/C letter
    expect(rows[0]!.fts).toMatch(/'ledger':\d+A/);            // title: weight A
  });

  it("CONTROL: without the fences the inactive AND invalidated rows WOULD match", async () => {
    // Makes every "retired.md / invalid.md is absent" assertion below mean
    // something: same match predicate, the ONLY difference is the missing
    // active/invalidated fence.
    const { rows } = await withSchema(c =>
      c.query<{ path: string }>(
        `SELECT d.path FROM documents d
          WHERE d.fts @@ websearch_to_tsquery('english', 'zebrafish')
          ORDER BY d.path`),
    );
    expect(rows.map(r => r.path)).toContain("retired.md");
    expect(rows.map(r => r.path)).toContain("invalid.md");
  });

  // =========================================================================
  // THE LOAD-BEARING CASE
  // =========================================================================

  it("FINDS a document whose distinctive term appears ONLY in the body", async () => {
    const out = await search("zebrafish", { collections: "research" });
    // body-only.md's title says "Quiet ledger" — nothing about zebrafish. If
    // the stored tsvector were title-only (the pre-007 defect) this is []. It is
    // also the arm's core acceptance: a lexical search over prose bodies.
    expect(out.map(r => r.displayPath)).toEqual(["research/body-only.md"]);
    expect(out[0]!.source).toBe("fts");
    expect(out[0]!.body).toContain("zebrafish larvae");
    expect(out[0]!.title).toBe("Quiet ledger");
  });

  // =========================================================================
  // THE WEIGHT PROOF — the instrument for the ts_rank_cd weight array
  // =========================================================================

  it("RANKS a TITLE match above a BODY-only match for the same term", async () => {
    // weight-title.md has "Marmalade" in the title (weight A = 1.0);
    // weight-body.md has it only in the body (weight D = 0.1). A:D = 10:1,
    // matching the sqlite arm's bm25(documents_fts, 10.0, 1.0). If the weights
    // array were wrong (or omitted, which defaults to the same array but would
    // stop being DERIVED) this ordering is what breaks.
    const out = await search("marmalade", { collections: "research" });
    expect(out.map(r => r.displayPath)).toEqual([
      "research/weight-title.md", "research/weight-body.md",
    ]);
    // Not merely "both returned" — the scores must actually separate, and by a
    // wide margin rather than a tie broken by the (collection, path) fallback.
    expect(out[0]!.score).toBeGreaterThan(out[1]!.score);
    expect(out[0]!.score).toBeGreaterThan(out[1]!.score * 2);
    // Sanity: the tie-break would have put weight-body FIRST alphabetically,
    // so the observed order cannot have come from the tie-break.
    expect("weight-body.md" < "weight-title.md").toBe(true);
  });

  // =========================================================================
  // Fences and filters, behaviourally
  // =========================================================================

  it("EXCLUDES inactive and invalidated documents from the results", async () => {
    const paths = (await search("zebrafish", { collections: "research" }))
      .map(r => r.displayPath);
    expect(paths).not.toContain("research/retired.md");
    expect(paths).not.toContain("research/invalid.md");
    expect(paths).toEqual(["research/body-only.md"]);
  });

  it("NARROWS to the requested collection — and the excluded row is one that would have ranked", async () => {
    // decisions/elsewhere.md carries "Marmalade" in its TITLE, so unfiltered it
    // ranks alongside the title match; a filter that did nothing would show it.
    const all = (await search("marmalade")).map(r => r.displayPath);
    expect(all).toContain("decisions/elsewhere.md");
    expect(all).toContain("research/weight-title.md");

    const narrowed = (await search("marmalade", { collections: "research" }))
      .map(r => r.displayPath);
    expect(narrowed).not.toContain("decisions/elsewhere.md");

    const other = await search("marmalade", { collections: ["decisions"] });
    expect(other.map(r => r.displayPath)).toEqual(["decisions/elsewhere.md"]);
  });

  it("honours the document limit", async () => {
    expect((await search("marmalade", { limit: 1 })).length).toBe(1);
    expect((await search("marmalade")).length).toBeGreaterThan(1);
  });

  // =========================================================================
  // websearch_to_tsquery is really PARSING
  // =========================================================================

  it("treats a QUOTED phrase differently from the same words unquoted", async () => {
    // phrase-adjacent.md: "quick brown fox".  phrase-split.md: "fox was brown".
    // Unquoted, websearch_to_tsquery ANDs the lexemes -> both match. Quoted, it
    // builds `brown <-> fox` -> only the adjacent one. If the constructor were
    // bypassed (a hand-rolled sanitizer stripping the quotes, say) these two
    // searches would return the same set.
    const unquoted = (await search("brown fox", { collections: "research" }))
      .map(r => r.displayPath).sort();
    expect(unquoted).toEqual(["research/phrase-adjacent.md", "research/phrase-split.md"]);

    const quoted = (await search('"brown fox"', { collections: "research" }))
      .map(r => r.displayPath);
    expect(quoted).toEqual(["research/phrase-adjacent.md"]);
    expect(quoted).not.toEqual(unquoted);
  });

  it("does not THROW on syntactically hostile input — the reason for websearch_to_tsquery", async () => {
    // `to_tsquery` raises a syntax error on this; websearch_to_tsquery does not.
    // A search box is exactly where such input arrives.
    const out = await searchDetailed("marmalade & ( ! ) \\ ''' <->", { collections: "research" });
    expect(out.degraded).toBe(false);
    expect(out.results.map(r => r.displayPath)).toContain("research/weight-title.md");
  });

  // =========================================================================
  // The degraded channel against the LIVE cluster
  // =========================================================================

  it("empty-tsquery, live: a query of only stopwords is DEGRADED, not empty", async () => {
    const out = await searchDetailed("the and of it", { collections: "research" });
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("empty-tsquery");
    expect(out.results).toEqual([]);
    expect(out.scannedRows).toBe(0);
  });

  it("GENUINE EMPTY, live: the scan really ran and matched zero documents", async () => {
    const out = await searchDetailed("pelargonium", { collections: "research" });
    expect(out.degraded).toBe(false);
    expect(out.degradedReason).toBeUndefined();
    expect(out.results).toEqual([]);
    expect(out.scannedRows).toBe(0);
    // Same [] as the case above; the LABEL is the entire difference, produced
    // here by real PostgreSQL state rather than by a fake's fixtures.
  });

  it("CONTROL: the healthy live search is degraded:false and counts pre-limit rows", async () => {
    const out = await searchDetailed("marmalade");
    expect(out.degraded).toBe(false);
    expect(out.degradedReason).toBeUndefined();
    expect(out.scannedRows).toBe(3); // two research + one decisions
    expect(out.results.length).toBe(3);
    // And the back-compat signature returns exactly `.results`.
    expect(await search("marmalade")).toEqual(out.results);
  });

  // =========================================================================
  // The statement_timeout actually fires, and is typed
  // =========================================================================

  it("surfaces a statement_timeout as PgFtsSearchTimeoutError — proven by inducing one", async () => {
    // INDUCING IT FOR REAL, by the same technique the vec suite established: a
    // second session takes ACCESS EXCLUSIVE on `documents`, which blocks the
    // ranked scan. A lock wait is exactly what statement_timeout is for, and it
    // is deterministic — unlike hoping an eight-row scan overruns a millisecond.
    const locker = await pool.connect();
    let caught: unknown;
    try {
      await locker.query(`SET search_path TO ${schema}, public`);
      await locker.query("BEGIN");
      await locker.query("LOCK TABLE documents IN ACCESS EXCLUSIVE MODE");

      const t0 = Date.now();
      try {
        await search("marmalade", { collections: "research", statementTimeoutMs: 400 });
      } catch (e) { caught = e; }
      const elapsed = Date.now() - t0;

      expect(caught).toBeInstanceOf(PgFtsSearchTimeoutError);
      const err = caught as PgFtsSearchTimeoutError;
      expect(err.timeoutMs).toBe(400);
      expect(err.scope).toBe("research");
      expect(err.stage).toBe("fts-scan");
      // Not a raw driver throw, NOT a silent empty result, and NOT the vector
      // arm's error class — a caller reading `name` learns which arm gave up.
      expect(err.name).toBe("PgFtsSearchTimeoutError");
      expect(err.message).toContain("Lexical");
      // It really was the server-side bound: bounded, and of the right order.
      expect(elapsed).toBeGreaterThanOrEqual(300);
      expect(elapsed).toBeLessThan(8000);
    } finally {
      await locker.query("ROLLBACK").catch(() => {});
      locker.release();
    }
    // RESTORE -> GREEN, in the same case: with the lock gone the identical call
    // returns results. That is what proves the throw came from the induced
    // condition and not from the timeout being permanently broken.
    const after = await search("marmalade", { collections: "research", statementTimeoutMs: 400 });
    expect(after.map(r => r.displayPath)).toEqual([
      "research/weight-title.md", "research/weight-body.md",
    ]);
  }, 20_000);

  it("leaves NO session GUC behind on the pooled connection it used", async () => {
    // SET LOCAL unwinds with the transaction; a session SET would still be
    // visible on the same client afterwards — the connection a later caller
    // would inherit.
    await withSchema(async c => {
      await pgSearchFts(c, "marmalade", { collections: "research", statementTimeoutMs: 400 });
      const { rows } = await c.query<{ v: string }>(
        "SELECT current_setting('statement_timeout') AS v");
      expect(rows[0]!.v).toBe("0");
      // And the connection is usable — not stuck in an aborted transaction.
      const { rows: ok } = await c.query<{ n: number }>("SELECT 1 AS n");
      expect(ok[0]!.n).toBe(1);
    });
  });
});
