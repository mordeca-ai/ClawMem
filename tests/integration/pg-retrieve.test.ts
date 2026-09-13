/**
 * `pg retrieve` integration tests (master-harness-2wx75 slice 7).
 *
 * House pattern (tests/integration/pg-search-reranked.test.ts): the LIVE
 * clawmem-pg cluster, a throwaway schema pinned via `setPgSchema`, SKIP when
 * CLAWMEM_PG_URL is unset.
 *
 * WHAT IS LIVE: the PRODUCTION `withClient` (the real vault pool, with the
 * schema knob), the real three read functions, the real contract mapping and
 * `retrieveCli`'s one-object stdout. WHAT IS INJECTED: the embedder (a fixed
 * 8-dim vector, so this suite does not report the GPU host's health) and the
 * reranker (deterministic reversal, so `applied` visibly reorders). The live
 * cross-encoder is exercised by the slice-7 live smoke, not here.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool, toVectorLiteral, withClient } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { pgSearchVecDetailed } from "../../src/pg/search.ts";
import { pgSearchFtsDetailed } from "../../src/pg/search-fts.ts";
import { pgSearchRerankedDetailed, type PgReranker } from "../../src/pg/search-reranked.ts";
import {
  EXIT_OK,
  EXIT_THREW,
  RETRIEVE_SCHEMA,
  retrieveCli,
  type RetrieveDeps,
  type RetrieveOutput,
} from "../../src/pg/retrieve.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 8;
const VAULT_MODEL = "embeddinggemma";
const d = URL_ ? describe : describe.skip;

function atAngle(deg: number): number[] {
  const t = (deg * Math.PI) / 180;
  const v = new Array(DIM).fill(0);
  v[0] = Math.cos(t);
  v[1] = Math.sin(t);
  return v;
}

const FIXTURES = [
  { path: "lexical-only.md", title: "Zebrafish ledger", body: "Counted twice before dawn.", angleDeg: undefined },
  { path: "notes/vector-only.md", title: "Quiet notes", body: "Nothing here names the fish at all.", angleDeg: 0 },
  { path: "both.md", title: "Unremarkable log", body: "The zebrafish larvae were tallied.", angleDeg: 30 },
] as const;

const CONTRACT_KEYS = [
  "schema", "mode", "query", "limit", "results", "degraded", "degradedReason",
  "rerankStatus", "timings", "error",
].sort();

d("pg retrieve against the live cluster", () => {
  let pool: pg.Pool;
  let schema: string;
  let rerankCalls = 0;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: URL_ });
    schema = `clawmem_retrievetest_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
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
          `INSERT INTO documents (collection, path, title, hash, active) VALUES ('research', $1, $2, $3, true)`,
          [fx.path, fx.title, hash],
        );
        if (fx.angleDeg !== undefined) {
          await c.query(
            `INSERT INTO content_vectors (hash, seq, pos, model, embedding) VALUES ($1, 0, 0, $2, $3::vector)`,
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

  /** Production withClient + real read functions; only embedder/reranker injected. */
  function liveDeps(over: Partial<RetrieveDeps> = {}): RetrieveDeps {
    return {
      withClient: (vault, fn) => withClient(vault, fn),
      embedder: () => ({ async embed() { return { embedding: atAngle(0), model: VAULT_MODEL }; } }),
      async reranker() {
        const r: PgReranker = async (_q, docs) => {
          rerankCalls++;
          return docs.map((doc, i) => ({ file: doc.file, score: (i + 1) / docs.length }));
        };
        return r;
      },
      searchFts: pgSearchFtsDetailed,
      searchVec: pgSearchVecDetailed,
      searchReranked: pgSearchRerankedDetailed,
      // The pool is shared across cases; afterAll closes it.
      async dispose() {},
      ...over,
    };
  }

  async function run(argv: string[], deps = liveDeps()) {
    const lines: string[] = [];
    const code = await retrieveCli(argv, deps, { stdout: s => lines.push(s) });
    expect(lines).toHaveLength(1);
    const out = JSON.parse(lines[0]!) as RetrieveOutput;
    expect(Object.keys(out).sort()).toEqual(CONTRACT_KEYS);
    expect(out.schema).toBe(RETRIEVE_SCHEMA);
    return { code, out };
  }

  it("search: the live FTS arm, <collection>/<path> files, rerankStatus null", async () => {
    const { code, out } = await run(["--mode", "search", "--query", "zebrafish", "--collection", "research"]);
    expect(code).toBe(EXIT_OK);
    expect(out.error).toBeNull();
    expect(out.degraded).toBe(false);
    expect(out.rerankStatus).toBeNull();
    expect(out.results.map(r => r.file).sort()).toEqual(["research/both.md", "research/lexical-only.md"]);
    for (const r of out.results) expect(typeof r.score).toBe("number");
  });

  it("search: an all-stopword query is degraded-empty with exit 0", async () => {
    const { code, out } = await run(["--mode", "search", "--query", "the and of"]);
    expect(code).toBe(EXIT_OK);
    expect(out.results).toEqual([]);
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("empty-tsquery");
  });

  it("vsearch: the live vector arm in distance order, nested path preserved, embedMs measured", async () => {
    const { code, out } = await run(["--mode", "vsearch", "--query", "zebrafish", "--collection", "research"]);
    expect(code).toBe(EXIT_OK);
    expect(out.error).toBeNull();
    expect(out.rerankStatus).toBeNull();
    expect(out.results.map(r => r.file)).toEqual(["research/notes/vector-only.md", "research/both.md"]);
    expect(out.results[0]!.score).toBeGreaterThan(out.results[1]!.score);
    expect(typeof out.timings.embedMs).toBe("number");
  });

  it("query: live hybrid + injected reranker => applied, timings populated", async () => {
    const before = rerankCalls;
    const { code, out } = await run([
      "--mode", "query", "--query", "zebrafish", "--collection", "research", "--deadline-ms", "5000",
    ]);
    expect(code).toBe(EXIT_OK);
    expect(out.error).toBeNull();
    expect(rerankCalls).toBe(before + 1);
    expect(out.rerankStatus).toBe("applied");
    expect(out.degraded).toBe(false);
    expect(out.results).toHaveLength(3);
    for (const k of ["totalMs", "hybridMs", "rerankMs", "embedMs"] as const) {
      expect(typeof out.timings[k]).toBe("number");
    }
  });

  it("query --no-rerank: live hybrid only, skipped-no-reranker, reranker untouched", async () => {
    const before = rerankCalls;
    const { code, out } = await run(["--mode", "query", "--query", "zebrafish", "--collection", "research", "--no-rerank"]);
    expect(code).toBe(EXIT_OK);
    expect(rerankCalls).toBe(before);
    expect(out.rerankStatus).toBe("skipped-no-reranker");
    expect(out.results[0]!.file).toBe("research/both.md");
  });

  it("thrown: an unconfigured nsfw vault is ONE JSON object with error and exit 2", async () => {
    const saved = { url: process.env.CLAWMEM_PG_NSFW_URL, host: process.env.CLAWMEM_PG_NSFW_HOST };
    delete process.env.CLAWMEM_PG_NSFW_URL;
    delete process.env.CLAWMEM_PG_NSFW_HOST;
    try {
      const { code, out } = await run(["--mode", "search", "--query", "zebrafish", "--vault", "nsfw"]);
      expect(code).toBe(EXIT_THREW);
      expect(out.results).toEqual([]);
      expect(out.error).toMatch(/^\w+: /);
    } finally {
      if (saved.url !== undefined) process.env.CLAWMEM_PG_NSFW_URL = saved.url;
      if (saved.host !== undefined) process.env.CLAWMEM_PG_NSFW_HOST = saved.host;
    }
  });
});
