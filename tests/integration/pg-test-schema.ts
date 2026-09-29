/**
 * Shared throwaway-schema harness for the PG integration suites
 * (master-harness-vn4rz.73).
 *
 * WHY THIS EXISTS. Each PG suite used to copy-paste the same setup:
 *
 *     CREATE SCHEMA s;  SET search_path TO s, public;  <apply every migration>
 *
 * with an afterAll that ran `DROP SCHEMA s CASCADE`. When the beforeAll ran
 * past bun's 5s default hook timeout, bun ran afterAll WHILE the migration loop
 * was still going: the schema was dropped, `s` vanished from the search_path,
 * and the next unqualified `CREATE OR REPLACE FUNCTION` resolved to `public` —
 * on 2026-09-29 that rewrote the PRODUCTION trigger functions
 * public.documents_fts_refresh / origin_documents_fts_refresh (pinned to the
 * dead schema) and every production write failed for ~5h.
 *
 * The four guards here close every link of that chain:
 *
 *  1. PRODUCTION GUARD — before any DDL, `current_database()` is checked; a
 *     production vault database (`clawmem`, `clawmem_nsfw`) THROWS unless
 *     CLAWMEM_PG_TEST_ALLOW_PRODUCTION=1. A PG gate pointed at production is an
 *     error, not a skip.
 *  2. ONE TRANSACTION — CREATE SCHEMA, every migration and the suite's seed run
 *     inside one BEGIN…COMMIT on one client with `SET LOCAL search_path`. Until
 *     COMMIT the schema is invisible to every other session, so a concurrent
 *     DROP cannot remove it, and the schema is always first on this client's
 *     path — an unqualified object can never fall through to `public`. After
 *     COMMIT the schema's existence and its FTS functions are asserted.
 *  3. TEARDOWN AWAITS SETUP — `setup()` records its in-flight promise and
 *     `teardown()` awaits it (swallowing its error) before the DROP, so a
 *     timed-out beforeAll can never race the drop.
 *  4. EXPLICIT HOOK TIMEOUT — PG_TEST_SETUP_TIMEOUT_MS, passed as bun's
 *     beforeAll(fn, timeout), sits far above the 5s default.
 *
 * NON-TRANSACTIONAL MIGRATIONS. Migration 002 declares
 * `-- clawmem:no-transaction` for `CREATE INDEX CONCURRENTLY`, which cannot run
 * inside a transaction block. For a throwaway schema whose table was created
 * in this same transaction (and is empty), CONCURRENTLY buys nothing — it only
 * avoids blocking concurrent writers, and there are none — so
 * `migrationSqlForTransaction` rewrites it to a plain `CREATE INDEX` with the
 * identical definition. Any OTHER statement that cannot run in a transaction
 * block is refused loudly rather than guessed at. The migration runner's real
 * CONCURRENTLY path is still exercised verbatim by pg-write-path's
 * "applies twice" test, which re-applies the unmodified files after setup.
 */

import pg from "pg";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";

/** The production vault databases no test may run DDL against. */
export const PRODUCTION_VAULT_DATABASES: readonly string[] = ["clawmem", "clawmem_nsfw"];
/** Explicit, operator-only override for the production guard. */
export const ALLOW_PRODUCTION_ENV = "CLAWMEM_PG_TEST_ALLOW_PRODUCTION";
/** bun beforeAll timeout for any hook that applies the migrations. */
export const PG_TEST_SETUP_TIMEOUT_MS = 120_000;
/**
 * bun afterAll timeout for any hook that runs DROP DATABASE. DROP DATABASE forces an
 * immediate checkpoint, so its latency scales with every dirty buffer on the shared
 * server, not with the test: the 2026-09-29 nightly full sweep measured a 9.0s
 * checkpoint and a 9.6s DROP, past bun's 5s default. The hook timed out, and the
 * second vault database was never dropped (master-harness-83qk0.120).
 */
export const PG_TEST_TEARDOWN_TIMEOUT_MS = 120_000;
/** Functions every fully-migrated schema must contain (the 2026-09-29 victims). */
export const EXPECTED_SCHEMA_FUNCTIONS: readonly string[] = [
  "documents_fts_refresh",
  "origin_documents_fts_refresh",
  "content_fts_cascade",
];

/** The one method the production guard needs — a pg Client/PoolClient or a test fake. */
export interface CurrentDatabaseQueryable {
  query(sql: string): Promise<{ rows: Array<Record<string, unknown>> }>;
}

export class ProductionDatabaseError extends Error {
  constructor(readonly database: string) {
    super(
      `Refusing to run PG integration DDL against the PRODUCTION database "${database}". ` +
        `The test suites create schemas and apply migrations; pointed at production they have ` +
        `already rewritten live trigger functions once (2026-09-29, master-harness-vn4rz.73). ` +
        `Point CLAWMEM_PG_URL at a throwaway database — e.g. run the suite through ` +
        `\`clawmem-lander pg-itest -- bun test …\`, which provisions and drops one — ` +
        `or, only if you are certain, set ${ALLOW_PRODUCTION_ENV}=1.`,
    );
    this.name = "ProductionDatabaseError";
  }
}

/**
 * Throw ProductionDatabaseError when the client is connected to a production
 * vault database, unless the explicit override env is exactly "1".
 * Returns the database name.
 */
export async function assertNotProductionDatabase(
  client: CurrentDatabaseQueryable,
  env: Record<string, string | undefined> = process.env,
): Promise<string> {
  const { rows } = await client.query("SELECT current_database() AS db");
  const db = rows[0]?.db;
  if (typeof db !== "string" || db.length === 0) {
    throw new Error("PG test guard: current_database() returned no name; refusing to run DDL.");
  }
  if (PRODUCTION_VAULT_DATABASES.includes(db) && env[ALLOW_PRODUCTION_ENV] !== "1") {
    throw new ProductionDatabaseError(db);
  }
  return db;
}

const NO_TRANSACTION_DIRECTIVE = /^--\s*clawmem:no-transaction\s*$/m;
const CREATE_INDEX_CONCURRENTLY = /\bCREATE(\s+UNIQUE)?\s+INDEX\s+CONCURRENTLY\b/gi;
/** Statements PostgreSQL refuses inside a transaction block (none may survive the rewrite). */
const NON_TRANSACTIONAL = [
  /\bCONCURRENTLY\b/i,
  /\bVACUUM\b/i,
  /\bCREATE\s+DATABASE\b/i,
  /\bDROP\s+DATABASE\b/i,
  /\bALTER\s+SYSTEM\b/i,
  /\bCREATE\s+TABLESPACE\b/i,
  /\bREINDEX\s+(DATABASE|SYSTEM)\b/i,
  /^\s*(BEGIN|COMMIT|ROLLBACK|START\s+TRANSACTION)\b/im,
];

/**
 * SQL with dollar-quoted bodies, `--` line comments and single-quoted literals
 * blanked — for the statement scan only (prose must not trip it).
 */
function executableText(sql: string): string {
  return sql
    .replace(/\$([A-Za-z_]*)\$[\s\S]*?\$\1\$/g, "''")
    .replace(/--[^\n]*/g, "")
    .replace(/'(?:[^']|'')*'/g, "''");
}

/**
 * One migration file's SQL, parameter-substituted and made safe to run inside
 * the setup transaction. Throws when the file needs something no transaction
 * can host.
 */
export function migrationSqlForTransaction(
  file: string,
  raw: string,
  schema: string,
  dim: number,
): string {
  let sql = substituteMigrationParams(raw, schema, dim);
  if (NO_TRANSACTION_DIRECTIVE.test(raw)) {
    // Empty table created in this same transaction: a plain build is equivalent.
    sql = sql.replace(
      CREATE_INDEX_CONCURRENTLY,
      (_m, unique: string | undefined) => `CREATE${unique ?? ""} INDEX`,
    );
  }
  const code = executableText(sql);
  for (const re of NON_TRANSACTIONAL) {
    if (re.test(code)) {
      throw new Error(
        `pg-test-schema: migration ${file} contains a statement that cannot run inside the ` +
          `setup transaction (${re}). Teach migrationSqlForTransaction how to host it safely — ` +
          `do NOT fall back to a non-transactional setup (master-harness-vn4rz.73).`,
      );
    }
  }
  return sql;
}

/** Double-quote a PostgreSQL identifier. */
export function quoteIdent(name: string): string {
  return `"${name.replaceAll('"', '""')}"`;
}

export interface PgTestSchemaOptions {
  /** Connection string (the suite's CLAWMEM_PG_URL). */
  url: string;
  /** Schema-name prefix; a timestamp + random suffix is appended. */
  prefix: string;
  /** Embedding dimension substituted for `:EMBED_DIM`. */
  dim: number;
  /** Fixture rows, run INSIDE the setup transaction with the schema on the search_path. */
  seed?: (c: pg.PoolClient, schema: string) => Promise<void>;
  /** Test seam: awaited after each migration file (inside the transaction). */
  afterMigration?: (file: string, index: number) => Promise<void>;
  /** Override the migrations directory (tests only). */
  migrationsDir?: string;
  /** Functions that must exist in the schema after COMMIT. */
  expectedFunctions?: readonly string[];
}

export interface PgTestSchema {
  /** The unique throwaway schema name. */
  readonly schema: string;
  /** Pool owned by the harness; ended by teardown(). */
  readonly pool: pg.Pool;
  /** Create + migrate + seed the schema atomically. Idempotent: returns the same promise. */
  setup(): Promise<void>;
  /** Resolves once any in-flight setup has finished, never rejects. */
  settled(): Promise<void>;
  /** Await setup, then DROP the schema and end the pool. Safe to call without setup. */
  teardown(): Promise<void>;
  /** Run `fn` on a pooled client whose search_path is `<schema>, public`. */
  withSchema<T>(fn: (c: pg.PoolClient) => Promise<T>): Promise<T>;
}

function uniqueSchemaName(prefix: string): string {
  const name = `${prefix}_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
  if (!/^[a-z_][a-z0-9_]{0,62}$/.test(name)) {
    throw new Error(`pg-test-schema: invalid schema name ${JSON.stringify(name)}`);
  }
  return name;
}

export function createPgTestSchema(opts: PgTestSchemaOptions): PgTestSchema {
  const schema = uniqueSchemaName(opts.prefix);
  const ident = quoteIdent(schema);
  const pool = new pg.Pool({ connectionString: opts.url });
  const dir = opts.migrationsDir ?? MIGRATIONS_DIR;
  const expected = opts.expectedFunctions ?? EXPECTED_SCHEMA_FUNCTIONS;
  let inFlight: Promise<void> | null = null;
  let tornDown: Promise<void> | null = null;

  async function assertMaterialised(c: pg.PoolClient): Promise<void> {
    const ns = await c.query("SELECT 1 FROM pg_namespace WHERE nspname = $1", [schema]);
    if (ns.rowCount !== 1) {
      throw new Error(`pg-test-schema: schema ${schema} does not exist after COMMIT`);
    }
    const { rows } = await c.query<{ proname: string }>(
      `SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = $1 AND p.proname = ANY($2::text[])`,
      [schema, [...expected]],
    );
    const found = new Set(rows.map((r) => r.proname));
    const missing = expected.filter((f) => !found.has(f));
    if (missing.length > 0) {
      throw new Error(
        `pg-test-schema: schema ${schema} is missing function(s) after COMMIT: ${missing.join(", ")}`,
      );
    }
  }

  async function runSetup(): Promise<void> {
    const files = readdirSync(dir)
      .filter((f) => f.endsWith(".sql"))
      .sort();
    const c = await pool.connect();
    try {
      await assertNotProductionDatabase(c);
      await c.query("BEGIN");
      try {
        await c.query(`CREATE SCHEMA ${ident}`);
        // SET LOCAL: reverts at COMMIT/ROLLBACK, so the pooled client carries no
        // lingering search_path, and while the transaction is open the schema
        // (created in it) is guaranteed to exist and to be first on the path.
        await c.query(`SET LOCAL search_path TO ${ident}, public`);
        for (const [i, f] of files.entries()) {
          const raw = readFileSync(join(dir, f), "utf-8");
          await c.query(migrationSqlForTransaction(f, raw, schema, opts.dim));
          if (opts.afterMigration) await opts.afterMigration(f, i);
        }
        if (opts.seed) await opts.seed(c, schema);
        await c.query("COMMIT");
      } catch (e) {
        await c.query("ROLLBACK").catch(() => {});
        throw e;
      }
      await assertMaterialised(c);
    } finally {
      c.release();
    }
  }

  const harness: PgTestSchema = {
    schema,
    pool,
    setup() {
      if (tornDown) return Promise.reject(new Error("pg-test-schema: setup() after teardown()"));
      inFlight ??= runSetup();
      return inFlight;
    },
    async settled() {
      if (inFlight) await inFlight.catch(() => {});
    },
    teardown() {
      tornDown ??= (async () => {
        await harness.settled();
        try {
          await pool.query(`DROP SCHEMA IF EXISTS ${ident} CASCADE`);
        } finally {
          await pool.end();
        }
      })();
      return tornDown;
    },
    async withSchema<T>(fn: (c: pg.PoolClient) => Promise<T>): Promise<T> {
      const c = await pool.connect();
      try {
        await c.query(`SET search_path TO ${ident}, public`);
        return await fn(c);
      } finally {
        c.release();
      }
    },
  };
  return harness;
}
