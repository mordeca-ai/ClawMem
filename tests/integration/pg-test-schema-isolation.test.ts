/**
 * Regression test for master-harness-vn4rz.73: a PG suite's throwaway-schema
 * setup must never be able to leak DDL into `public`, even when its teardown
 * races it (the 2026-09-29 production write outage).
 *
 * Two tiers:
 *  - DB-free (always runs): the production guard, and the rewrite that lets
 *    the no-transaction migration run inside the setup transaction.
 *  - Live PG (SKIPS when CLAWMEM_PG_URL is unset — run it through
 *    `clawmem-lander pg-itest`, which points it at a THROWAWAY database): pause
 *    setup between migration files, race a teardown / a bare DROP SCHEMA
 *    against it, let it finish, and prove `public` is byte-for-byte unchanged,
 *    nothing in `public` names the test schema, and no schema is leaked.
 *
 * The pause sits after 004 and before 005, the first file that (re)defines
 * documents_fts_refresh / origin_documents_fts_refresh — the functions the
 * incident rewrote in production.
 */

import { describe, it, expect } from "bun:test";
import pg from "pg";
import { readFileSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR } from "../../src/pg/migrate.ts";
import {
  ALLOW_PRODUCTION_ENV,
  PG_TEST_SETUP_TIMEOUT_MS,
  ProductionDatabaseError,
  assertNotProductionDatabase,
  createPgTestSchema,
  migrationSqlForTransaction,
  type CurrentDatabaseQueryable,
  type PgTestSchema,
} from "./pg-test-schema.ts";

const URL_ = process.env.CLAWMEM_PG_URL;
const DIM = 768;

function fakeClient(db: string): CurrentDatabaseQueryable & { sql: string[] } {
  const sql: string[] = [];
  return {
    sql,
    async query(text: string) {
      sql.push(text);
      return { rows: [{ db }] };
    },
  };
}

describe("PG test harness — production guard (DB-free)", () => {
  for (const prod of ["clawmem", "clawmem_nsfw"]) {
    it(`throws for current_database() = '${prod}'`, async () => {
      const c = fakeClient(prod);
      await expect(assertNotProductionDatabase(c, {})).rejects.toBeInstanceOf(
        ProductionDatabaseError,
      );
      await expect(assertNotProductionDatabase(c, {})).rejects.toThrow(/PRODUCTION database/);
      expect(c.sql[0]).toMatch(/current_database\(\)/);
    });
  }

  it("the explicit override env (exactly '1') allows production", async () => {
    await expect(
      assertNotProductionDatabase(fakeClient("clawmem"), { [ALLOW_PRODUCTION_ENV]: "1" }),
    ).resolves.toBe("clawmem");
  });

  it("any other override value still refuses", async () => {
    for (const v of ["", "0", "true", "yes"]) {
      await expect(
        assertNotProductionDatabase(fakeClient("clawmem_nsfw"), { [ALLOW_PRODUCTION_ENV]: v }),
      ).rejects.toBeInstanceOf(ProductionDatabaseError);
    }
  });

  it("a throwaway database passes (exact-name match, not prefix)", async () => {
    for (const db of [
      "clawmem_lander_itest_1_2",
      "clawmem_nsfw_lander_itest_1_2",
      "clawmemx",
      "postgres",
    ]) {
      await expect(assertNotProductionDatabase(fakeClient(db), {})).resolves.toBe(db);
    }
  });

  it("an empty current_database() refuses rather than guessing", async () => {
    await expect(assertNotProductionDatabase(fakeClient(""), {})).rejects.toThrow(/no name/);
  });
});

describe("PG test harness — transactional migration text (DB-free)", () => {
  it("rewrites 002's CREATE INDEX CONCURRENTLY to a plain build with the same definition", () => {
    const f = "002_vector_index_hnsw.sql";
    const raw = readFileSync(join(MIGRATIONS_DIR, f), "utf-8");
    const sql = migrationSqlForTransaction(f, raw, "s", DIM);
    const code = sql
      .split("\n")
      .filter((l) => !l.trim().startsWith("--"))
      .join("\n");
    expect(code).not.toMatch(/CONCURRENTLY/i);
    expect(code).toMatch(
      /CREATE INDEX IF NOT EXISTS content_vectors_embedding_hnsw_idx\s+ON content_vectors USING hnsw/,
    );
  });

  it("rewrites 013's DROP INDEX CONCURRENTLY inside the isolated schema transaction", () => {
    const f = "013_drop_full_content_vectors_hnsw.sql";
    const raw = readFileSync(join(MIGRATIONS_DIR, f), "utf-8");
    const sql = migrationSqlForTransaction(f, raw, "s", DIM);
    const code = sql.split("\n").filter(l => !l.trim().startsWith("--")).join("\n");
    expect(code).not.toMatch(/CONCURRENTLY/i);
    expect(code).toMatch(/DROP INDEX IF EXISTS content_vectors_embedding_hnsw_idx/);
  });

  it("refuses a statement no transaction can host", () => {
    expect(() => migrationSqlForTransaction("x.sql", "VACUUM documents;", "s", DIM)).toThrow(
      /cannot run inside/,
    );
    expect(() =>
      migrationSqlForTransaction(
        "y.sql",
        "ALTER TABLE t DETACH PARTITION p CONCURRENTLY;",
        "s",
        DIM,
      ),
    ).toThrow(/cannot run inside/);
  });

  it("does not trip on prose in comments, literals or function bodies", () => {
    const sql =
      "-- VACUUM and CONCURRENTLY in a comment\n" +
      "COMMENT ON TABLE t IS 'never VACUUM FULL';\n" +
      "CREATE FUNCTION f() RETURNS void AS $$ BEGIN PERFORM 1; END $$ LANGUAGE plpgsql;";
    expect(() => migrationSqlForTransaction("z.sql", sql, "s", DIM)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Live race tests
// ---------------------------------------------------------------------------

interface PublicSnapshot {
  functions: string[];
  relations: string[];
  types: string[];
}

/** Every non-extension object in `public`, with function bodies + config hashed. */
async function snapshotPublic(pool: pg.Pool): Promise<PublicSnapshot> {
  const notExt = (cls: string, col: string) =>
    `NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = '${cls}'::regclass AND d.objid = ${col} AND d.deptype = 'e')`;
  const f = await pool.query<{ s: string }>(
    `SELECT p.oid::regprocedure::text || ' ' || md5(p.prosrc || coalesce(p.proconfig::text, '')) AS s
       FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
      WHERE n.nspname = 'public' AND ${notExt("pg_proc", "p.oid")} ORDER BY 1`,
  );
  const r = await pool.query<{ s: string }>(
    `SELECT c.relname || ':' || c.relkind::text AS s
       FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'public' AND ${notExt("pg_class", "c.oid")} ORDER BY 1`,
  );
  const t = await pool.query<{ s: string }>(
    `SELECT t.typname AS s FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
      WHERE n.nspname = 'public' AND ${notExt("pg_type", "t.oid")} ORDER BY 1`,
  );
  return {
    functions: f.rows.map((x) => x.s),
    relations: r.rows.map((x) => x.s),
    types: t.rows.map((x) => x.s),
  };
}

/** Anything in `public` whose definition or config names the test schema. */
async function publicReferencesTo(pool: pg.Pool, schema: string): Promise<string[]> {
  const { rows } = await pool.query<{ s: string }>(
    `SELECT 'function ' || p.oid::regprocedure::text AS s
       FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
      WHERE n.nspname = 'public'
        AND (p.prosrc LIKE '%' || $1 || '%' OR coalesce(p.proconfig::text, '') LIKE '%' || $1 || '%')
     UNION ALL
     SELECT 'view ' || c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'public' AND c.relkind IN ('v', 'm')
        AND pg_get_viewdef(c.oid) LIKE '%' || $1 || '%'`,
    [schema],
  );
  return rows.map((r) => r.s);
}

async function schemaExists(pool: pg.Pool, schema: string): Promise<boolean> {
  const { rowCount } = await pool.query("SELECT 1 FROM pg_namespace WHERE nspname = $1", [schema]);
  return rowCount === 1;
}

/** A gate the setup seam blocks on after `pauseAfter`, reporting when it got there. */
function pauseAfterMigration(pauseAfter: string) {
  let reached!: () => void;
  let release!: () => void;
  const reachedP = new Promise<void>((r) => (reached = r));
  const releaseP = new Promise<void>((r) => (release = r));
  return {
    reached: reachedP,
    release,
    hook: async (file: string) => {
      if (file.startsWith(pauseAfter)) {
        reached();
        await releaseP;
      }
    },
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const d = URL_ ? describe : describe.skip;

d("PG test harness — setup/teardown race cannot touch public (live)", () => {
  async function withObserver<T>(fn: (obs: pg.Pool) => Promise<T>): Promise<T> {
    const obs = new pg.Pool({ connectionString: URL_, max: 2 });
    try {
      return await fn(obs);
    } finally {
      await obs.end();
    }
  }

  it(
    "a teardown racing an in-flight setup waits for it, drops the schema, and leaves public unchanged",
    async () => {
      await withObserver(async (obs) => {
        const before = await snapshotPublic(obs);
        const gate = pauseAfterMigration("004_");
        const h: PgTestSchema = createPgTestSchema({
          url: URL_!,
          prefix: "clawmem_race_td",
          dim: DIM,
          afterMigration: gate.hook,
        });
        const setup = h.setup();
        await gate.reached; // setup is mid-flight: 001..004 applied, 005.. pending
        const teardown = h.teardown(); // what bun does when beforeAll times out
        await sleep(750); // give an un-awaiting teardown every chance to DROP now
        gate.release();
        const [s, t] = await Promise.allSettled([setup, teardown]);

        expect(await snapshotPublic(obs)).toEqual(before);
        expect(await publicReferencesTo(obs, h.schema)).toEqual([]);
        expect(await schemaExists(obs, h.schema)).toBe(false); // no leaked schema either
        expect(t.status).toBe("fulfilled");
        expect(s.status).toBe("fulfilled");
      });
    },
    PG_TEST_SETUP_TIMEOUT_MS,
  );

  it(
    "a bare DROP SCHEMA issued mid-setup cannot remove the uncommitted schema; public unchanged",
    async () => {
      await withObserver(async (obs) => {
        const before = await snapshotPublic(obs);
        const gate = pauseAfterMigration("004_");
        const h = createPgTestSchema({
          url: URL_!,
          prefix: "clawmem_race_drop",
          dim: DIM,
          afterMigration: gate.hook,
        });
        try {
          const setup = h.setup();
          await gate.reached;
          // A teardown that forgot to await setup: DROP from ANOTHER session, now.
          await obs.query(`DROP SCHEMA IF EXISTS "${h.schema}" CASCADE`);
          gate.release();
          const [s] = await Promise.allSettled([setup]);

          expect(await snapshotPublic(obs)).toEqual(before);
          expect(await publicReferencesTo(obs, h.schema)).toEqual([]);
          // The DROP saw nothing; setup committed a complete schema.
          expect(s.status).toBe("fulfilled");
          expect(await schemaExists(obs, h.schema)).toBe(true);
          const { rows } = await h.withSchema((c) =>
            c.query<{ n: number }>(
              `SELECT count(*)::int n FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = $1 AND p.proname IN ('documents_fts_refresh', 'origin_documents_fts_refresh')`,
              [h.schema],
            ),
          );
          expect(rows[0]!.n).toBe(2);
        } finally {
          await h.teardown();
        }
        expect(await schemaExists(obs, h.schema)).toBe(false);
      });
    },
    PG_TEST_SETUP_TIMEOUT_MS,
  );

  it(
    "setup fails loudly when the committed schema lacks an expected function",
    async () => {
      const h = createPgTestSchema({
        url: URL_!,
        prefix: "clawmem_race_expect",
        dim: DIM,
        expectedFunctions: ["documents_fts_refresh", "no_such_function_vn4rz73"],
      });
      try {
        await expect(h.setup()).rejects.toThrow(
          /missing function\(s\) after COMMIT: no_such_function_vn4rz73/,
        );
      } finally {
        await h.teardown();
      }
    },
    PG_TEST_SETUP_TIMEOUT_MS,
  );
});
