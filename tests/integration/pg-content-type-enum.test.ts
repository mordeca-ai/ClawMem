/**
 * content_type enum extension + conform layer (master-harness-vn4rz.35).
 *
 * WHY A SIBLING FILE and not an extension of pg-write-path.test.ts: that suite
 * is already 823 lines and is built around ONE throwaway *schema* inside
 * whatever database CLAWMEM_PG_URL points at. This bead needs something it
 * structurally cannot provide — a whole ephemeral *database* that is stepped
 * through the migration sequence in two stages (001+002, seed pre-amendment
 * rows, then 003) so the BACKFILL can be observed repairing real rows. Bolting
 * a second, differently-shaped fixture onto that file would have made both
 * harder to read. Different fixture, different file.
 *
 * SAFETY: the ephemeral database is created and unconditionally dropped, with
 * backends force-terminated first so a leaked connection cannot wedge the drop.
 * Nothing here ever touches the live `clawmem`, `clawmem_nsfw` or
 * `source_corpora` databases.
 *
 * The DB half SKIPS (not fails) when CLAWMEM_PG_ADMIN_URL is unset, so
 * `bun test` on a machine without the cluster stays green. The drift guard and
 * the pure narrowing tests ALWAYS run — they need no database at all, which is
 * the entire point of keeping the SQL parseable.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync } from "fs";
import { join } from "path";
import { MIGRATIONS_DIR } from "../../src/pg/migrate.ts";
import { closePool } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import {
  CONTENT_TYPES,
  CONTENT_TYPE_CONFORM,
  narrowContentType,
  upsertDocument,
} from "../../src/pg/write.ts";

const MIGRATION_003 = "003_content_type_enum_extend.sql";
const DIM = 768;

/** The ten synonym mappings, restated here so the test is not the code's echo. */
const EXPECTED_CONFORM: Record<string, string> = {
  "planning": "plan",
  "queue-plan": "plan",
  "run": "eval-run",
  "run-report": "eval-run",
  "synthesis": "deductive",
  "research-synthesis": "deductive",
  "memo": "deductive",
  "reference": "hub",
  "runbook": "hub",
  "operations": "handoff",
};

const EXPECTED_NEW = ["eval-run", "observation", "plan", "retro"] as const;

function migrationSql(file: string): string {
  return readFileSync(join(MIGRATIONS_DIR, file), "utf-8").replaceAll(":EMBED_DIM", String(DIM));
}

/**
 * Parse the admitted value list straight out of 003's CHECK constraint.
 *
 * Anchored on `CHECK (content_type IN ( ... ))` specifically, so the backfill's
 * VALUES list — which also contains quoted strings — cannot be picked up by
 * accident.
 */
export function admittedValuesFromMigration(sql: string): string[] {
  const m = /CHECK\s*\(\s*content_type\s+IN\s*\(([^)]*)\)\s*\)/i.exec(sql);
  if (!m) {
    throw new Error(
      `Could not find a \`CHECK (content_type IN (...))\` clause in ${MIGRATION_003}. ` +
      `The drift guard parses that clause; if the DDL shape changed, the guard must ` +
      `change with it rather than silently passing.`,
    );
  }
  return [...m[1]!.matchAll(/'([^']*)'/g)].map(x => x[1]!);
}

// =========================================================================
// Drift guard — no database required. This is the cheap honest version of
// "the TS enum and the SQL constraint must never disagree".
// =========================================================================

describe("content_type enum drift guard", () => {
  it("CONTENT_TYPES is set-equal to the CHECK constraint in 003", () => {
    const fromSql = admittedValuesFromMigration(migrationSql(MIGRATION_003));
    expect(new Set(fromSql).size).toBe(fromSql.length); // no dupes in the SQL
    expect([...fromSql].sort()).toEqual([...CONTENT_TYPES].sort());
  });

  it("the admitted set is exactly the 17 types + the unknown sink", () => {
    expect(CONTENT_TYPES.length).toBe(18);
    expect(CONTENT_TYPES).toContain("unknown");
    for (const v of EXPECTED_NEW) expect(CONTENT_TYPES).toContain(v);
  });

  it("CONTENT_TYPES is alphabetically ordered", () => {
    expect([...CONTENT_TYPES]).toEqual([...CONTENT_TYPES].sort());
  });

  it("every CONTENT_TYPE_CONFORM value is admitted", () => {
    for (const [k, v] of Object.entries(CONTENT_TYPE_CONFORM)) {
      expect({ k, admitted: (CONTENT_TYPES as readonly string[]).includes(v) })
        .toEqual({ k, admitted: true });
    }
  });

  it("no CONTENT_TYPE_CONFORM key is itself admitted (an admitted key is dead code)", () => {
    for (const k of Object.keys(CONTENT_TYPE_CONFORM)) {
      // Rule 2 of narrowContentType wins before the conform lookup, so a key
      // that is also an admitted value could never be reached.
      expect({ k, shadowed: (CONTENT_TYPES as readonly string[]).includes(k) })
        .toEqual({ k, shadowed: false });
    }
  });

  it("the conform map is exactly the ten ADR-0058-amendment mappings", () => {
    expect({ ...CONTENT_TYPE_CONFORM }).toEqual(EXPECTED_CONFORM);
  });

  it("003's backfill covers every conform key and every newly-admitted value", () => {
    const sql = migrationSql(MIGRATION_003);
    const backfill = sql.slice(sql.indexOf("UPDATE documents"));
    for (const k of Object.keys(EXPECTED_CONFORM)) expect(backfill).toContain(`'${k}'`);
    for (const v of EXPECTED_NEW) expect(backfill).toContain(`'${v}'`);
    // session-transcript is deliberately absent from the DDL itself. (It is
    // NAMED in the header comment on purpose — the comment is what stops the
    // next reader from "helpfully" adding it — so comments are stripped first.)
    const ddl = sql.split("\n").filter(l => !l.trim().startsWith("--")).join("\n");
    expect(ddl).not.toMatch(/session-transcript/);
  });

  it("003 does NOT declare the no-transaction directive (it is a plain ALTER)", () => {
    expect(migrationSql(MIGRATION_003)).not.toMatch(/^--\s*clawmem:no-transaction\s*$/m);
  });
});

// =========================================================================
// The narrowing function itself — pure, no database.
// =========================================================================

describe("narrowContentType conform layer", () => {
  it("falsy raw stays unknown with a null raw", () => {
    for (const v of [null, undefined, ""]) {
      expect(narrowContentType(v)).toEqual({ contentType: "unknown", raw: null });
    }
  });

  it("every admitted value round-trips as itself with a NULL raw", () => {
    for (const v of CONTENT_TYPES) {
      expect({ v, ...narrowContentType(v) }).toEqual({ v, contentType: v, raw: null });
    }
  });

  it("every conform key maps to its admitted value AND preserves the raw", () => {
    for (const [k, mapped] of Object.entries(EXPECTED_CONFORM)) {
      expect({ k, ...narrowContentType(k) }).toEqual({ k, contentType: mapped, raw: k });
    }
  });

  it("session-transcript and junk stay unknown with the raw preserved", () => {
    for (const v of ["session-transcript", "totally-invented-nonsense-9f2"]) {
      expect(narrowContentType(v)).toEqual({ contentType: "unknown", raw: v });
    }
  });
});

// =========================================================================
// The database half — an EPHEMERAL database, created and unconditionally
// dropped. Never the live clawmem.
// =========================================================================

const ADMIN_URL = process.env.CLAWMEM_PG_ADMIN_URL;
const dbDescribe = ADMIN_URL ? describe : describe.skip;

/** Rows the PRE-amendment reindex would have parked in the unknown sink. */
const SEED_UNKNOWN_RAWS = [...Object.keys(EXPECTED_CONFORM), ...EXPECTED_NEW];
const SEED_UNMAPPABLE = "session-transcript";

dbDescribe("003 against an ephemeral database", () => {
  const dbName = `clawmem_enumtest_${process.pid}_${Math.floor(Math.random() * 1e6)}`;
  let admin: pg.Pool;
  let ephemeral: pg.Pool;
  let ephemeralUrl: string;
  let priorUrl: string | undefined;

  async function applyFile(c: pg.PoolClient, file: string): Promise<void> {
    await c.query(migrationSql(file));
  }

  async function seedDoc(c: pg.PoolClient, path: string, raw: string): Promise<void> {
    const hash = `h_${path}`;
    await c.query("INSERT INTO content (hash, doc) VALUES ($1, $2) ON CONFLICT DO NOTHING", [hash, "#"]);
    await c.query(
      `INSERT INTO documents (collection, path, title, hash, content_type, content_type_raw)
       VALUES ('seed', $1, $1, $2, 'unknown', $3)`,
      [path, hash, raw],
    );
  }

  beforeAll(async () => {
    // Any pool a previously-run test file left open must go before we re-point
    // CLAWMEM_PG_URL, or the write path would keep talking to the old database.
    setPgSchema(null);
    await closePool();

    admin = new pg.Pool({ connectionString: ADMIN_URL });
    await admin.query(`CREATE DATABASE ${dbName}`);

    const u = new URL(ADMIN_URL!);
    u.pathname = `/${dbName}`;
    ephemeralUrl = u.toString();
    ephemeral = new pg.Pool({ connectionString: ephemeralUrl });

    const c = await ephemeral.connect();
    try {
      // 001 builds a pgvector column and a gin_trgm_ops index; a virgin
      // database has neither extension installed.
      await c.query("CREATE EXTENSION IF NOT EXISTS vector");
      await c.query("CREATE EXTENSION IF NOT EXISTS pg_trgm");
      // STAGE 1 — the world as the in-flight reindex saw it: the narrow enum.
      await applyFile(c, "001_core_schema.sql");
      await applyFile(c, "002_vector_index_hnsw.sql");
      for (const raw of SEED_UNKNOWN_RAWS) await seedDoc(c, `${raw}.md`, raw);
      await seedDoc(c, "transcript.md", SEED_UNMAPPABLE);
      // A row that already carries a decided type must not be disturbed. It
      // lives in its OWN collection so it cannot contaminate the seed-row
      // assertions below — its raw ('memo') is deliberately a conform key.
      await c.query("INSERT INTO content (hash, doc) VALUES ('h_keep', '#') ON CONFLICT DO NOTHING");
      await c.query(
        `INSERT INTO documents (collection, path, title, hash, content_type, content_type_raw)
         VALUES ('keep', 'keep.md', 'keep', 'h_keep', 'note', 'memo')`,
      );
      // STAGE 2 — the amendment lands.
      await applyFile(c, MIGRATION_003);
    } finally {
      c.release();
    }

    priorUrl = process.env.CLAWMEM_PG_URL;
    process.env.CLAWMEM_PG_URL = ephemeralUrl;
  });

  afterAll(async () => {
    if (priorUrl === undefined) delete process.env.CLAWMEM_PG_URL;
    else process.env.CLAWMEM_PG_URL = priorUrl;
    await closePool();
    try { await ephemeral?.end(); } catch { /* teardown is best-effort */ }
    if (admin) {
      // Force-terminate first: one leaked backend is enough to make DROP
      // DATABASE hang forever and leave a stray database behind.
      try {
        await admin.query(
          "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1",
          [dbName],
        );
      } catch { /* best-effort */ }
      await admin.query(`DROP DATABASE IF EXISTS ${dbName}`);
      await admin.end();
    }
  });

  async function withEphemeral<T>(fn: (c: pg.PoolClient) => Promise<T>): Promise<T> {
    const c = await ephemeral.connect();
    try { return await fn(c); } finally { c.release(); }
  }

  // ---- the widened CHECK -------------------------------------------------

  it("the live constraint text lists exactly the 18 admitted values", async () => {
    const { rows } = await withEphemeral(c =>
      c.query<{ def: string }>(
        `SELECT pg_get_constraintdef(oid) def FROM pg_constraint
          WHERE conname = 'documents_content_type_check'`,
      ),
    );
    expect(rows).toHaveLength(1);
    const listed = [...rows[0]!.def.matchAll(/'([^']*)'::text/g)].map(m => m[1]!);
    expect([...new Set(listed)].sort()).toEqual([...CONTENT_TYPES].sort());
  });

  it("a real INSERT ACCEPTS every one of the 18 admitted values", async () => {
    await withEphemeral(async c => {
      await c.query("INSERT INTO content (hash, doc) VALUES ('h_ok', '#') ON CONFLICT DO NOTHING");
      for (const v of CONTENT_TYPES) {
        await c.query(
          `INSERT INTO documents (collection, path, title, hash, content_type)
           VALUES ('accept', $1, $1, 'h_ok', $2)`,
          [`accept-${v}.md`, v],
        );
      }
      const { rows } = await c.query<{ n: string }>(
        "SELECT count(*)::text n FROM documents WHERE collection = 'accept'",
      );
      expect(rows[0]!.n).toBe(String(CONTENT_TYPES.length));
    });
  });

  it("a real INSERT REJECTS a value outside the admitted set", async () => {
    await withEphemeral(async c => {
      await c.query("INSERT INTO content (hash, doc) VALUES ('h_bad', '#') ON CONFLICT DO NOTHING");
      let err: unknown;
      try {
        await c.query(
          `INSERT INTO documents (collection, path, title, hash, content_type)
           VALUES ('reject', 'reject.md', 'reject', 'h_bad', 'session-transcript')`,
        );
      } catch (e) { err = e; }
      // Not merely "it threw": the error must be THE check constraint, and the
      // row must not exist.
      expect((err as { code?: string })?.code).toBe("23514");
      expect(String((err as Error)?.message)).toMatch(/documents_content_type_check/);
      const { rows } = await c.query("SELECT 1 FROM documents WHERE collection = 'reject'");
      expect(rows).toEqual([]);
    });
  });

  // ---- the backfill ------------------------------------------------------

  it("the backfill repaired every conform key, preserving content_type_raw", async () => {
    const { rows } = await withEphemeral(c =>
      c.query<{ content_type: string; content_type_raw: string | null }>(
        `SELECT content_type, content_type_raw FROM documents
          WHERE collection = 'seed' AND content_type_raw = ANY($1) ORDER BY content_type_raw`,
        [Object.keys(EXPECTED_CONFORM)],
      ),
    );
    const got = Object.fromEntries(rows.map(r => [r.content_type_raw, r.content_type]));
    expect(got).toEqual(EXPECTED_CONFORM);
    // The audit trail survived in EVERY repaired row.
    expect(rows.every(r => r.content_type_raw !== null)).toBe(true);
    expect(rows).toHaveLength(Object.keys(EXPECTED_CONFORM).length);
  });

  it("the backfill repaired every newly-admitted value to itself, raw intact", async () => {
    const { rows } = await withEphemeral(c =>
      c.query<{ content_type: string; content_type_raw: string | null }>(
        `SELECT content_type, content_type_raw FROM documents
          WHERE collection = 'seed' AND content_type_raw = ANY($1)`,
        [[...EXPECTED_NEW]],
      ),
    );
    expect(rows).toHaveLength(EXPECTED_NEW.length);
    for (const r of rows) expect(r.content_type).toBe(r.content_type_raw);
  });

  it("the backfill LEAVES ALONE an unmappable unknown row (session-transcript)", async () => {
    const { rows } = await withEphemeral(c =>
      c.query<{ content_type: string; content_type_raw: string | null }>(
        "SELECT content_type, content_type_raw FROM documents WHERE path = 'transcript.md'",
      ),
    );
    expect(rows[0]).toEqual({ content_type: "unknown", content_type_raw: SEED_UNMAPPABLE });
  });

  it("the backfill does NOT overwrite a row that already carries a decided type", async () => {
    // path keep.md is content_type='note' with raw='memo'. 'memo' conforms to
    // 'deductive', but the WHERE content_type = 'unknown' guard must protect it.
    const { rows } = await withEphemeral(c =>
      c.query<{ content_type: string }>("SELECT content_type FROM documents WHERE path = 'keep.md'"),
    );
    expect(rows[0]!.content_type).toBe("note");
  });

  it("003 is re-runnable: a second apply changes nothing", async () => {
    const snapshot = () =>
      withEphemeral(c =>
        c.query<{ path: string; content_type: string; content_type_raw: string | null }>(
          `SELECT path, content_type, content_type_raw FROM documents ORDER BY path`,
        ),
      ).then(r => r.rows);
    const before = await snapshot();
    await withEphemeral(c => applyFile(c, MIGRATION_003));
    expect(await snapshot()).toEqual(before);
    // ...and the constraint is still exactly one row in pg_constraint.
    const { rows } = await withEphemeral(c =>
      c.query("SELECT 1 FROM pg_constraint WHERE conname = 'documents_content_type_check'"),
    );
    expect(rows).toHaveLength(1);
  });

  // ---- the write path, end to end ---------------------------------------

  it("upsertDocument lands each newly-admitted value as itself with a NULL raw", async () => {
    for (const v of EXPECTED_NEW) {
      await upsertDocument({
        collection: "wp", path: `new-${v}.md`, title: v, hash: `hw_${v}`, body: "#",
        contentTypeRaw: v,
      });
    }
    const { rows } = await withEphemeral(c =>
      c.query<{ path: string; content_type: string; content_type_raw: string | null }>(
        "SELECT path, content_type, content_type_raw FROM documents WHERE collection = 'wp' ORDER BY path",
      ),
    );
    expect(rows).toEqual(
      [...EXPECTED_NEW].sort().map(v => ({
        path: `new-${v}.md`, content_type: v, content_type_raw: null,
      })),
    );
  });

  it("upsertDocument CONFORMS each of the ten synonyms and keeps the original raw", async () => {
    for (const k of Object.keys(EXPECTED_CONFORM)) {
      await upsertDocument({
        collection: "cf", path: `cf-${k}.md`, title: k, hash: `hc_${k}`, body: "#",
        contentTypeRaw: k,
      });
    }
    const { rows } = await withEphemeral(c =>
      c.query<{ content_type: string; content_type_raw: string | null }>(
        "SELECT content_type, content_type_raw FROM documents WHERE collection = 'cf'",
      ),
    );
    expect(Object.fromEntries(rows.map(r => [r.content_type_raw, r.content_type])))
      .toEqual(EXPECTED_CONFORM);
  });

  it("upsertDocument still sinks session-transcript and junk to unknown with raw kept", async () => {
    for (const v of [SEED_UNMAPPABLE, "totally-invented-nonsense-9f2"]) {
      await upsertDocument({
        collection: "sink", path: `sink-${v}.md`, title: v, hash: `hs_${v}`, body: "#",
        contentTypeRaw: v,
      });
    }
    const { rows } = await withEphemeral(c =>
      c.query<{ content_type: string; content_type_raw: string | null }>(
        "SELECT content_type, content_type_raw FROM documents WHERE collection = 'sink' ORDER BY content_type_raw",
      ),
    );
    expect(rows).toEqual([
      { content_type: "unknown", content_type_raw: SEED_UNMAPPABLE },
      { content_type: "unknown", content_type_raw: "totally-invented-nonsense-9f2" },
    ]);
  });
});
