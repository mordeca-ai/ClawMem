/**
 * Origin-load write read-back integration tests (master-harness-vn4rz.109).
 *
 * Uses an EPHEMERAL scratch database created and dropped per run.
 * Follows the pattern from tests/integration/pg-vault-isolation.test.ts:
 * never connects to live `clawmem` / `clawmem_nsfw` production databases.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import {
  assertNotProductionDatabase,
  PG_TEST_SETUP_TIMEOUT_MS,
  PG_TEST_TEARDOWN_TIMEOUT_MS,
} from "./pg-test-schema.ts";
import { readFileSync, readdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { randomBytes } from "crypto";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { resetVaultCache } from "../../src/pg/vaults.ts";
import { hashContent } from "../../src/indexer.ts";
import { loadOriginCollection, OriginWriteReadBackError } from "../../src/pg/origin.ts";

const REPO = "/home/bj/claude/master-harness";
const DIM = 768;

function superuserUrl(): string | null {
  const explicit = process.env.CLAWMEM_PG_SUPERUSER_URL?.trim();
  if (explicit) return explicit;
  try {
    const pw = readFileSync(`${REPO}/.secrets/local/clawmem-pg/superuser.pw`, "utf-8").trim();
    if (!pw) return null;
    return `postgres://postgres:${encodeURIComponent(pw)}@127.0.0.1:5433/postgres`;
  } catch {
    return null;
  }
}

const SU = superuserUrl();
if (!SU) {
  console.log(
    "[SKIP] pg-origin-readback: no superuser connection " +
      "(CLAWMEM_PG_SUPERUSER_URL unset and .secrets/local/clawmem-pg/superuser.pw unreadable) — " +
      "ephemeral scratch database cannot be created.",
  );
}
const d = SU ? describe : describe.skip;

const suffix = `${process.pid}_${randomBytes(4).toString("hex")}`;
const SCRATCH_DB = `clawmem_vaulttest_${suffix}`;

const saved: Record<string, string | undefined> = {};
const ENV_KEYS = [
  "CLAWMEM_CONFIG_DIR",
  "CLAWMEM_PG_URL",
  "CLAWMEM_PG_NSFW_URL",
  "CLAWMEM_PG_HOST",
  "CLAWMEM_PG_DATABASE",
  "CLAWMEM_PG_USER",
  "CLAWMEM_PG_PASSWORD",
];

function setEnv(k: string, v: string | undefined): void {
  if (v === undefined) delete process.env[k];
  else process.env[k] = v;
}

function dbUrl(db: string): string {
  const u = new URL(SU!);
  u.pathname = `/${db}`;
  return u.toString();
}

d("PG origin write read-back guard", () => {
  let admin: pg.Pool;
  let configDir: string;
  let fixtureDir: string;

  async function migrateInto(db: string): Promise<void> {
    const p = new pg.Pool({ connectionString: dbUrl(db) });
    try {
      const c = await p.connect();
      try {
        await assertNotProductionDatabase(c);
        await c.query("CREATE EXTENSION IF NOT EXISTS vector");
        await c.query("CREATE EXTENSION IF NOT EXISTS pg_trgm");
        for (const f of readdirSync(MIGRATIONS_DIR)
          .filter((f) => f.endsWith(".sql"))
          .sort()) {
          const sql = substituteMigrationParams(
            readFileSync(join(MIGRATIONS_DIR, f), "utf-8"),
            undefined,
            DIM,
          );
          await c.query(sql);
        }
      } finally {
        c.release();
      }
    } finally {
      await p.end();
    }
  }

  beforeAll(async () => {
    for (const k of ENV_KEYS) saved[k] = process.env[k];
    setPgSchema(null);

    admin = new pg.Pool({ connectionString: SU! });
    await admin.query(`CREATE DATABASE ${SCRATCH_DB}`);
    await migrateInto(SCRATCH_DB);

    configDir = mkdtempSync(join(tmpdir(), "clawmem-orig-cfg-"));
    fixtureDir = mkdtempSync(join(tmpdir(), "clawmem-orig-fix-"));

    writeFileSync(
      join(configDir, "index.yml"),
      `collections:\n` +
        `  origin-rb-test:\n    path: ${fixtureDir}\n    pattern: "**/*.md"\n` +
        `  origin-rb-fresh:\n    path: ${fixtureDir}\n    pattern: "**/*.md"\n`,
      "utf-8",
    );
    setEnv("CLAWMEM_CONFIG_DIR", configDir);
    for (const k of [
      "CLAWMEM_PG_HOST",
      "CLAWMEM_PG_DATABASE",
      "CLAWMEM_PG_USER",
      "CLAWMEM_PG_PASSWORD",
    ]) {
      setEnv(k, undefined);
    }
    setEnv("CLAWMEM_PG_URL", dbUrl(SCRATCH_DB));
    resetVaultCache();

    // 3 small origin records with frontmatter (ts, session_id, content_type) and distinct bodies
    writeFileSync(
      join(fixtureDir, "session_001.md"),
      `---\nts: "2026-03-01T10:00:00Z"\nsession_id: "sess-001"\ncontent_type: "session-transcript"\ntitle: "First session record"\n---\nFirst transcript content body here.\n`,
      "utf-8",
    );
    writeFileSync(
      join(fixtureDir, "session_002.md"),
      `---\nts: "2026-03-02T11:00:00Z"\nsession_id: "sess-002"\ncontent_type: "session-transcript"\ntitle: "Second session record"\n---\nSecond transcript content body here.\n`,
      "utf-8",
    );
    writeFileSync(
      join(fixtureDir, "session_003.md"),
      `---\nts: "2026-04-01T12:00:00Z"\nsession_id: "sess-003"\ncontent_type: "session-transcript"\ntitle: "Third session record"\n---\nThird transcript content body here.\n`,
      "utf-8",
    );
  }, PG_TEST_SETUP_TIMEOUT_MS);

  afterAll(async () => {
    await closePool();
    for (const k of ENV_KEYS) setEnv(k, saved[k]);
    resetVaultCache();
    if (configDir) rmSync(configDir, { recursive: true, force: true });
    if (fixtureDir) rmSync(fixtureDir, { recursive: true, force: true });
    if (admin) {
      await admin.query(
        `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1`,
        [SCRATCH_DB],
      );
      await admin.query(`DROP DATABASE IF EXISTS ${SCRATCH_DB}`);
      await admin.end();
    }
  }, PG_TEST_TEARDOWN_TIMEOUT_MS);

  it("POSITIVE: loadOriginCollection writes 3 rows; every row's hash == sha256 of its file; no throw", async () => {
    const stats = await loadOriginCollection({
      collection: "origin-rb-test",
      root: fixtureDir,
      pattern: "**/*.md",
      skipEmbed: true,
    });
    expect(stats.recordsWritten).toBe(3);

    const pool = new pg.Pool({ connectionString: dbUrl(SCRATCH_DB) });
    try {
      const { rows } = await pool.query<{ path: string; hash: string }>(
        "SELECT path, hash FROM origin_documents WHERE collection = $1 ORDER BY path",
        ["origin-rb-test"],
      );
      expect(rows.length).toBe(3);
      for (const row of rows) {
        const fileContent = readFileSync(join(fixtureDir, row.path), "utf-8");
        const expectedHash = hashContent(fileContent);
        expect(row.hash).toBe(expectedHash);
      }
    } finally {
      await pool.end();
    }
  });

  it("NEGATIVE / positive control: tamper trigger forces read-back rejection and roll back", async () => {
    const pool = new pg.Pool({ connectionString: dbUrl(SCRATCH_DB) });
    try {
      // Tamper trigger sets NEW.title := 'tampered'
      await pool.query(`
        CREATE OR REPLACE FUNCTION tamper_title_trigger() RETURNS trigger AS $$
        BEGIN
          NEW.title := 'tampered';
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
      `);

      let triggerAttached = false;
      try {
        await pool.query(`
          CREATE TRIGGER trg_tamper_title
          BEFORE INSERT OR UPDATE ON origin_documents
          FOR EACH ROW EXECUTE FUNCTION tamper_title_trigger();
        `);
        triggerAttached = true;
      } catch {
        // If PG refuses BEFORE row trigger on partitioned parent, put it on partitions
        const { rows: partRows } = await pool.query<{ relname: string }>(`
          SELECT c.relname
          FROM pg_inherits i
          JOIN pg_class c ON i.inhrelid = c.oid
          JOIN pg_class p ON i.inhparent = p.oid
          WHERE p.relname = 'origin_documents'
        `);
        for (const pr of partRows) {
          await pool.query(`
            CREATE TRIGGER trg_tamper_title
            BEFORE INSERT OR UPDATE ON ${pr.relname}
            FOR EACH ROW EXECUTE FUNCTION tamper_title_trigger();
          `);
        }
        triggerAttached = true;
      }
      expect(triggerAttached).toBe(true);

      // Reload into a FRESH collection name: loadOriginCollection must reject
      // with OriginWriteReadBackError naming 'title'
      let thrown: unknown = null;
      try {
        await loadOriginCollection({
          collection: "origin-rb-fresh",
          root: fixtureDir,
          pattern: "**/*.md",
          skipEmbed: true,
        });
      } catch (err) {
        thrown = err;
      }

      expect(thrown).toBeInstanceOf(OriginWriteReadBackError);
      const readBackErr = thrown as OriginWriteReadBackError;
      expect(readBackErr.message).toContain("title");
      expect(readBackErr.mismatches).toContain("title");

      // origin_documents must hold ZERO rows for that collection (rolled back)
      const { rows: countRows } = await pool.query<{ n: string }>(
        "SELECT count(*)::text AS n FROM origin_documents WHERE collection = $1",
        ["origin-rb-fresh"],
      );
      expect(Number(countRows[0]!.n)).toBe(0);

      // Drop the trigger
      try {
        await pool.query("DROP TRIGGER IF EXISTS trg_tamper_title ON origin_documents");
      } catch {}
      const { rows: partRows } = await pool.query<{ relname: string }>(`
        SELECT c.relname
        FROM pg_inherits i
        JOIN pg_class c ON i.inhrelid = c.oid
        JOIN pg_class p ON i.inhparent = p.oid
        WHERE p.relname = 'origin_documents'
      `);
      for (const pr of partRows) {
        await pool.query(`DROP TRIGGER IF EXISTS trg_tamper_title ON ${pr.relname}`);
      }
      await pool.query("DROP FUNCTION IF EXISTS tamper_title_trigger()");
    } finally {
      await pool.end();
    }
  });
});
