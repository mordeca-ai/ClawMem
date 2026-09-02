/**
 * Vault isolation integration tests (master-harness-0ynkd, ADR-0162 §1).
 *
 * Runs against the LIVE clawmem-pg cluster (127.0.0.1:5433) using TWO EPHEMERAL
 * DATABASES created and dropped per run — never the live `clawmem` /
 * `clawmem_nsfw` vaults. Both sides of every comparison are fixtures we own,
 * which is what makes "zero private rows in the sfw vault" a statement about
 * this code rather than about whatever the reindex happens to have written.
 *
 * SKIPS WITH A REASON when the substrate is unreachable (no superuser URL, no
 * container) — never silently passes. A skip that should have been a run is a
 * false green, so the reason is printed.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { readFileSync, readdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { randomBytes } from "crypto";
import { MIGRATIONS_DIR } from "../../src/pg/migrate.ts";
import { closePool, withClient } from "../../src/pg/client.ts";
import { resolvePgConfig, setPgSchema } from "../../src/pg/config.ts";
import { resetVaultCache } from "../../src/pg/vaults.ts";
import {
  assertVaultDatabase,
  insertEmbeddingsBatch,
  upsertDocument,
} from "../../src/pg/write.ts";
import {
  PgWrongDatabaseError,
  PrivateContentRoutingError,
  VaultDatabaseCollisionError,
  VaultNotConfiguredError,
} from "../../src/pg/errors.ts";

const REPO = "/home/bj/claude/master-harness";
const DIM = 768;
const MODEL = "embeddinggemma";

/**
 * Superuser URL: explicit env wins, else assembled from the repo's gitignored
 * password file. Absent = unreachable substrate = skip with a reason.
 */
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
    "[SKIP] pg-vault-isolation: no superuser connection " +
    "(CLAWMEM_PG_SUPERUSER_URL unset and .secrets/local/clawmem-pg/superuser.pw unreadable) — " +
    "ephemeral vault databases cannot be created.",
  );
}
const d = SU ? describe : describe.skip;

const suffix = `${process.pid}_${randomBytes(4).toString("hex")}`;
const SFW_DB = `clawmem_vaulttest_${suffix}`;
const NSFW_DB = `clawmem_vaulttest_nsfw_${suffix}`;

const saved: Record<string, string | undefined> = {};
const ENV_KEYS = [
  "CLAWMEM_CONFIG_DIR", "CLAWMEM_PG_URL", "CLAWMEM_PG_NSFW_URL",
  "CLAWMEM_PG_HOST", "CLAWMEM_PG_DATABASE", "CLAWMEM_PG_USER", "CLAWMEM_PG_PASSWORD",
];

function setEnv(k: string, v: string | undefined): void {
  if (v === undefined) delete process.env[k];
  else process.env[k] = v;
}

d("PG vault isolation", () => {
  let admin: pg.Pool;
  let configDir: string;

  function dbUrl(db: string): string {
    const u = new URL(SU!);
    u.pathname = `/${db}`;
    return u.toString();
  }

  async function migrateInto(db: string): Promise<void> {
    const p = new pg.Pool({ connectionString: dbUrl(db) });
    try {
      const c = await p.connect();
      try {
        // The extensions are bootstrap's job in production
        // (infra/clawmem-pg/bootstrap/1?-*.sql), not the migrations' — a fresh
        // ephemeral database has neither, so the fixture stands them up itself.
        await c.query("CREATE EXTENSION IF NOT EXISTS vector");
        await c.query("CREATE EXTENSION IF NOT EXISTS pg_trgm");
        for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
          const sql = readFileSync(join(MIGRATIONS_DIR, f), "utf-8")
            .replaceAll(":EMBED_DIM", String(DIM));
          await c.query(sql);
        }
      } finally {
        c.release();
      }
    } finally {
      await p.end();
    }
  }

  async function count(db: string, sql: string, params: unknown[] = []): Promise<number> {
    const p = new pg.Pool({ connectionString: dbUrl(db) });
    try {
      const { rows } = await p.query<{ n: string }>(sql, params as never[]);
      return Number(rows[0]?.n ?? -1);
    } finally {
      await p.end();
    }
  }

  beforeAll(async () => {
    for (const k of ENV_KEYS) saved[k] = process.env[k];
    setPgSchema(null);

    admin = new pg.Pool({ connectionString: SU! });
    await admin.query(`CREATE DATABASE ${SFW_DB}`);
    await admin.query(`CREATE DATABASE ${NSFW_DB}`);
    await migrateInto(SFW_DB);
    await migrateInto(NSFW_DB);

    // A real config file: bj-corpus declared nsfw, docs left undeclared (sfw).
    configDir = mkdtempSync(join(tmpdir(), "clawmem-vaultint-"));
    writeFileSync(
      join(configDir, "index.yml"),
      `collections:\n` +
      `  bj-corpus:\n    path: ${REPO}\n    pattern: "intelligence/personal/bj/approved/**/*.md"\n    vault: nsfw\n` +
      `  docs:\n    path: ${REPO}/library/reference/documentation\n    pattern: "**/*.md"\n`,
      "utf-8",
    );
    setEnv("CLAWMEM_CONFIG_DIR", configDir);
    for (const k of ["CLAWMEM_PG_HOST", "CLAWMEM_PG_DATABASE", "CLAWMEM_PG_USER", "CLAWMEM_PG_PASSWORD"]) {
      setEnv(k, undefined);
    }
    setEnv("CLAWMEM_PG_URL", dbUrl(SFW_DB));
    setEnv("CLAWMEM_PG_NSFW_URL", dbUrl(NSFW_DB));
    resetVaultCache();
  });

  afterAll(async () => {
    await closePool();
    for (const k of ENV_KEYS) setEnv(k, saved[k]);
    resetVaultCache();
    if (configDir) rmSync(configDir, { recursive: true, force: true });
    if (admin) {
      for (const db of [SFW_DB, NSFW_DB]) {
        await admin.query(
          `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1`, [db],
        );
        await admin.query(`DROP DATABASE IF EXISTS ${db}`);
      }
      await admin.end();
    }
  });

  // =========================================================================
  // The core assertion: private content lands in the nsfw database ONLY
  // =========================================================================

  it("routes a bj-corpus document into the NSFW database and NOT the sfw one", async () => {
    await upsertDocument({
      collection: "bj-corpus",
      path: "intelligence/personal/bj/approved/probe.md",
      title: "probe", hash: "h_priv", body: "# probe",
    });
    await closePool(); // release the pooled connections before counting

    expect(await count(NSFW_DB, "SELECT count(*)::text AS n FROM documents WHERE collection = $1", ["bj-corpus"])).toBe(1);
    // THE probe this whole bead exists for, expressed against fixtures we own.
    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM documents WHERE collection = $1", ["bj-corpus"])).toBe(0);
    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM documents WHERE path LIKE 'intelligence/personal/bj/%'")).toBe(0);
  });

  it("routes an ordinary collection into the SFW database and NOT the nsfw one", async () => {
    await upsertDocument({
      collection: "docs", path: "blazor/overview.md",
      title: "overview", hash: "h_pub", body: "# overview",
    });
    await closePool();

    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM documents WHERE collection = $1", ["docs"])).toBe(1);
    expect(await count(NSFW_DB, "SELECT count(*)::text AS n FROM documents WHERE collection = $1", ["docs"])).toBe(0);
  });

  it("routes a bj-corpus fragment's VECTORS to the nsfw database too", async () => {
    await insertEmbeddingsBatch([{
      collection: "bj-corpus",
      path: "intelligence/personal/bj/approved/probe.md",
      hash: "h_priv", seq: 0, pos: 0,
      embedding: new Array(DIM).fill(0.25), model: MODEL,
    }]);
    await closePool();

    expect(await count(NSFW_DB, "SELECT count(*)::text AS n FROM content_vectors WHERE hash = $1", ["h_priv"])).toBe(1);
    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM content_vectors")).toBe(0);
  });

  // =========================================================================
  // The belt: current_database() inside the transaction
  // =========================================================================

  it("assertVaultDatabase PASSES on a correctly-wired connection", async () => {
    const db = await withClient("sfw", c => assertVaultDatabase(c, "sfw"));
    expect(db).toBe(SFW_DB);
    const ndb = await withClient("nsfw", c => assertVaultDatabase(c, "nsfw"));
    expect(ndb).toBe(NSFW_DB);
  });

  it("assertVaultDatabase THROWS naming BOTH databases on a mis-wired connection", async () => {
    // The proven-RED half: a real client attached to the SFW database, asked to
    // vouch for the nsfw vault. This is what a copy-pasted connection string
    // looks like from inside the transaction.
    let err: unknown;
    try {
      await withClient("sfw", c => assertVaultDatabase(c, "nsfw"));
    } catch (e) { err = e; }
    expect(err).toBeInstanceOf(PgWrongDatabaseError);
    expect((err as Error).message).toContain(SFW_DB);
    expect((err as Error).message).toContain(NSFW_DB);
  });

  // =========================================================================
  // Refusals, proven against the real substrate
  // =========================================================================

  it("REFUSES both vaults pointing at one database (the belt cannot see this one)", () => {
    expect(() => {
      // Same database for both — the boundary silently removed.
      const env = {
        CLAWMEM_PG_URL: dbUrl(SFW_DB),
        CLAWMEM_PG_NSFW_URL: dbUrl(SFW_DB),
      } as NodeJS.ProcessEnv;
      resolvePgConfig("nsfw", env);
    }).toThrow(VaultDatabaseCollisionError);
  });

  it("REFUSES a routed-nsfw write when the nsfw connection is absent — nothing lands in sfw", async () => {
    await closePool();
    const keep = process.env.CLAWMEM_PG_NSFW_URL;
    delete process.env.CLAWMEM_PG_NSFW_URL;
    try {
      await expect(upsertDocument({
        collection: "bj-corpus",
        path: "intelligence/personal/bj/approved/nofallback.md",
        title: "nf", hash: "h_nofb", body: "# nf",
      })).rejects.toBeInstanceOf(VaultNotConfiguredError);
    } finally {
      process.env.CLAWMEM_PG_NSFW_URL = keep;
      await closePool();
    }
    // The load-bearing half: the refusal did not become a write somewhere else.
    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM documents WHERE hash = $1", ["h_nofb"])).toBe(0);
    expect(await count(NSFW_DB, "SELECT count(*)::text AS n FROM documents WHERE hash = $1", ["h_nofb"])).toBe(0);
  });

  it("REFUSES a private document whose collection is mis-declared sfw — nothing lands anywhere", async () => {
    await closePool();
    const other = mkdtempSync(join(tmpdir(), "clawmem-vaultint-bad-"));
    writeFileSync(
      join(other, "index.yml"),
      `collections:\n  bj-corpus:\n    path: ${REPO}\n    pattern: "**/*.md"\n    vault: sfw\n`,
      "utf-8",
    );
    const keep = process.env.CLAWMEM_CONFIG_DIR;
    process.env.CLAWMEM_CONFIG_DIR = other;
    resetVaultCache();
    try {
      await expect(upsertDocument({
        collection: "bj-corpus",
        path: "intelligence/personal/bj/approved/misdeclared.md",
        title: "md", hash: "h_misdecl", body: "# md",
      })).rejects.toBeInstanceOf(PrivateContentRoutingError);
    } finally {
      setEnv("CLAWMEM_CONFIG_DIR", keep);
      resetVaultCache();
      rmSync(other, { recursive: true, force: true });
      await closePool();
    }
    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM documents WHERE hash = $1", ["h_misdecl"])).toBe(0);
    expect(await count(NSFW_DB, "SELECT count(*)::text AS n FROM documents WHERE hash = $1", ["h_misdecl"])).toBe(0);
  });
});
