/**
 * Idempotent migration runner for the clawmem PG schema (master-harness-vn4rz.7).
 *
 * Contract:
 *  - Migrations are numbered `NNN_slug.sql` files in migrations/, applied in
 *    lexicographic order.
 *  - `schema_migrations` RECORDS what was applied. Idempotence is not re-inferred
 *    from the catalog on every run; it is read from that table.
 *  - Each file is ALSO written to be independently idempotent (IF NOT EXISTS /
 *    OR REPLACE), so a half-recorded state converges rather than wedging.
 *  - Running the apply twice in a row succeeds both times with no drift. There is
 *    a test that runs it twice and diffs the resulting catalog.
 *  - A file declaring `-- clawmem:no-transaction` runs OUTSIDE a transaction
 *    (required for CREATE INDEX CONCURRENTLY).
 *  - After a CONCURRENTLY build, pg_index.indisvalid is checked. An INVALID index
 *    is silently ignored by the planner — a green migration that leaves one behind
 *    would degrade every vector query to a brute-force scan with no error.
 */

import { createHash } from "crypto";
import { readdirSync, readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import type { PoolClient } from "pg";
import { embedDim, pgSchema } from "./config.ts";
import { withClient } from "./client.ts";
import type { Vault } from "./vaults.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
export const MIGRATIONS_DIR = join(HERE, "..", "..", "migrations");

export interface MigrationFile {
  version: string;
  path: string;
  sql: string;
  checksum: string;
  noTransaction: boolean;
}

export interface ApplyResult {
  applied: string[];
  skipped: string[];
}

/**
 * Substitute the migration parameters into one migration file's raw text.
 *
 * Exported because the integration suites apply the migration files directly
 * against a throwaway schema, and a second, hand-maintained copy of this
 * substitution in each of those suites is exactly how migration 007's
 * `:CLAWMEM_SCHEMA` reached the server unsubstituted the first time
 * (master-harness-vn4rz.46). One substitution, one home.
 *
 * `schema` defaults to the configured target schema and `dim` to the
 * configured embedding dimension; callers that apply the files to a schema
 * they created themselves (before `setPgSchema` is in effect), or with a
 * fixture dimension, MUST pass those explicitly.
 */
export function substituteMigrationParams(
  raw: string,
  schema: string = pgSchema() ?? "public",
  dim: number = embedDim(),
): string {
  return raw
    .replaceAll(":EMBED_DIM", String(dim))
    .replaceAll(":CLAWMEM_SCHEMA", `"${schema}"`);
}

/** Read + parameter-substitute every migration file, in version order. */
export function loadMigrations(dir: string = MIGRATIONS_DIR): MigrationFile[] {
  // `:CLAWMEM_SCHEMA` is the deployment's OWN schema (master-harness-vn4rz.46).
  // Migration 005 hardened the FTS trigger functions against CVE-2018-1058 by
  // pinning `search_path` and hard-qualifying `public.content` in the body —
  // correct for a deployment that lives in `public`, but it pinned the
  // FUNCTIONS to the literal name `public`, so a deployment pointed at another
  // schema (including the integration suite's throwaway one, via setPgSchema)
  // resolved the body's table in the WRONG schema, found nothing, coalesced to
  // '' and silently stored a title-only tsvector. Interpolating the target
  // schema keeps the hardening — search_path is still pinned, just to the
  // schema the connection actually uses — without pinning the deployment to
  // one schema NAME.
  const schema = pgSchema() ?? "public";
  const dim = embedDim();
  return readdirSync(dir)
    .filter(f => f.endsWith(".sql"))
    .sort()
    .map(f => {
      const path = join(dir, f);
      const raw = readFileSync(path, "utf-8");
      // The only two substitutions, both sourced from the single named
      // accessors in config.ts so each has exactly one authoritative home:
      // `:EMBED_DIM` (embedding dimension) and `:CLAWMEM_SCHEMA` (target
      // schema, quoted as an identifier).
      const sql = substituteMigrationParams(raw, schema, dim);
      return {
        version: f.replace(/\.sql$/, ""),
        path,
        sql,
        // Checksum the SUBSTITUTED text: changing EMBED_DIM or the target
        // schema is a schema change, and the recorded checksum must notice it.
        checksum: createHash("sha256").update(sql, "utf-8").digest("hex").slice(0, 16),
        noTransaction: /^--\s*clawmem:no-transaction\s*$/m.test(raw),
      };
    });
}

async function ensureTrackingTable(c: PoolClient): Promise<void> {
  await c.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      version    text        PRIMARY KEY,
      applied_at timestamptz NOT NULL DEFAULT now(),
      checksum   text        NOT NULL
    )
  `);
}

/**
 * Refuse to record a CONCURRENTLY-built index that came back INVALID. Postgres
 * leaves such an index in place and the planner ignores it — no error, no plan,
 * just a silent brute-force scan forever.
 */
async function assertNoInvalidIndexes(c: PoolClient): Promise<void> {
  const { rows } = await c.query<{ relname: string }>(`
    SELECT ci.relname
      FROM pg_index i
      JOIN pg_class ci ON ci.oid = i.indexrelid
      JOIN pg_class ct ON ct.oid = i.indrelid
      JOIN pg_namespace n ON n.oid = ct.relnamespace
     WHERE n.nspname = current_schema()
       AND NOT i.indisvalid
  `);
  if (rows.length > 0) {
    throw new Error(
      `Migration left INVALID index(es): ${rows.map(r => r.relname).join(", ")}. ` +
      `CREATE INDEX CONCURRENTLY failed. The planner silently ignores an invalid ` +
      `index, so this is NOT safe to record as applied. Drop it and re-run: ` +
      rows.map(r => `DROP INDEX ${r.relname};`).join(" "),
    );
  }
}

/**
 * Apply every pending migration to ONE vault's database.
 *
 * `vault` is explicit (master-harness-0ynkd) because provisioning the nsfw
 * vault is a first-class operation, not an accident of which env var happened
 * to be exported: `bun src/pg/cli.ts migrate --vault nsfw` reads the
 * CLAWMEM_PG_NSFW_* namespace and applies the SAME migration files, which is
 * what makes the two schemas comparable rather than merely similar.
 */
export async function applyMigrations(
  dir: string = MIGRATIONS_DIR,
  vault: Vault = "sfw",
): Promise<ApplyResult> {
  const files = loadMigrations(dir);
  const applied: string[] = [];
  const skipped: string[] = [];

  await withClient(vault, async c => {
    await ensureTrackingTable(c);
    const { rows } = await c.query<{ version: string; checksum: string }>(
      "SELECT version, checksum FROM schema_migrations",
    );
    const done = new Map(rows.map(r => [r.version, r.checksum]));

    for (const m of files) {
      const prior = done.get(m.version);
      if (prior !== undefined) {
        if (prior !== m.checksum) {
          throw new Error(
            `Migration ${m.version} was applied with checksum ${prior} but the file on ` +
            `disk now checksums ${m.checksum}. An applied migration was edited in place. ` +
            `Add a new numbered migration instead — re-applying an edited one is how a ` +
            `schema silently diverges between environments.`,
          );
        }
        skipped.push(m.version);
        continue;
      }

      if (m.noTransaction) {
        await c.query(m.sql);
        await assertNoInvalidIndexes(c);
        await c.query(
          "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)",
          [m.version, m.checksum],
        );
      } else {
        await c.query("BEGIN");
        try {
          await c.query(m.sql);
          await c.query(
            "INSERT INTO schema_migrations (version, checksum) VALUES ($1, $2)",
            [m.version, m.checksum],
          );
          await c.query("COMMIT");
        } catch (e) {
          await c.query("ROLLBACK");
          throw e;
        }
      }
      applied.push(m.version);
    }
  });

  return { applied, skipped };
}
