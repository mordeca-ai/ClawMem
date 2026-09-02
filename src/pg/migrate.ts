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
import { embedDim } from "./config.js";
import { withClient } from "./client.js";

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

/** Read + parameter-substitute every migration file, in version order. */
export function loadMigrations(dir: string = MIGRATIONS_DIR): MigrationFile[] {
  const dim = embedDim();
  return readdirSync(dir)
    .filter(f => f.endsWith(".sql"))
    .sort()
    .map(f => {
      const path = join(dir, f);
      const raw = readFileSync(path, "utf-8");
      // The ONLY substitution. `:EMBED_DIM` comes from the single named constant
      // in config.ts so the dimension has exactly one authoritative home.
      const sql = raw.replaceAll(":EMBED_DIM", String(dim));
      return {
        version: f.replace(/\.sql$/, ""),
        path,
        sql,
        // Checksum the SUBSTITUTED text: changing EMBED_DIM is a schema change,
        // and the recorded checksum must notice it.
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

export async function applyMigrations(dir: string = MIGRATIONS_DIR): Promise<ApplyResult> {
  const files = loadMigrations(dir);
  const applied: string[] = [];
  const skipped: string[] = [];

  await withClient(async c => {
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
