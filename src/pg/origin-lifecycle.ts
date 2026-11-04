/** Synchronize SQLite lifecycle decisions independently of file watermarks. */
import { Database } from "bun:sqlite";
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { withTransaction } from "./client.ts";
import type { Vault } from "./vaults.ts";

export type LifecycleRow = { path: string; title: string | null; active: boolean; archived_at: string | null };

export function defaultSqlitePath(): string {
  if (process.env.INDEX_PATH) return process.env.INDEX_PATH;
  return resolve(process.env.XDG_CACHE_HOME || resolve(homedir(), ".cache"), "clawmem/index.sqlite");
}

export function readOriginLifecycle(sqlitePath: string, collection: string): LifecycleRow[] {
  if (!existsSync(sqlitePath)) throw new Error(`SQLite vault does not exist: ${sqlitePath}`);
  const db = new Database(sqlitePath, { readonly: true });
  try {
    const rows = db.query(`SELECT path, title, active, archived_at FROM documents WHERE collection = ? ORDER BY path`)
      .all(collection) as { path: string; title: string | null; active: number; archived_at: string | null }[];
    return rows.map(r => ({ path: r.path, title: r.title, active: r.active === 1, archived_at: r.archived_at }));
  } finally {
    db.close();
  }
}

export async function syncOriginLifecycle(
  collection: string, sqlitePath: string, vault: Vault = "sfw",
): Promise<{ sourceRows: number; changedRows: number }> {
  const rows = readOriginLifecycle(sqlitePath, collection);
  let changedRows = 0;
  // A bounded JSON batch avoids one round trip per transcript and one huge bind.
  // Which revision stays active: the one whose title matches SQLite, then the newest ingested.
  // Newest alone is wrong (vn4rz.148): a stale-raw load (vn4rz.114) wrote another file's text and
  // title under this path, later than the correct row. Title is the comparison because the two
  // tiers' hashes never match: SQLite hashes the body, origin hashes the raw file.
  for (let i = 0; i < rows.length; i += 1000) {
    const batch = rows.slice(i, i + 1000);
    changedRows += await withTransaction(vault, async c => {
      const r = await c.query(
        `WITH source AS (
           SELECT * FROM jsonb_to_recordset($2::jsonb)
             AS s(path text, title text, active boolean, archived_at timestamptz)
         ), ranked AS (
           SELECT o.id, s.active, s.archived_at,
                  row_number() OVER (
                    PARTITION BY o.path
                    ORDER BY (o.title IS NOT DISTINCT FROM s.title) DESC, o.ingested_at DESC, o.id DESC
                  ) AS revision
           FROM origin_documents AS o
           JOIN source AS s ON s.path = o.path
           WHERE o.collection = $1
         )
         UPDATE origin_documents AS o
         SET active = (r.active AND r.revision = 1), archived_at = r.archived_at
         FROM ranked AS r
         WHERE o.id = r.id
           AND (o.active, o.archived_at)
             IS DISTINCT FROM ((r.active AND r.revision = 1), r.archived_at)`,
        [collection, JSON.stringify(batch)],
      );
      return r.rowCount ?? 0;
    });
  }
  return { sourceRows: rows.length, changedRows };
}
