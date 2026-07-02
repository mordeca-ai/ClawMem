/**
 * ClawMem Backup — Periodic snapshot of the SQLite index (master-harness-t5i0).
 *
 * Prior to this module there was no backup/snapshot mechanism for the ClawMem
 * index at all — the incident that motivated this hardening pass (a blind
 * DB-wide cleanup hard-deleting 3566 pre-existing document rows) had no
 * recovery path because nothing had ever snapshotted the DB. `.backup()` is
 * not exposed on bun:sqlite's Database class; `VACUUM INTO <file>` is the
 * bun-native equivalent of `sqlite3 <db> ".backup <dest>"` — a single
 * transactional statement that produces a consistent, compacted copy of the
 * live database without requiring exclusive access or an external `sqlite3`
 * binary.
 *
 * Usage: `clawmem backup [--dest <dir>] [--keep <n>]` (see clawmem.ts).
 */

import { Database } from "bun:sqlite";
import { existsSync, mkdirSync, readdirSync, statSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

export const DEFAULT_BACKUP_RETENTION = 7;

// index-YYYYMMDDTHHMMSS.sqlite, with an optional -N disambiguator for
// same-second collisions (e.g. rapid test invocations).
const BACKUP_FILE_RE = /^index-\d{8}T\d{6}(-\d+)?\.sqlite$/;

/**
 * Resolve the default backup directory: CLAWMEM_BACKUP_DIR env override, else
 * <XDG_CACHE_HOME or ~/.cache>/clawmem/backups.
 */
export function getDefaultBackupDir(): string {
  const override = Bun.env.CLAWMEM_BACKUP_DIR;
  if (override) return override;
  const cacheDir = Bun.env.XDG_CACHE_HOME || join(homedir(), ".cache");
  return join(cacheDir, "clawmem", "backups");
}

function timestampSlug(d: Date): string {
  // 2026-07-02T06:53:12.345Z -> 20260702T065312
  return d.toISOString().replace(/[-:]/g, "").split(".")[0]!;
}

export interface BackupResult {
  /** Absolute path to the newly created backup file. */
  path: string;
  /** Size of the backup file in bytes. */
  sizeBytes: number;
  /** Absolute paths of older backups pruned by retention, oldest first. */
  prunedFiles: string[];
}

export interface BackupOptions {
  /** Destination directory. Defaults to getDefaultBackupDir(). */
  destDir?: string;
  /** Number of backups to retain (including the one just created). Default 7. Must be >= 1. */
  retentionCount?: number;
  /** Injectable clock for deterministic tests. */
  now?: () => Date;
}

/**
 * Create a consistent snapshot of `db` at `destDir/index-<timestamp>.sqlite`
 * via `VACUUM INTO`, then prune old backups down to `retentionCount`.
 */
export function createBackup(db: Database, opts: BackupOptions = {}): BackupResult {
  const destDir = opts.destDir || getDefaultBackupDir();
  const retentionCount = opts.retentionCount ?? DEFAULT_BACKUP_RETENTION;
  if (!Number.isInteger(retentionCount) || retentionCount < 1) {
    throw new Error(`retentionCount must be an integer >= 1 (got ${retentionCount})`);
  }

  if (!existsSync(destDir)) {
    mkdirSync(destDir, { recursive: true });
  }

  const slug = timestampSlug(opts.now?.() ?? new Date());
  let destPath = join(destDir, `index-${slug}.sqlite`);
  let disambiguator = 1;
  while (existsSync(destPath)) {
    destPath = join(destDir, `index-${slug}-${disambiguator}.sqlite`);
    disambiguator++;
  }

  db.prepare(`VACUUM INTO ?`).run(destPath);

  const sizeBytes = statSync(destPath).size;
  const prunedFiles = pruneBackups(destDir, retentionCount);

  return { path: destPath, sizeBytes, prunedFiles };
}

/**
 * List backup files in `destDir`, newest first. Filenames are
 * lexicographically sortable by their embedded timestamp.
 */
export function listBackups(destDir: string): string[] {
  if (!existsSync(destDir)) return [];
  return readdirSync(destDir)
    .filter(f => BACKUP_FILE_RE.test(f))
    .sort()
    .reverse();
}

/**
 * Delete backups beyond `retentionCount` (oldest first), keeping the
 * `retentionCount` newest. Returns the absolute paths of files removed.
 */
export function pruneBackups(destDir: string, retentionCount: number): string[] {
  const backups = listBackups(destDir); // newest first
  const toDelete = backups.slice(retentionCount); // everything past the retained window
  const removed: string[] = [];
  for (const f of toDelete) {
    const p = join(destDir, f);
    unlinkSync(p);
    removed.push(p);
  }
  return removed;
}
