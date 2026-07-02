/**
 * Unit tests for src/backup.ts (master-harness-t5i0, fix #3).
 *
 * Prior to this module there was no snapshot mechanism for the ClawMem SQLite
 * index at all — the incident that motivated this hardening pass had no
 * recovery path because nothing had ever backed up the DB. These tests cover:
 *  - a backup produces an openable, non-empty SQLite file with the source data
 *  - default dest dir honors CLAWMEM_BACKUP_DIR
 *  - retention prunes down to N, oldest first
 *  - same-second collisions get a disambiguating suffix instead of clobbering
 *  - retentionCount validation
 */
import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdtempSync, rmSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createBackup,
  listBackups,
  pruneBackups,
  getDefaultBackupDir,
  DEFAULT_BACKUP_RETENTION,
} from "../../src/backup.ts";

function makeSourceDb(): Database {
  const db = new Database(":memory:");
  db.exec("CREATE TABLE t (a INTEGER, b TEXT)");
  db.exec("INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three')");
  return db;
}

describe("createBackup", () => {
  let dir: string;
  let db: Database;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "clawmem-backup-test-"));
    db = makeSourceDb();
  });

  afterEach(() => {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("produces an openable SQLite file containing the source data", () => {
    const result = createBackup(db, { destDir: dir });

    expect(existsSync(result.path)).toBe(true);
    expect(result.sizeBytes).toBeGreaterThan(0);

    const backupDb = new Database(result.path, { readonly: true });
    const rows = backupDb.prepare("SELECT a, b FROM t ORDER BY a").all() as { a: number; b: string }[];
    expect(rows).toEqual([
      { a: 1, b: "one" },
      { a: 2, b: "two" },
      { a: 3, b: "three" },
    ]);
    backupDb.close();
  });

  it("creates the destination directory if it does not exist", () => {
    const nested = join(dir, "nested", "backups");
    expect(existsSync(nested)).toBe(false);

    const result = createBackup(db, { destDir: nested });

    expect(existsSync(nested)).toBe(true);
    expect(existsSync(result.path)).toBe(true);
  });

  it("names backups with a sortable timestamp under index-*.sqlite", () => {
    const now = new Date("2026-07-02T06:53:12.000Z");
    const result = createBackup(db, { destDir: dir, now: () => now });
    expect(result.path).toMatch(/index-20260702T065312\.sqlite$/);
  });

  it("disambiguates same-second collisions instead of clobbering an existing backup", () => {
    const fixedNow = new Date("2026-07-02T06:53:12.000Z");
    const first = createBackup(db, { destDir: dir, now: () => fixedNow, retentionCount: 10 });
    const second = createBackup(db, { destDir: dir, now: () => fixedNow, retentionCount: 10 });

    expect(first.path).not.toBe(second.path);
    expect(existsSync(first.path)).toBe(true);
    expect(existsSync(second.path)).toBe(true);
  });

  it("retention prunes down to N, keeping the newest and removing the oldest", () => {
    const timestamps = [
      new Date("2026-07-01T00:00:00.000Z"),
      new Date("2026-07-02T00:00:00.000Z"),
      new Date("2026-07-03T00:00:00.000Z"),
      new Date("2026-07-04T00:00:00.000Z"),
    ];

    let lastResult;
    for (const ts of timestamps) {
      lastResult = createBackup(db, { destDir: dir, now: () => ts, retentionCount: 2 });
    }

    // After 4 backups with retentionCount=2, only the 2 newest should remain.
    const remaining = listBackups(dir);
    expect(remaining).toHaveLength(2);
    expect(remaining[0]).toMatch(/20260704/); // newest first
    expect(remaining[1]).toMatch(/20260703/);
    // The 4th backup pushes the count to 3; retention=2 prunes the oldest (07-01 was
    // already pruned by the 3rd backup, so this prunes 07-02).
    expect(lastResult!.prunedFiles).toHaveLength(1);
    expect(lastResult!.prunedFiles[0]).toMatch(/20260702/);

    // Sanity: files on disk match listBackups
    const onDisk = readdirSync(dir).filter(f => f.startsWith("index-"));
    expect(onDisk).toHaveLength(2);
  });

  it("prunedFiles reports the exact backups removed by retention", () => {
    createBackup(db, { destDir: dir, now: () => new Date("2026-07-01T00:00:00.000Z"), retentionCount: 1 });
    const second = createBackup(db, { destDir: dir, now: () => new Date("2026-07-02T00:00:00.000Z"), retentionCount: 1 });

    expect(second.prunedFiles).toHaveLength(1);
    expect(second.prunedFiles[0]).toMatch(/20260701/);
    expect(existsSync(second.prunedFiles[0]!)).toBe(false);
  });

  it("defaults retention to DEFAULT_BACKUP_RETENTION (7) when unspecified", () => {
    for (let i = 1; i <= 9; i++) {
      createBackup(db, { destDir: dir, now: () => new Date(`2026-07-0${i}T00:00:00.000Z`) });
    }
    expect(listBackups(dir)).toHaveLength(DEFAULT_BACKUP_RETENTION);
  });

  it("throws on a non-positive retentionCount", () => {
    expect(() => createBackup(db, { destDir: dir, retentionCount: 0 })).toThrow();
    expect(() => createBackup(db, { destDir: dir, retentionCount: -1 })).toThrow();
  });
});

describe("pruneBackups", () => {
  it("returns an empty array for a directory with no backups", () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-backup-empty-"));
    try {
      expect(pruneBackups(dir, 5)).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("getDefaultBackupDir", () => {
  const original = Bun.env.CLAWMEM_BACKUP_DIR;

  afterEach(() => {
    if (original === undefined) delete Bun.env.CLAWMEM_BACKUP_DIR;
    else Bun.env.CLAWMEM_BACKUP_DIR = original;
  });

  it("honors CLAWMEM_BACKUP_DIR override", () => {
    Bun.env.CLAWMEM_BACKUP_DIR = "/tmp/custom-clawmem-backups";
    expect(getDefaultBackupDir()).toBe("/tmp/custom-clawmem-backups");
  });

  it("defaults to <cache>/clawmem/backups when unset", () => {
    delete Bun.env.CLAWMEM_BACKUP_DIR;
    const dir = getDefaultBackupDir();
    expect(dir.endsWith(join("clawmem", "backups"))).toBe(true);
  });
});
