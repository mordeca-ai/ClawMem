// Regression coverage for Issue #13 (yoloshii/ClawMem#13):
// PRAGMA ordering race between concurrent Stop-hook subprocesses calling
// initializeDatabase() on the same SQLite file.
//
// Pre-fix, `PRAGMA busy_timeout = 15000` was set AFTER `PRAGMA journal_mode = WAL`,
// so the journal_mode statement could not benefit from the busy handler and
// concurrent openers returned SQLITE_BUSY immediately (default busy callback
// is NULL → SQLITE_BUSY returns without waiting). Post-fix, busy_timeout is
// the first statement on the connection in both initializeDatabase() (writable
// path) and createStore() readonly branch.
//
// Two layers of coverage:
//   1. Source-text assertion — deterministic, catches an accidental re-swap
//      of the PRAGMA order without runtime timing.
//   2. Subprocess concurrent disk init — spawns 3 short Bun processes that
//      each call createStore(path) on the SAME on-disk DB file. Asserts all
//      three succeed without SQLITE_BUSY. Mirrors the real production
//      scenario (separate `clawmem hook X` subprocesses) more faithfully than
//      an in-process Promise.all could (bun:sqlite db.exec is synchronous, so
//      in-process "concurrent" calls serialize on the JS event loop and do
//      not contend on the SQLite file lock). :memory: stores have no
//      file-system lock and cannot reproduce this bug at all — that's why
//      pre-existing tests/integration/store.test.ts missed this regression.

import { describe, test, expect } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const REPO_ROOT = resolve(import.meta.dir, "..", "..");
const STORE_PATH = resolve(REPO_ROOT, "src/store.ts");
const WORKER_PATH = resolve(REPO_ROOT, "tests/helpers/concurrent-init-worker.ts");

describe("Issue #13 — PRAGMA ordering protects against concurrent-init SQLITE_BUSY", () => {
  test("initializeDatabase sets busy_timeout BEFORE journal_mode in source (assertion gate)", () => {
    const source = readFileSync(STORE_PATH, "utf8");
    const funcStart = source.indexOf("function initializeDatabase(");
    expect(funcStart).toBeGreaterThanOrEqual(0);

    // Walk braces to find the matching close.
    let depth = 0;
    let funcEnd = -1;
    let started = false;
    for (let i = funcStart; i < source.length; i++) {
      const ch = source[i];
      if (ch === "{") {
        depth++;
        started = true;
      } else if (ch === "}") {
        depth--;
        if (started && depth === 0) {
          funcEnd = i;
          break;
        }
      }
    }
    expect(funcEnd).toBeGreaterThan(funcStart);

    const body = source.slice(funcStart, funcEnd);
    const busyTimeoutIdx = body.search(/PRAGMA\s+busy_timeout/i);
    const journalModeIdx = body.search(/PRAGMA\s+journal_mode/i);

    expect(busyTimeoutIdx).toBeGreaterThanOrEqual(0);
    expect(journalModeIdx).toBeGreaterThanOrEqual(0);
    expect(busyTimeoutIdx).toBeLessThan(journalModeIdx);
  });

  test("createStore readonly branch sets busy_timeout BEFORE journal_mode in source (assertion gate)", () => {
    const source = readFileSync(STORE_PATH, "utf8");

    // Anchor on the "// Readonly:" comment that marks the readonly branch.
    const branchStart = source.indexOf("// Readonly:");
    expect(branchStart).toBeGreaterThanOrEqual(0);

    // The readonly branch's 4 statements fit comfortably in ~800 chars after
    // the comment. Slice and check PRAGMA ordering within that window.
    const slice = source.slice(branchStart, branchStart + 800);
    const busyTimeoutIdx = slice.search(/PRAGMA\s+busy_timeout/i);
    const journalModeIdx = slice.search(/PRAGMA\s+journal_mode/i);

    expect(busyTimeoutIdx).toBeGreaterThanOrEqual(0);
    expect(journalModeIdx).toBeGreaterThanOrEqual(0);
    expect(busyTimeoutIdx).toBeLessThan(journalModeIdx);
  });

  test("3 concurrent createStore() subprocesses on the same DB file all succeed without SQLITE_BUSY", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-issue-13-"));
    const dbPath = join(dir, "test.sqlite");

    try {
      type Result = { code: number; stderr: string; stdout: string };

      const launchWorker = (): Promise<Result> => new Promise<Result>((resolveProc) => {
        // process.execPath is the Bun binary when this test is run via `bun test`.
        const p = spawn(process.execPath, ["run", WORKER_PATH, dbPath], {
          stdio: ["ignore", "pipe", "pipe"],
        });
        let stderr = "";
        let stdout = "";
        p.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
        p.stdout.on("data", (chunk) => { stdout += chunk.toString(); });
        p.on("close", (code) => resolveProc({ code: code ?? -1, stderr, stdout }));
      });

      // Launch all three in parallel — separate OS processes, separate SQLite
      // connections, contending on the file lock for PRAGMA journal_mode=WAL.
      const results = await Promise.all([launchWorker(), launchWorker(), launchWorker()]);

      for (const r of results) {
        if (r.code !== 0) {
          throw new Error(
            `Worker subprocess failed (exit=${r.code}). stderr=${r.stderr.trim()} stdout=${r.stdout.trim()}`,
          );
        }
        expect(r.code).toBe(0);
        expect(r.stderr.toLowerCase()).not.toContain("sqlite_busy");
        expect(r.stderr.toLowerCase()).not.toContain("database is locked");
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
