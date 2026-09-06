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
// Follow-up (master-harness-3xw7n): PRAGMA ordering was necessary but NOT
// sufficient. `PRAGMA journal_mode = WAL` takes an exclusive lock on SQLite's
// deadlock-avoidance path, which does NOT invoke the busy handler, so
// busy_timeout cannot absorb the cold DELETE->WAL transition race. Both call
// sites now route through store.ts's setWalJournalMode(), which does read-first
// + bounded jittered SQLITE_BUSY retry. The source-text gates below were
// STRENGTHENED accordingly: they still assert busy_timeout precedes the WAL
// work at each call site, and additionally assert that (a) neither call site
// issues a raw `PRAGMA journal_mode = ...` write (i.e. cannot bypass the
// helper) and (b) inside the helper the read precedes the write and the
// bounded-retry branch is present.
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

/** Return the source text of the block starting at `signature` (brace-walked). */
function functionBody(source: string, signature: string): string {
  const start = source.indexOf(signature);
  expect(start).toBeGreaterThanOrEqual(0);
  let depth = 0;
  let end = -1;
  let started = false;
  for (let i = start; i < source.length; i++) {
    const ch = source[i];
    if (ch === "{") {
      depth++;
      started = true;
    } else if (ch === "}") {
      depth--;
      if (started && depth === 0) {
        end = i;
        break;
      }
    }
  }
  expect(end).toBeGreaterThan(start);
  return source.slice(start, end);
}

describe("Issue #13 — PRAGMA ordering protects against concurrent-init SQLITE_BUSY", () => {
  test("initializeDatabase sets busy_timeout BEFORE the WAL work, and cannot bypass the helper (assertion gate)", () => {
    const source = readFileSync(STORE_PATH, "utf8");
    const body = functionBody(source, "function initializeDatabase(");

    const busyTimeoutIdx = body.search(/PRAGMA\s+busy_timeout/i);
    const walCallIdx = body.indexOf("setWalJournalMode(");

    expect(busyTimeoutIdx).toBeGreaterThanOrEqual(0);
    expect(walCallIdx).toBeGreaterThanOrEqual(0);
    // Original invariant, expressed against the call site that now owns the WAL work.
    expect(busyTimeoutIdx).toBeLessThan(walCallIdx);
    // Strengthening: no raw journal_mode WRITE may be issued here — all WAL
    // transitions must go through the retrying helper.
    expect(body).not.toMatch(/PRAGMA\s+journal_mode\s*=/i);
  });

  test("createStore readonly branch sets busy_timeout BEFORE the WAL work, and cannot bypass the helper (assertion gate)", () => {
    const source = readFileSync(STORE_PATH, "utf8");

    // Anchor on the "// Readonly:" comment that marks the readonly branch.
    const branchStart = source.indexOf("// Readonly:");
    expect(branchStart).toBeGreaterThanOrEqual(0);

    // The readonly branch's statements fit comfortably in ~800 chars after
    // the comment. Slice and check ordering within that window.
    const slice = source.slice(branchStart, branchStart + 800);
    const busyTimeoutIdx = slice.search(/PRAGMA\s+busy_timeout/i);
    const walCallIdx = slice.indexOf("setWalJournalMode(");

    expect(busyTimeoutIdx).toBeGreaterThanOrEqual(0);
    expect(walCallIdx).toBeGreaterThanOrEqual(0);
    expect(busyTimeoutIdx).toBeLessThan(walCallIdx);
    expect(slice).not.toMatch(/PRAGMA\s+journal_mode\s*=/i);
  });

  test("setWalJournalMode reads journal_mode BEFORE writing it, and retries on SQLITE_BUSY (assertion gate)", () => {
    const source = readFileSync(STORE_PATH, "utf8");
    const body = functionBody(source, "export function setWalJournalMode(");

    // Read-first: the PRAGMA journal_mode QUERY must precede the WAL WRITE.
    const readIdx = body.search(/query\(\s*["'`]PRAGMA\s+journal_mode["'`]\s*\)/i);
    const writeIdx = body.search(/PRAGMA\s+journal_mode\s*=\s*WAL/i);
    expect(readIdx).toBeGreaterThanOrEqual(0);
    expect(writeIdx).toBeGreaterThanOrEqual(0);
    expect(readIdx).toBeLessThan(writeIdx);

    // Bounded jittered retry on BUSY, with a rethrow escape — never a bare loop.
    expect(body).toMatch(/SQLITE_BUSY/);
    expect(body).toMatch(/database is locked/i);
    expect(body).toMatch(/budgetMs/);
    expect(body).toMatch(/throw err/);
    expect(body).toMatch(/sleepSync/);
  });

  test("N concurrent createStore() subprocesses on the same DB file all succeed without SQLITE_BUSY", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-issue-13-"));
    const dbPath = join(dir, "test.sqlite");

    try {
      type Result = { code: number; stderr: string; stdout: string };

      // Start barrier: every worker spins until this instant before calling
      // createStore(), so the N connections contend on the cold DELETE->WAL
      // transition instead of arriving spread out by Bun startup jitter.
      const startAtMs = Date.now() + 1500;

      const launchWorker = (): Promise<Result> => new Promise<Result>((resolveProc) => {
        // process.execPath is the Bun binary when this test is run via `bun test`.
        const p = spawn(process.execPath, ["run", WORKER_PATH, dbPath, String(startAtMs)], {
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
      const CONCURRENCY = Number(process.env.CLAWMEM_TEST_CONCURRENT_INIT_WORKERS ?? 5);
      const results = await Promise.all(Array.from({ length: CONCURRENCY }, () => launchWorker()));

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
