// Unit coverage for setWalJournalMode() — the read-first + bounded-jittered-retry
// helper that closes the cold DELETE->WAL transition race (master-harness-3xw7n,
// follow-up to Issue #13).
//
// These tests drive the helper with a fake connection so the retry branch is
// exercised DETERMINISTICALLY (the integration test proves the same code under
// real multi-process contention, but cannot guarantee the backoff path runs on
// any given rep). Dead retry code is the thing being ruled out here.

import { describe, test, expect } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setWalJournalMode } from "../../src/store.ts";

type FakeDb = {
  query: (sql: string) => { get: () => { journal_mode?: string } | null };
  exec: (sql: string) => void;
  writes: string[];
};

/** Fake connection: reports `modes[i]` on the i-th read, throws `errors[i]` on the i-th write. */
function fakeDb(opts: {
  modes: string[];
  writeResults: (Error | null)[];
}): FakeDb {
  let readIdx = 0;
  let writeIdx = 0;
  const writes: string[] = [];
  return {
    writes,
    query: (_sql: string) => ({
      get: () => {
        const m = opts.modes[Math.min(readIdx++, opts.modes.length - 1)];
        return m === undefined ? null : { journal_mode: m };
      },
    }),
    exec: (sql: string) => {
      writes.push(sql);
      const r = opts.writeResults[Math.min(writeIdx++, opts.writeResults.length - 1)];
      if (r) throw r;
    },
  };
}

function busyError(): Error & { code: string } {
  const e = new Error("database is locked") as Error & { code: string };
  e.code = "SQLITE_BUSY";
  return e;
}

describe("setWalJournalMode", () => {
  test("read-first: an already-WAL database issues NO write and never retries", () => {
    const db = fakeDb({ modes: ["wal"], writeResults: [] });
    const res = setWalJournalMode(db as never, 1000);
    expect(res.outcome).toBe("already-wal");
    expect(res.attempts).toBe(1);
    expect(res.retries).toBe(0);
    expect(db.writes).toEqual([]);
  });

  test("read-first is case-insensitive (WAL / WaL)", () => {
    for (const mode of ["WAL", "WaL"]) {
      const db = fakeDb({ modes: [mode], writeResults: [] });
      expect(setWalJournalMode(db as never, 1000).outcome).toBe("already-wal");
      expect(db.writes).toEqual([]);
    }
  });

  test("cold DB transitions with a single write and no retry", () => {
    const db = fakeDb({ modes: ["delete"], writeResults: [null] });
    const res = setWalJournalMode(db as never, 1000);
    expect(res.outcome).toBe("transitioned");
    expect(res.attempts).toBe(1);
    expect(res.retries).toBe(0);
    expect(db.writes).toEqual(["PRAGMA journal_mode = WAL"]);
  });

  test("RETRY PATH: SQLITE_BUSY twice then success — backoff branch executes", () => {
    const db = fakeDb({
      modes: ["delete", "delete", "delete"],
      writeResults: [busyError(), busyError(), null],
    });
    const started = Date.now();
    const res = setWalJournalMode(db as never, 5000);
    const elapsed = Date.now() - started;
    expect(res.outcome).toBe("transitioned");
    expect(res.attempts).toBe(3);
    expect(res.retries).toBe(2); // two backoff sleeps actually happened
    expect(db.writes.length).toBe(3);
    expect(elapsed).toBeGreaterThanOrEqual(5); // at least the first 5ms backoff slept
  });

  test("RETRY PATH re-checks read-first each pass: a peer winning the transition ends the loop with no further write", () => {
    const db = fakeDb({
      modes: ["delete", "wal"],
      writeResults: [busyError()],
    });
    const res = setWalJournalMode(db as never, 5000);
    expect(res.outcome).toBe("already-wal");
    expect(res.attempts).toBe(2);
    expect(res.retries).toBe(1);
    expect(db.writes.length).toBe(1); // only the first, failed, attempt wrote
  });

  test("a non-BUSY error is rethrown immediately and never swallowed", () => {
    const boom = new Error("attempt to write a readonly database") as Error & { code: string };
    boom.code = "SQLITE_READONLY";
    const db = fakeDb({ modes: ["delete"], writeResults: [boom] });
    expect(() => setWalJournalMode(db as never, 5000)).toThrow("attempt to write a readonly database");
    expect(db.writes.length).toBe(1); // no retry
  });

  test("the retry loop is BOUNDED: an exhausted budget rethrows the original BUSY error", () => {
    const db = fakeDb({
      modes: ["delete"],
      writeResults: [busyError()], // always busy
    });
    const started = Date.now();
    expect(() => setWalJournalMode(db as never, 40)).toThrow("database is locked");
    const elapsed = Date.now() - started;
    expect(elapsed).toBeLessThan(5000); // terminated on budget, did not loop forever
    expect(db.writes.length).toBeGreaterThan(1); // it DID retry before giving up
  });

  test("against a real on-disk SQLite file: cold open transitions, warm reopen takes the read-first path", () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-wal-helper-"));
    try {
      const path = join(dir, "t.sqlite");
      const a = new Database(path);
      a.exec("PRAGMA busy_timeout = 5000");
      expect(setWalJournalMode(a, 5000).outcome).toBe("transitioned");
      a.close();

      const b = new Database(path);
      b.exec("PRAGMA busy_timeout = 5000");
      const res = setWalJournalMode(b, 5000);
      expect(res.outcome).toBe("already-wal");
      expect(res.retries).toBe(0);
      b.close();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
