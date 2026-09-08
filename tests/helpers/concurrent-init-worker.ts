// Worker script for tests/integration/store-concurrent-init.test.ts.
// Invoked as: bun run tests/helpers/concurrent-init-worker.ts <dbPath> [startAtEpochMs]
//
// The optional start-barrier (startAtEpochMs) removes Bun process-startup and
// module-load jitter from the race window: every worker spin-waits until the
// same wall-clock instant and only THEN calls createStore(), so all N
// connections reach `PRAGMA journal_mode = WAL` within microseconds of each
// other. Without the barrier the workers arrive hundreds of milliseconds
// apart and the cold DELETE->WAL transition race is almost never observed
// (measured: 0/60 reps unbarriered vs. a reliably reproducing barriered run).
// Calls createStore(dbPath) on the shared DB file, closes the connection,
// exits 0 on success or 1 with a one-line error on stderr on failure.
// Regression coverage for Issue #13.

import { createStore } from "../../src/store.ts";

const dbPath = process.argv[2];
if (!dbPath) {
  console.error("Usage: bun run tests/helpers/concurrent-init-worker.ts <db-path>");
  process.exit(2);
}

const startAtMs = Number(process.argv[3] ?? 0);
if (Number.isFinite(startAtMs) && startAtMs > 0) {
  // Busy-wait (not setTimeout) so the resume is as tight as the clock allows.
  while (Date.now() < startAtMs) { /* spin */ }
}

try {
  const store = createStore(dbPath);
  store.close();
  process.exit(0);
} catch (err) {
  const e = err as Error & { code?: string };
  console.error(`error: ${e.message}; code: ${e.code ?? "unknown"}`);
  process.exit(1);
}
