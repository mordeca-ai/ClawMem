// Worker script for tests/integration/store-concurrent-init.test.ts.
// Invoked as: bun run tests/helpers/concurrent-init-worker.ts <dbPath>
// Calls createStore(dbPath) on the shared DB file, closes the connection,
// exits 0 on success or 1 with a one-line error on stderr on failure.
// Regression coverage for Issue #13.

import { createStore } from "../../src/store.ts";

const dbPath = process.argv[2];
if (!dbPath) {
  console.error("Usage: bun run tests/helpers/concurrent-init-worker.ts <db-path>");
  process.exit(2);
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
