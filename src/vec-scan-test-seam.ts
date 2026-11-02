/**
 * Test seam for the vector-leg deadline (master-harness-vn4rz.131). A no-op unless one of these
 * is set; nothing in production sets them.
 *
 *   CLAWMEM_VEC_SCAN_TEST_BLOCK_MS     block the calling thread this long just before the scan
 *   CLAWMEM_VEC_SCAN_TEST_BLOCK_MATCH  only block when the query text contains this string
 *   CLAWMEM_VEC_SCAN_TEST_LOG          append one line per scan naming the thread: main|worker
 *
 * Bun.sleepSync on purpose: it is a native sleep that Worker.terminate() cannot interrupt, the
 * same way it cannot interrupt a vec0 scan inside sqlite3_step (measured on Bun 1.3.13).
 */
import { appendFileSync } from "node:fs";

export function vecScanTestSeam(query: string, env: Record<string, string | undefined> = process.env): void {
  const log = env.CLAWMEM_VEC_SCAN_TEST_LOG;
  if (log) appendFileSync(log, `${Bun.isMainThread ? "main" : "worker"}\n`);
  const ms = Number(env.CLAWMEM_VEC_SCAN_TEST_BLOCK_MS);
  if (!(ms > 0)) return;
  const match = env.CLAWMEM_VEC_SCAN_TEST_BLOCK_MATCH;
  if (match && !query.includes(match)) return;
  Bun.sleepSync(ms);
}
