import { afterEach, describe, expect, it, spyOn } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, LLM_CACHE_MAX_ROWS, setCachedResult, type Store } from "../../src/store.ts";

/**
 * master-harness-uvnhf.17 — reranker scores are written per document and per query. At a 1000-row
 * cap they evicted everything else in about 38 minutes, so a query expansion cached by a job that
 * runs every 30 minutes hit or missed by chance. Retention has to outlast a daily repeat.
 */

const opened: Array<{ store: Store; dir: string }> = [];

function openStore(): Store {
  const dir = mkdtempSync(join(tmpdir(), "llm-cache-retention-"));
  const store = createStore(join(dir, "index.sqlite"));
  opened.push({ store, dir });
  return store;
}

function seed(store: Store, rows: number): void {
  const insert = store.db.prepare(`INSERT INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)`);
  const start = Date.parse("2026-01-01T00:00:00.000Z");
  store.db.transaction(() => {
    for (let i = 0; i < rows; i++) insert.run(`seed-${i}`, "0.5", new Date(start + i).toISOString());
  })();
}

function count(store: Store): number {
  return (store.db.prepare(`SELECT count(*) AS n FROM llm_cache`).get() as { n: number }).n;
}

function has(store: Store, hash: string): boolean {
  return store.db.prepare(`SELECT 1 FROM llm_cache WHERE hash = ?`).get(hash) !== null;
}

afterEach(() => {
  for (const { store, dir } of opened.splice(0)) {
    store.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("llm_cache retention", () => {
  it("keeps a day of writes at the rate seen in production", () => {
    const peakWritesPerHour = 2200;
    expect(LLM_CACHE_MAX_ROWS).toBeGreaterThanOrEqual(peakWritesPerHour * 22);
  });

  it("a purge keeps every row under the cap, so an older expansion survives", () => {
    const store = openStore();
    seed(store, 1500);
    const random = spyOn(Math, "random").mockReturnValue(0);
    try {
      setCachedResult(store.db, "newest", "0.9");
    } finally {
      random.mockRestore();
    }
    expect(count(store)).toBe(1501);
    expect(has(store, "seed-0")).toBe(true);
  });

  it("a purge trims to the cap and drops the oldest rows first", () => {
    const store = openStore();
    seed(store, LLM_CACHE_MAX_ROWS + 10);
    const random = spyOn(Math, "random").mockReturnValue(0);
    try {
      setCachedResult(store.db, "newest", "0.9");
    } finally {
      random.mockRestore();
    }
    expect(count(store)).toBe(LLM_CACHE_MAX_ROWS);
    expect(has(store, "seed-10")).toBe(false);
    expect(has(store, "seed-11")).toBe(true);
    expect(has(store, "newest")).toBe(true);
  });

  it("most writes do not purge, because a purge at this size is slow", () => {
    const store = openStore();
    seed(store, LLM_CACHE_MAX_ROWS + 10);
    const random = spyOn(Math, "random").mockReturnValue(0.002);
    try {
      setCachedResult(store.db, "newest", "0.9");
    } finally {
      random.mockRestore();
    }
    expect(count(store)).toBe(LLM_CACHE_MAX_ROWS + 11);
  });
});
