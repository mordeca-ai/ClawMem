import { describe, it, expect, beforeEach, afterAll } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * master-harness-apktc — getContextForFile() memo.
 *
 * After hxa17's loadConfig memo, getContextForFile still cost ~2.9 ms/row (three
 * stat + structuredClone round-trips through the collections API per call) and it runs
 * per row per leg across all six retrieval legs. The config-derived half of the answer
 * (collection resolution + global/prefix context assembly) is now memoized per process,
 * keyed on the config-file identity stamp, and bounded. The DB existence check stays
 * OUTSIDE the memo so a deactivated document is never served stale context.
 *
 * These tests prove: (1) the memo holds the per-row cost under 1 ms on a realistic
 * repeated-filepath workload, (2) any context / collection config change busts it,
 * (3) DB-side state is never memoized, (4) the memo is bounded.
 */

import { createStore, getContextForFile, clearContextMemo, contextMemoStats, CONTEXT_MEMO_MAX_ENTRIES, type Store } from "../../src/store.ts";
import { clearConfigCache, addContext, removeContext, loadConfig } from "../../src/collections.ts";
import { hashContent } from "../../src/indexer.ts";

const ORIGINAL_CONFIG_DIR = process.env.CLAWMEM_CONFIG_DIR;
const tmpRoot = mkdtempSync(join(tmpdir(), "clawmem-apktc-"));

const N_COLLECTIONS = 40;
const DOCS_PER_COLLECTION = 20;

function collName(i: number): string {
  return `coll${i}`;
}

/** Live-vault shape: 40 collections in index.yml (the live vault has 39), each with a handful of prefix contexts. */
function writeFixtureConfig(configDir: string): void {
  const lines: string[] = ["global_context: \"Global vault context for every document.\"", "collections:"];
  for (let i = 0; i < N_COLLECTIONS; i++) {
    lines.push(`  ${collName(i)}:`);
    lines.push(`    path: ${join(tmpRoot, "fs", collName(i))}`);
    lines.push(`    pattern: "**/*.md"`);
    lines.push(`    content_type: notes`);
    lines.push(`    vault: default`);
    lines.push(`    context:`);
    lines.push(`      /: "Collection ${i} root context."`);
    lines.push(`      /notes: "Collection ${i} notes context."`);
    lines.push(`      /notes/deep: "Collection ${i} deep notes context."`);
    lines.push(`      /archive: "Collection ${i} archive context."`);
    lines.push(`      /projects: "Collection ${i} projects context."`);
  }
  writeFileSync(join(configDir, "index.yml"), lines.join("\n") + "\n", "utf-8");
}

function seedDocs(store: Store): void {
  const now = new Date().toISOString();
  for (let i = 0; i < N_COLLECTIONS; i++) {
    for (let d = 0; d < DOCS_PER_COLLECTION; d++) {
      const path = d % 3 === 0 ? `notes/deep/doc${d}.md` : d % 3 === 1 ? `notes/doc${d}.md` : `projects/doc${d}.md`;
      const body = `body ${i}/${path}`;
      const hash = hashContent(body);
      store.insertContent(hash, body, now);
      store.insertDocument(collName(i), path, path, hash, now, now);
    }
  }
}

/** 60 lookups over 20 distinct paths (mixed virtual + filesystem), the way six legs re-hit the same docs. */
function workload(): string[] {
  const distinct: string[] = [];
  for (let k = 0; k < 20; k++) {
    const i = k % N_COLLECTIONS;
    const d = (k * 7) % DOCS_PER_COLLECTION;
    const path = d % 3 === 0 ? `notes/deep/doc${d}.md` : d % 3 === 1 ? `notes/doc${d}.md` : `projects/doc${d}.md`;
    distinct.push(k % 2 === 0 ? `clawmem://${collName(i)}/${path}` : join(tmpRoot, "fs", collName(i), path));
  }
  const out: string[] = [];
  for (let n = 0; n < 60; n++) out.push(distinct[(n * 13) % distinct.length]!);
  return out;
}

let configDir: string;
let store: Store;

beforeEach(() => {
  configDir = mkdtempSync(join(tmpRoot, "cfg-"));
  mkdirSync(configDir, { recursive: true });
  writeFixtureConfig(configDir);
  process.env.CLAWMEM_CONFIG_DIR = configDir;
  delete process.env.CLAWMEM_DISABLE_CONTEXT_MEMO;
  clearConfigCache();
  clearContextMemo();
  store = createStore(":memory:");
  seedDocs(store);
});

afterAll(() => {
  if (ORIGINAL_CONFIG_DIR === undefined) delete process.env.CLAWMEM_CONFIG_DIR;
  else process.env.CLAWMEM_CONFIG_DIR = ORIGINAL_CONFIG_DIR;
  clearConfigCache();
  clearContextMemo();
  rmSync(tmpRoot, { recursive: true, force: true });
});

describe("getContextForFile memo (master-harness-apktc)", () => {
  it("returns the assembled global + prefix context for virtual and filesystem paths", () => {
    const expected = "Global vault context for every document.\n\nCollection 3 root context.\n\nCollection 3 notes context.\n\nCollection 3 deep notes context.";
    expect(getContextForFile(store.db, "clawmem://coll3/notes/deep/doc0.md")).toBe(expected);
    expect(getContextForFile(store.db, join(tmpRoot, "fs", "coll3", "notes/deep/doc0.md"))).toBe(expected);
    // Unknown collection / path outside every collection → null, memoized or not.
    expect(getContextForFile(store.db, "clawmem://nope/a.md")).toBeNull();
    expect(getContextForFile(store.db, "/nowhere/a.md")).toBeNull();
    expect(getContextForFile(store.db, "clawmem://nope/a.md")).toBeNull();
  });

  it("60 calls over repeated filepaths average < 1 ms/row", () => {
    const paths = workload();
    // Prime the YAML parser + hxa17's loadConfig cache (NOT the context memo): the first
    // loadConfig in a process pays a one-off cold parse/JIT (tens of ms) that old and new code
    // pay identically, and the live ~2.9 ms/row figure was a steady-state measurement.
    loadConfig();
    const t0 = performance.now();
    let hits = 0;
    for (const p of paths) if (getContextForFile(store.db, p) !== null) hits++;
    const perRow = (performance.now() - t0) / paths.length;
    console.log(`[apktc] getContextForFile 60x mean=${perRow.toFixed(4)} ms/row memo=${process.env.CLAWMEM_DISABLE_CONTEXT_MEMO === "true" ? "off" : "on"}`);
    expect(hits).toBe(paths.length);
    // Deterministic half (immune to host load): 20 distinct filepaths → 20 misses, 40 hits.
    const stats = contextMemoStats();
    expect({ hits: stats.hits, misses: stats.misses, bypassed: stats.bypassed }).toEqual({ hits: 40, misses: 20, bypassed: 0 });
    // Timing half (acceptance A2): < 1 ms/row mean. Pre-memo the same workload measured 0.58–1.86 ms/row.
    expect(perRow).toBeLessThan(1);
  });

  it("a context config change busts the memo (addContext / removeContext)", () => {
    const fp = "clawmem://coll2/notes/doc1.md";
    const before = getContextForFile(store.db, fp);
    expect(before).toContain("Collection 2 notes context.");
    expect(addContext("coll2", "/notes", "REPLACED notes context.")).toBe(true);
    const after = getContextForFile(store.db, fp);
    expect(contextMemoStats().invalidations).toBeGreaterThanOrEqual(1);
    expect(after).toContain("REPLACED notes context.");
    expect(after).not.toContain("Collection 2 notes context.");
    expect(removeContext("coll2", "/notes")).toBe(true);
    expect(getContextForFile(store.db, fp)).not.toContain("notes context.");
  });

  it("an out-of-band config file edit busts the memo", () => {
    const fp = "clawmem://coll1/projects/doc2.md";
    expect(getContextForFile(store.db, fp)).toContain("Global vault context");
    writeFileSync(join(configDir, "index.yml"), `global_context: "Hand-edited global."\ncollections:\n  coll1:\n    path: /x\n    pattern: "**/*.md"\n`, "utf-8");
    expect(getContextForFile(store.db, fp)).toBe("Hand-edited global.");
  });

  it("DB state is never memoized: a deactivated document stops yielding context", () => {
    const fp = "clawmem://coll4/notes/doc1.md";
    expect(getContextForFile(store.db, fp)).not.toBeNull();
    store.deactivateDocument("coll4", "notes/doc1.md", "archive");
    expect(getContextForFile(store.db, fp)).toBeNull();
  });

  it("memoized results are identical to the bypassed (uncached) path", () => {
    const paths = [...workload(), "clawmem://nope/a.md", "/nowhere/a.md", "clawmem://coll0/missing.md", ""];
    const memoized = paths.map(p => getContextForFile(store.db, p));
    process.env.CLAWMEM_DISABLE_CONTEXT_MEMO = "true";
    try {
      expect(paths.map(p => getContextForFile(store.db, p))).toEqual(memoized);
      expect(contextMemoStats().size).toBe(0);
    } finally {
      delete process.env.CLAWMEM_DISABLE_CONTEXT_MEMO;
    }
  });

  it("the memo is bounded", () => {
    for (let n = 0; n < CONTEXT_MEMO_MAX_ENTRIES + 500; n++) getContextForFile(store.db, `clawmem://coll0/missing/${n}.md`);
    expect(contextMemoStats().size).toBe(CONTEXT_MEMO_MAX_ENTRIES);
  });
});
