/**
 * Unit tests for the master-harness-t5i0 hardening pass:
 *
 *  1. Collection-scoped purge API (store.purgeCollection)
 *  2. Mandatory explicit scope on deleteInactiveDocuments / cleanupOrphanedContent /
 *     cleanupOrphanedVectors, plus the archived-row carve-out (default: spared;
 *     opts.includeArchived: true removes them too).
 *
 * The incident: a bare, unscoped call to deleteInactiveDocuments() +
 * cleanupOrphanedContent() + cleanupOrphanedVectors() hard-deleted 3566
 * pre-existing inactive/archived document rows when only a 50-doc disposable
 * smoke collection was meant to be purged. These tests assert the guard rails
 * that make that mistake structurally impossible to repeat.
 */

import { describe, it, expect, beforeEach } from "bun:test";
import {
  deleteInactiveDocuments,
  cleanupOrphanedContent,
  cleanupOrphanedVectors,
  purgeCollection,
  type Store,
} from "../../src/store.ts";
import { createTestStore } from "../helpers/test-store.ts";

// =============================================================================
// Seeding helpers — direct SQL so we control active / archived_at precisely.
// (collection, path) is UNIQUE, so one row per path holds the full lifecycle
// state we need for a given test case.
// =============================================================================

interface SeedRow {
  collection: string;
  path: string;
  active: 0 | 1;
  archivedAt?: string | null;
  hash?: string; // defaults to a unique hash per row
}

function seedRow(store: Store, row: SeedRow): { id: number; hash: string } {
  const hash = row.hash ?? `hash-${row.collection}-${row.path}-${Math.random().toString(36).slice(2)}`;
  const now = new Date().toISOString();

  // Insert content only if this hash hasn't been inserted yet (dedup, matches
  // production content-addressing).
  const exists = store.db.prepare(`SELECT 1 FROM content WHERE hash = ?`).get(hash);
  if (!exists) {
    store.db.prepare(`INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)`).run(hash, `body for ${row.path}`, now);
  }

  store.db.prepare(`
    INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active, archived_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(row.collection, row.path, row.path, hash, now, now, row.active, row.archivedAt ?? null);

  const id = (store.db.prepare(
    `SELECT id FROM documents WHERE collection = ? AND path = ?`
  ).get(row.collection, row.path) as { id: number }).id;

  return { id, hash };
}

function countDocs(store: Store, collection?: string): number {
  if (collection) {
    return (store.db.prepare(`SELECT COUNT(*) as c FROM documents WHERE collection = ?`).get(collection) as { c: number }).c;
  }
  return (store.db.prepare(`SELECT COUNT(*) as c FROM documents`).get() as { c: number }).c;
}

function countContent(store: Store): number {
  return (store.db.prepare(`SELECT COUNT(*) as c FROM content`).get() as { c: number }).c;
}

// =============================================================================
// purgeCollection — the scoped hard-delete API (fix #1)
// =============================================================================

describe("purgeCollection", () => {
  let store: Store;
  beforeEach(() => {
    store = createTestStore();
  });

  it("hard-deletes only the named collection's documents, active and inactive", () => {
    seedRow(store, { collection: "smoke", path: "a.md", active: 1 });
    seedRow(store, { collection: "smoke", path: "b.md", active: 0 }); // forgotten
    seedRow(store, { collection: "smoke", path: "c.md", active: 0, archivedAt: new Date().toISOString() }); // archived
    seedRow(store, { collection: "keep-me", path: "d.md", active: 1 });
    seedRow(store, { collection: "keep-me", path: "e.md", active: 0 });

    const result = purgeCollection(store.db, "smoke");

    expect(result.documents).toBe(3);
    expect(countDocs(store, "smoke")).toBe(0);
    expect(countDocs(store, "keep-me")).toBe(2); // untouched — this is the exact incident fix
  });

  it("deletes now-orphaned content rows scoped to the purged collection only", () => {
    seedRow(store, { collection: "smoke", path: "a.md", active: 1, hash: "hash-solo" });
    seedRow(store, { collection: "keep-me", path: "d.md", active: 1, hash: "hash-shared" });
    // second doc in the purged collection shares a hash with a surviving collection —
    // that content must be SPARED because another live document still references it.
    seedRow(store, { collection: "smoke", path: "b.md", active: 1, hash: "hash-shared" });

    expect(countContent(store)).toBe(2);

    const result = purgeCollection(store.db, "smoke");

    expect(result.content).toBe(1); // only hash-solo, not hash-shared
    expect(countContent(store)).toBe(1);
    expect((store.db.prepare(`SELECT 1 FROM content WHERE hash = 'hash-shared'`).get())).toBeTruthy();
    expect((store.db.prepare(`SELECT 1 FROM content WHERE hash = 'hash-solo'`).get())).toBeFalsy();
  });

  it("deletes now-orphaned vector rows scoped to the purged collection", () => {
    seedRow(store, { collection: "smoke", path: "a.md", active: 1, hash: "hash-vec" });
    store.ensureVecTable(2);
    store.insertEmbedding("hash-vec", 0, 0, new Float32Array([1, 0]), "test", new Date().toISOString());

    const before = (store.db.prepare(`SELECT COUNT(*) as c FROM content_vectors`).get() as { c: number }).c;
    expect(before).toBe(1);

    const result = purgeCollection(store.db, "smoke");
    expect(result.vectors).toBe(1);

    const after = (store.db.prepare(`SELECT COUNT(*) as c FROM content_vectors`).get() as { c: number }).c;
    expect(after).toBe(0);
  });

  it("refuses a missing (empty) collection name", () => {
    expect(() => purgeCollection(store.db, "")).toThrow();
    expect(() => purgeCollection(store.db, "   ")).toThrow();
  });

  it("refuses an ambiguous (wildcard-bearing) collection name", () => {
    seedRow(store, { collection: "smoke-1", path: "a.md", active: 1 });
    seedRow(store, { collection: "smoke-2", path: "b.md", active: 1 });
    expect(() => purgeCollection(store.db, "smoke-*")).toThrow();
    // and it must NOT have deleted anything on the way to throwing
    expect(countDocs(store, "smoke-1")).toBe(1);
    expect(countDocs(store, "smoke-2")).toBe(1);
  });

  it("refuses a collection name that matches no documents", () => {
    seedRow(store, { collection: "other", path: "a.md", active: 1 });
    expect(() => purgeCollection(store.db, "does-not-exist")).toThrow();
    expect(countDocs(store, "other")).toBe(1);
  });
});

// =============================================================================
// deleteInactiveDocuments — mandatory scope + archived carve-out (fix #2)
// =============================================================================

describe("deleteInactiveDocuments scope guard", () => {
  let store: Store;
  beforeEach(() => {
    store = createTestStore();
  });

  it("throws when called with an invalid/missing scope (simulates a bare call bypassing the type system)", () => {
    seedRow(store, { collection: "smoke", path: "a.md", active: 0 });
    // Cast to `any` to simulate what a bare `deleteInactiveDocuments()` call would
    // be at runtime if someone bypassed the compiler (e.g. plain JS caller, or a
    // `// @ts-ignore`) — the runtime guard must still refuse it.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(() => (deleteInactiveDocuments as any)(store.db)).toThrow();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(() => (deleteInactiveDocuments as any)(store.db, undefined)).toThrow();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(() => (deleteInactiveDocuments as any)(store.db, {})).toThrow();
  });

  it("all-scope, default options: deletes forgotten (active=0, archived_at NULL) rows but spares archived rows", () => {
    seedRow(store, { collection: "a", path: "forgotten.md", active: 0, archivedAt: null });
    seedRow(store, { collection: "a", path: "archived.md", active: 0, archivedAt: new Date().toISOString() });
    seedRow(store, { collection: "a", path: "active.md", active: 1 });

    const deleted = deleteInactiveDocuments(store.db, { all: true });

    expect(deleted).toBe(1); // only the forgotten row
    expect(countDocs(store, "a")).toBe(2); // archived.md + active.md survive
    const remaining = store.db.prepare(`SELECT path FROM documents WHERE collection = 'a'`).all() as { path: string }[];
    expect(remaining.map(r => r.path).sort()).toEqual(["active.md", "archived.md"]);
  });

  it("all-scope with includeArchived=true also deletes archived rows", () => {
    seedRow(store, { collection: "a", path: "forgotten.md", active: 0, archivedAt: null });
    seedRow(store, { collection: "a", path: "archived.md", active: 0, archivedAt: new Date().toISOString() });
    seedRow(store, { collection: "a", path: "active.md", active: 1 });

    const deleted = deleteInactiveDocuments(store.db, { all: true }, { includeArchived: true });

    expect(deleted).toBe(2);
    expect(countDocs(store, "a")).toBe(1); // only active.md
  });

  it("collection-scope only touches the named collection, and still spares archived rows by default", () => {
    seedRow(store, { collection: "smoke", path: "forgotten.md", active: 0, archivedAt: null });
    seedRow(store, { collection: "smoke", path: "archived.md", active: 0, archivedAt: new Date().toISOString() });
    seedRow(store, { collection: "other", path: "forgotten.md", active: 0, archivedAt: null });

    const deleted = deleteInactiveDocuments(store.db, { collection: "smoke" });

    expect(deleted).toBe(1);
    expect(countDocs(store, "smoke")).toBe(1); // archived.md survives
    expect(countDocs(store, "other")).toBe(1); // untouched — different collection
  });

  it("refuses an empty collection-scope name", () => {
    expect(() => deleteInactiveDocuments(store.db, { collection: "" })).toThrow();
  });

  it("refuses a wildcard-bearing collection-scope name", () => {
    expect(() => deleteInactiveDocuments(store.db, { collection: "smoke-*" })).toThrow();
  });
});

// =============================================================================
// cleanupOrphanedContent — mandatory scope + archived carve-out
// =============================================================================

describe("cleanupOrphanedContent scope guard", () => {
  let store: Store;
  beforeEach(() => {
    store = createTestStore();
  });

  it("throws on invalid/missing scope", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(() => (cleanupOrphanedContent as any)(store.db)).toThrow();
  });

  it("all-scope, default options: content referenced only by an archived doc is spared", () => {
    seedRow(store, { collection: "a", path: "archived.md", active: 0, archivedAt: new Date().toISOString(), hash: "hash-archived" });
    seedRow(store, { collection: "a", path: "forgotten.md", active: 0, archivedAt: null, hash: "hash-forgotten" });

    const deleted = cleanupOrphanedContent(store.db, { all: true });

    expect(deleted).toBe(1); // only hash-forgotten
    expect(store.db.prepare(`SELECT 1 FROM content WHERE hash = 'hash-archived'`).get()).toBeTruthy();
    expect(store.db.prepare(`SELECT 1 FROM content WHERE hash = 'hash-forgotten'`).get()).toBeFalsy();
  });

  it("all-scope with includeArchived=true also drops content only referenced by archived docs", () => {
    seedRow(store, { collection: "a", path: "archived.md", active: 0, archivedAt: new Date().toISOString(), hash: "hash-archived" });

    const deleted = cleanupOrphanedContent(store.db, { all: true }, { includeArchived: true });
    expect(deleted).toBe(1);
    expect(store.db.prepare(`SELECT 1 FROM content WHERE hash = 'hash-archived'`).get()).toBeFalsy();
  });

  it("collection-scope restricts the sweep to hashes that belonged to that collection", () => {
    seedRow(store, { collection: "smoke", path: "forgotten.md", active: 0, archivedAt: null, hash: "hash-smoke" });
    seedRow(store, { collection: "other", path: "forgotten.md", active: 0, archivedAt: null, hash: "hash-other" });

    const deleted = cleanupOrphanedContent(store.db, { collection: "smoke" });

    expect(deleted).toBe(1);
    expect(store.db.prepare(`SELECT 1 FROM content WHERE hash = 'hash-smoke'`).get()).toBeFalsy();
    expect(store.db.prepare(`SELECT 1 FROM content WHERE hash = 'hash-other'`).get()).toBeTruthy();
  });
});

// =============================================================================
// cleanupOrphanedVectors — mandatory scope + archived carve-out
// =============================================================================

describe("cleanupOrphanedVectors scope guard", () => {
  let store: Store;
  beforeEach(() => {
    store = createTestStore();
    store.ensureVecTable(2);
  });

  it("throws on invalid/missing scope", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(() => (cleanupOrphanedVectors as any)(store.db)).toThrow();
  });

  function seedVectorDoc(row: SeedRow) {
    const { hash } = seedRow(store, row);
    store.insertEmbedding(hash, 0, 0, new Float32Array([1, 0]), "test", new Date().toISOString());
    return hash;
  }

  it("all-scope, default options: vectors for archived docs are spared", () => {
    seedVectorDoc({ collection: "a", path: "archived.md", active: 0, archivedAt: new Date().toISOString(), hash: "hash-archived-v" });
    seedVectorDoc({ collection: "a", path: "forgotten.md", active: 0, archivedAt: null, hash: "hash-forgotten-v" });

    const deleted = cleanupOrphanedVectors(store.db, { all: true });

    expect(deleted).toBe(1);
    const remaining = store.db.prepare(`SELECT hash FROM content_vectors`).all() as { hash: string }[];
    expect(remaining.map(r => r.hash)).toEqual(["hash-archived-v"]);
  });

  it("all-scope with includeArchived=true removes archived-doc vectors too", () => {
    seedVectorDoc({ collection: "a", path: "archived.md", active: 0, archivedAt: new Date().toISOString(), hash: "hash-archived-v2" });

    const deleted = cleanupOrphanedVectors(store.db, { all: true }, { includeArchived: true });
    expect(deleted).toBe(1);
    expect(store.db.prepare(`SELECT 1 FROM content_vectors WHERE hash = 'hash-archived-v2'`).get()).toBeFalsy();
  });

  it("collection-scope restricts the sweep", () => {
    seedVectorDoc({ collection: "smoke", path: "forgotten.md", active: 0, archivedAt: null, hash: "hash-smoke-v" });
    seedVectorDoc({ collection: "other", path: "forgotten.md", active: 0, archivedAt: null, hash: "hash-other-v" });

    const deleted = cleanupOrphanedVectors(store.db, { collection: "smoke" });

    expect(deleted).toBe(1);
    expect(store.db.prepare(`SELECT 1 FROM content_vectors WHERE hash = 'hash-smoke-v'`).get()).toBeFalsy();
    expect(store.db.prepare(`SELECT 1 FROM content_vectors WHERE hash = 'hash-other-v'`).get()).toBeTruthy();
  });

  it("returns 0 when vectors_vec table does not exist", () => {
    const bareStore = createTestStore(); // ensureVecTable never called
    seedRow(bareStore, { collection: "a", path: "x.md", active: 0, archivedAt: null });
    expect(cleanupOrphanedVectors(bareStore.db, { all: true })).toBe(0);
  });
});
