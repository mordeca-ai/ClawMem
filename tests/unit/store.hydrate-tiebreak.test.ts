import { describe, it, expect } from "bun:test";

/**
 * master-harness-hxa17 — hydrateVecResults must impose a TOTAL order.
 *
 * Cosine distances tie exactly and often: one live query carried a 19-fragment group
 * at distance 0.5426244139671326 against a limit of 20. With a distance-only
 * comparator, which of a tied group survives `.slice(limit)` is decided by SQL row
 * order — and that order was an artifact of the query plan (`SCAN d USING INDEX
 * idx_documents_effective_time`), so it tracked documents-table effective-time order
 * and re-rolled on any documents write. Recall on a tie boundary was therefore
 * silently write-order dependent: a genuinely DIFFERENT DOCUMENT could come back.
 *
 * The fix breaks exact-distance ties on `filepath`, which is unique after the
 * per-filepath dedupe that already ran, making (bestDist, filepath) a total order.
 *
 * These assert the CORRECT behavior, not the old behavior — they fail against the
 * pre-fix source, which returned the tied docs in SQL row order (insertion order
 * here) rather than lexicographic order.
 */

import { createStore, hydrateVecResults, type Store } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";

const MODEL = "test-model";
const DIM = 4;

/** Seed one doc with a single fragment, and return its hash_seq key. */
function seedFragment(store: Store, col: string, path: string): string {
  const body = `body of ${col}/${path}`;
  const hash = hashContent(body + col + path);
  const now = new Date().toISOString();
  store.insertContent(hash, body, now);
  store.insertDocument(col, path, path, hash, now, now);
  store.ensureVecTable(DIM);
  store.insertEmbedding(hash, 0, 0, new Float32Array([1, 0, 0, 0]), MODEL, now, "section", undefined, `${col}/${path}`);
  return `${hash}_0`;
}

// Inserted in REVERSE-lexicographic order on purpose: SQL row order (what the old
// distance-only sort fell through to) is insertion order here, so the pre-fix code
// returns z/y/x and the fixed code returns v/w/x. The test discriminates.
const PATHS = ["z.md", "y.md", "x.md", "w.md", "v.md"];
const TIED_DISTANCE = 0.5426244139671326;

function seedTieGroup(): { store: Store; vecResults: { hash_seq: string; distance: number }[] } {
  const store = createStore(":memory:");
  const vecResults = PATHS.map(p => ({ hash_seq: seedFragment(store, "user", p), distance: TIED_DISTANCE }));
  return { store, vecResults };
}

describe("hydrateVecResults — deterministic tie-break (master-harness-hxa17)", () => {
  it("resolves an exact-distance tie group crossing the limit by filepath, not by row order", () => {
    const { store, vecResults } = seedTieGroup();
    // 5 candidates, all at the same distance, limit 3 → the tie group crosses the boundary.
    const out = hydrateVecResults(store.db, vecResults, 3);
    expect(out.length).toBe(3);
    expect(out.every(r => r.score === 1 - TIED_DISTANCE)).toBe(true);
    // Lexicographically smallest three filepaths, in order — NOT the insertion order z/y/x.
    expect(out.map(r => r.filepath)).toEqual([
      "clawmem://user/v.md",
      "clawmem://user/w.md",
      "clawmem://user/x.md",
    ]);
  });

  it("is stable across repeated calls", () => {
    const { store, vecResults } = seedTieGroup();
    const first = hydrateVecResults(store.db, vecResults, 3).map(r => r.filepath);
    for (let i = 0; i < 5; i++) {
      expect(hydrateVecResults(store.db, vecResults, 3).map(r => r.filepath)).toEqual(first);
    }
  });

  it("is independent of the order the tied candidates arrive in", () => {
    const { store, vecResults } = seedTieGroup();
    const expected = hydrateVecResults(store.db, vecResults, 3).map(r => r.filepath);
    expect(hydrateVecResults(store.db, [...vecResults].reverse(), 3).map(r => r.filepath)).toEqual(expected);
    // a rotation, so neither original nor reversed order is accidentally reproduced
    const rotated = [...vecResults.slice(2), ...vecResults.slice(0, 2)];
    expect(hydrateVecResults(store.db, rotated, 3).map(r => r.filepath)).toEqual(expected);
  });

  it("still ranks by distance first — the tie-break only applies to exact ties", () => {
    const { store, vecResults } = seedTieGroup();
    // Give 'z.md' (lexicographically last) a strictly better distance: it must lead.
    const withWinner = vecResults.map(r => r.hash_seq === vecResults[0]!.hash_seq ? { ...r, distance: 0.1 } : r);
    const out = hydrateVecResults(store.db, withWinner, 3);
    expect(out[0]!.filepath).toBe("clawmem://user/z.md");
    expect(out.map(r => r.filepath).slice(1)).toEqual(["clawmem://user/v.md", "clawmem://user/w.md"]);
  });
});
