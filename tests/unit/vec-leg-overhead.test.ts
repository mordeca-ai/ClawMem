/**
 * master-harness-vn4rz.89 — fixed per-leg overhead on the vector path.
 *
 * Two cuts, both verdict-preserving:
 *  (1) the W1 model guard first reads the UNJOINED distinct model set; only when that is not
 *      exactly [endpointModel] does it run the active-scoped getVecModels() join (the slow half).
 *  (2) an INCLUDE-scoped search served by the pre-filter path no longer pays the vec0 count(*).
 *
 * The verdict tests pin the exact semantics of the old guard (inactive docs never count); the
 * statement-spy tests pin that the slow statements are actually skipped.
 */
import { describe, it, expect, afterEach } from "bun:test";
import {
  createStore,
  searchVec,
  searchVecDetailed,
  VecReadModelMismatchError,
  DEFAULT_EMBED_MODEL,
  type Store,
} from "../../src/store.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";
import { hashContent } from "../../src/indexer.ts";

function seed(store: Store, col: string, path: string, body: string): string {
  const hash = hashContent(body + path);
  const now = new Date().toISOString();
  store.insertContent(hash, body, now);
  store.insertDocument(col, path, path, hash, now, now);
  return hash;
}
const vec = (dim: number) => new Float32Array(Array.from({ length: dim }, (_, i) => (i === 0 ? 1 : 0)));

// Lay down a vector under a foreign model below the write guard (same fixture shape as
// embed-dimension-safety.test.ts: the write path refuses a second model by design).
function seedForeignModelVectorRaw(store: Store, hash: string, model: string): void {
  store.db.prepare(`INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`).run(`${hash}_0`, vec(4));
  store.db.prepare(
    `INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at, fragment_type, fragment_label, canonical_id, embed_input_fp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run(hash, 0, 1, model, new Date().toISOString(), "full", null, null, null);
}

function mockEmbedLlm(model: string): unknown {
  return { embed: async () => ({ embedding: [1, 0, 0, 0], model }) };
}

// Record every SQL text prepared on this connection from now on.
function spyPrepared(store: Store): string[] {
  const seen: string[] = [];
  const db = store.db as unknown as { prepare: (sql: string) => unknown };
  const orig = db.prepare.bind(db);
  db.prepare = (sql: string) => { seen.push(sql); return orig(sql); };
  return seen;
}
const isActiveJoinGuard = (sql: string) => /SELECT DISTINCT cv\.model/.test(sql) && /JOIN documents/.test(sql);
const isVecCount = (sql: string) => /count\(\*\)[\s\S]*FROM vectors_vec\b/.test(sql);

/** Active a.md under model-A; inactive b.md under `inactiveModel`. */
function mixedVault(inactiveModel: string): Store {
  const store = createStore(":memory:");
  store.ensureVecTable(4);
  const hA = seed(store, "c", "a.md", "doc a body");
  store.insertEmbedding(hA, 0, 1, vec(4), "model-A", new Date().toISOString(), "full", null, "c/a.md");
  const hB = seed(store, "c", "b.md", "doc b body");
  seedForeignModelVectorRaw(store, hB, inactiveModel);
  store.deactivateDocument("c", "b.md", "forget");
  return store;
}

describe("vn4rz.89 — W1 guard fast path keeps the exact verdict", () => {
  afterEach(() => setDefaultLlamaCpp(null));

  it("single-model vault: passes WITHOUT running the active-scoped join", async () => {
    const store = createStore(":memory:");
    store.ensureVecTable(4);
    const h = seed(store, "c", "a.md", "doc a body");
    store.insertEmbedding(h, 0, 1, vec(4), "model-A", new Date().toISOString(), "full", null, "c/a.md");
    setDefaultLlamaCpp(mockEmbedLlm("model-A") as any);
    const seen = spyPrepared(store);
    const r = await searchVec(store.db, "q", DEFAULT_EMBED_MODEL, 10);
    expect(r.length).toBeGreaterThanOrEqual(1);
    expect(seen.some(isActiveJoinGuard)).toBe(false);
  });

  it("a foreign model only on an INACTIVE doc does not throw (falls back to the exact join)", async () => {
    const store = mixedVault("model-B");
    setDefaultLlamaCpp(mockEmbedLlm("model-A") as any);
    const seen = spyPrepared(store);
    const r = await searchVec(store.db, "q", DEFAULT_EMBED_MODEL, 10);
    expect(r.length).toBeGreaterThanOrEqual(1);
    expect(seen.some(isActiveJoinGuard)).toBe(true); // the disagreement took the exact path
  });

  it("endpoint matching only the INACTIVE doc's model still throws", async () => {
    const store = mixedVault("model-B");
    setDefaultLlamaCpp(mockEmbedLlm("model-B") as any);
    await expect(searchVec(store.db, "q", DEFAULT_EMBED_MODEL, 10)).rejects.toBeInstanceOf(VecReadModelMismatchError);
  });

  it("single-model vault with a different endpoint model still throws", async () => {
    const store = createStore(":memory:");
    store.ensureVecTable(4);
    const h = seed(store, "c", "a.md", "doc a body");
    store.insertEmbedding(h, 0, 1, vec(4), "model-A", new Date().toISOString(), "full", null, "c/a.md");
    setDefaultLlamaCpp(mockEmbedLlm("model-B") as any);
    await expect(searchVec(store.db, "q", DEFAULT_EMBED_MODEL, 10)).rejects.toBeInstanceOf(VecReadModelMismatchError);
  });
});

describe("vn4rz.89 — scoped pre-filter path skips the vec0 count(*)", () => {
  afterEach(() => setDefaultLlamaCpp(null));

  function twoCollections(): Store {
    const store = createStore(":memory:");
    store.ensureVecTable(4);
    for (const [col, path] of [["small", "s.md"], ["big", "b1.md"], ["big", "b2.md"]] as const) {
      const h = seed(store, col, path, `body ${col} ${path}`);
      store.insertEmbedding(h, 0, 1, vec(4), "model-A", new Date().toISOString(), "full", null, `${col}/${path}`);
    }
    return store;
  }

  it("collection-scoped search returns the scope's docs and never counts vectors_vec", async () => {
    const store = twoCollections();
    setDefaultLlamaCpp(mockEmbedLlm("model-A") as any);
    const seen = spyPrepared(store);
    const det = await searchVecDetailed(store.db, "q", DEFAULT_EMBED_MODEL, 5, { collections: ["small"] });
    expect(det.results.map(r => r.collectionName)).toEqual(["small"]);
    expect(seen.some(isVecCount)).toBe(false);
  });

  it("unscoped search still counts vectors_vec (escalation bound)", async () => {
    const store = twoCollections();
    setDefaultLlamaCpp(mockEmbedLlm("model-A") as any);
    const seen = spyPrepared(store);
    const det = await searchVecDetailed(store.db, "q", DEFAULT_EMBED_MODEL, 5, {});
    expect(det.results.length).toBe(3);
    expect(seen.some(isVecCount)).toBe(true);
  });
});
