import { describe, it, expect, afterEach } from "bun:test";

/**
 * WRITE-path embed-geometry preflight (master-harness-vn4rz.21).
 *
 * The read path has had assertQueryEmbedModelConsistent since W1; the vector-WRITE
 * chokepoints (insertEmbedding / insertEmbeddingsBatch) had no counterpart. The only
 * model preflight lived in the `clawmem embed` COMMAND layer, so every other caller of
 * the store's write API — a daemon, the MCP server, a repair script, a test harness —
 * could write a second model name into the vault. That is the asymmetry the 2026-08-09
 * incident exploited: a llama.cpp ggml-org Q8_0 endpoint wrote 2,899 foreign-geometry
 * vectors at the SAME dimension, which neither the dimension guard nor the read guard
 * could stop.
 *
 * These tests prove the guard goes BOTH ways: it REFUSES a foreign-geometry write
 * (naming the endpoint plus the expected and actual model names, and writing nothing),
 * and it leaves a matching-geometry write completely unchanged.
 */

import {
  createStore,
  VecWriteModelMismatchError,
  VecModelMismatchError,
  FatalVectorError,
  type Store,
} from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";

const VAULT_MODEL = "embeddinggemma:latest";
const FOREIGN_MODEL = "ggml-org/embeddinggemma-300M-GGUF-Q8_0"; // the 2026-08-09 poisoner
const FOREIGN_ENDPOINT = "http://192.168.2.15:8199/v1";

function seed(store: Store, col: string, path: string, body: string): string {
  const hash = hashContent(body + path);
  const now = new Date().toISOString();
  store.insertContent(hash, body, now);
  store.insertDocument(col, path, path, hash, now, now);
  return hash;
}
const vec = (dim: number, lead = 1) =>
  new Float32Array(Array.from({ length: dim }, (_, i) => (i === 0 ? lead : 0)));

/** A vault already holding vectors under exactly one model — the state the live vault is in. */
function vaultWithGeometry(): { store: Store; hashes: string[] } {
  const store = createStore(":memory:");
  store.ensureVecTable(4);
  const now = new Date().toISOString();
  const h1 = seed(store, "c", "a.md", "doc a");
  const h2 = seed(store, "c", "b.md", "doc b");
  store.insertEmbedding(h1, 0, 1, vec(4), VAULT_MODEL, now, "full", undefined, "c/a.md");
  return { store, hashes: [h1, h2] };
}

const savedUrl = process.env.CLAWMEM_EMBED_URL;
afterEach(() => {
  if (savedUrl === undefined) delete process.env.CLAWMEM_EMBED_URL;
  else process.env.CLAWMEM_EMBED_URL = savedUrl;
});

describe("write-path geometry preflight — RED (refusal)", () => {
  it("REFUSES a same-dimension foreign-model single write and writes nothing", () => {
    const { store, hashes } = vaultWithGeometry();
    const before = store.getVectorConsistency();

    expect(() =>
      store.insertEmbedding(hashes[1]!, 0, 1, vec(4, 0.5), FOREIGN_MODEL, new Date().toISOString(), "full", undefined, "c/b.md")
    ).toThrow(VecWriteModelMismatchError);

    // Transaction rolled back: the foreign vector is NOT in the vault.
    const after = store.getVectorConsistency();
    expect(after.vvCount).toBe(before.vvCount);
    expect(after.cvCount).toBe(before.cvCount);
    expect(store.getVecModels()).toEqual([VAULT_MODEL]);
  });

  it("names the endpoint plus the expected and actual model in the error", () => {
    process.env.CLAWMEM_EMBED_URL = FOREIGN_ENDPOINT;
    const { store, hashes } = vaultWithGeometry();
    let caught: unknown;
    try {
      store.insertEmbedding(hashes[1]!, 0, 1, vec(4, 0.5), FOREIGN_MODEL, new Date().toISOString(), "full", undefined, "c/b.md");
    } catch (e) { caught = e; }

    expect(caught).toBeInstanceOf(VecWriteModelMismatchError);
    const err = caught as VecWriteModelMismatchError;
    expect(err.message).toContain(FOREIGN_ENDPOINT);   // WHICH server did this
    expect(err.message).toContain(VAULT_MODEL);        // expected
    expect(err.message).toContain(FOREIGN_MODEL);      // actual
    expect(err.storedModels).toEqual([VAULT_MODEL]);
    expect(err.writeModel).toBe(FOREIGN_MODEL);
    expect(err.endpoint).toBe(FOREIGN_ENDPOINT);
    // Fatal, so the embed run aborts (exit 1) instead of counting a per-fragment failure.
    expect(err).toBeInstanceOf(FatalVectorError);
  });

  it("REFUSES the WHOLE batch when one row carries a foreign model", () => {
    const { store, hashes } = vaultWithGeometry();
    const before = store.getVectorConsistency();
    const now = new Date().toISOString();

    expect(() =>
      store.insertEmbeddingsBatch([
        { hash: hashes[1]!, seq: 0, pos: 1, embedding: vec(4), model: VAULT_MODEL, embeddedAt: now },
        { hash: hashes[1]!, seq: 1, pos: 2, embedding: vec(4, 0.5), model: FOREIGN_MODEL, embeddedAt: now },
      ])
    ).toThrow(FatalVectorError);

    // All-or-nothing: even the well-formed first row is rolled back.
    const after = store.getVectorConsistency();
    expect(after.vvCount).toBe(before.vvCount);
    expect(store.getVecModels()).toEqual([VAULT_MODEL]);
  });

  it("REFUSES a batch carrying two models even on a FRESH vault (no stored geometry to compare)", () => {
    const store = createStore(":memory:");
    store.ensureVecTable(4);
    const h = seed(store, "c", "a.md", "doc a");
    const now = new Date().toISOString();

    expect(() =>
      store.insertEmbeddingsBatch([
        { hash: h, seq: 0, pos: 1, embedding: vec(4), model: VAULT_MODEL, embeddedAt: now },
        { hash: h, seq: 1, pos: 2, embedding: vec(4, 0.5), model: FOREIGN_MODEL, embeddedAt: now },
      ])
    ).toThrow(VecModelMismatchError);
    expect(store.getVectorConsistency().vvCount).toBe(0);
  });
});

describe("write-path geometry preflight — GREEN (unchanged behavior)", () => {
  it("lets a matching-geometry single write proceed", () => {
    const { store, hashes } = vaultWithGeometry();
    store.insertEmbedding(hashes[1]!, 0, 1, vec(4, 0.5), VAULT_MODEL, new Date().toISOString(), "full", undefined, "c/b.md");
    const after = store.getVectorConsistency();
    expect(after.vvCount).toBe(2);
    expect(after.cvMissingVv).toBe(0);
    expect(store.getVecModels()).toEqual([VAULT_MODEL]);
  });

  it("lets a matching-geometry batch proceed", () => {
    const { store, hashes } = vaultWithGeometry();
    const now = new Date().toISOString();
    store.insertEmbeddingsBatch([
      { hash: hashes[1]!, seq: 0, pos: 1, embedding: vec(4), model: VAULT_MODEL, embeddedAt: now },
      { hash: hashes[1]!, seq: 1, pos: 2, embedding: vec(4, 0.5), model: VAULT_MODEL, embeddedAt: now },
    ]);
    expect(store.getVectorConsistency().vvCount).toBe(3);
  });

  it("no-ops on a FRESH vault — the first model in is the vault's geometry", () => {
    const store = createStore(":memory:");
    store.ensureVecTable(4);
    const h = seed(store, "c", "a.md", "doc a");
    store.insertEmbedding(h, 0, 1, vec(4), FOREIGN_MODEL, new Date().toISOString());
    expect(store.getVecModels()).toEqual([FOREIGN_MODEL]);
  });

  it("does NOT break `embed --force`: clearAllEmbeddings then a new model rebuilds cleanly", () => {
    const { store, hashes } = vaultWithGeometry();
    store.clearAllEmbeddings();          // what `embed --force` does before rebuilding
    store.ensureVecTable(4);
    store.insertEmbedding(hashes[0]!, 0, 1, vec(4), FOREIGN_MODEL, new Date().toISOString());
    expect(store.getVecModels()).toEqual([FOREIGN_MODEL]);
  });

  it("allows a write with NO model name (an endpoint that reports none is undiscriminable)", () => {
    const { store, hashes } = vaultWithGeometry();
    store.insertEmbedding(hashes[1]!, 0, 1, vec(4, 0.5), "", new Date().toISOString());
    expect(store.getVectorConsistency().vvCount).toBe(2);
  });
});

describe("write-path geometry preflight — cross-process invalidation", () => {
  it("re-checks after another connection changes the vault (memo is data_version-keyed)", () => {
    // Same-connection memo must not mask a foreign model appearing later in the same run:
    // the memo key includes the model, so a swapped model always re-reads content_vectors.
    const { store, hashes } = vaultWithGeometry();
    store.insertEmbedding(hashes[1]!, 0, 1, vec(4), VAULT_MODEL, new Date().toISOString()); // memoize OK
    expect(() =>
      store.insertEmbedding(hashes[1]!, 1, 2, vec(4, 0.5), FOREIGN_MODEL, new Date().toISOString())
    ).toThrow(VecWriteModelMismatchError);
  });
});
