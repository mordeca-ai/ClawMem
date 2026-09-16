import { describe, it, expect, afterEach } from "bun:test";

/**
 * The write fence names the arm that PRODUCED the vector (master-harness-vn4rz.44).
 *
 * The write-path mismatch errors used to label the culprit with
 * `process.env.CLAWMEM_EMBED_URL || "the local in-process embedder"`. With the URL set but the
 * remote endpoint on cooldown, LlamaCpp.embed() falls back to the LOCAL arm — and the refusal
 * then blamed the (innocent) remote server and told the operator to re-point CLAWMEM_EMBED_URL.
 * The producing arm now rides on EmbeddingResult.endpoint into the write, and the env var is
 * never consulted to name it.
 */

import type { PoolClient } from "pg";
import { LlamaCpp, LOCAL_EMBED_ARM_LABEL, UNREPORTED_EMBED_ARM_LABEL } from "../../src/llm.ts";
import { createStore, VecWriteModelMismatchError, type Store } from "../../src/store.ts";
import { assertWriteEmbedModelConsistent } from "../../src/pg/write.ts";
import { PgVecWriteModelMismatchError } from "../../src/pg/errors.ts";
import { hashContent } from "../../src/indexer.ts";

const VAULT_MODEL = "nomic-embed-text"; // deliberately NOT the local arm's model
const DUMMY_URL = "http://dummy-embed.invalid:9999";

const savedEnv = { url: process.env.CLAWMEM_EMBED_URL, noLocal: process.env.CLAWMEM_NO_LOCAL_MODELS };
const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  if (savedEnv.url === undefined) delete process.env.CLAWMEM_EMBED_URL;
  else process.env.CLAWMEM_EMBED_URL = savedEnv.url;
  if (savedEnv.noLocal === undefined) delete process.env.CLAWMEM_NO_LOCAL_MODELS;
  else process.env.CLAWMEM_NO_LOCAL_MODELS = savedEnv.noLocal;
});

/** Remote configured at DUMMY_URL but in cooldown; local context stubbed (no model load). */
function llmOnLocalFallback(): LlamaCpp {
  const llm = new LlamaCpp({ remoteEmbedUrl: DUMMY_URL });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (llm as any).ensureEmbedContext = async () => ({
    getEmbeddingFor: async (_t: string) => ({ vector: new Float32Array(4).fill(0.1) }),
  });
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (llm as any).markRemoteEmbedDown();
  return llm;
}

function vaultHolding(model: string): { store: Store; hash: string } {
  const store = createStore(":memory:");
  store.ensureVecTable(4);
  const now = new Date().toISOString();
  const seed = (path: string) => {
    const h = hashContent("body " + path);
    store.insertContent(h, "body " + path, now);
    store.insertDocument("c", path, path, h, now, now);
    return h;
  };
  store.insertEmbedding(seed("a.md"), 0, 1, new Float32Array([1, 0, 0, 0]), model, now);
  return { store, hash: seed("b.md") };
}

function caught(fn: () => unknown): unknown {
  try { fn(); } catch (e) { return e; }
  return undefined;
}

/** Minimal PoolClient: the fence only runs `SELECT DISTINCT model FROM content_vectors`. */
function pgClientHolding(models: string[]): PoolClient {
  return { query: async () => ({ rows: models.map(model => ({ model })) }) } as unknown as PoolClient;
}

describe("sqlite write fence names the producing arm", () => {
  it("local fallback arm with CLAWMEM_EMBED_URL set: names the local embedder, NOT the URL", async () => {
    process.env.CLAWMEM_EMBED_URL = DUMMY_URL;
    delete process.env.CLAWMEM_NO_LOCAL_MODELS;
    const result = await llmOnLocalFallback().embed("hello");
    expect(result).not.toBeNull();
    expect(result!.endpoint).toBe(LOCAL_EMBED_ARM_LABEL);

    const { store, hash } = vaultHolding(VAULT_MODEL);
    const err = caught(() => store.insertEmbedding(
      hash, 0, 1, new Float32Array(result!.embedding), result!.model, new Date().toISOString(),
      undefined, undefined, undefined, undefined, undefined, result!.endpoint,
    ));
    expect(err).toBeInstanceOf(VecWriteModelMismatchError);
    const e = err as VecWriteModelMismatchError;
    expect(e.endpoint).toBe(LOCAL_EMBED_ARM_LABEL);
    expect(e.message).toContain(LOCAL_EMBED_ARM_LABEL);
    expect(e.message).not.toContain(DUMMY_URL);
    expect(e.message).not.toContain("Point CLAWMEM_EMBED_URL at the vault's model");
  });

  it("batch path carries the arm too (local-arm label, URL in env)", async () => {
    process.env.CLAWMEM_EMBED_URL = DUMMY_URL;
    delete process.env.CLAWMEM_NO_LOCAL_MODELS;
    const [r] = await llmOnLocalFallback().embedBatch(["hello"]);
    expect(r!.endpoint).toBe(LOCAL_EMBED_ARM_LABEL);
    const { store, hash } = vaultHolding(VAULT_MODEL);
    const err = caught(() => store.insertEmbeddingsBatch([{
      hash, seq: 0, pos: 1, embedding: new Float32Array(r!.embedding), model: r!.model,
      embeddedAt: new Date().toISOString(), endpoint: r!.endpoint,
    }]));
    expect(err).toBeInstanceOf(VecWriteModelMismatchError);
    expect((err as Error).message).toContain(LOCAL_EMBED_ARM_LABEL);
    expect((err as Error).message).not.toContain(DUMMY_URL);
  });

  it("positive control: the remote arm is still named by the URL it actually fetched", async () => {
    const remoteUrl = "http://remote-embed.invalid:8080";
    process.env.CLAWMEM_EMBED_URL = DUMMY_URL; // env differs from the arm's URL on purpose
    globalThis.fetch = (async () => new Response(
      JSON.stringify({ data: [{ embedding: [0.1, 0.1, 0.1, 0.1], index: 0 }], model: "foreign-model" }),
      { status: 200, headers: { "content-type": "application/json" } },
    )) as unknown as typeof fetch;
    const result = await new LlamaCpp({ remoteEmbedUrl: remoteUrl }).embed("hello");
    expect(result!.endpoint).toBe(remoteUrl);

    const { store, hash } = vaultHolding(VAULT_MODEL);
    const err = caught(() => store.insertEmbedding(
      hash, 0, 1, new Float32Array(result!.embedding), result!.model, new Date().toISOString(),
      undefined, undefined, undefined, undefined, undefined, result!.endpoint,
    ));
    expect(err).toBeInstanceOf(VecWriteModelMismatchError);
    expect((err as Error).message).toContain(remoteUrl);
    expect((err as Error).message).not.toContain(DUMMY_URL);
    expect((err as Error).message).toContain("Point CLAWMEM_EMBED_URL at the vault's model");
  });

  it("an unreported arm is said to be unreported, never guessed from CLAWMEM_EMBED_URL", () => {
    process.env.CLAWMEM_EMBED_URL = DUMMY_URL;
    const { store, hash } = vaultHolding(VAULT_MODEL);
    const err = caught(() => store.insertEmbedding(hash, 0, 1, new Float32Array(4), "foreign-model", new Date().toISOString()));
    expect(err).toBeInstanceOf(VecWriteModelMismatchError);
    expect((err as VecWriteModelMismatchError).endpoint).toBe(UNREPORTED_EMBED_ARM_LABEL);
    expect((err as Error).message).not.toContain(DUMMY_URL);
  });
});

describe("PG write fence names the producing arm (no live PG needed)", () => {
  it("local-arm write with CLAWMEM_EMBED_URL set: names the local embedder, NOT the URL", async () => {
    process.env.CLAWMEM_EMBED_URL = DUMMY_URL;
    const err = await assertWriteEmbedModelConsistent(pgClientHolding([VAULT_MODEL]), "embeddinggemma", LOCAL_EMBED_ARM_LABEL)
      .then(() => undefined, (e: unknown) => e);
    expect(err).toBeInstanceOf(PgVecWriteModelMismatchError);
    expect((err as PgVecWriteModelMismatchError).endpoint).toBe(LOCAL_EMBED_ARM_LABEL);
    expect((err as Error).message).toContain(LOCAL_EMBED_ARM_LABEL);
    expect((err as Error).message).not.toContain(DUMMY_URL);
  });

  it("positive control: a remote-arm write is named by its URL", async () => {
    process.env.CLAWMEM_EMBED_URL = DUMMY_URL;
    const remoteUrl = "http://remote-embed.invalid:8080";
    const err = await assertWriteEmbedModelConsistent(pgClientHolding([VAULT_MODEL]), "foreign-model", remoteUrl)
      .then(() => undefined, (e: unknown) => e);
    expect(err).toBeInstanceOf(PgVecWriteModelMismatchError);
    expect((err as Error).message).toContain(remoteUrl);
    expect((err as Error).message).not.toContain(DUMMY_URL);
  });

  it("matching model passes regardless of arm", async () => {
    await assertWriteEmbedModelConsistent(pgClientHolding([VAULT_MODEL]), VAULT_MODEL, LOCAL_EMBED_ARM_LABEL);
  });
});
