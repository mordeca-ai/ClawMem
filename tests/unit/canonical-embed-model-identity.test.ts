import { describe, it, expect, afterEach } from "bun:test";

/**
 * Canonical embedding-model IDENTITY (master-harness-yidbh).
 *
 * ONE physical embedding model was recorded under FOUR different identity strings depending
 * on which producer made the vector: the in-process node-llama-cpp arm wrote the hf: URI, the
 * remote arm wrote whatever the endpoint echoed (`embeddinggemma` OR ollama's `embeddinggemma:latest`),
 * and — when the endpoint reported no model at all — the arm fell back to the endpoint URL, which
 * is not a model identity in the first place. All four are the same 768-dim geometry, so the
 * dimension guard could never see the difference. That is the defect class that recurred three
 * times (master-harness-92t7, ich.2, 54rlt) through three different doors.
 *
 * The live trigger is the TRANSPORT-failure fallback: `bin/clawmem` defaults CLAWMEM_EMBED_URL to
 * http://localhost:8088; when nothing answers there, ECONNREFUSED marks remote down and the
 * in-process arm serves the fragment — writing the hf: URI into a vault whose identity is
 * `embeddinggemma`. Pre-vn4rz.21 that poisoned silently; post-.21 it fails the whole embed closed.
 *
 * These tests prove: the aliases collapse (AC1), nothing else is invented (AC2), a URL is refused
 * (AC3), all four producers emit the canonical id (AC4), the legitimate local fallback now simply
 * WORKS against a live-shaped vault (AC5), a genuinely foreign model is still refused (AC6), and
 * the live configuration's recorded string is unchanged so zero re-embed is required (AC8).
 */

import {
  canonicalEmbedModelId,
  EmbedModelIdentityError,
  DEFAULT_EMBED_MODEL,
  LlamaCpp,
} from "../../src/llm.ts";
import { createStore, VecWriteModelMismatchError, type Store } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";

const CANON = "embeddinggemma";
const HF_URI = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const OLLAMA_TAG = "embeddinggemma:latest";

// ---------------------------------------------------------------------------
// AC1 / AC2 / AC3 / AC8 — the normalization function itself
// ---------------------------------------------------------------------------

describe("canonicalEmbedModelId — AC1 identity collapse", () => {
  it("collapses all THREE known spellings of the one physical model to one id", () => {
    expect(canonicalEmbedModelId(HF_URI)).toBe(CANON);
    expect(canonicalEmbedModelId(CANON)).toBe(CANON);
    expect(canonicalEmbedModelId(OLLAMA_TAG)).toBe(CANON);
    // …and therefore to each other, which is the property the write fence relies on.
    expect(canonicalEmbedModelId(HF_URI)).toBe(canonicalEmbedModelId(CANON));
    expect(canonicalEmbedModelId(OLLAMA_TAG)).toBe(canonicalEmbedModelId(CANON));
  });

  it("is case- and whitespace-insensitive for the alias lookup", () => {
    expect(canonicalEmbedModelId("  EmbeddingGemma:Latest  ")).toBe(CANON);
    expect(canonicalEmbedModelId(HF_URI.toUpperCase())).toBe(CANON);
  });
});

describe("canonicalEmbedModelId — AC2 no invention", () => {
  it("passes an UNKNOWN model through unchanged (lowercased/trimmed only)", () => {
    expect(canonicalEmbedModelId("nomic-embed-text")).toBe("nomic-embed-text");
    expect(canonicalEmbedModelId("  BGE-M3  ")).toBe("bge-m3");
  });

  it("keeps two distinct unknown models DISTINCT", () => {
    expect(canonicalEmbedModelId("nomic-embed-text")).not.toBe(canonicalEmbedModelId("bge-m3"));
  });

  it("does NOT generically strip a tag — a pinned tag can be a different snapshot", () => {
    // "embeddinggemma:latest" is an EXPLICIT table entry; an unrelated tagged model is not,
    // so it must survive intact rather than being merged with its untagged sibling.
    expect(canonicalEmbedModelId("nomic-embed-text:latest")).toBe("nomic-embed-text:latest");
    expect(canonicalEmbedModelId("nomic-embed-text:v1.5")).not.toBe(canonicalEmbedModelId("nomic-embed-text"));
  });

  it("treats blank/null/undefined as the UNNAMED identity", () => {
    expect(canonicalEmbedModelId(undefined)).toBe("");
    expect(canonicalEmbedModelId(null)).toBe("");
    expect(canonicalEmbedModelId("   ")).toBe("");
  });
});

describe("canonicalEmbedModelId — AC3 a URL can never be an identity", () => {
  it("throws EmbedModelIdentityError for an endpoint URL", () => {
    expect(() => canonicalEmbedModelId("http://192.168.2.15:8088")).toThrow(EmbedModelIdentityError);
    expect(() => canonicalEmbedModelId("https://api.example.com/v1")).toThrow(EmbedModelIdentityError);
    expect(() => canonicalEmbedModelId("  HTTP://localhost:8088/v1  ")).toThrow(EmbedModelIdentityError);
  });
});

describe("canonicalEmbedModelId — AC8 zero migration", () => {
  it("records exactly what BOTH live vaults already hold, so no re-embed is required", () => {
    // The live config: DEFAULT_EMBED_MODEL for the in-process arm, "embeddinggemma" echoed by
    // the ollama endpoints. Both vaults store the string "embeddinggemma" today.
    expect(canonicalEmbedModelId(DEFAULT_EMBED_MODEL)).toBe("embeddinggemma");
    expect(canonicalEmbedModelId("embeddinggemma")).toBe("embeddinggemma");
  });
});

// ---------------------------------------------------------------------------
// AC4 — all four producers
// ---------------------------------------------------------------------------

/** Stub the node-llama-cpp embedding context so the LOCAL arm can be driven without a GPU,
 *  a model download, or node-llama-cpp being loadable at all. Everything else in embedLocal /
 *  embedLocalBatch (including the model-string it records) is the real code path. */
function withStubbedLocalContext(llm: LlamaCpp, dim = 4): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (llm as any).ensureEmbedContext = async () => ({
    getEmbeddingFor: async (_t: string) => ({ vector: new Float32Array(dim).fill(0.1) }),
  });
}

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

/** Mock an OpenAI-shaped /v1/embeddings endpoint. `model` undefined => reply omits the field. */
function mockEmbedEndpoint(model: string | undefined, dim = 4): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  globalThis.fetch = (async (_url: any, init: any) => {
    const body = JSON.parse(init.body as string) as { input: string | string[] };
    const inputs = Array.isArray(body.input) ? body.input : [body.input];
    const payload: Record<string, unknown> = {
      data: inputs.map((_t, index) => ({ embedding: Array(dim).fill(0.1), index })),
    };
    if (model !== undefined) payload.model = model;
    return new Response(JSON.stringify(payload), { status: 200, headers: { "content-type": "application/json" } });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }) as any;
}

describe("AC4 — all four EmbeddingResult producers record the CANONICAL id", () => {
  it("embedLocal records the canonical id, not the raw hf: URI", async () => {
    const llm = new LlamaCpp({});
    withStubbedLocalContext(llm);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const r = await (llm as any).embedLocal("hello");
    expect(r?.model).toBe(CANON);
    expect(r?.model).not.toContain("hf:");
  });

  it("embedLocalBatch records the canonical id for every fragment", async () => {
    const llm = new LlamaCpp({});
    withStubbedLocalContext(llm);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rs = await (llm as any).embedLocalBatch(["a", "b", "c"]);
    expect(rs.map((r: { model: string }) => r.model)).toEqual([CANON, CANON, CANON]);
  });

  it("resolves the local identity ONCE in the constructor (hot-path invariant)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((new LlamaCpp({}) as any).embedModelId).toBe(CANON);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((new LlamaCpp({ embedModel: OLLAMA_TAG }) as any).embedModelId).toBe(CANON);
  });

  it("embedRemote canonicalizes the endpoint's echoed model (ollama ':latest' form)", async () => {
    mockEmbedEndpoint(OLLAMA_TAG);
    const llm = new LlamaCpp({ remoteEmbedUrl: "http://127.0.0.1:11435", remoteEmbedModel: OLLAMA_TAG });
    const r = await llm.embed("hello");
    expect(r?.model).toBe(CANON);
  });

  it("embedRemoteBatch canonicalizes the endpoint's echoed model for every row", async () => {
    mockEmbedEndpoint(OLLAMA_TAG);
    const llm = new LlamaCpp({ remoteEmbedUrl: "http://127.0.0.1:11435", remoteEmbedModel: OLLAMA_TAG });
    const rs = await llm.embedBatch(["a", "b"]);
    expect(rs.map(r => r?.model)).toEqual([CANON, CANON]);
  });

  it("falls back to the REQUESTED model, never the URL, when the reply omits `model`", async () => {
    mockEmbedEndpoint(undefined);
    const url = "http://127.0.0.1:11435";
    const llm = new LlamaCpp({ remoteEmbedUrl: url, remoteEmbedModel: OLLAMA_TAG });
    const single = await llm.embed("hello");
    const batch = await llm.embedBatch(["a", "b"]);
    expect(single?.model).toBe(CANON);
    expect(batch.map(r => r?.model)).toEqual([CANON, CANON]);
    // The old third door: `data.model || this.remoteEmbedUrl!` recorded the endpoint URL.
    expect(single?.model).not.toContain(url);
  });

  it("an unknown remote model still passes through — the endpoint stays authoritative", async () => {
    mockEmbedEndpoint("Nomic-Embed-Text");
    const llm = new LlamaCpp({ remoteEmbedUrl: "http://127.0.0.1:11435", remoteEmbedModel: "nomic-embed-text" });
    expect((await llm.embed("hello"))?.model).toBe("nomic-embed-text");
  });
});

// ---------------------------------------------------------------------------
// AC5 / AC6 — end-to-end against the store's write fence
// ---------------------------------------------------------------------------

function seed(store: Store, col: string, path: string, body: string): string {
  const hash = hashContent(body + path);
  const now = new Date().toISOString();
  store.insertContent(hash, body, now);
  store.insertDocument(col, path, path, hash, now, now);
  return hash;
}

/** A vault shaped like the LIVE ones: every vector recorded under the string "embeddinggemma". */
function liveShapedVault(): { store: Store; hashes: string[] } {
  const store = createStore(":memory:");
  store.ensureVecTable(4);
  const h1 = seed(store, "c", "a.md", "doc a");
  const h2 = seed(store, "c", "b.md", "doc b");
  store.insertEmbedding(h1, 0, 1, new Float32Array([1, 0, 0, 0]), CANON, new Date().toISOString(), "full", undefined, "c/a.md");
  return { store, hashes: [h1, h2] };
}

describe("AC5 — the original bug is dead: the local fallback no longer poisons or fails closed", () => {
  it("a vault holding `embeddinggemma` ACCEPTS a vector produced by the in-process local arm", async () => {
    const { store, hashes } = liveShapedVault();
    expect(store.getVecModels()).toEqual([CANON]);

    // Exactly the live trigger: remote embed endpoint unreachable → in-process arm serves the
    // fragment. Pre-fix this recorded "hf:ggml-org/..." — a SECOND identity for the SAME model,
    // which poisoned the vault (pre-.21) or aborted the whole embed run (post-.21).
    const llm = new LlamaCpp({});
    withStubbedLocalContext(llm);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const produced = await (llm as any).embedLocal("doc b") as { embedding: number[]; model: string };

    expect(() =>
      store.insertEmbedding(
        hashes[1]!, 0, 1, new Float32Array(produced.embedding), produced.model,
        new Date().toISOString(), "full", undefined, "c/b.md",
      )
    ).not.toThrow();

    // One identity, still `embeddinggemma` — no migration, no re-embed, no heterogeneity.
    expect(store.getVecModels()).toEqual([CANON]);
    expect(store.getVectorConsistency().vvCount).toBe(2);
  });

  it("the ollama ':latest' remote reply lands in the SAME vault without a second identity", async () => {
    const { store, hashes } = liveShapedVault();
    mockEmbedEndpoint(OLLAMA_TAG);
    const llm = new LlamaCpp({ remoteEmbedUrl: "http://127.0.0.1:11435", remoteEmbedModel: OLLAMA_TAG });
    const r = (await llm.embed("doc b"))!;
    store.insertEmbedding(hashes[1]!, 0, 1, new Float32Array(r.embedding), r.model, new Date().toISOString());
    expect(store.getVecModels()).toEqual([CANON]);
  });
});

describe("AC6 — the fence did NOT get weaker", () => {
  it("still REFUSES a genuinely foreign model and writes nothing", async () => {
    const { store, hashes } = liveShapedVault();
    const before = store.getVectorConsistency();

    mockEmbedEndpoint("nomic-embed-text");
    const llm = new LlamaCpp({ remoteEmbedUrl: "http://127.0.0.1:11435", remoteEmbedModel: "nomic-embed-text" });
    const r = (await llm.embed("doc b"))!;
    expect(r.model).toBe("nomic-embed-text"); // canonicalization did not merge it into embeddinggemma

    expect(() =>
      store.insertEmbedding(hashes[1]!, 0, 1, new Float32Array(r.embedding), r.model, new Date().toISOString())
    ).toThrow(VecWriteModelMismatchError);

    expect(store.getVectorConsistency().vvCount).toBe(before.vvCount);
    expect(store.getVecModels()).toEqual([CANON]);
  });
});
