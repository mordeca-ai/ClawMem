import { afterEach, beforeEach, describe, expect, it } from "bun:test";

import { LlamaCpp, resetEmbedContextCache } from "../../src/llm.ts";
import { EMBED_CHARS_PER_TOKEN, FALLBACK_EMBED_CONTEXT_TOKENS } from "../../src/limits.ts";

/**
 * master-harness-vn4rz.42 — bound the embed input at the chokepoint.
 *
 * The splitter's `full` fragment is the only class that bypasses chunkContent,
 * so it can hand the embed path up to MAX_SPLITTER_INPUT_CHARS (500_000)
 * characters. An ollama endpoint serving a small-context embed model does not
 * reject such a body — it stalls, the remote-fetch deadline fires, the abort is
 * classified as transport, the remote-embed breaker trips, and every remaining
 * fragment fails fast (measured cascade: 4,200 embedded vs 4,332 failures).
 *
 * These tests prove:
 *  - A1: an input far larger than the model context is truncated to the DERIVED
 *        budget BEFORE the outbound fetch (asserted on the body the endpoint
 *        actually received).
 *  - A2: the budget is read from the endpoint's advertised `context_length`
 *        (/api/show), not from a constant — two endpoints advertising different
 *        distinctive contexts produce two different budgets, and neither equals
 *        the fallback. A missing /api/show falls back to the named constant.
 *  - A3: an oversized fragment no longer triggers the breaker cascade — a
 *        stalling endpoint (stalls on anything over its context) still embeds
 *        the oversized fragment AND every normal fragment behind it.
 */

/** A distinctive advertised context no fallback path could ever produce. */
const DISTINCT_CONTEXT_TOKENS = 777;
const DISTINCT_CONTEXT_TOKENS_B = 333;
const FALLBACK_BUDGET_CHARS = FALLBACK_EMBED_CONTEXT_TOKENS * EMBED_CHARS_PER_TOKEN;

function budgetFor(tokens: number): number {
  return tokens * EMBED_CHARS_PER_TOKEN;
}

interface StubOptions {
  /** context_length advertised by /api/show; null = serve 404 (non-ollama endpoint). */
  contextTokens: number | null;
  /** When set, /v1/embeddings NEVER responds for an input longer than this. */
  stallOverChars?: number;
}

function startStubEmbedServer(opts: StubOptions) {
  const received: string[] = [];
  let showCalls = 0;
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    idleTimeout: 0,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/api/show") {
        showCalls++;
        if (opts.contextTokens === null) return new Response("not found", { status: 404 });
        return Response.json({
          model_info: {
            "general.architecture": "gemma3",
            "gemma3.embedding_length": 768,
            "gemma3.context_length": opts.contextTokens,
          },
        });
      }
      if (url.pathname === "/v1/embeddings") {
        const body = (await req.json()) as { input: string | string[] };
        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        received.push(...inputs);
        if (opts.stallOverChars !== undefined && inputs.some(i => i.length > opts.stallOverChars!)) {
          // Model the measured failure: the endpoint accepts the body and never
          // answers, so the client's deadline is what ends the call.
          await new Promise(() => {});
        }
        return Response.json({
          data: inputs.map((_, i) => ({ embedding: [0.1, 0.2, 0.3], index: i })),
          model: "embeddinggemma",
        });
      }
      return new Response("not found", { status: 404 });
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}`,
    received,
    get showCalls() { return showCalls; },
    stop: () => server.stop(true),
  };
}

const ENV_KEYS = [
  "CLAWMEM_EMBED_MAX_CHARS",
  "CLAWMEM_EMBED_MAX_TOKENS",
  "CLAWMEM_EMBED_CHARS_PER_TOKEN",
  "CLAWMEM_NO_LOCAL_MODELS",
  "CLAWMEM_REMOTE_FETCH_TIMEOUT_MS",
] as const;
const saved: Record<string, string | undefined> = {};

beforeEach(() => {
  for (const k of ENV_KEYS) {
    saved[k] = process.env[k];
    delete process.env[k];
  }
  // Never load or download a local GGUF from a unit test.
  process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
  resetEmbedContextCache();
});

afterEach(() => {
  for (const k of ENV_KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k]!;
  }
  resetEmbedContextCache();
});

describe("vn4rz.42 A1: oversized embed input is truncated before the fetch", () => {
  it("a 200k-char input reaches the endpoint at exactly the derived budget", async () => {
    const stub = startStubEmbedServer({ contextTokens: DISTINCT_CONTEXT_TOKENS });
    try {
      // Comparable to the measured 127,832-char `full` fragment.
      const oversized = "the quick brown fox jumps over the lazy dog. ".repeat(4500);
      expect(oversized.length).toBeGreaterThan(190_000);

      const llm = new LlamaCpp({
        remoteEmbedUrl: stub.url,
        remoteEmbedModel: "embeddinggemma",
      });
      const result = await llm.embed(oversized);

      expect(result).not.toBeNull();
      expect(stub.received).toHaveLength(1);
      // The load-bearing assertion: what the endpoint RECEIVED, not what returned.
      expect(stub.received[0]!.length).toBe(budgetFor(DISTINCT_CONTEXT_TOKENS));
      // Prefix-preserving truncation, not a re-chunk.
      expect(oversized.startsWith(stub.received[0]!)).toBe(true);
    } finally {
      stub.stop();
    }
  }, 20_000);

  it("the batch path is bounded identically (every element, not just the first)", async () => {
    const stub = startStubEmbedServer({ contextTokens: DISTINCT_CONTEXT_TOKENS });
    try {
      const oversized = "x".repeat(150_000);
      const normal = "a short fragment";
      const llm = new LlamaCpp({
        remoteEmbedUrl: stub.url,
        remoteEmbedModel: "embeddinggemma",
      });
      const results = await llm.embedBatch([oversized, normal, oversized]);

      expect(results.filter(r => r !== null)).toHaveLength(3);
      expect(stub.received).toHaveLength(3);
      expect(stub.received[0]!.length).toBe(budgetFor(DISTINCT_CONTEXT_TOKENS));
      expect(stub.received[1]!.length).toBe(normal.length);
      expect(stub.received[2]!.length).toBe(budgetFor(DISTINCT_CONTEXT_TOKENS));
    } finally {
      stub.stop();
    }
  }, 20_000);
});

describe("vn4rz.42 A2: the budget is DERIVED from the endpoint's advertised context", () => {
  it("two endpoints advertising different contexts produce different budgets (neither is the fallback)", async () => {
    const a = startStubEmbedServer({ contextTokens: DISTINCT_CONTEXT_TOKENS });
    const b = startStubEmbedServer({ contextTokens: DISTINCT_CONTEXT_TOKENS_B });
    try {
      const oversized = "y".repeat(120_000);

      const llmA = new LlamaCpp({ remoteEmbedUrl: a.url, remoteEmbedModel: "embeddinggemma" });
      expect(await llmA.embed(oversized)).not.toBeNull();

      const llmB = new LlamaCpp({ remoteEmbedUrl: b.url, remoteEmbedModel: "embeddinggemma" });
      expect(await llmB.embed(oversized)).not.toBeNull();

      expect(a.received[0]!.length).toBe(budgetFor(DISTINCT_CONTEXT_TOKENS));
      expect(b.received[0]!.length).toBe(budgetFor(DISTINCT_CONTEXT_TOKENS_B));
      // The control: a hardcoded budget would make these equal, and would equal
      // the fallback. Both must be false.
      expect(a.received[0]!.length).not.toBe(b.received[0]!.length);
      expect(a.received[0]!.length).not.toBe(FALLBACK_BUDGET_CHARS);
      expect(b.received[0]!.length).not.toBe(FALLBACK_BUDGET_CHARS);
    } finally {
      a.stop();
      b.stop();
    }
  }, 20_000);

  it("the /api/show probe is memoized per (endpoint, model) — one call across many embeds", async () => {
    const stub = startStubEmbedServer({ contextTokens: DISTINCT_CONTEXT_TOKENS });
    try {
      const llm = new LlamaCpp({ remoteEmbedUrl: stub.url, remoteEmbedModel: "embeddinggemma" });
      for (let i = 0; i < 5; i++) expect(await llm.embed("z".repeat(50_000))).not.toBeNull();
      expect(stub.showCalls).toBe(1);
      expect(stub.received.every(r => r.length === budgetFor(DISTINCT_CONTEXT_TOKENS))).toBe(true);
    } finally {
      stub.stop();
    }
  }, 20_000);

  it("an endpoint with no /api/show falls back to the named constant, still bounded", async () => {
    const stub = startStubEmbedServer({ contextTokens: null });
    try {
      const llm = new LlamaCpp({ remoteEmbedUrl: stub.url, remoteEmbedModel: "embeddinggemma" });
      expect(await llm.embed("w".repeat(130_000))).not.toBeNull();
      expect(stub.received[0]!.length).toBe(FALLBACK_BUDGET_CHARS);
    } finally {
      stub.stop();
    }
  }, 20_000);

  it("CLAWMEM_EMBED_MAX_CHARS still wins over the advertised context", async () => {
    process.env.CLAWMEM_EMBED_MAX_CHARS = "512";
    const stub = startStubEmbedServer({ contextTokens: DISTINCT_CONTEXT_TOKENS });
    try {
      const llm = new LlamaCpp({ remoteEmbedUrl: stub.url, remoteEmbedModel: "embeddinggemma" });
      expect(await llm.embed("v".repeat(90_000))).not.toBeNull();
      expect(stub.received[0]!.length).toBe(512);
    } finally {
      stub.stop();
    }
  }, 20_000);
});

describe("vn4rz.42 A3: no breaker cascade behind an oversized fragment", () => {
  it("an oversized fragment followed by normal fragments all embed successfully", async () => {
    // Short deadline so a RED run (unbounded input -> stall) fails in seconds
    // rather than at the 60s production default.
    process.env.CLAWMEM_REMOTE_FETCH_TIMEOUT_MS = "2000";
    const stub = startStubEmbedServer({
      contextTokens: DISTINCT_CONTEXT_TOKENS,
      // Exactly the failure mechanism: the endpoint stalls on anything past its
      // advertised context, it never returns an error.
      stallOverChars: budgetFor(DISTINCT_CONTEXT_TOKENS),
    });
    try {
      const llm = new LlamaCpp({ remoteEmbedUrl: stub.url, remoteEmbedModel: "embeddinggemma" });

      const oversized = await llm.embed("q".repeat(127_832));
      expect(oversized).not.toBeNull();

      // The cascade victims: everything behind the oversized fragment.
      for (let i = 0; i < 5; i++) {
        const r = await llm.embed(`normal fragment ${i}`);
        expect(r).not.toBeNull();
      }
      expect(stub.received).toHaveLength(6);
    } finally {
      stub.stop();
    }
  }, 30_000);
});
