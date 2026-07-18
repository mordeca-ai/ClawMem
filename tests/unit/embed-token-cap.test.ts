import { afterEach, describe, expect, it } from "bun:test";

import { LlamaCpp, resolveEmbedModelTokenCeiling } from "../../src/llm.ts";

/**
 * Tests for 5r0rd (follow-up to the lth retro — primary-fragment embed
 * failures at the model's token ceiling): truncateForEmbed's only defense
 * used to be char-based (maxRemoteEmbedChars, calibrated for EmbeddingGemma's
 * 2048-token context, not nomic-embed-text's 8192). Failure mode (b) from the
 * retro is structurally uncatchable by char truncation: high-entropy content
 * (UUIDs/paths/numeric scales) tokenizes far past a model's token ceiling
 * while sitting UNDER the char cap, so char truncation never fires.
 *
 * This file proves:
 *  - AC1: a token-aware cap catches failure mode (b) (stubbed tokenizer, no
 *    model load) where the char-only path would not.
 *  - AC2: resolveEmbedModelTokenCeiling resolves the right ceiling per model,
 *    with CLAWMEM_EMBED_MAX_TOKENS winning when set.
 *  - AC3: a tokenizer failure (CLAWMEM_NO_LOCAL_MODELS=true, no real model)
 *    falls back to the existing char-based cap without throwing.
 */

// A subclass that overrides the tokenizer seam with a fake 1-char-per-token
// tokenizer — deliberately WORSE than any real tokenizer's char/token ratio,
// modeling the retro's "high-entropy content tokenizes far past the assumed
// ~3 chars/token" finding. No real model is loaded.
class StubTokenizerLlamaCpp extends LlamaCpp {
  protected override async truncateToTokenCeiling(text: string, maxTokens: number): Promise<string> {
    // 1 token per char (worst case): anything longer than maxTokens chars
    // is "over ceiling" under this stub and must be sliced to exactly
    // maxTokens tokens (== maxTokens chars here).
    if (text.length <= maxTokens) return text;
    return text.slice(0, maxTokens);
  }
}

// A subclass whose tokenizer seam always throws — simulates the tokenizer
// being genuinely unavailable independent of CLAWMEM_NO_LOCAL_MODELS (e.g. a
// corrupt cached GGUF), to prove the char-cap fallback is unconditional.
class ThrowingTokenizerLlamaCpp extends LlamaCpp {
  protected override async truncateToTokenCeiling(): Promise<string> {
    throw new Error("simulated tokenizer failure");
  }
}

const originalNoLocalModels = process.env.CLAWMEM_NO_LOCAL_MODELS;
const originalMaxTokensEnv = process.env.CLAWMEM_EMBED_MAX_TOKENS;

function restoreEnv() {
  if (originalNoLocalModels === undefined) delete process.env.CLAWMEM_NO_LOCAL_MODELS;
  else process.env.CLAWMEM_NO_LOCAL_MODELS = originalNoLocalModels;
  if (originalMaxTokensEnv === undefined) delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
  else process.env.CLAWMEM_EMBED_MAX_TOKENS = originalMaxTokensEnv;
}

/** A healthy /v1/embeddings stub server that echoes back the received input. */
function startEchoEmbedServer() {
  const received: string[] = [];
  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    async fetch(req) {
      const body = (await req.json()) as { input: string | string[] };
      if (Array.isArray(body.input)) received.push(...body.input);
      else received.push(body.input);
      const inputs = Array.isArray(body.input) ? body.input : [body.input];
      return new Response(
        JSON.stringify({
          data: inputs.map((_, i) => ({ embedding: [0.1, 0.2, 0.3], index: i })),
          model: "test-embed",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}`,
    received,
    stop: () => server.stop(true),
  };
}

describe("per-model embed token ceiling resolution (5r0rd AC2)", () => {
  afterEach(restoreEnv);

  it("resolves nomic-embed-text to 8192", () => {
    delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
    expect(resolveEmbedModelTokenCeiling("nomic-embed-text")).toBe(8192);
    expect(resolveEmbedModelTokenCeiling("Nomic-Embed-Text-v1.5")).toBe(8192); // case-insensitive substring
  });

  it("resolves embeddinggemma to 2048", () => {
    delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
    expect(resolveEmbedModelTokenCeiling("embeddinggemma")).toBe(2048);
    expect(resolveEmbedModelTokenCeiling("EmbeddingGemma-300M")).toBe(2048);
  });

  it("resolves an unknown model to the conservative 2048 default", () => {
    delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
    expect(resolveEmbedModelTokenCeiling("some-unheard-of-embed-model")).toBe(2048);
    expect(resolveEmbedModelTokenCeiling(undefined)).toBe(2048);
  });

  it("resolves granite-family to 512", () => {
    delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
    expect(resolveEmbedModelTokenCeiling("granite-embedding-278m")).toBe(512);
  });

  it("CLAWMEM_EMBED_MAX_TOKENS env override wins over the model map", () => {
    process.env.CLAWMEM_EMBED_MAX_TOKENS = "999";
    expect(resolveEmbedModelTokenCeiling("nomic-embed-text")).toBe(999);
    expect(resolveEmbedModelTokenCeiling("embeddinggemma")).toBe(999);
    expect(resolveEmbedModelTokenCeiling("anything-unknown")).toBe(999);
  });

  it("a non-positive/invalid CLAWMEM_EMBED_MAX_TOKENS is ignored (falls back to the map)", () => {
    process.env.CLAWMEM_EMBED_MAX_TOKENS = "0";
    expect(resolveEmbedModelTokenCeiling("embeddinggemma")).toBe(2048);
    process.env.CLAWMEM_EMBED_MAX_TOKENS = "not-a-number";
    expect(resolveEmbedModelTokenCeiling("embeddinggemma")).toBe(2048);
  });
});

describe("token-aware truncateForEmbed via embed() (5r0rd AC1)", () => {
  afterEach(restoreEnv);

  it(
    "AC1: high-entropy text UNDER the char cap but OVER the model's token ceiling is truncated by the token-aware path",
    async () => {
      delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
      // embeddinggemma ceiling = 2048 (tokens). Build a UUID-dense string
      // that is well under maxRemoteEmbedChars (default 6000 chars) but,
      // under the stub's 1-token-per-char worst case, is "3000 tokens" —
      // over the 2048 ceiling. A char-only cap (6000 chars) would NOT
      // truncate this at all.
      const uuidDense = Array.from({ length: 100 }, () => crypto.randomUUID()).join("-");
      expect(uuidDense.length).toBeLessThan(6000); // confirms it's under the char cap
      expect(uuidDense.length).toBeGreaterThan(2048); // and over the (stubbed) token ceiling

      const echo = startEchoEmbedServer();
      try {
        const llm = new StubTokenizerLlamaCpp({
          remoteEmbedUrl: echo.url,
          remoteEmbedModel: "embeddinggemma",
        });
        const result = await llm.embed(uuidDense);
        expect(result).not.toBeNull();
        expect(echo.received).toHaveLength(1);
        // Truncated to <= ceiling tokens (== <= ceiling chars under the stub's 1:1 mapping).
        expect(echo.received[0]!.length).toBe(2048);
        expect(echo.received[0]!.length).toBeLessThan(uuidDense.length);
      } finally {
        echo.stop();
      }
    },
    10_000
  );

  it(
    "text within the token ceiling is left untouched",
    async () => {
      delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
      const short = "a short fragment of text";
      const echo = startEchoEmbedServer();
      try {
        const llm = new StubTokenizerLlamaCpp({
          remoteEmbedUrl: echo.url,
          remoteEmbedModel: "embeddinggemma",
        });
        await llm.embed(short);
        expect(echo.received[0]).toBe(short);
      } finally {
        echo.stop();
      }
    },
    10_000
  );

  it(
    "AC3: a tokenizer failure falls back to the char-based cap without throwing (CLAWMEM_NO_LOCAL_MODELS=true)",
    async () => {
      process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
      process.env.CLAWMEM_EMBED_MAX_CHARS = process.env.CLAWMEM_EMBED_MAX_CHARS || "6000";
      // Real LlamaCpp (no stub): tokenize()/countTokens() call
      // ensureGenerateModel() -> resolveModel(), which throws immediately
      // when CLAWMEM_NO_LOCAL_MODELS=true (no download allowed). This
      // exercises the REAL default tokenizer seam's failure path, not a
      // simulated one.
      const oversizedChars = "x".repeat(7000); // over the 6000-char cap
      const echo = startEchoEmbedServer();
      try {
        const llm = new LlamaCpp({
          remoteEmbedUrl: echo.url,
          remoteEmbedModel: "embeddinggemma",
        });
        const result = await llm.embed(oversizedChars);
        expect(result).not.toBeNull(); // no exception escaped truncateForEmbed
        expect(echo.received[0]!.length).toBe(6000); // bounded by the char-cap fallback
      } finally {
        echo.stop();
      }
    },
    10_000
  );

  it(
    "AC3 (explicit tokenizer-throw seam): a thrown tokenizer error still bounds input via the char cap",
    async () => {
      delete process.env.CLAWMEM_NO_LOCAL_MODELS;
      process.env.CLAWMEM_EMBED_MAX_CHARS = process.env.CLAWMEM_EMBED_MAX_CHARS || "6000";
      const oversizedChars = "y".repeat(6500);
      const echo = startEchoEmbedServer();
      try {
        // embeddinggemma (ceiling 2048) so the char-capped text (6000 chars)
        // is still ABOVE the token ceiling and the tokenizer seam actually
        // gets invoked (and throws) rather than being skipped by the
        // char<=ceiling short-circuit.
        const llm = new ThrowingTokenizerLlamaCpp({
          remoteEmbedUrl: echo.url,
          remoteEmbedModel: "embeddinggemma",
        });
        const result = await llm.embed(oversizedChars);
        expect(result).not.toBeNull();
        expect(echo.received[0]!.length).toBe(6000);
      } finally {
        echo.stop();
      }
    },
    10_000
  );

  it(
    "cloud embedding (API key set) bypasses truncation entirely, as before",
    async () => {
      delete process.env.CLAWMEM_EMBED_MAX_TOKENS;
      const echo = startEchoEmbedServer();
      try {
        const llm = new StubTokenizerLlamaCpp({
          remoteEmbedUrl: echo.url,
          remoteEmbedApiKey: "test-key",
          remoteEmbedModel: "embeddinggemma",
        });
        const long = "z".repeat(3000); // over the stub's 2048 "ceiling"
        await llm.embed(long);
        expect(echo.received[0]).toBe(long); // untouched — cloud handles its own window
      } finally {
        echo.stop();
      }
    },
    10_000
  );
});
