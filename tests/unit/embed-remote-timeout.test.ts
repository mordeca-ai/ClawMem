import { afterEach, describe, expect, it } from "bun:test";

import { LlamaCpp } from "../../src/llm.ts";

/**
 * Tests for 1d1fn (follow-up to 62xr.5 — embed timeout storm, finance corpus
 * 0 fragments, 38 fails, ~3hr): embed/LLM/rerank fetches must carry an
 * explicit deadline, and a timeout DOMException must be classified as a
 * transport failure so the circuit breaker trips on the FIRST hang instead
 * of every fragment eating a full unbounded wait.
 *
 * The hanging endpoint here is a raw Bun.listen() TCP socket that accepts
 * the connection and then never writes a response and never closes — this
 * is deliberately NOT Bun.serve() (which imposes its own short server-side
 * idleTimeout, ~10s in this Bun version, and would mask the bug by
 * producing an ECONNRESET the existing code already classified correctly).
 * A raw accept-and-silence socket reproduces the actual 62xr.5 shape: a
 * remote GPU server that is alive at the TCP layer (GPU-contention queued)
 * but never sends bytes back. Empirically probed in this session: an
 * un-timed-out fetch() against this fixture did not resolve within 300s
 * (see agents/skills/debugging/artifacts/clawmem/retro.md 62xr.5 section).
 */

type HangingServer = {
  url: string;
  connectionCount: () => number;
  stop: () => void;
};

function startHangingServer(): HangingServer {
  let connections = 0;
  const server = Bun.listen({
    hostname: "127.0.0.1",
    port: 0,
    socket: {
      open() {
        connections++;
      },
      data() {
        /* swallow the request; never respond */
      },
      close() {},
      error() {},
    },
  });
  return {
    url: `http://127.0.0.1:${server.port}`,
    connectionCount: () => connections,
    stop: () => server.stop(true),
  };
}

const originalNoLocalModels = process.env.CLAWMEM_NO_LOCAL_MODELS;

describe("remote embed fetch timeout (1d1fn)", () => {
  afterEach(() => {
    if (originalNoLocalModels === undefined) delete process.env.CLAWMEM_NO_LOCAL_MODELS;
    else process.env.CLAWMEM_NO_LOCAL_MODELS = originalNoLocalModels;
  });

  it(
    "AC1: embed() against a hung server aborts at the configured timeout, not Bun's unbounded wait",
    async () => {
      process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
      const hung = startHangingServer();
      try {
        const llm = new LlamaCpp({
          remoteEmbedUrl: hung.url,
          remoteFetchTimeoutMs: 300,
        });

        const start = Date.now();
        const result = await llm.embed("test fragment");
        const elapsed = Date.now() - start;

        expect(result).toBeNull();
        // Must abort at ~the configured deadline, with generous slack for
        // CI jitter — and nowhere close to Bun's multi-minute idle default.
        expect(elapsed).toBeGreaterThanOrEqual(250);
        expect(elapsed).toBeLessThan(3000);
      } finally {
        hung.stop();
      }
    },
    10_000
  );

  it(
    // Contract updated at the v0.17.0 upstream sync (9jyc0): an abort/timeout is now
    // classified as caller-driven cancellation, NOT a transport failure — it must NOT
    // trip the 60s remote-down cooldown (which would needlessly force local fallback
    // on the query path whose deadline merely elapsed). The 1d1fn guarantee that
    // survives is the per-fetch deadline itself: every fragment fails at the bound,
    // never at Bun's unbounded connect/idle wait.
    "AC2 (v0.17 revision): timeout is cancellation — bounded per-fragment failure, breaker NOT tripped",
    async () => {
      process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
      const hung = startHangingServer();
      try {
        const llm = new LlamaCpp({
          remoteEmbedUrl: hung.url,
          remoteFetchTimeoutMs: 300,
        });

        // First fragment eats the timeout; classified as cancellation, no cooldown.
        const first = await llm.embed("fragment 1");
        expect(first).toBeNull();
        const connectionsAfterFirst = hung.connectionCount();
        expect(connectionsAfterFirst).toBeGreaterThanOrEqual(1);

        // A subsequent fragment is still ATTEMPTED (breaker untripped — upstream v0.17
        // contract) and again fails at ~the configured bound, not an unbounded wait.
        const start = Date.now();
        const second = await llm.embed("fragment 2");
        const elapsed = Date.now() - start;

        expect(second).toBeNull();
        expect(elapsed).toBeGreaterThanOrEqual(250); // real attempt, bounded by the deadline
        expect(elapsed).toBeLessThan(3000);
        expect(hung.connectionCount()).toBeGreaterThan(connectionsAfterFirst); // new socket — no cooldown short-circuit
      } finally {
        hung.stop();
      }
    },
    10_000
  );

  it(
    "control: a healthy endpoint still embeds successfully with the timeout wired in",
    async () => {
      const healthy = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch() {
          return new Response(
            JSON.stringify({ data: [{ embedding: [0.1, 0.2, 0.3] }], model: "test-embed" }),
            { status: 200, headers: { "Content-Type": "application/json" } }
          );
        },
      });
      try {
        const llm = new LlamaCpp({
          remoteEmbedUrl: `http://127.0.0.1:${healthy.port}`,
          remoteFetchTimeoutMs: 300,
        });

        const result = await llm.embed("healthy fragment");
        expect(result).not.toBeNull();
        expect(result?.embedding).toEqual([0.1, 0.2, 0.3]);
      } finally {
        healthy.stop(true);
      }
    },
    10_000
  );

  it(
    "AC2 (batch path): embedRemoteBatch on a hung server also trips the breaker via isTransportError",
    async () => {
      process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
      const hung = startHangingServer();
      try {
        const llm = new LlamaCpp({
          remoteEmbedUrl: hung.url,
          remoteFetchTimeoutMs: 300,
        });

        const start = Date.now();
        const results = await llm.embedBatch(["a", "b", "c"]);
        const elapsed = Date.now() - start;

        expect(results).toEqual([null, null, null]);
        expect(elapsed).toBeLessThan(3000);

        // Second batch call should short-circuit (breaker tripped).
        const start2 = Date.now();
        const results2 = await llm.embedBatch(["d", "e"]);
        const elapsed2 = Date.now() - start2;
        expect(results2).toEqual([null, null]);
        expect(elapsed2).toBeLessThan(150);
      } finally {
        hung.stop();
      }
    },
    10_000
  );
});
