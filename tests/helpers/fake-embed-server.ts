/**
 * Deterministic fake OpenAI-compatible `/v1/embeddings` endpoint for exit-code
 * contract tests (master-harness-zkjyh).
 *
 * WHY a real HTTP server and not a stubbed LlamaCpp: the exit-code contract is a
 * PROCESS-level contract (`clawmem embed` must exit non-zero), so the tests spawn
 * the real CLI as a subprocess. A subprocess cannot receive an in-process mock, so
 * the seam has to be the wire.
 *
 * WHY a bag-of-words hash embedding and not a constant/content-hash vector: the
 * geometry canary (src/canary.ts) runs BEFORE every branch these tests exercise and
 * FAILS CLOSED. Its four pair-separation margins require an embedding that genuinely
 * discriminates related from unrelated text. A content-hash vector scores ~0 on every
 * margin and aborts the run at the canary — every downstream assertion would then be
 * testing the canary, not the branch under test. Hashed word-token vectors clear the
 * 0.10 absolute floor with min margin ~0.236 at dim 512 and 1024 (measured), so the
 * canary passes and control reaches the branch being tested.
 *
 * The margin is dim-SENSITIVE (a hash collision between the `rel_a`/`unrel` probe tokens
 * drops m_rel to 0.046 at dims 128/256/384/768 — which aborts at the canary and would
 * silently misattribute every downstream assertion). `CANARY_SAFE_DIMS` below is the
 * verified set, and `assertCanarySafeDim` is asserted in the test suite so a future dim
 * change cannot reintroduce that false attribution.
 */

export type FakeEmbedOptions = {
  /** Output dimension. MUST be one of CANARY_SAFE_DIMS (see module docblock). */
  dim?: number;
  /** Value reported as `model` in the response body (drives the model-identity guard). */
  model?: string;
  /**
   * Inputs (exact, post-formatting) that the server should FAIL.
   * Single-embed requests answer HTTP 500; batch requests silently omit the index,
   * which is exactly how a real partial-batch failure reaches runBatchedEmbed
   * (`results[i] === null` → failedFragments++).
   */
  failInputs?: (input: string) => boolean;
};

export type FakeEmbedServer = {
  url: string;
  /** Every input the server was asked to embed, in order. */
  seen: string[];
  stop: () => void;
};

/** Deterministic hashed word-token embedding; see module docblock for the rationale. */
export function fakeEmbedVector(text: string, dim: number): number[] {
  const v = new Float64Array(dim);
  for (const tok of text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)) {
    let h = 2166136261;
    for (let i = 0; i < tok.length; i++) {
      h ^= tok.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    v[(h >>> 0) % dim]! += 1;
  }
  let norm = 0;
  for (const x of v) norm += x * x;
  norm = Math.sqrt(norm) || 1;
  return Array.from(v, x => x / norm);
}

/** Dimensions measured to clear every canary pair-separation floor. */
export const CANARY_SAFE_DIMS = [320, 448, 512, 1024, 1536, 2048] as const;

/**
 * `POST /v1/rerank` scorer, needed ONLY so `clawmem doctor` can reach a zero-issue
 * state for the negative control. It is a GOLDEN-SET LOOKUP, not a model: the doctor
 * probe uses deliberate HARD negatives, which no bag-of-words scorer can separate
 * (measured min margin -0.38 vs the required +0.25), and the reranker is not the
 * subject of these tests. Docs arrive truncated to 400 chars, so match by prefix.
 * Anything not in the golden set gets a neutral 0.5.
 */
let _goldenPairs: { relevant: string; hardNegative: string }[] | null = null;
export function goldenRerankScore(text: string): number {
  if (!_goldenPairs) {
    const raw = Bun.file(new URL("../../src/health/rerank-golden.json", import.meta.url).pathname);
    // Synchronous read keeps the request handler simple; the file is a few KB.
    const parsed = JSON.parse(require("node:fs").readFileSync(raw.name!, "utf-8")) as {
      triples: { relevant: string; hardNegative: string }[];
    };
    _goldenPairs = parsed.triples;
  }
  for (const t of _goldenPairs) {
    if (t.relevant.startsWith(text) || text.startsWith(t.relevant.slice(0, 400))) return 0.95;
    if (t.hardNegative.startsWith(text) || text.startsWith(t.hardNegative.slice(0, 400))) return 0.10;
  }
  return 0.5;
}

export function startFakeEmbedServer(opts: FakeEmbedOptions = {}): FakeEmbedServer {
  const dim = opts.dim ?? 512;
  const model = opts.model ?? "fake-embed-a";
  const shouldFail = opts.failInputs ?? (() => false);
  const seen: string[] = [];

  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname.endsWith("/v1/rerank")) {
        const rr = (await req.json()) as { query: string; documents: string[] };
        return Response.json({
          results: rr.documents.map((text, index) => ({ index, relevance_score: goldenRerankScore(text) })),
        });
      }
      if (!url.pathname.endsWith("/v1/embeddings")) {
        return new Response("not found", { status: 404 });
      }
      const body = (await req.json()) as { input: string | string[] };
      const inputs = Array.isArray(body.input) ? body.input : [body.input];
      seen.push(...inputs);

      if (!Array.isArray(body.input)) {
        // Single-embed path (canary probes, dimension probe).
        if (shouldFail(inputs[0]!)) return new Response("injected failure", { status: 500 });
        return Response.json({
          data: [{ embedding: fakeEmbedVector(inputs[0]!, dim), index: 0 }],
          model,
        });
      }
      // Batch path: omit failing indices entirely — the OpenAI-compat client maps
      // present `index` values back into a null-filled array, so an omitted index
      // becomes a null result (one failed fragment) without failing the whole batch.
      const data = inputs
        .map((t, index) => ({ embedding: fakeEmbedVector(t, dim), index }))
        .filter((_, i) => !shouldFail(inputs[i]!));
      return Response.json({ data, model });
    },
  });

  return {
    url: `http://127.0.0.1:${server.port}`,
    seen,
    stop: () => server.stop(true),
  };
}
