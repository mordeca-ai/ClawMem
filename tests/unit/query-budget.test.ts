import { afterAll, beforeAll, describe, expect, it, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, expandQueryCacheKey } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import {
  QueryBudget,
  QueryBudgetExpired,
  queryBudgetMs,
  withinBudget,
} from "../../src/query-budget.ts";
import { fakeEmbedVector, goldenRerankScore } from "../helpers/fake-embed-server.ts";

/**
 * master-harness-vn4rz.134 — one whole-request budget for POST /query.
 *
 * Unit half: the budget clock and withinBudget with an injected clock.
 * End-to-end half: the real `clawmem serve` in a subprocess on a throwaway vault, with the
 * blocking seam from vn4rz.131 and a fake rerank endpoint whose delay a test sets. Bounds are
 * stated as budget + a tolerance that dwarfs host jitter, and each RED control uses a delay far
 * past it, so no assertion depends on how loaded the host is.
 */

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

describe("queryBudgetMs", () => {
  it("is off for unset, empty, zero and anything not a plain integer", () => {
    for (const v of [undefined, "", "  ", "0", "-1", "1.5", "7000ms", "soon"]) {
      expect(queryBudgetMs({ CLAWMEM_QUERY_BUDGET_MS: v })).toBe(0);
    }
  });
  it("parses a positive integer", () => {
    expect(queryBudgetMs({ CLAWMEM_QUERY_BUDGET_MS: " 6500 " })).toBe(6500);
  });
  it("fromEnv is null when off", () => {
    expect(QueryBudget.fromEnv(0, {})).toBeNull();
    expect(QueryBudget.fromEnv(0, { CLAWMEM_QUERY_BUDGET_MS: "10" })).not.toBeNull();
  });
});

describe("QueryBudget", () => {
  it("counts from arrival, not from construction", () => {
    let now = 1000;
    const b = new QueryBudget(500, 800, () => now);
    expect(b.remainingMs()).toBe(300);
    expect(b.expired()).toBe(false);
    now = 1300;
    expect(b.expired()).toBe(true);
    expect(b.remainingMs()).toBe(0);
  });
});

describe("withinBudget", () => {
  it("returns the value when the work finishes in time", async () => {
    const b = new QueryBudget(1000);
    expect(await withinBudget(b, "t", async () => "ok")).toBe("ok");
  });
  it("aborts the signal and rejects at the budget", async () => {
    const b = new QueryBudget(30);
    let signal: AbortSignal | undefined;
    const err = await withinBudget(b, "rerank", (s) => {
      signal = s;
      return new Promise<never>(() => {});
    }).catch((e) => e);
    expect(err).toBeInstanceOf(QueryBudgetExpired);
    expect(signal!.aborted).toBe(true);
  });
  it("does not start work on a spent budget", async () => {
    let ran = false;
    const b = new QueryBudget(5, 0, () => 100);
    const err = await withinBudget(b, "t", async () => {
      ran = true;
    }).catch((e) => e);
    expect(err).toBeInstanceOf(QueryBudgetExpired);
    expect(ran).toBe(false);
  });
});

// ---------------------------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------------------------

const REPO_ROOT = new URL("../..", import.meta.url).pathname;
const ROOT = mkdtempSync(join(tmpdir(), "clawmem-query-budget-"));
const DB = join(ROOT, "index.sqlite");
const DIM = 512;
const MODEL = "fake-embed-a";
const BLOCKED = "zzblock";
const Q_OK = "orbital ferry schedule";
const Q_BLOCK = `orbital ferry schedule ${BLOCKED}`;
const Q_RERANK = "orbital ferry cutoffprobe";

let fake: ReturnType<typeof Bun.serve>;
let rerankDelayMs = 0;
let rerankCalls = 0;

function seedVault(): void {
  mkdirSync(join(ROOT, "alpha"), { recursive: true });
  writeFileSync(
    join(ROOT, "config.yaml"),
    [
      "collections:",
      "  alpha:",
      `    path: ${join(ROOT, "alpha")}`,
      '    pattern: "**/*.md"',
      "",
    ].join("\n"),
  );
  const store = createStore(DB);
  const now = new Date().toISOString();
  store.ensureVecTable(DIM);
  store.db.transaction(() => {
    const docs: Record<string, string> = {
      "ferry.md": `# Orbital ferry schedule ${BLOCKED}\n\nThe orbital ferry cutoffprobe departs every six hours.\n`,
      "dock.md": `# Docking bay rota\n\nFerry crews rotate through the docking bays weekly.\n`,
      "cargo.md": `# Cargo manifest\n\nCargo on the orbital ferry is weighed before departure.\n`,
    };
    for (const [path, body] of Object.entries(docs)) {
      const hash = hashContent(body + path);
      store.insertContent(hash, body, now);
      store.insertDocument("alpha", path, path, hash, now, now);
      store.insertEmbedding(
        hash,
        0,
        1,
        new Float32Array(fakeEmbedVector(body, DIM)),
        MODEL,
        now,
        "full",
        undefined,
        `alpha/${path}`,
      );
    }
    // Known at once (llm_cache hit, no expansion), so no test waits on a live LLM call.
    for (const q of [Q_OK, Q_BLOCK, Q_RERANK]) {
      store.db
        .prepare(`INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)`)
        .run(expandQueryCacheKey(q), "[]", now);
    }
  })();
  store.close();
}

function serveEnv(extra: Record<string, string>): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (
      v !== undefined &&
      !k.startsWith("CLAWMEM_") &&
      k !== "INDEX_PATH" &&
      !k.endsWith("SESSION_ID")
    )
      env[k] = v;
  }
  const url = `http://127.0.0.1:${fake.port}`;
  return {
    ...env,
    INDEX_PATH: DB,
    CLAWMEM_CONFIG_DIR: ROOT,
    CLAWMEM_FOCUS_ROOT: join(ROOT, "focus"),
    CLAWMEM_EMBED_URL: url,
    CLAWMEM_LLM_URL: url,
    CLAWMEM_RERANK_URL: url,
    CLAWMEM_NO_LOCAL_MODELS: "true",
    CLAWMEM_DISABLE_FTS_BYPASS: "true",
    CLAWMEM_QUERY_TIMING: "1",
    CLAWMEM_VEC_SCAN_TEST_BLOCK_MATCH: BLOCKED,
    ...extra,
  };
}

type Timing = { counts: Record<string, number>; stages: Record<string, { ms: number }> };
type Hit = { file: string; searchScore: number };
type Reply = { status: number; body: any; ms: number };

type Serve = {
  post: (query: string) => Promise<Reply>;
  timings: () => Promise<Timing[]>;
};

async function withServe(
  extra: Record<string, string>,
  fn: (s: Serve) => Promise<void>,
): Promise<void> {
  const probe = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: () => new Response("") });
  const port = probe.port as number;
  probe.stop(true);
  const proc = Bun.spawn(
    [process.execPath, "src/clawmem.ts", "serve", "--host", "127.0.0.1", "--port", String(port)],
    { cwd: REPO_ROOT, env: serveEnv(extra), stdout: "ignore", stderr: "pipe" },
  );
  const stderr = new Response(proc.stderr).text();
  const base = `http://127.0.0.1:${port}`;
  try {
    const up = Date.now() + 30_000;
    for (;;) {
      try {
        if ((await fetch(`${base}/health`)).ok) break;
      } catch {
        /* not listening yet */
      }
      if (Date.now() > up) throw new Error("serve never became healthy");
      await sleep(100);
    }
    let finished: string | undefined;
    await fn({
      post: async (query) => {
        const t0 = performance.now();
        const res = await fetch(`${base}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, limit: 5 }),
        });
        const body = await res.json();
        return { status: res.status, body, ms: performance.now() - t0 };
      },
      timings: async () => {
        proc.kill(9);
        await proc.exited;
        finished ??= await stderr;
        return finished
          .split("\n")
          .filter((l) => l.includes("[query-timing]"))
          .map((l) => JSON.parse(l.slice(l.indexOf("{"))) as Timing);
      },
    });
  } finally {
    proc.kill(9);
    await proc.exited;
  }
}

const hitsOf = (body: any): Hit[] => (Array.isArray(body) ? body : body.hits);

beforeAll(() => {
  fake = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    async fetch(req) {
      const path = new URL(req.url).pathname;
      if (path.endsWith("/v1/rerank")) {
        rerankCalls++;
        const rr = (await req.json()) as { documents: string[] };
        if (rerankDelayMs > 0) await sleep(rerankDelayMs);
        return Response.json({
          results: rr.documents.map((t, index) => ({
            index,
            relevance_score: goldenRerankScore(t),
          })),
        });
      }
      if (path.endsWith("/v1/embeddings")) {
        const body = (await req.json()) as { input: string | string[] };
        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        return Response.json({
          data: inputs.map((t, index) => ({ embedding: fakeEmbedVector(t, DIM), index })),
          model: MODEL,
        });
      }
      return new Response("not found", { status: 404 });
    },
  });
  seedVault();
}, 120_000);

afterAll(() => {
  fake.stop(true);
  rmSync(ROOT, { recursive: true, force: true });
});

// Positive control: QUERY_BUDGET_TEST_CONTROL=off leaves the budget unset, and B1 and B2 go RED.
const BUDGET = (ms: number) => (process.env.QUERY_BUDGET_TEST_CONTROL === "off" ? "0" : String(ms));
const TOLERANCE_MS = 1000;

describe("clawmem serve POST /query under CLAWMEM_QUERY_BUDGET_MS", () => {
  test("B1: 4 concurrent POSTs with a blocked scan all return within budget + tolerance", async () => {
    const D = 2000;
    await withServe(
      {
        CLAWMEM_QUERY_BUDGET_MS: BUDGET(D),
        // The vn4rz.131 deadline at the same value: alone it stacks at D, 2D, ...
        CLAWMEM_QUERY_VEC_DEADLINE_MS: String(D),
        CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "6000",
      },
      async (s) => {
        const replies = await Promise.all([1, 2, 3, 4].map(() => s.post(Q_BLOCK)));
        expect(Math.max(...replies.map((r) => r.ms))).toBeLessThan(D + TOLERANCE_MS);
        for (const r of replies) {
          expect(r.status).toBe(200);
          expect(r.body.degraded).toBe(true);
          expect(r.body.degraded_reason).toBe("query_budget_expired");
        }
        // The request that ran returns what it had: the FTS hit. The queued ones had nothing yet.
        const withHits = replies.filter((r) => hitsOf(r.body).length > 0);
        expect(withHits.length).toBeGreaterThanOrEqual(1);
        expect(hitsOf(withHits[0]!.body).map((h) => h.file)).toContain("alpha/ferry.md");
        const timings = await s.timings();
        expect(timings.length).toBe(4);
        for (const t of timings) expect(t.counts.query_budget_expired).toBe(1);
      },
    );
  }, 90_000);

  test("B2: a slow rerank is cut at the remaining budget; unreranked hits, degraded", async () => {
    const budget = 2000;
    rerankDelayMs = 4500; // far past budget + tolerance, far under the 60 s fetch bound
    rerankCalls = 0;
    try {
      await withServe({ CLAWMEM_QUERY_BUDGET_MS: BUDGET(budget) }, async (s) => {
        const r = await s.post(Q_RERANK);
        expect(r.status).toBe(200);
        expect(r.ms).toBeLessThan(budget + TOLERANCE_MS);
        expect(r.body.degraded).toBe(true);
        expect(r.body.degraded_reason).toBe("query_budget_expired");
        expect(hitsOf(r.body).length).toBeGreaterThan(0);
        expect(rerankCalls).toBeGreaterThan(0);
        const [t] = await s.timings();
        expect(t!.counts.query_budget_expired).toBe(1);
        expect(t!.counts.vec_deadline_expired).toBe(0);
        expect(t!.stages.rerank!.ms).toBeLessThan(budget + TOLERANCE_MS);
      });
    } finally {
      rerankDelayMs = 0;
    }
  }, 90_000);

  test("B3: a budget that is not spent changes nothing: same ranking, bare array, zero count", async () => {
    const run = async (extra: Record<string, string>) => {
      const out: unknown[] = [];
      let counts: Record<string, number> = {};
      await withServe(extra, async (s) => {
        const r = await s.post(Q_OK);
        expect(r.status).toBe(200);
        expect(Array.isArray(r.body)).toBe(true);
        out.push(hitsOf(r.body).map((h) => [h.file, h.searchScore]));
        counts = (await s.timings())[0]!.counts;
      });
      return { out, counts };
    };
    const on = await run({ CLAWMEM_QUERY_BUDGET_MS: "60000" });
    const off = await run({});
    expect((on.out[0] as unknown[]).length).toBeGreaterThan(0);
    expect(on.out).toEqual(off.out);
    expect(on.counts.query_budget_expired).toBe(0);
  }, 90_000);

  test("C: env unset is today's behaviour: bare array, no budget count, requests serialize", async () => {
    await withServe({ CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "1500" }, async (s) => {
      const t0 = performance.now();
      const replies = await Promise.all([s.post(Q_BLOCK), s.post(Q_BLOCK)]);
      for (const r of replies) {
        expect(r.status).toBe(200);
        expect(Array.isArray(r.body)).toBe(true);
      }
      // Two blocked scans, one at a time: the second waits its turn, as before.
      expect(performance.now() - t0).toBeGreaterThanOrEqual(3000);
      const timings = await s.timings();
      for (const t of timings)
        expect(Object.keys(t.counts).sort()).toEqual(["lex_legs", "vec_legs"]);
    });
  }, 90_000);
});
