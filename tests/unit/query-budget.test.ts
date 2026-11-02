import { afterAll, beforeAll, describe, expect, it, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, expandQueryCacheKey } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import {
  QueryBudget,
  QueryBudgetExpired,
  QUERY_BUDGET_GRACE_MS,
  QUERY_BUDGET_MAX_MS,
  queryBudgetMs,
  submitWithBudget,
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
  it("is off for a value setTimeout cannot hold (2^31 ms and up), like any invalid value", () => {
    // setTimeout with 2^31 or more fires at once, so such a budget would cut every request.
    expect(QUERY_BUDGET_MAX_MS).toBe(2147483647);
    expect(queryBudgetMs({ CLAWMEM_QUERY_BUDGET_MS: "2147483647" })).toBe(2147483647);
    for (const v of ["2147483648", "3000000000", "99999999999999999999"]) {
      expect(queryBudgetMs({ CLAWMEM_QUERY_BUDGET_MS: v })).toBe(0);
    }
    expect(QueryBudget.fromEnv(0, { CLAWMEM_QUERY_BUDGET_MS: "3000000000" })).toBeNull();
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

describe("submitWithBudget", () => {
  // Real timers at tens of milliseconds against pipelines that settle seconds later: a loaded host
  // can slip a timer by a few hundred ms, never by the gap these tests leave.
  const LATE_MS = 2000;
  const mk = (
    budgetMs: number,
    queue: Promise<unknown>,
    run: () => Promise<string>,
    graceMs = 60,
  ) => {
    let unstarted = 0;
    const sub = submitWithBudget<string>({
      budget: new QueryBudget(budgetMs),
      queue,
      run,
      unstarted: async () => {
        unstarted++;
        return "spent";
      },
      empty: () => "empty",
      graceMs,
    });
    return { ...sub, unstartedCalls: () => unstarted };
  };

  it("returns the pipeline's answer when it finishes within the budget, and never answers twice", async () => {
    const sub = mk(200, Promise.resolve(), async () => {
      await sleep(20);
      return "ran";
    });
    expect(await sub.result).toBe("ran");
    await sleep(500); // past budget and grace: no timer may fire an answer after the fact
    expect(sub.unstartedCalls()).toBe(0);
  });

  it("answers a started request at budget + grace when its pipeline cannot be cut", async () => {
    // The backstop: this pipeline ignores the budget and settles long after it.
    const sub = mk(40, Promise.resolve(), async () => {
      await sleep(LATE_MS);
      return "late";
    });
    expect(await sub.result).toBe("spent");
    expect(sub.unstartedCalls()).toBe(1);
    expect(await sub.tail).toBeUndefined(); // the queue still waits for the pipeline itself
  });

  it("answers a queued request at its budget, without waiting for the ones ahead", async () => {
    let ran = false;
    const ahead = new Promise<void>((resolve) => setTimeout(resolve, LATE_MS));
    const sub = mk(40, ahead, async () => {
      ran = true;
      return "ran";
    });
    expect(await sub.result).toBe("spent");
    expect(ran).toBe(false);
    expect(sub.unstartedCalls()).toBe(1);
    // Its turn comes later and finds the answer gone: nothing runs and nothing is counted again.
    expect(await sub.tail).toBeUndefined();
    expect(ran).toBe(false);
    expect(sub.unstartedCalls()).toBe(1);
  });

  it("does not start a pipeline whose budget ran out before its turn", async () => {
    let ran = false;
    const b = new QueryBudget(5, 0, () => 100); // already spent when the turn comes
    let unstarted = 0;
    const sub = submitWithBudget<string>({
      budget: b,
      queue: Promise.resolve(),
      run: async () => {
        ran = true;
        return "ran";
      },
      unstarted: async () => {
        unstarted++;
        return "spent";
      },
      empty: () => "empty",
    });
    expect(await sub.result).toBe("spent");
    expect(ran).toBe(false);
    expect(unstarted).toBe(1);
  });

  it("passes a pipeline's own failure through and still frees the queue", async () => {
    const sub = mk(500, Promise.resolve(), async () => {
      throw new Error("boom");
    });
    expect((await sub.result.catch((e) => e)).message).toBe("boom");
    expect(await sub.tail).toBeUndefined();
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
const Q_LEG = "orbital ferry zzlegfail";

let fake: ReturnType<typeof Bun.serve>;
let rerankDelayMs = 0;
let rerankCalls = 0;
let rerankAborts = 0; // rerank requests the client closed before the fake answered

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
      "ferry.md": `# Orbital ferry schedule ${BLOCKED}\n\nThe orbital ferry cutoffprobe departs every six hours zzlegfail.\n`,
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
    for (const q of [Q_OK, Q_BLOCK, Q_RERANK, Q_LEG]) {
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
  /** Headers go out now and the body `bodyDelayMs` later: the request has arrived but is unread. */
  postSlowBody: (query: string, bodyDelayMs: number) => Promise<Reply>;
  timings: () => Promise<Timing[]>;
  /** The serve's stderr, once it has been stopped. */
  log: () => Promise<string>;
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
      postSlowBody: async (query, bodyDelayMs) => {
        const t0 = performance.now();
        const payload = new TextEncoder().encode(JSON.stringify({ query, limit: 5 }));
        const res = await fetch(`${base}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: new ReadableStream({
            async start(ctrl) {
              await sleep(bodyDelayMs);
              ctrl.enqueue(payload);
              ctrl.close();
            },
          }),
          // A streamed request body needs duplex; Bun's types do not list it.
          ...({ duplex: "half" } as object),
        });
        const body = await res.json();
        return { status: res.status, body, ms: performance.now() - t0 };
      },
      log: async () => {
        proc.kill(9);
        await proc.exited;
        finished ??= await stderr;
        return finished;
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
        req.signal.addEventListener("abort", () => rerankAborts++);
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
// Host load of 20-40 lags timers by hundreds of ms; the RED controls miss the bound by seconds.
const TOLERANCE_MS = 1500;

describe("store.rerank with failOnAbort (the budget's abort must not fall through to the local reranker)", () => {
  const docs = Array.from({ length: 3 }, (_, i) => ({ file: `f${i}`, text: `doc ${i}` }));
  const saved: Record<string, string | undefined> = {};
  const ENV = ["CLAWMEM_RERANK_URL", "CLAWMEM_NO_LOCAL_MODELS"];
  beforeAll(() => {
    for (const k of ENV) saved[k] = process.env[k];
    process.env.CLAWMEM_RERANK_URL = `http://127.0.0.1:${fake.port}`;
    process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
  });
  afterAll(() => {
    for (const k of ENV) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  test("S1: an aborted remote rerank rejects, and the fetch ends at the abort", async () => {
    rerankDelayMs = 4500;
    const st = createStore(DB);
    try {
      const ctrl = new AbortController();
      setTimeout(() => ctrl.abort(), 200);
      const t0 = performance.now();
      const outcome = await st
        .rerank("abort probe", docs, undefined, undefined, {
          signal: ctrl.signal,
          failOnAbort: true,
        })
        .then(
          () => "resolved" as const,
          (e: Error) => e,
        );
      expect(outcome).not.toBe("resolved");
      expect((outcome as Error).name).toMatch(/Abort|Timeout/);
      // Far under the fake's 4500 ms: the reply never waited for the endpoint.
      expect(performance.now() - t0).toBeLessThan(3000);
    } finally {
      rerankDelayMs = 0;
      st.close();
    }
  }, 30_000);
});

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
    rerankAborts = 0;
    try {
      await withServe({ CLAWMEM_QUERY_BUDGET_MS: BUDGET(budget) }, async (s) => {
        // Cold start (store, workers) is not what this test times: warm with a fast rerank.
        rerankDelayMs = 0;
        await s.post(Q_OK);
        rerankDelayMs = 4500;
        rerankCalls = 0;
        rerankAborts = 0;
        const r = await s.post(Q_RERANK);
        expect(r.status).toBe(200);
        expect(r.ms).toBeLessThan(budget + TOLERANCE_MS);
        expect(r.body.degraded).toBe(true);
        expect(r.body.degraded_reason).toBe("query_budget_expired");
        expect(hitsOf(r.body).length).toBeGreaterThan(0);
        expect(rerankCalls).toBeGreaterThan(0);
        // The fetch is cancelled, not just abandoned: the fake sees the client close while it is
        // still sleeping on its 4500 ms delay. Without the signal it would sleep that out. Checked
        // before timings(), which kills the serve and would close the connection itself.
        const until = Date.now() + 1500;
        while (rerankAborts === 0 && Date.now() < until) await sleep(50);
        expect(rerankAborts).toBeGreaterThan(0);
        const t = (await s.timings()).at(-1)!;
        expect(t.counts.query_budget_expired).toBe(1);
        expect(t.counts.vec_deadline_expired).toBe(0);
        expect(t.stages.rerank!.ms).toBeLessThan(budget + TOLERANCE_MS);
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

  test("B4: the budget's own cut is counted as query_budget_expired, not vec_deadline_expired", async () => {
    // No vector deadline of its own: the budget is what ends the blocked scan.
    await withServe(
      { CLAWMEM_QUERY_BUDGET_MS: "1500", CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "6000" },
      async (s) => {
        const r = await s.post(Q_BLOCK);
        expect(r.status).toBe(200);
        expect(r.body.degraded_reason).toBe("query_budget_expired");
        const [t] = await s.timings();
        expect(t!.counts.query_budget_expired).toBe(1);
        expect(t!.counts.vec_deadline_expired).toBe(0);
      },
    );
  }, 90_000);

  test("B5: a vector deadline tighter than the budget keeps its own reason and count", async () => {
    await withServe(
      {
        CLAWMEM_QUERY_BUDGET_MS: "20000",
        CLAWMEM_QUERY_VEC_DEADLINE_MS: "1000",
        CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "3000",
      },
      async (s) => {
        const r = await s.post(Q_BLOCK);
        expect(r.status).toBe(200);
        expect(r.body.degraded_reason).toBe("vec_deadline_expired");
        const [t] = await s.timings();
        expect(t!.counts.vec_deadline_expired).toBe(1);
        expect(t!.counts.query_budget_expired).toBe(0);
      },
    );
  }, 90_000);

  test("B6: the budget counts from arrival: a request read late is answered without running", async () => {
    // Headers at t=0, body at t=3000, budget 1500. Measured from the handler call (the body read)
    // the budget would be whole and the pipeline would run; from arrival it is already spent.
    rerankCalls = 0;
    await withServe({ CLAWMEM_QUERY_BUDGET_MS: "1500" }, async (s) => {
      const r = await s.postSlowBody(Q_RERANK, 3000);
      expect(r.status).toBe(200);
      expect(r.body.degraded).toBe(true);
      expect(r.body.degraded_reason).toBe("query_budget_expired");
      expect(hitsOf(r.body)).toEqual([]);
      expect(rerankCalls).toBe(0);
      const [t] = await s.timings();
      expect(t!.counts.query_budget_unstarted).toBe(1);
      expect(t!.counts.vec_legs).toBeUndefined();
    });
  }, 90_000);

  test("B7: a budget spent during FTS is cut there: no FTS hits, vector stage and rerank skipped, still a 200", async () => {
    // FTS runs in a worker (vn4rz.135), so the budget cuts the stall instead of waiting it out.
    rerankCalls = 0;
    await withServe(
      {
        CLAWMEM_QUERY_BUDGET_MS: "600",
        CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS: "4000",
        CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MATCH: "cutoffprobe",
      },
      async (s) => {
        const r = await s.post(Q_RERANK);
        expect(r.status).toBe(200);
        expect(r.ms).toBeLessThan(600 + TOLERANCE_MS); // far short of the 4000 ms stall
        expect(r.body.degraded).toBe(true);
        expect(r.body.degraded_reason).toBe("query_budget_expired");
        expect(hitsOf(r.body)).toEqual([]);
        expect(rerankCalls).toBe(0);
        const [t] = await s.timings();
        expect(t!.counts.query_budget_expired).toBe(1);
        expect(t!.counts.fts_cut).toBe(1);
        expect(t!.counts.vec_legs).toBe(0);
        expect(t!.counts.query_budget_unstarted).toBeUndefined();
        expect(t!.stages.fts).toBeDefined();
        expect(await s.log()).toContain("fts worker started");
      },
    );
  }, 90_000);

  test("A: FTS stalled 5000 ms inside the worker: A answers within budget + grace, B (sent at t=1000) is stamped on arrival and answers within its own budget + grace; both counted", async () => {
    const D = 2000;
    await withServe(
      {
        CLAWMEM_QUERY_BUDGET_MS: String(D),
        CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS: "5000",
        CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MATCH: "cutoffprobe",
      },
      async (s) => {
        const t0 = performance.now();
        const a = s.post(Q_RERANK);
        await sleep(1000);
        const bSentAt = performance.now() - t0;
        const b = await s.post(Q_RERANK);
        const ra = await a;
        // The bug: A answers at ~5045 ms, and B, read only after the stall, starts its clock at ~5000 ms.
        const bound = D + QUERY_BUDGET_GRACE_MS + 500;
        expect(ra.status).toBe(200);
        expect(ra.ms).toBeLessThan(bound);
        expect(b.status).toBe(200);
        expect(b.ms).toBeLessThan(bound);
        // B's budget counts from its arrival at t=1000: it ends near t=3000, not near t=7000.
        expect(bSentAt).toBeLessThan(1000 + 500);
        for (const r of [ra, b]) {
          expect(r.body.degraded).toBe(true);
          expect(r.body.degraded_reason).toBe("query_budget_expired");
        }
        const timings = await s.timings();
        expect(timings.length).toBe(2);
        for (const t of timings) {
          expect(t.counts.query_budget_expired).toBe(1);
          expect(t.counts.fts_cut).toBe(1);
        }
      },
    );
  }, 90_000);

  test("B8: a failed vector leg stays the reply's reason when the budget also cuts the request", async () => {
    // A directory as the seam's log path makes every worker leg throw a plain (non-fatal) error.
    // Rerank then outlasts the budget. The hook counts vec_leg_failed toward its DOWN streak, so
    // the reply must keep it; the timing row counts both reasons.
    rerankDelayMs = 4500;
    try {
      await withServe(
        { CLAWMEM_QUERY_BUDGET_MS: "2000", CLAWMEM_VEC_SCAN_TEST_LOG: ROOT },
        async (s) => {
          // Cold start must not eat the budget before the vector stage runs: warm with a fast rerank.
          rerankDelayMs = 0;
          await s.post(Q_OK);
          rerankDelayMs = 4500;
          const r = await s.post(Q_LEG);
          expect(r.status).toBe(200);
          expect(r.body.degraded).toBe(true);
          expect(r.body.degraded_reason).toBe("vec_leg_failed");
          expect(hitsOf(r.body).length).toBeGreaterThan(0);
          const t = (await s.timings()).at(-1)!;
          expect(t.counts.vec_leg_failed).toBe(1);
          expect(t.counts.query_budget_expired).toBe(1);
        },
      );
    } finally {
      rerankDelayMs = 0;
    }
  }, 90_000);

  test("B9: a failed vector leg alone is vec_leg_failed with no budget cut counted", async () => {
    await withServe(
      { CLAWMEM_QUERY_BUDGET_MS: "20000", CLAWMEM_VEC_SCAN_TEST_LOG: ROOT },
      async (s) => {
        const r = await s.post(Q_LEG);
        expect(r.body.degraded_reason).toBe("vec_leg_failed");
        const [t] = await s.timings();
        expect(t!.counts.vec_leg_failed).toBe(1);
        expect(t!.counts.query_budget_expired).toBe(0);
      },
    );
  }, 90_000);

  test("B10: an FTS worker that fails falls back to the main thread: a 200 with the FTS hits, not an error, counted", async () => {
    await withServe(
      {
        CLAWMEM_QUERY_BUDGET_MS: "60000",
        CLAWMEM_QUERY_BUDGET_TEST_FTS_FAIL_MATCH: "cutoffprobe",
      },
      async (s) => {
        const r = await s.post(Q_RERANK);
        expect(r.status).toBe(200);
        expect(hitsOf(r.body).map((h) => h.file)).toContain("alpha/ferry.md");
        const [t] = await s.timings();
        expect(t!.counts.fts_worker_failed).toBe(1);
        expect(t!.counts.fts_cut).toBe(0);
      },
    );
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
      // No budget: FTS stayed on the main thread and no FTS worker was ever started (vn4rz.135).
      expect(await s.log()).not.toContain("fts worker started");
    });
  }, 90_000);
});
