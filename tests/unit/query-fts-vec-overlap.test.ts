import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, expandQueryCacheKey } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { ftsVecOverlapEnabled } from "../../src/vec-deadline.ts";
import { fakeEmbedVector, goldenRerankScore } from "../helpers/fake-embed-server.ts";

/**
 * master-harness-vn4rz.150: under a request budget with the vector deadline on, FTS and the
 * original query's vector leg run side by side, so a request costs about max(fts, vec) instead of
 * fts + vec. The expansion legs still wait for FTS.
 *
 * End to end against the real `clawmem serve` on a throwaway vault, like query-budget.test.ts.
 * FTS is stalled inside its worker (CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS) and the vector scan
 * inside its worker (CLAWMEM_VEC_SCAN_TEST_BLOCK_MS), each by a native sleep that a loaded host
 * cannot shorten. The assertions read the timing row's stage windows (start_ms, ms) first, and
 * the wall clock second with a margin of seconds, so they do not measure how busy the host is.
 */

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

const REPO_ROOT = new URL("../..", import.meta.url).pathname;
const ROOT = mkdtempSync(join(tmpdir(), "clawmem-fts-vec-overlap-"));
const DB = join(ROOT, "index.sqlite");
const DIM = 512;
const MODEL = "fake-embed-a";
const FTS_MATCH = "cutoffprobe";
const VEC_MATCH = "zzblock";
// Cached empty expansion, so no test waits on a live LLM call.
const Q_BOTH = `orbital ferry ${FTS_MATCH} ${VEC_MATCH}`;
const Q_FTS_ONLY = `orbital ferry ${FTS_MATCH} cutonly`;
const Q_FTS_FAIL = "orbital ferry workerfail";
const Q_WARM = "orbital ferry warmup";
// A strong BM25 hit (both terms in one title among ~50 fillers) and a weak one, neither cached.
const Q_STRONG = "zanzibar protocol";
const Q_WEAK = "assorted planning notes";
const STALL_MS = 2000;

let fake: ReturnType<typeof Bun.serve>;
let llmCalls = 0;

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
    const embedded: Record<string, string> = {
      "ferry.md": `# Orbital ferry schedule ${VEC_MATCH}\n\nThe orbital ferry ${FTS_MATCH} cutonly departs every six hours workerfail.\n`,
      "dock.md": `# Docking bay rota\n\nFerry crews rotate through the docking bays weekly.\n`,
      "zanzibar-protocol.md":
        "zanzibar protocol reference. The zanzibar protocol governs handshake ordering.\n",
    };
    for (const [path, body] of Object.entries(embedded)) {
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
    // IDF headroom: a strong BM25 signal needs a corpus of real size.
    for (let i = 0; i < 50; i++) {
      const body = `assorted planning notes number ${i} about schedules, follow-ups, cooking, travel bookings, and reading lists`;
      const hash = hashContent(body + i);
      store.insertContent(hash, body, now);
      store.insertDocument("alpha", `filler-${i}.md`, `filler-${i}.md`, hash, now, now);
    }
    for (const q of [Q_BOTH, Q_FTS_ONLY, Q_FTS_FAIL, Q_WARM]) {
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
    CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MATCH: FTS_MATCH,
    CLAWMEM_VEC_SCAN_TEST_BLOCK_MATCH: VEC_MATCH,
    ...extra,
  };
}

type Stage = { start_ms: number; ms: number };
type Timing = { counts: Record<string, number>; stages: Record<string, Stage> };
type Reply = { status: number; body: any; ms: number };
type Serve = {
  post: (query: string) => Promise<Reply>;
  /** The serve's rows for every request so far, once it has been stopped. */
  timings: () => Promise<Timing[]>;
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
    {
      cwd: REPO_ROOT,
      env: serveEnv(extra),
      stdout: "ignore",
      stderr: "pipe",
    },
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
    const stop = async () => {
      proc.kill(9);
      await proc.exited;
      finished ??= await stderr;
      return finished;
    };
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
      log: stop,
      timings: async () =>
        (await stop())
          .split("\n")
          .filter((l) => l.includes("[query-timing]"))
          .map((l) => JSON.parse(l.slice(l.indexOf("{"))) as Timing),
    });
  } finally {
    proc.kill(9);
    await proc.exited;
  }
}

const hitsOf = (body: any): { file: string }[] => (Array.isArray(body) ? body : body.hits);
const end = (s: Stage) => s.start_ms + s.ms;
/** Overlap of two stage windows in ms (rounded stamps: allow a ms or two of slack in callers). */
const overlapMs = (a: Stage, b: Stage) =>
  Math.min(end(a), end(b)) - Math.max(a.start_ms, b.start_ms);

const BUDGETED = { CLAWMEM_QUERY_BUDGET_MS: "9000", CLAWMEM_QUERY_VEC_DEADLINE_MS: "7000" };
const STALLS = {
  CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS: String(STALL_MS),
  CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: String(STALL_MS),
};

beforeAll(() => {
  fake = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    async fetch(req) {
      const path = new URL(req.url).pathname;
      if (path.endsWith("/v1/rerank")) {
        const rr = (await req.json()) as { documents: string[] };
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
      if (path.endsWith("/chat/completions")) {
        llmCalls++;
        return new Response("no expansion model here", { status: 500 });
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

describe("ftsVecOverlapEnabled", () => {
  test("is on unless the knob says off", () => {
    expect(ftsVecOverlapEnabled({})).toBe(true);
    expect(ftsVecOverlapEnabled({ CLAWMEM_QUERY_FTS_VEC_OVERLAP: "" })).toBe(true);
    expect(ftsVecOverlapEnabled({ CLAWMEM_QUERY_FTS_VEC_OVERLAP: "1" })).toBe(true);
    expect(ftsVecOverlapEnabled({ CLAWMEM_QUERY_FTS_VEC_OVERLAP: "garbage" })).toBe(true);
    for (const v of ["0", "false", "OFF", " no "]) {
      expect(ftsVecOverlapEnabled({ CLAWMEM_QUERY_FTS_VEC_OVERLAP: v })).toBe(false);
    }
  });
});

describe("FTS beside the original vector leg (vn4rz.150)", () => {
  test("a: slow FTS and a slow vector leg cost about the longer one, not the sum", async () => {
    await withServe({ ...BUDGETED, ...STALLS }, async (s) => {
      expect((await s.post(Q_WARM)).status).toBe(200); // spawns the FTS and vector workers
      const r = await s.post(Q_BOTH);
      expect(r.status).toBe(200);
      expect(hitsOf(r.body).map((h) => h.file)).toContain("alpha/ferry.md");
      const [, t] = await s.timings();
      const { fts, vec_legs } = t!.stages;
      // The vector stage started before FTS ended, and the two windows share most of the stall.
      expect(vec_legs!.start_ms).toBeLessThan(end(fts!));
      expect(overlapMs(fts!, vec_legs!)).toBeGreaterThan(STALL_MS * 0.6);
      expect(fts!.ms).toBeGreaterThanOrEqual(STALL_MS - 100);
      expect(vec_legs!.ms).toBeGreaterThanOrEqual(STALL_MS - 100);
      // Serial costs at least 2 x STALL_MS; the margin of 0.35 x STALL_MS is far above host jitter.
      expect(r.ms).toBeLessThan(2 * STALL_MS - 0.35 * STALL_MS);
      expect(t!.counts.fts_vec_overlap).toBe(1);
    });
  }, 90_000);

  test("b: FTS cut by the budget while the vector leg finished in time: the vector hits stay, the reply is degraded", async () => {
    await withServe(
      {
        CLAWMEM_QUERY_BUDGET_MS: "2500",
        CLAWMEM_QUERY_VEC_DEADLINE_MS: "7000",
        CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS: "9000",
      },
      async (s) => {
        expect((await s.post(Q_WARM)).status).toBe(200);
        const r = await s.post(Q_BOTH.replace(VEC_MATCH, "vecfast"));
        expect(r.status).toBe(200);
        expect(r.ms).toBeLessThan(2500 + 2500); // cut at the budget, not at the 9000 ms stall
        expect(r.body.degraded).toBe(true);
        expect(r.body.degraded_reason).toBe("query_budget_expired");
        // FTS gave nothing (cut), so every hit here came from the vector leg.
        expect(hitsOf(r.body).length).toBeGreaterThan(0);
        const [, t] = await s.timings();
        expect(t!.counts.fts_cut).toBe(1);
        expect(t!.counts.fts_vec_overlap).toBe(1);
        expect(t!.counts.query_budget_expired).toBe(1);
        expect(t!.counts.vec_legs).toBe(1);
      },
    );
  }, 90_000);

  test("c: a strong FTS signal skips expansion, and the original vector leg still ran", async () => {
    await withServe({ ...BUDGETED, CLAWMEM_DISABLE_FTS_BYPASS: "false" }, async (s) => {
      expect((await s.post(Q_WARM)).status).toBe(200);
      const before = llmCalls;
      const r = await s.post(Q_STRONG);
      expect(r.status).toBe(200);
      expect(llmCalls).toBe(before); // expandQuery never reached the LLM
      expect(hitsOf(r.body).map((h) => h.file)).toContain("alpha/zanzibar-protocol.md");
      // Control: a weak FTS result on the same serve does ask for an expansion, so the counter works.
      const beforeWeak = llmCalls;
      expect((await s.post(Q_WEAK)).status).toBe(200);
      expect(llmCalls).toBeGreaterThan(beforeWeak);
      const rows = await s.timings();
      const strong = rows[1]!;
      expect(strong.counts.fts_vec_overlap).toBe(1);
      expect(strong.counts.vec_legs).toBe(1); // the original query's leg, no expansion legs
      expect(strong.stages.expand).toBeUndefined();
      expect(strong.stages.vec_legs).toBeDefined();
      expect(rows[2]!.stages.expand).toBeDefined();
    });
  }, 90_000);

  test("d: without a budget the stages run in the old order, FTS then the vector stage", async () => {
    for (const extra of [{}, { CLAWMEM_QUERY_VEC_DEADLINE_MS: "7000" }] as Record<
      string,
      string
    >[]) {
      await withServe(extra, async (s) => {
        expect((await s.post(Q_BOTH)).status).toBe(200);
        const [t] = await s.timings();
        expect(t!.counts.fts_vec_overlap).toBeUndefined();
        const { fts, vec_legs } = t!.stages;
        expect(vec_legs!.start_ms).toBeGreaterThanOrEqual(end(fts!) - 2); // stamps are rounded to 1 ms
      });
    }
  }, 90_000);

  test("d2: a budget with the vector deadline off keeps the old order and says so", async () => {
    await withServe({ CLAWMEM_QUERY_BUDGET_MS: "9000" }, async (s) => {
      expect((await s.post(Q_BOTH)).status).toBe(200);
      const [t] = await s.timings();
      expect(t!.counts.fts_vec_overlap).toBe(0);
      expect(t!.stages.vec_legs!.start_ms).toBeGreaterThanOrEqual(end(t!.stages.fts!) - 2);
    });
  }, 90_000);

  test("e: CLAWMEM_QUERY_FTS_VEC_OVERLAP=0 restores the serial order and the serial cost", async () => {
    await withServe({ ...BUDGETED, ...STALLS, CLAWMEM_QUERY_FTS_VEC_OVERLAP: "0" }, async (s) => {
      expect((await s.post(Q_WARM)).status).toBe(200);
      const r = await s.post(Q_BOTH);
      expect(r.status).toBe(200);
      const [, t] = await s.timings();
      expect(t!.counts.fts_vec_overlap).toBe(0);
      expect(t!.stages.vec_legs!.start_ms).toBeGreaterThanOrEqual(end(t!.stages.fts!) - 2);
      expect(r.ms).toBeGreaterThanOrEqual(2 * STALL_MS - 100); // both stalls, one after the other
    });
  }, 90_000);

  test("f: an FTS worker failure still falls back to the main thread while the vector stage runs beside it", async () => {
    await withServe(
      { ...BUDGETED, CLAWMEM_QUERY_BUDGET_TEST_FTS_FAIL_MATCH: "workerfail" },
      async (s) => {
        expect((await s.post(Q_WARM)).status).toBe(200);
        const r = await s.post(Q_FTS_FAIL);
        expect(r.status).toBe(200);
        expect(r.body.degraded).toBeUndefined();
        expect(hitsOf(r.body).map((h) => h.file)).toContain("alpha/ferry.md");
        const [, t] = await s.timings();
        expect(t!.counts.fts_worker_failed).toBe(1);
        expect(t!.counts.fts_vec_overlap).toBe(1);
        expect(t!.counts.vec_legs).toBe(1);
      },
    );
  }, 90_000);
});
