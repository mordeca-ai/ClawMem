import { afterAll, beforeAll, describe, expect, it, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, expandQueryCacheKey, FatalVectorError } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import {
  runVecStageWithDeadline,
  VEC_DEADLINE_MAX_ABANDONED,
  VecDeadlineLegs,
  vecDeadlineMs,
  type VecDeadlinePool,
} from "../../src/vec-deadline.ts";
import { fakeEmbedVector, goldenRerankScore } from "../helpers/fake-embed-server.ts";

/**
 * master-harness-vn4rz.131 — a deadline on the hybrid query's vector legs.
 *
 * Unit half: the deadline owner with a fake pool (no timing beyond a short timer).
 * End-to-end half: the real `clawmem serve` in a subprocess, on a throwaway vault. The delay is
 * injected by src/vec-scan-test-seam.ts: a native sleep on the thread about to scan, which also
 * logs which thread that was. So "stays in process" and "never falls back in process" are read off
 * that log, and no test here depends on how loaded the host is.
 */

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

type FakePool = VecDeadlinePool & { closes: number; exits: Array<() => void>; legs: string[] };

function fakePool(run: (q: string) => Promise<unknown> = async () => []): FakePool {
  const pool: FakePool = {
    size: 1,
    closes: 0,
    exits: [],
    legs: [],
    warm: async () => {},
    close: () => {
      pool.closes++;
    },
    closeTracked: () => {
      pool.closes++;
      return [new Promise<void>((resolve) => pool.exits.push(resolve))];
    },
    runLegOn: async (q) => {
      pool.legs.push(q);
      return (await run(q)) as never;
    },
    runLegScoped: async (q) => {
      pool.legs.push(q);
      return (await run(q)) as never;
    },
    releaseScoped: () => {},
  };
  return pool;
}

const never = () => new Promise<never>(() => {});

describe("vecDeadlineMs", () => {
  it("is off for unset, empty, zero and anything not a plain integer", () => {
    for (const v of [undefined, "", "  ", "0", "-1", "1.5", "4500ms", "soon"]) {
      expect(vecDeadlineMs({ CLAWMEM_QUERY_VEC_DEADLINE_MS: v })).toBe(0);
    }
  });
  it("parses a positive integer", () => {
    expect(vecDeadlineMs({ CLAWMEM_QUERY_VEC_DEADLINE_MS: " 4500 " })).toBe(4500);
  });
});

describe("VecDeadlineLegs", () => {
  it("returns the value in time and keeps the same pool for the next query", async () => {
    const pools: FakePool[] = [];
    const legs = new VecDeadlineLegs({
      createPool: () => {
        const p = fakePool();
        pools.push(p);
        return p;
      },
    });
    expect(await legs.run(1000, async () => "fine")).toEqual({ status: "ok", value: "fine" });
    expect(await legs.run(1000, async () => "again")).toEqual({ status: "ok", value: "again" });
    expect(pools).toHaveLength(1);
    expect(pools[0]!.closes).toBe(0);
  });

  it("on expiry retires the pool, and the next query gets a new one", async () => {
    const pools: FakePool[] = [];
    const legs = new VecDeadlineLegs({
      createPool: () => {
        const p = fakePool();
        pools.push(p);
        return p;
      },
    });
    expect(await legs.run(5, never)).toEqual({ status: "expired" });
    expect(pools[0]!.closes).toBe(1);
    expect(await legs.run(1000, async () => "recovered")).toEqual({
      status: "ok",
      value: "recovered",
    });
    expect(pools).toHaveLength(2);
    expect(pools[1]!.closes).toBe(0);
  });

  it("sheds once too many abandoned scans are running, and resumes as they exit", async () => {
    const pools: FakePool[] = [];
    const legs = new VecDeadlineLegs({
      createPool: () => {
        const p = fakePool();
        pools.push(p);
        return p;
      },
    });
    for (let i = 0; i < VEC_DEADLINE_MAX_ABANDONED; i++)
      expect((await legs.run(5, never)).status).toBe("expired");
    expect(legs.abandonedScans()).toBe(VEC_DEADLINE_MAX_ABANDONED);

    let started = false;
    const shed = await legs.run(1000, async () => {
      started = true;
      return "no";
    });
    expect(shed).toEqual({ status: "shed" });
    expect(started).toBe(false);
    const poolsWhenShed = pools.length;

    pools[0]!.exits[0]!();
    await sleep(0);
    expect(legs.abandonedScans()).toBe(VEC_DEADLINE_MAX_ABANDONED - 1);
    expect(await legs.run(1000, async () => "back")).toEqual({ status: "ok", value: "back" });
    expect(pools.length).toBe(poolsWhenShed + 1);
  });

  it("stops counting an abandoned worker whose close never arrives", async () => {
    let now = 1000;
    const legs = new VecDeadlineLegs({
      createPool: () => fakePool(),
      now: () => now,
      abandonedStaleMs: 500,
      maxAbandoned: 1,
    });
    await legs.run(5, never);
    expect((await legs.run(1000, async () => "x")).status).toBe("shed");
    now += 501;
    expect((await legs.run(1000, async () => "x")).status).toBe("ok");
  });

  it("restores a fatal vector error from a worker, with its class and message", async () => {
    const legs = new VecDeadlineLegs({ createPool: () => fakePool() });
    const fromWorker = Object.assign(new Error("Embedding model changed"), {
      fatalName: "VecReadModelMismatchError",
    });
    const caught = await legs
      .run(1000, async () => {
        throw fromWorker;
      })
      .catch((e) => e);
    expect(caught).toBeInstanceOf(FatalVectorError);
    expect(caught.name).toBe("VecReadModelMismatchError");
    expect(caught.message).toBe("Embedding model changed");
  });

  it("a non-fatal leg failure retires the pool and reports failed", async () => {
    const pools: FakePool[] = [];
    const legs = new VecDeadlineLegs({
      createPool: () => {
        const p = fakePool();
        pools.push(p);
        return p;
      },
    });
    const res = await legs.run(1000, async () => {
      throw new Error("Worker spawn failed");
    });
    expect(res.status).toBe("failed");
    expect(pools[0]!.closes).toBe(1);
  });

  it("keepWarm warms a worker at once and a fresh one after an expiry", async () => {
    const warmed: Array<string | undefined> = [];
    const legs = new VecDeadlineLegs({
      createPool: () => ({
        ...fakePool(),
        warm: async (_n: number, path?: string) => {
          warmed.push(path);
        },
      }),
    });
    legs.keepWarm("/vault.sqlite");
    expect(warmed).toEqual(["/vault.sqlite"]);
    await legs.run(5, never);
    expect(warmed).toEqual(["/vault.sqlite", "/vault.sqlite"]);
  });
});

describe("runVecStageWithDeadline", () => {
  const stage = (legs: VecDeadlineLegs, deadlineMs: number, expansion: Promise<string[]>) =>
    runVecStageWithDeadline({
      legs,
      deadlineMs,
      original: "orig",
      expansion,
      legsOf: (e: string[]) => e,
      legOn: (pool) => (q) => pool.runLegOn(q, { dbPath: "", model: "m", limit: 1 }),
      warm: 0,
    });

  it("keeps leg order when nothing expires", async () => {
    const pool = fakePool(async (q) => [q]);
    const res = await stage(
      new VecDeadlineLegs({ createPool: () => pool }),
      1000,
      Promise.resolve(["a", "b"]),
    );
    expect(res.degraded).toBeUndefined();
    expect(res.results as unknown).toEqual([["orig"], ["a"], ["b"]]);
  });

  it("drops the vector results but keeps an expansion that landed in time", async () => {
    const pool = fakePool(() => never());
    const res = await stage(
      new VecDeadlineLegs({ createPool: () => pool }),
      20,
      Promise.resolve(["a"]),
    );
    expect(res).toEqual({ expansion: ["a"], results: [], degraded: "vec_deadline_expired" });
  });

  it("keeps the original leg when it finished but the expansion is still in flight at the deadline", async () => {
    // vn4rz.134 finding 4: the original leg does not wait for the expansion, so it can be done by
    // the deadline; its scan used to be thrown away with the unfinished ones.
    const pool = fakePool(async (q) => [q]);
    const expansion = new Promise<string[]>(() => {});
    const res = await stage(new VecDeadlineLegs({ createPool: () => pool }), 30, expansion);
    expect(res.degraded).toBe("vec_deadline_expired");
    expect(res.expansion).toBeUndefined();
    expect(res.results as unknown).toEqual([["orig"]]);
  });

  it("does not wait for an expansion still in flight, nor start its legs when it lands", async () => {
    const pools: FakePool[] = [];
    let land!: (e: string[]) => void;
    const expansion = new Promise<string[]>((resolve) => {
      land = resolve;
    });
    const legs = new VecDeadlineLegs({
      createPool: () => {
        const p = fakePool(() => never());
        pools.push(p);
        return p;
      },
    });
    const res = await stage(legs, 20, expansion);
    expect(res).toEqual({ expansion: undefined, results: [], degraded: "vec_deadline_expired" });
    expect(pools[0]!.legs).toEqual(["orig"]);
    land(["late"]);
    await sleep(0);
    // The late leg asks the retired pool (the real one rejects it as closed), never a new pool.
    expect(pools).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------------------------
// End to end: the real serve, on a throwaway vault
// ---------------------------------------------------------------------------------------------

const REPO_ROOT = new URL("../..", import.meta.url).pathname;
const ROOT = mkdtempSync(join(tmpdir(), "clawmem-vec-deadline-"));
const DB = join(ROOT, "index.sqlite");
const SCAN_LOG = join(ROOT, "scan.log");
const DIM = 512;
const MODEL = "fake-embed-a";
const BLOCKED = "zzblock"; // A query containing this word blocks in the seam.
const Q_OK = "orbital ferry schedule";
const Q_BLOCK = `orbital ferry schedule ${BLOCKED}`;
const Q_LIVE_OK = "orbital ferry liveone";
const Q_LIVE_BLOCK = `orbital ferry livetwo ${BLOCKED}`;

let fake: ReturnType<typeof Bun.serve>;
let fakeModel = MODEL;

function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seedVault(): void {
  mkdirSync(join(ROOT, "alpha"), { recursive: true });
  mkdirSync(join(ROOT, "beta"), { recursive: true });
  writeFileSync(
    join(ROOT, "config.yaml"),
    [
      "collections:",
      "  alpha:",
      `    path: ${join(ROOT, "alpha")}`,
      '    pattern: "**/*.md"',
      "  beta:",
      `    path: ${join(ROOT, "beta")}`,
      '    pattern: "**/*.md"',
      "",
    ].join("\n"),
  );
  const store = createStore(DB);
  const now = new Date().toISOString();
  store.ensureVecTable(DIM);
  store.db.transaction(() => {
    // Three vec0 chunks of another collection, so `alpha` is a small share of the table and the
    // chunk-targeted scoped scan serves it.
    const r = mulberry(131);
    const fillerBody = "assorted planning notes";
    const fillerHash = hashContent(fillerBody);
    store.insertContent(fillerHash, fillerBody, now);
    store.insertDocument("beta", "filler.md", "Filler", fillerHash, now, now);
    const v = new Float32Array(DIM);
    for (let seq = 0; seq < 3100; seq++) {
      for (let i = 0; i < DIM; i++) v[i] = r() * 2 - 1;
      store.insertEmbedding(
        fillerHash,
        seq,
        seq,
        v,
        MODEL,
        now,
        "full",
        undefined,
        "beta/filler.md",
      );
    }
    const docs: Record<string, string> = {
      "ferry.md": `# Orbital ferry schedule ${BLOCKED}\n\nThe orbital ferry liveone livetwo departs every six hours.\n`,
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
    // Known at once (llm_cache hit, no expansion): the path that used to run one leg in process.
    for (const q of [Q_OK, Q_BLOCK]) {
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
    CLAWMEM_VEC_SCAN_TEST_LOG: SCAN_LOG,
    CLAWMEM_VEC_SCAN_TEST_BLOCK_MATCH: BLOCKED,
    ...extra,
  };
}

const scanThreads = (): string[] => {
  try {
    return readFileSync(SCAN_LOG, "utf8").split("\n").filter(Boolean);
  } catch {
    return [];
  }
};

type Timing = { counts: Record<string, number> };
type Hit = { file: string; searchScore: number };
type QueryBody = Hit[] | { degraded: boolean; degraded_reason: string; hits: Hit[] };

type Serve = {
  post: (query: string, collection?: string) => Promise<{ status: number; body: any; ms: number }>;
  timings: () => Promise<Timing[]>;
};

async function withServe(
  extra: Record<string, string>,
  fn: (s: Serve) => Promise<void>,
): Promise<void> {
  writeFileSync(SCAN_LOG, "");
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
      post: async (query, collection) => {
        const t0 = performance.now();
        const res = await fetch(`${base}/query`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, limit: 5, ...(collection ? { collection } : {}) }),
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

const hitsOf = (body: QueryBody): Hit[] => (Array.isArray(body) ? body : body.hits);
const ranking = (body: QueryBody) => hitsOf(body).map((h) => [h.file, h.searchScore]);

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
      if (path.endsWith("/chat/completions")) {
        // A live LLM call: slow enough that the expansion is still in flight when the leg starts.
        await sleep(150);
        return Response.json({
          choices: [
            { message: { content: "lex: orbital ferry\nvec: orbital ferry departure times" } },
          ],
        });
      }
      if (path.endsWith("/v1/embeddings")) {
        const body = (await req.json()) as { input: string | string[] };
        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        return Response.json({
          data: inputs.map((t, index) => ({ embedding: fakeEmbedVector(t, DIM), index })),
          model: fakeModel,
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

// Positive control: VEC_DEADLINE_TEST_CONTROL=off turns the A1 case's deadline off, and it goes RED.
const A1_DEADLINE_MS = process.env.VEC_DEADLINE_TEST_CONTROL === "off" ? "0" : "500";
const A1_BLOCK_MS = "6000";

describe("clawmem serve POST /query under CLAWMEM_QUERY_VEC_DEADLINE_MS", () => {
  test("A1: a blocked scan returns FTS hits, flagged degraded, within deadline + 1000 ms", async () => {
    await withServe(
      {
        CLAWMEM_QUERY_VEC_DEADLINE_MS: A1_DEADLINE_MS,
        CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: A1_BLOCK_MS,
      },
      async (s) => {
        const { status, body, ms } = await s.post(Q_BLOCK);
        expect(status).toBe(200);
        expect(ms).toBeLessThan(500 + 1000);
        expect(body.degraded).toBe(true);
        expect(body.degraded_reason).toBe("vec_deadline_expired");
        expect(body.hits.map((h: Hit) => h.file)).toContain("alpha/ferry.md");
        const [timing] = await s.timings();
        expect(timing!.counts.vec_deadline_expired).toBe(1);
        expect(timing!.counts.vec_legs).toBe(0);
      },
    );
  }, 60_000);

  test("A2: env unset keeps the single leg in process, blocking, in the bare array shape", async () => {
    await withServe({ CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "1500" }, async (s) => {
      const { status, body, ms } = await s.post(Q_BLOCK);
      expect(status).toBe(200);
      expect(Array.isArray(body)).toBe(true);
      expect(ms).toBeGreaterThanOrEqual(1500);
      expect(scanThreads()).toEqual(["main"]);
      const [timing] = await s.timings();
      // No budget and no deadline: no budget or deadline counts, only the queue wait serve always stamps.
      expect(Object.keys(timing!.counts).sort()).toEqual([
        "lex_legs",
        "query_queue_wait_ms",
        "vec_legs",
      ]);
    });
  }, 60_000);

  test("A2: deadline 0 is the same as unset", async () => {
    await withServe({ CLAWMEM_QUERY_VEC_DEADLINE_MS: "0" }, async (s) => {
      const { body } = await s.post(Q_OK);
      expect(Array.isArray(body)).toBe(true);
      expect(scanThreads()).toEqual(["main"]);
    });
  }, 60_000);

  test("the ranking is the same with a deadline that does not expire, on every leg shape", async () => {
    const shapes: Array<[string, string?]> = [[Q_OK], [Q_OK, "alpha"], [Q_LIVE_OK]];
    const run = async (extra: Record<string, string>) => {
      const out: unknown[] = [];
      await withServe(extra, async (s) => {
        for (const [q, c] of shapes) {
          const { status, body } = await s.post(q, c);
          expect(status).toBe(200);
          expect(Array.isArray(body)).toBe(true);
          out.push(ranking(body));
        }
      });
      return out;
    };
    const on = await run({ CLAWMEM_QUERY_VEC_DEADLINE_MS: "60000" });
    expect(scanThreads().length).toBeGreaterThan(0);
    expect(scanThreads()).not.toContain("main");
    const off = await run({});
    expect((on[0] as unknown[]).length).toBeGreaterThan(0);
    expect(on).toEqual(off);
  }, 120_000);

  describe("A3: expiry never reruns the scan on the main thread", () => {
    // Long enough that the leg has embedded and reached the seam, far short of the block.
    const deadline = {
      CLAWMEM_QUERY_VEC_DEADLINE_MS: "1500",
      CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "10000",
    };

    test("the single-leg shape", async () => {
      await withServe(deadline, async (s) => {
        const { body } = await s.post(Q_BLOCK);
        expect(body.degraded).toBe(true);
        await sleep(600); // Long enough for a fallback leg to have embedded and reached the seam.
        expect(scanThreads()).toEqual(["worker"]);
      });
    }, 60_000);

    test("the scoped shape, which the chunk-targeted scan serves when nothing expires", async () => {
      // The fixture really does select the scoped path, off and on.
      await withServe({}, async (s) => {
        await s.post(Q_OK, "alpha");
        expect(scanThreads()).toEqual(["main"]);
        const [timing] = await s.timings();
        expect(timing!.counts.scope_scans).toBeGreaterThan(0);
      });
      await withServe({ CLAWMEM_QUERY_VEC_DEADLINE_MS: "60000" }, async (s) => {
        await s.post(Q_OK, "alpha");
        expect(scanThreads()).toEqual(["worker"]);
        const [timing] = await s.timings();
        expect(timing!.counts.scope_scans).toBeGreaterThan(0);
      });
      await withServe(deadline, async (s) => {
        const { body } = await s.post(Q_BLOCK, "alpha");
        expect(body.degraded).toBe(true);
        expect(body.hits.map((h: Hit) => h.file)).toContain("alpha/ferry.md");
        await sleep(600);
        expect(scanThreads()).toEqual(["worker"]);
      });
    }, 90_000);

    test("the live-expansion shape", async () => {
      await withServe(deadline, async (s) => {
        const { status, body } = await s.post(Q_LIVE_BLOCK);
        expect(status).toBe(200);
        expect(body.degraded).toBe(true);
        await sleep(600);
        expect(scanThreads().length).toBeGreaterThan(0);
        expect(scanThreads()).not.toContain("main");
      });
    }, 60_000);
  });

  test("A4: the next request after an expiry succeeds, with vector legs, on a working pool", async () => {
    await withServe(
      { CLAWMEM_QUERY_VEC_DEADLINE_MS: "5000", CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "30000" },
      async (s) => {
        expect((await s.post(Q_BLOCK)).body.degraded).toBe(true);
        const next = await s.post(Q_OK);
        expect(next.status).toBe(200);
        expect(Array.isArray(next.body)).toBe(true);
        expect(hitsOf(next.body).length).toBeGreaterThan(0);
        const timings = await s.timings();
        expect(timings[1]!.counts.vec_deadline_expired).toBe(0);
        expect(timings[1]!.counts.vec_legs).toBe(1);
      },
    );
    expect(scanThreads()).toEqual(["worker", "worker"]);
  }, 60_000);

  test("abandoned scans are bounded: past the cap a query sheds without starting a scan", async () => {
    await withServe(
      { CLAWMEM_QUERY_VEC_DEADLINE_MS: "1500", CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "120000" },
      async (s) => {
        // A leg still embedding at the deadline dies at once and leaves nothing behind, so count
        // what reached the scan rather than assume each expiry did.
        let shed: Awaited<ReturnType<Serve["post"]>> | undefined;
        for (let i = 0; i < VEC_DEADLINE_MAX_ABANDONED + 4 && !shed; i++) {
          const res = await s.post(Q_BLOCK);
          if (res.body.degraded_reason === "vec_deadline_shed") shed = res;
          else expect(res.body.degraded_reason).toBe("vec_deadline_expired");
        }
        expect(shed).toBeDefined();
        expect(shed!.status).toBe(200);
        expect(shed!.body.hits.map((h: Hit) => h.file)).toContain("alpha/ferry.md");
        expect(scanThreads().length).toBe(VEC_DEADLINE_MAX_ABANDONED);
        const timings = await s.timings();
        expect(timings.at(-1)!.counts.vec_deadline_shed).toBe(1);
      },
    );
  }, 90_000);

  test("an abandoned worker stops counting when its scan really ends", async () => {
    const blockMs = 4000;
    await withServe(
      {
        CLAWMEM_QUERY_VEC_DEADLINE_MS: "1500",
        CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: String(blockMs),
      },
      async (s) => {
        for (let i = 0; i < VEC_DEADLINE_MAX_ABANDONED; i++) await s.post(Q_BLOCK);
        await sleep(blockMs + 2000);
        // Whether this query itself beats the deadline does not matter here; it must get to try.
        const back = await s.post(Q_OK);
        expect(back.status).toBe(200);
        expect(back.body.degraded_reason).not.toBe("vec_deadline_shed");
        expect(scanThreads().length).toBeGreaterThan(VEC_DEADLINE_MAX_ABANDONED - 1);
      },
    );
  }, 60_000);

  test("a fatal vector error surfaces exactly as it does with the deadline off", async () => {
    fakeModel = "some-other-model";
    try {
      const run = async (extra: Record<string, string>) => {
        let out: { status: number; body: unknown } | undefined;
        await withServe(extra, async (s) => {
          out = await s.post(Q_OK);
        });
        return out!;
      };
      const off = await run({});
      const on = await run({ CLAWMEM_QUERY_VEC_DEADLINE_MS: "60000" });
      expect(off.status).toBe(500);
      expect(String((off.body as { error: string }).error)).toContain("clawmem embed --force");
      expect(on.status).toBe(off.status);
      expect(on.body).toEqual(off.body);
    } finally {
      fakeModel = MODEL;
    }
  }, 90_000);
});

describe("clawmem query (CLI) under the deadline", () => {
  it("prints the usual array, notes the drop on stderr, and exits without waiting for the scan", async () => {
    writeFileSync(SCAN_LOG, "");
    const t0 = performance.now();
    const proc = Bun.spawn([process.execPath, "src/clawmem.ts", "query", Q_BLOCK, "--json"], {
      cwd: REPO_ROOT,
      env: serveEnv({
        CLAWMEM_QUERY_VEC_DEADLINE_MS: "5000",
        CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "60000",
      }),
      stdout: "pipe",
      stderr: "pipe",
    });
    const [stdout, stderr, code] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);
    expect(code, stderr).toBe(0);
    expect(performance.now() - t0).toBeLessThan(60_000);
    expect((JSON.parse(stdout) as Hit[]).map((h) => h.file)).toContain("alpha/ferry.md");
    expect(stderr).toContain("vec_deadline_expired");
    expect(scanThreads()).toEqual(["worker"]);
  }, 90_000);

  // vn4rz.135: a cut during worker load holds the terminate until the load ends, and the leg already
  // queued on that worker can start its scan first. That scan must not hold the CLI open either.
  it("exits without waiting for a scan when the deadline cuts during worker load", async () => {
    writeFileSync(SCAN_LOG, "");
    const t0 = performance.now();
    const proc = Bun.spawn([process.execPath, "src/clawmem.ts", "query", Q_BLOCK, "--json"], {
      cwd: REPO_ROOT,
      env: serveEnv({
        CLAWMEM_QUERY_VEC_DEADLINE_MS: "1",
        CLAWMEM_VEC_SCAN_TEST_BLOCK_MS: "60000",
      }),
      stdout: "pipe",
      stderr: "pipe",
    });
    const [stdout, stderr, code] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ]);
    expect(code, stderr).toBe(0);
    expect(performance.now() - t0).toBeLessThan(45_000);
    expect((JSON.parse(stdout) as Hit[]).map((h) => h.file)).toContain("alpha/ferry.md");
    expect(stderr).toContain("vec_deadline_expired");
  }, 90_000);
});
