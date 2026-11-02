import { afterAll, beforeAll, describe, expect, it } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { FtsLegs, type FtsWorkerHandle } from "../../src/fts-legs.ts";
import { createStore } from "../../src/store.ts";
import { WORKER_LOADED_MESSAGE } from "../../src/worker-load-guard.ts";
import { hashContent } from "../../src/indexer.ts";

/**
 * master-harness-vn4rz.135 — FTS off the main thread, under the query budget.
 *
 * Two halves. Fake workers drive the cap, the late-result and the crash rules exactly. Real
 * workers prove the answer equals the main thread's on a seeded vault and that a store-open
 * failure comes back as a failure instead of a hang.
 */

class FakeWorker implements FtsWorkerHandle {
  onmessage: FtsWorkerHandle["onmessage"] = null;
  onerror: FtsWorkerHandle["onerror"] = null;
  posted: any[] = [];
  terminated = false;
  private closers: (() => void)[] = [];
  postMessage(msg: unknown) {
    this.posted.push(msg);
  }
  terminate() {
    this.terminated = true;
  }
  addEventListener(_: "close", fn: () => void) {
    this.closers.push(fn);
  }
  /** The thread really exits (the statement returned). */
  exit() {
    for (const fn of this.closers) fn();
  }
  reply(data: unknown) {
    this.onmessage?.({ data });
  }
  get searches() {
    return this.posted.filter((m) => m.query !== undefined);
  }
}

/** `loadSignal: false` leaves the workers loading forever, for the terminate-during-load rules. */
function fakePool(max: number, loadSignal = true) {
  const made: FakeWorker[] = [];
  let peakLive = 0;
  const legs = new FtsLegs({
    maxWorkers: max,
    spawn: () => {
      const w = new FakeWorker();
      made.push(w);
      // A real worker posts this as the last statement of its module; a microtask stands in.
      if (loadSignal) queueMicrotask(() => w.reply(WORKER_LOADED_MESSAGE));
      return w;
    },
  });
  const note = () => {
    peakLive = Math.max(peakLive, legs.totalCount());
  };
  return { legs, made, note, peak: () => peakLive };
}

const row = (name: string) => ({ filepath: `clawmem://a/${name}` }) as any;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

describe("FtsLegs with fake workers", () => {
  it("creates nothing until asked: no worker for an owner that is never used", () => {
    const { legs, made } = fakePool(3);
    expect(legs.spawned).toBe(0);
    expect(made.length).toBe(0);
  });

  it("answers from the worker and keeps it for the next request", async () => {
    const { legs, made } = fakePool(3);
    const p = legs.search("db", "q1", 10, ["a"], 5000);
    made[0]!.reply({ id: made[0]!.searches[0].id, ok: true, results: [row("x")] });
    expect(await p).toEqual({ status: "ok", results: [row("x")] });
    expect(made[0]!.searches[0]).toMatchObject({
      dbPath: "db",
      query: "q1",
      limit: 10,
      collections: ["a"],
    });
    const p2 = legs.search("db", "q2", 10, undefined, 5000);
    expect(made.length).toBe(1);
    made[0]!.reply({ id: made[0]!.searches[1].id, ok: true, results: [] });
    expect((await p2).status).toBe("ok");
  });

  it("does not take a worker's warm-up reply for the answer to a request", async () => {
    const { legs, made } = fakePool(3);
    legs.keepWarm("db");
    expect(made[0]!.posted[0]).toMatchObject({ warm: true });
    const p = legs.search("db", "q", 10, undefined, 5000);
    made[0]!.reply({ id: 0, ok: true, results: null }); // the warm-up lands after the request was posted
    made[0]!.reply({ id: made[0]!.searches[0].id, ok: true, results: [row("real")] });
    expect(await p).toEqual({ status: "ok", results: [row("real")] });
    expect(made.length).toBe(1); // the request used the warming worker
  });

  it("cuts at the budget, terminates the worker, and never lets its late answer reach a later request", async () => {
    const { legs, made } = fakePool(3);
    const p1 = legs.search("db", "slow", 10, undefined, 30);
    expect(await p1).toEqual({ status: "expired" });
    expect(made[0]!.terminated).toBe(true);
    expect(legs.abandonedCount()).toBe(1);
    const p2 = legs.search("db", "next", 10, undefined, 5000);
    expect(made.length).toBe(2); // a fresh worker, not the abandoned one
    // The abandoned thread answers late, with the id of the request that was cut.
    made[0]!.reply({ id: made[0]!.searches[0].id, ok: true, results: [row("STALE")] });
    made[1]!.reply({ id: made[1]!.searches[0].id, ok: true, results: [row("fresh")] });
    expect(await p2).toEqual({ status: "ok", results: [row("fresh")] });
  });

  it("a request that finds no budget left starts nothing", async () => {
    const { legs, made } = fakePool(3);
    expect(await legs.search("db", "q", 10, undefined, 0)).toEqual({ status: "expired" });
    expect(made.length).toBe(0);
  });

  it("holds live + abandoned workers under the cap and sheds a request at it without starting FTS", async () => {
    const { legs, made, note, peak } = fakePool(2);
    expect((await legs.search("db", "a", 10, undefined, 20)).status).toBe("expired");
    note();
    expect((await legs.search("db", "b", 10, undefined, 20)).status).toBe("expired");
    note();
    // Two abandoned threads are still inside their statements: the cap is reached.
    expect(legs.abandonedCount()).toBe(2);
    const spawnedBefore = made.length;
    const shed = await legs.search("db", "c", 10, undefined, 5000);
    note();
    expect(shed).toEqual({ status: "shed" });
    expect(made.length).toBe(spawnedBefore);
    for (const w of made) expect(w.searches.some((m) => m.query === "c")).toBe(false);
    expect(peak()).toBeLessThanOrEqual(2);
    // One thread finally exits: a request is accepted again.
    made[0]!.exit();
    expect(legs.abandonedCount()).toBe(1);
    const p = legs.search("db", "d", 10, undefined, 5000);
    const w = made[made.length - 1]!;
    w.reply({ id: w.searches[0].id, ok: true, results: [] });
    expect((await p).status).toBe("ok");
    expect(legs.totalCount()).toBeLessThanOrEqual(2);
  });

  it("stops counting an abandoned worker whose close event never came", async () => {
    let now = 1000;
    const made: FakeWorker[] = [];
    const legs = new FtsLegs({
      maxWorkers: 1,
      abandonedStaleMs: 500,
      now: () => now,
      spawn: () => {
        const w = new FakeWorker();
        made.push(w);
        return w;
      },
    });
    expect((await legs.search("db", "a", 10, undefined, 10)).status).toBe("expired");
    expect((await legs.search("db", "b", 10, undefined, 10)).status).toBe("shed");
    now += 501;
    expect(legs.abandonedCount()).toBe(0);
  });

  it("a worker error degrades to a failure at once, drops the worker, and the next request gets a new one", async () => {
    const { legs, made } = fakePool(3);
    const t0 = performance.now();
    const p = legs.search("db", "q", 10, undefined, 5000);
    made[0]!.onerror?.({ message: "boom" });
    const r = await p;
    expect(r.status).toBe("failed");
    expect(performance.now() - t0).toBeLessThan(1000); // not the 5000 ms budget
    expect(made[0]!.terminated).toBe(true);
    const p2 = legs.search("db", "q", 10, undefined, 5000);
    expect(made.length).toBe(2);
    made[1]!.reply({ id: made[1]!.searches[0].id, ok: true, results: [] });
    expect((await p2).status).toBe("ok");
  });

  it("a spawn that throws is a failure, not a hang", async () => {
    const legs = new FtsLegs({
      spawn: () => {
        throw new Error("no threads");
      },
    });
    expect((await legs.search("db", "q", 10, undefined, 5000)).status).toBe("failed");
  });

  it("re-warms a worker after a cut, within the cap", async () => {
    const { legs, made } = fakePool(3);
    legs.keepWarm("db");
    made[0]!.reply({ id: 0, ok: true, results: null });
    const p = legs.search("db", "q", 10, undefined, 20);
    expect((await p).status).toBe("expired");
    await sleep(0);
    expect(made.length).toBe(2);
    expect(made[1]!.posted[0]).toMatchObject({ warm: true });
  });
});

// ---------------------------------------------------------------------------------------------
// Real workers
// ---------------------------------------------------------------------------------------------

const ROOT = mkdtempSync(join(tmpdir(), "clawmem-fts-legs-"));
const DB = join(ROOT, "index.sqlite");
const savedEnv: Record<string, string | undefined> = {};

beforeAll(() => {
  for (const k of ["CLAWMEM_CONFIG_DIR", "INDEX_PATH", "CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS"])
    savedEnv[k] = process.env[k];
  delete process.env.CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS;
  process.env.CLAWMEM_CONFIG_DIR = ROOT;
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
  store.db.transaction(() => {
    const docs: [string, string, string][] = [
      [
        "alpha",
        "ferry.md",
        "# Orbital ferry schedule\n\nThe orbital ferry departs every six hours.\n",
      ],
      [
        "alpha",
        "dock.md",
        "# Docking bay rota\n\nFerry crews rotate through the docking bays weekly.\n",
      ],
      [
        "beta",
        "cargo.md",
        "# Cargo manifest\n\nCargo on the orbital ferry is weighed before departure.\n",
      ],
      ["beta", "other.md", "# Unrelated\n\nNothing about shipping here.\n"],
    ];
    for (const [coll, path, body] of docs) {
      const hash = hashContent(body + path);
      store.insertContent(hash, body, now);
      store.insertDocument(coll, path, path, hash, now, now);
    }
  })();
  store.close();
}, 120_000);

afterAll(() => {
  for (const [k, v] of Object.entries(savedEnv)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  rmSync(ROOT, { recursive: true, force: true });
});

describe("FtsLegs with a real worker", () => {
  it("returns exactly what the main thread's searchFTS returns, scoped and unscoped", async () => {
    const legs = new FtsLegs();
    const main = createStore(DB);
    try {
      for (const [q, limit, cols] of [
        ["orbital ferry", 20, undefined],
        ["orbital ferry", 20, ["alpha"]],
        ["orbital ferry", 20, ["alpha", "beta"]],
        ["cargo", 1, undefined],
        ["nomatchzzz", 20, undefined],
        ["the", 20, undefined],
      ] as [string, number, string[] | undefined][]) {
        const got = await legs.search(DB, q, limit, cols, 20_000);
        expect(got.status).toBe("ok");
        const want = main.searchFTS(q, limit, undefined, cols);
        expect((got as { results: unknown[] }).results).toEqual(want);
        if (q === "orbital ferry" && !cols) expect(want.length).toBeGreaterThan(1);
      }
      expect(legs.spawned).toBe(1); // one warm worker served all of them
    } finally {
      main.close();
      legs.close();
    }
  }, 60_000);

  it("is read-only: the worker's handle cannot write", async () => {
    const legs = new FtsLegs();
    try {
      const before = createStore(DB);
      const n = (before.db.prepare("SELECT count(*) AS n FROM documents").get() as { n: number }).n;
      before.close();
      expect((await legs.search(DB, "ferry", 5, undefined, 20_000)).status).toBe("ok");
      const after = createStore(DB);
      expect(
        (after.db.prepare("SELECT count(*) AS n FROM documents").get() as { n: number }).n,
      ).toBe(n);
      after.close();
    } finally {
      legs.close();
    }
  }, 60_000);

  it("a vault that cannot be opened is a failure at once, and the next request still works", async () => {
    const legs = new FtsLegs();
    try {
      const t0 = performance.now();
      const bad = await legs.search(
        join(ROOT, "missing", "nope.sqlite"),
        "ferry",
        5,
        undefined,
        20_000,
      );
      expect(bad.status).toBe("failed");
      expect(performance.now() - t0).toBeLessThan(15_000);
      expect((await legs.search(DB, "ferry", 5, undefined, 20_000)).status).toBe("ok");
    } finally {
      legs.close();
    }
  }, 60_000);

  it("a stall inside the worker is cut at the budget and the worker is counted as abandoned", async () => {
    process.env.CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS = "3000";
    const legs = new FtsLegs();
    try {
      legs.keepWarm(DB);
      await sleep(500); // let the warm-up open the vault so the budget below is all stall
      const t0 = performance.now();
      const r = await legs.search(DB, "ferry", 5, undefined, 500);
      expect(r.status).toBe("expired");
      expect(performance.now() - t0).toBeLessThan(2500); // the stall is 3000 ms
      expect(legs.abandonedCount()).toBe(1);
    } finally {
      delete process.env.CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS;
      legs.close();
    }
  }, 60_000);
});

describe("FtsLegs and the terminate-during-load rule (vn4rz.135)", () => {
  it("a cut worker that is still loading is not terminated until it has loaded, and stays counted", async () => {
    const { legs, made } = fakePool(3, false);
    const p = legs.search("db", "q", 10, undefined, 20);
    expect(await p).toEqual({ status: "expired" });
    expect(made[0]!.terminated).toBe(false);
    expect(legs.abandonedCount()).toBe(1);
    expect(legs.totalCount()).toBe(1);
    made[0]!.reply(WORKER_LOADED_MESSAGE);
    expect(made[0]!.terminated).toBe(true);
    made[0]!.exit();
    expect(legs.abandonedCount()).toBe(0);
  });

  it("the loaded signal is not a reply: it cannot resolve a pending request", async () => {
    const { legs, made } = fakePool(3, false);
    let done = false;
    const p = legs.search("db", "q", 10, undefined, 5000).then((o) => {
      done = true;
      return o;
    });
    made[0]!.reply(WORKER_LOADED_MESSAGE);
    await sleep(10);
    expect(done).toBe(false);
    made[0]!.reply({ id: made[0]!.searches[0].id, ok: true, results: [row("x")] });
    expect(await p).toEqual({ status: "ok", results: [row("x")] });
  });

  it("a silent exit (close, no error) drops the slot and fails the pending request", async () => {
    const { legs, made } = fakePool(1);
    const p = legs.search("db", "q", 10, undefined, 5000);
    made[0]!.exit();
    const out = await p;
    expect(out.status).toBe("failed");
    expect(legs.liveCount()).toBe(0);
    expect(legs.totalCount()).toBe(0);
    // The next request gets a fresh worker instead of a wedged slot.
    const next = legs.search("db", "q2", 10, undefined, 5000);
    expect(made.length).toBe(2);
    made[1]!.reply({ id: made[1]!.searches[0].id, ok: true, results: [] });
    expect((await next).status).toBe("ok");
  });

  it("a synchronous postMessage throw resolves as failed and leaves no timer or pending", async () => {
    // Throws on the first search post only.
    const w = new FakeWorker();
    let first = true;
    w.postMessage = () => {
      if (first) {
        first = false;
        throw new Error("post failed");
      }
    };
    const l2 = new FtsLegs({ maxWorkers: 2, spawn: () => w });
    const t0 = Date.now();
    const out = await l2.search("db", "q", 10, undefined, 5000);
    expect(out.status).toBe("failed");
    expect(Date.now() - t0).toBeLessThan(1000); // not the 5 s budget: the timer was cleared and unused
    expect(w.terminated).toBe(false); // still loading: held, not terminated
    expect(l2.liveCount()).toBe(0); // failed: the worker was dropped, not kept
  });
});
