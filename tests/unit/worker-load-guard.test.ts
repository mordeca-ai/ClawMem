import { describe, expect, it } from "bun:test";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  isWorkerLoadedMessage,
  WORKER_LOADED_MESSAGE,
  WorkerLoadGuard,
} from "../../src/worker-load-guard.ts";

/**
 * master-harness-vn4rz.135 — a worker is never terminated while it is still loading (Bun 1.3.13
 * panics the process: exit 132, SavedSourceMap.putValue). Deterministic: a fake worker counts the
 * terminate calls that reach it, and the test chooses when the loaded signal, an error or a close
 * arrives.
 */

class FakeWorker {
  terminateCalls = 0;
  private closers: (() => void)[] = [];
  terminate() {
    this.terminateCalls++;
  }
  addEventListener(_: "close", fn: () => void) {
    this.closers.push(fn);
  }
  exit() {
    for (const fn of this.closers) fn();
  }
}

describe("WorkerLoadGuard", () => {
  it("does not forward a terminate that arrives before the worker has loaded", () => {
    const w = new FakeWorker();
    const g = new WorkerLoadGuard(w);
    g.terminate();
    g.terminate();
    expect(w.terminateCalls).toBe(0);
    expect(g.isDeferred).toBe(true);
    expect(g.isSettled).toBe(false);
  });

  it("forwards the held terminate exactly once when the loaded signal arrives", () => {
    const w = new FakeWorker();
    const g = new WorkerLoadGuard(w);
    g.terminate();
    expect(g.observe(WORKER_LOADED_MESSAGE)).toBe(true);
    expect(w.terminateCalls).toBe(1);
    g.observe(WORKER_LOADED_MESSAGE);
    g.terminate();
    expect(w.terminateCalls).toBe(1);
    expect(g.isDeferred).toBe(false);
  });

  it("terminates a loaded worker at once", () => {
    const w = new FakeWorker();
    const g = new WorkerLoadGuard(w);
    g.observe(WORKER_LOADED_MESSAGE);
    g.terminate();
    expect(w.terminateCalls).toBe(1);
  });

  it("a close before the load settles it without any later terminate", () => {
    const w = new FakeWorker();
    const g = new WorkerLoadGuard(w);
    g.terminate();
    w.exit();
    expect(g.isSettled).toBe(true);
    g.observe(WORKER_LOADED_MESSAGE);
    g.terminate();
    expect(w.terminateCalls).toBe(0);
  });

  it("a load error settles it and releases the held terminate", () => {
    const w = new FakeWorker();
    const g = new WorkerLoadGuard(w);
    g.terminate();
    g.failed();
    expect(w.terminateCalls).toBe(1);
  });

  it("does not wait on a clock: a long-held terminate stays held", async () => {
    const w = new FakeWorker();
    const g = new WorkerLoadGuard(w);
    g.terminate();
    await new Promise((r) => setTimeout(r, 120));
    expect(w.terminateCalls).toBe(0);
  });

  it("consumes only the loaded signal: replies and look-alikes pass through", () => {
    const g = new WorkerLoadGuard(new FakeWorker());
    expect(g.observe({ id: 1, ok: true, results: [] })).toBe(false);
    expect(g.observe({ id: 0, ok: true, results: null })).toBe(false);
    expect(g.observe(null)).toBe(false);
    expect(g.observe("loaded")).toBe(false);
    expect(g.isSettled).toBe(false);
    expect(isWorkerLoadedMessage(WORKER_LOADED_MESSAGE)).toBe(true);
    // Replies carry an id; the signal has none, so no pending request can match it.
    expect("id" in WORKER_LOADED_MESSAGE).toBe(false);
  });
});

describe("a worker cut during load, on a real Worker", () => {
  // The terminate is held until the loaded signal. The leg already queued on the worker can start
  // first, and terminate does not stop a thread inside a blocking call: that thread must not hold
  // the process open. The busy main thread forces the order.
  it("does not hold the process open for the scan the worker started first", async () => {
    const dir = mkdtempSync(join(tmpdir(), "load-guard-"));
    const guardPath = resolve(
      import.meta.dir,
      "../../src/worker-load-guard.ts",
    );
    const started = join(dir, "scan-started");
    try {
      writeFileSync(
        join(dir, "w.ts"),
        `import { writeFileSync } from "node:fs";
         import { postWorkerLoaded } from ${JSON.stringify(guardPath)};
         self.onmessage = () => {
           writeFileSync(${JSON.stringify(started)}, "1");
           Bun.sleepSync(30_000);
         };
         postWorkerLoaded();`,
      );
      writeFileSync(
        join(dir, "host.ts"),
        `import { WorkerLoadGuard } from ${JSON.stringify(guardPath)};
         const w = new Worker(${JSON.stringify(join(dir, "w.ts"))});
         const g = new WorkerLoadGuard(w);
         w.onmessage = (ev) => { g.observe(ev.data); };
         w.postMessage("leg");
         g.terminate();
         setTimeout(() => Bun.sleepSync(2_000), 0);`,
      );
      const t0 = performance.now();
      const proc = Bun.spawn([process.execPath, join(dir, "host.ts")], {
        stdout: "ignore",
        stderr: "pipe",
      });
      const stderr = new Response(proc.stderr).text();
      const code = await proc.exited;
      expect(code, await stderr).toBe(0);
      expect(existsSync(started)).toBe(true); // the control: the scan did start before the terminate
      expect(performance.now() - t0).toBeLessThan(20_000);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
