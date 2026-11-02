/**
 * Never terminate a worker before its module graph has loaded (master-harness-vn4rz.135).
 *
 * On Bun 1.3.13, `Worker.terminate()` while the worker is still loading its modules can panic the
 * whole process (SIGILL, exit 132, in `SavedSourceMap.putValue` under a thread-pool
 * `TranspilerJob`: the worker's transpile races the teardown). Measured with the real
 * vec-leg-worker: terminated during load, at least 27 panics; terminated after a warm reply while
 * fresh workers loaded beside it, 0 of 45; never terminated, 0 of 45. So a terminate that arrives
 * early is held until the worker says it has loaded, then forwarded once.
 *
 * Each worker script posts `WORKER_LOADED_MESSAGE` as the LAST statement of its module evaluation
 * (`postWorkerLoaded()`). The owner of the Worker passes every message through `observe()`, which
 * consumes that signal so it can never be read as a reply. A load error or `close` also counts as
 * settled: nothing is left loading, so a held terminate is forwarded (or dropped if the thread is
 * already gone). There is deliberately no timer that forces a terminate: that would bring the
 * crash back on a slow host.
 */

/** Distinct from every reply shape: replies carry `id` and `ok`, this carries neither. */
export const WORKER_LOADED_MESSAGE = Object.freeze({ __clawmemWorkerLoaded: true });

/** Worker side: the last statement of the script. */
export function postWorkerLoaded(): void {
  (globalThis as unknown as { postMessage(m: unknown): void }).postMessage(WORKER_LOADED_MESSAGE);
}

export function isWorkerLoadedMessage(data: unknown): boolean {
  return typeof data === "object" && data !== null && (data as Record<string, unknown>).__clawmemWorkerLoaded === true;
}

export interface TerminableWorker {
  terminate(): void;
  unref?(): void;
  addEventListener(type: "close", fn: () => void): void;
}

export class WorkerLoadGuard {
  private settled = false;
  private wantTerminate = false;
  private forwarded = false;

  constructor(private readonly worker: TerminableWorker) {
    // A thread that has exited is not loading. Registered at spawn, before `close` can fire.
    try {
      worker.addEventListener("close", () => this.closed());
    } catch {}
  }

  /** True when the worker has finished loading, or failed or closed. */
  get isSettled(): boolean {
    return this.settled;
  }

  /** True when a terminate has been asked for and is still waiting for the load to finish. */
  get isDeferred(): boolean {
    return this.wantTerminate && !this.forwarded;
  }

  /** Pass each message from the worker here. Returns true if it was the loaded signal (consumed). */
  observe(data: unknown): boolean {
    if (!isWorkerLoadedMessage(data)) return false;
    this.settle();
    return true;
  }

  /** The thread has exited: settled, and there is nothing left to terminate. */
  private closed(): void {
    this.settled = true;
    this.forwarded = true;
  }

  /** The worker raised an error event: nothing is loading any more. */
  failed(): void {
    this.settle();
  }

  /** Terminate now if loaded; otherwise once, when it settles. Safe to call repeatedly. */
  terminate(): void {
    this.wantTerminate = true;
    if (this.settled) this.forward();
  }

  private settle(): void {
    this.settled = true;
    if (this.wantTerminate) this.forward();
  }

  private forward(): void {
    if (this.forwarded) return;
    this.forwarded = true;
    // Terminate does not stop a thread inside a blocking call, and a leg queued during load can
    // start before this runs: that thread must not hold a CLI process open after its answer. Not
    // unreffed earlier, so the process cannot exit while the worker is still loading.
    try {
      this.worker.unref?.();
    } catch {}
    try {
      this.worker.terminate();
    } catch {}
  }
}
