/**
 * FTS off the main thread, for the query budget (master-harness-vn4rz.135).
 *
 * `Store.searchFTS` is one synchronous SQLite call. On the main thread it blocks every timer and
 * every incoming request until it returns, so CLAWMEM_QUERY_BUDGET_MS could neither cut it nor
 * stamp the arrival of a request that came in meanwhile. With a budget set, `cmdQuery` runs it here
 * instead: a read-only worker answers, and the caller awaits that against the budget it has left.
 *
 * A cut worker is terminated, but terminate() does not free it. Measured on Bun 1.3.13: it returns
 * in about 3 ms, no late message is delivered, and the thread keeps running its sqlite3_step to the
 * end; its `close` event fires only when the step returns (a 12.6 s statement closed at 13.5 s
 * after a terminate at 0.3 s). So an abandoned FTS still burns a core for its full length. This
 * class counts such workers from terminate to `close` and keeps live + abandoned workers under a
 * hard cap; at the cap a request is shed (no FTS started) instead of adding another.
 *
 * A worker is never terminated while it is still loading: on Bun 1.3.13 that panics the whole
 * process. `WorkerLoadGuard` (src/worker-load-guard.ts) holds an early terminate until the worker
 * posts its loaded signal, so a cut worker that was still booting stays in the abandoned count
 * until the signal arrives, is terminated, and its `close` fires. There is no timed force.
 *
 * FTS has its own workers, not a message on the vector-leg pool: the pool's workers may be inside
 * an abandoned scan for minutes, and an FTS queued behind one would be cut by the budget without
 * ever having run.
 */
import type { SearchResult } from "./store.ts";
import { WorkerLoadGuard } from "./worker-load-guard.ts";

/** Live plus abandoned FTS workers allowed at once. */
export const FTS_MAX_WORKERS = 3;
/** Stop counting an abandoned worker after this long, so a lost `close` cannot shed forever. */
export const FTS_ABANDONED_STALE_MS = 300_000;

export type FtsOutcome =
  | { status: "ok"; results: SearchResult[] }
  | { status: "expired" }
  | { status: "shed" }
  | { status: "failed"; error: Error };

export interface FtsWorkerHandle {
  postMessage(msg: unknown): void;
  terminate(): void;
  onmessage: ((ev: { data: any }) => void) | null;
  onerror: ((ev: { error?: unknown; message?: string }) => void) | null;
  addEventListener(type: "close", fn: () => void): void;
}

export interface FtsLegsOpts {
  /** Test hook; the default starts src/fts-worker.ts. */
  spawn?: () => FtsWorkerHandle;
  maxWorkers?: number;
  abandonedStaleMs?: number;
  now?: () => number;
  /** Called each time a worker is started. */
  onSpawn?: () => void;
}

interface Slot {
  worker: FtsWorkerHandle;
  pending?: { id: number; resolve: (o: FtsOutcome) => void };
  guard: WorkerLoadGuard;
  dead: boolean;
}

function spawnFtsWorker(): FtsWorkerHandle {
  // The environment as it is now: a worker's default copy can predate a later change to it.
  const env = Object.fromEntries(
    Object.entries(process.env).filter(([, v]) => v !== undefined),
  ) as Record<string, string>;
  return new Worker(new URL("./fts-worker.ts", import.meta.url).href, {
    env,
  }) as unknown as FtsWorkerHandle;
}

export class FtsLegs {
  private readonly slots = new Set<Slot>();
  private readonly abandoned = new Set<{ since: number }>();
  private readonly spawnWorker: () => FtsWorkerHandle;
  private readonly max: number;
  private readonly staleMs: number;
  private readonly now: () => number;
  private readonly onSpawn: (() => void) | undefined;
  private warmDbPath: string | undefined;
  private nextId = 1;
  private closed = false;
  /** Workers ever started, for tests and observability. */
  spawned = 0;

  constructor(opts: FtsLegsOpts = {}) {
    this.spawnWorker = opts.spawn ?? spawnFtsWorker;
    this.max = Math.max(1, opts.maxWorkers ?? FTS_MAX_WORKERS);
    this.staleMs = opts.abandonedStaleMs ?? FTS_ABANDONED_STALE_MS;
    this.now = opts.now ?? (() => Date.now());
    this.onSpawn = opts.onSpawn;
  }

  /** Terminated workers whose thread is still inside its statement. */
  abandonedCount(): number {
    const cutoff = this.now() - this.staleMs;
    for (const entry of this.abandoned) if (entry.since <= cutoff) this.abandoned.delete(entry);
    return this.abandoned.size;
  }

  /** Live workers (idle or busy), not counting abandoned ones. */
  liveCount(): number {
    return this.slots.size;
  }

  /** Live plus abandoned: what the cap limits. */
  totalCount(): number {
    return this.slots.size + this.abandonedCount();
  }

  /** Long-lived owner (serve): keep one worker up with the vault open, now and after each cut. */
  keepWarm(dbPath: string): void {
    this.warmDbPath = dbPath;
    this.warmUp();
  }

  private warmUp(): void {
    if (this.closed || this.warmDbPath === undefined) return;
    if (this.slots.size > 0 || this.totalCount() >= this.max) return;
    const slot = this.spawn();
    if (slot) {
      try {
        slot.worker.postMessage({ id: 0, warm: true, dbPath: this.warmDbPath });
      } catch {
        this.drop(slot);
      }
    }
  }

  private spawn(): Slot | undefined {
    let worker: FtsWorkerHandle;
    try {
      worker = this.spawnWorker();
    } catch {
      return undefined;
    }
    this.spawned++;
    this.onSpawn?.();
    const slot: Slot = { worker, guard: new WorkerLoadGuard(worker), dead: false };
    worker.onmessage = (ev) => {
      const msg = ev.data;
      if (slot.guard.observe(msg)) return; // the loaded signal is never a reply
      const pending = slot.pending;
      // A reply for anything but the request this worker is on (a warm-up, or a request that was
      // already cut) is dropped: a late answer can never land on a later request.
      if (!pending || pending.id !== msg?.id || slot.dead) return;
      slot.pending = undefined;
      if (msg.ok) pending.resolve({ status: "ok", results: msg.results as SearchResult[] });
      else pending.resolve({ status: "failed", error: new Error(String(msg.error)) });
    };
    worker.onerror = (ev) => {
      slot.guard.failed();
      const pending = slot.pending;
      slot.pending = undefined;
      this.drop(slot);
      pending?.resolve({
        status: "failed",
        error: ev.error instanceof Error ? ev.error : new Error(ev.message || "FTS worker error"),
      });
    };
    // A thread that exits without an error event (a silent exit) must not wedge its slot: drop it
    // and fail whatever it was running, so the caller falls back instead of waiting out a budget.
    worker.addEventListener("close", () => {
      slot.dead = true;
      this.slots.delete(slot);
      const pending = slot.pending;
      slot.pending = undefined;
      pending?.resolve({ status: "failed", error: new Error("FTS worker exited") });
    });
    this.slots.add(slot);
    return slot;
  }

  private drop(slot: Slot): void {
    slot.dead = true;
    this.slots.delete(slot);
    slot.guard.terminate(); // held until the worker has loaded
  }

  /** Terminate a busy worker and count it until its thread really exits. */
  private abandon(slot: Slot): void {
    const entry = { since: this.now() };
    this.abandoned.add(entry);
    slot.worker.addEventListener("close", () => this.abandoned.delete(entry));
    slot.pending = undefined;
    this.drop(slot);
    this.warmUp();
  }

  /**
   * Run `searchFTS(query, limit, undefined, collections)` in a worker, giving up after `budgetMs`.
   * Never rejects. "shed" means nothing was started: the cap is reached and none of the live
   * workers is free.
   */
  async search(
    dbPath: string,
    query: string,
    limit: number,
    collections: string[] | undefined,
    budgetMs: number,
  ): Promise<FtsOutcome> {
    if (this.closed) return { status: "failed", error: new Error("FTS workers closed") };
    if (budgetMs <= 0) return { status: "expired" };
    let slot = [...this.slots].find((s) => !s.pending);
    if (!slot) {
      if (this.totalCount() >= this.max) return { status: "shed" };
      slot = this.spawn();
      if (!slot) return { status: "failed", error: new Error("FTS worker spawn failed") };
    }
    const id = this.nextId++;
    const mine = slot;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const answer = new Promise<FtsOutcome>((resolve) => {
      mine.pending = { id, resolve };
      timer = setTimeout(() => resolve({ status: "expired" }), budgetMs);
    });
    try {
      mine.worker.postMessage({ id, dbPath, query, limit, collections });
    } catch (e) {
      const p = mine.pending;
      mine.pending = undefined;
      p?.resolve({ status: "failed", error: e instanceof Error ? e : new Error(String(e)) });
    }
    const outcome = await answer;
    clearTimeout(timer);
    if (outcome.status === "expired") this.abandon(mine);
    else if (outcome.status === "failed" && !mine.dead) this.drop(mine);
    return outcome;
  }

  close(): void {
    this.closed = true;
    this.warmDbPath = undefined;
    for (const slot of [...this.slots]) {
      const pending = slot.pending;
      slot.pending = undefined;
      this.drop(slot);
      pending?.resolve({ status: "failed", error: new Error("FTS workers closed") });
    }
  }
}
