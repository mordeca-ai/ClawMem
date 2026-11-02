/**
 * Deadline on the hybrid query's vector legs (master-harness-vn4rz.131).
 *
 * CLAWMEM_QUERY_VEC_DEADLINE_MS bounds the whole vector stage of `cmdQuery`: unset or 0 is off and
 * cmdQuery runs its older branches untouched. When on, every leg runs in a Worker, because a scan
 * on the main thread would block the very timer meant to end it. On expiry the workers are
 * terminated, the vector results are dropped and the query carries on with FTS, fusion and rerank.
 * Expiry never falls back to an in-process leg: that would rerun the scan on the main thread.
 *
 * Worker.terminate() does not stop a blocked scan. Measured on Bun 1.3.13: terminate() returns in
 * under 10 ms, but a worker inside sqlite3_step keeps scanning until the step returns, and its
 * `close` event fires only then. So an abandoned scan still burns CPU for its full length. This
 * class counts those workers from terminate to `close`, and sheds the vector stage (degraded,
 * without starting a scan) once too many are still running, so they cannot stack up.
 */
import { FatalVectorError } from "./store.ts";
import { runVecLegsOverlapped, type WorkerLegPool } from "./vec-legs.ts";

/** Why a response carries no vector results. */
export type VecDegradedReason = "vec_deadline_expired" | "vec_deadline_shed" | "vec_leg_failed";

/** Abandoned scans allowed to run at once before new queries skip the vector stage. */
export const VEC_DEADLINE_MAX_ABANDONED = 2;
/** Stop counting an abandoned worker after this long, so a lost `close` cannot shed forever. */
export const VEC_DEADLINE_ABANDONED_STALE_MS = 300_000;

/** CLAWMEM_QUERY_VEC_DEADLINE_MS: a positive integer, or 0 (off) for anything else. */
export function vecDeadlineMs(env: Record<string, string | undefined> = process.env): number {
  const raw = env.CLAWMEM_QUERY_VEC_DEADLINE_MS?.trim();
  if (!raw || !/^\d+$/.test(raw)) return 0;
  return Number.parseInt(raw, 10);
}

/** A worker's error crossed the thread boundary as text: restore the fatal class it had there. */
function fatalFromWorker(err: unknown): FatalVectorError | null {
  const name = (err as { fatalName?: string } | null)?.fatalName;
  if (!name) return null;
  const fatal = new FatalVectorError((err as Error).message);
  fatal.name = name;
  return fatal;
}

export type VecDeadlinePool = Pick<
  WorkerLegPool,
  "size" | "warm" | "close" | "closeTracked" | "runLegOn" | "runLegScoped" | "releaseScoped"
>;

export type VecDeadlineOutcome<T> =
  | { status: "ok"; value: T }
  | { status: "expired" }
  | { status: "shed" }
  | { status: "failed"; error: Error };

export interface VecDeadlineLegsOpts {
  createPool: () => VecDeadlinePool;
  maxAbandoned?: number;
  abandonedStaleMs?: number;
  now?: () => number;
}

/** Owns the leg pool across queries and tracks the scans abandoned on expiry. */
export class VecDeadlineLegs {
  private pool: VecDeadlinePool | null = null;
  private warmDbPath: string | undefined;
  private readonly abandoned = new Set<{ since: number }>();
  private readonly maxAbandoned: number;
  private readonly staleMs: number;
  private readonly now: () => number;

  constructor(private readonly opts: VecDeadlineLegsOpts) {
    this.maxAbandoned = opts.maxAbandoned ?? VEC_DEADLINE_MAX_ABANDONED;
    this.staleMs = opts.abandonedStaleMs ?? VEC_DEADLINE_ABANDONED_STALE_MS;
    this.now = opts.now ?? (() => Date.now());
  }

  /** Workers terminated but still inside their scan. */
  abandonedScans(): number {
    const cutoff = this.now() - this.staleMs;
    for (const entry of this.abandoned) if (entry.since <= cutoff) this.abandoned.delete(entry);
    return this.abandoned.size;
  }

  /** Long-lived owner (serve): keep one worker ready now and again after each expiry. */
  keepWarm(dbPath: string): void {
    this.warmDbPath = dbPath;
    this.warmUp();
  }

  private warmUp(): void {
    if (this.warmDbPath === undefined) return;
    this.pool ??= this.opts.createPool();
    void this.pool.warm(1, this.warmDbPath);
  }

  private retirePool(): void {
    const pool = this.pool;
    this.pool = null;
    if (!pool) return;
    for (const exited of pool.closeTracked()) {
      const entry = { since: this.now() };
      this.abandoned.add(entry);
      void exited.then(() => this.abandoned.delete(entry));
    }
    this.warmUp();
  }

  /**
   * Run `work` on the pool, bounded by `deadlineMs`. Expiry and a non-fatal leg failure both retire
   * the pool, so no leg of this query can outlive it; a fatal vector error is rethrown.
   */
  async run<T>(
    deadlineMs: number,
    work: (pool: VecDeadlinePool) => Promise<T>,
  ): Promise<VecDeadlineOutcome<T>> {
    if (this.abandonedScans() >= this.maxAbandoned) return { status: "shed" };
    const pool = (this.pool ??= this.opts.createPool());
    let timer: ReturnType<typeof setTimeout> | undefined;
    const expired = new Promise<"expired">((resolve) => {
      timer = setTimeout(() => resolve("expired"), deadlineMs);
    });
    try {
      const value = await Promise.race([work(pool), expired]);
      if (value === "expired") {
        this.retirePool();
        return { status: "expired" };
      }
      return { status: "ok", value: value as T };
    } catch (err) {
      this.retirePool();
      const fatal = fatalFromWorker(err);
      if (fatal) throw fatal;
      if (err instanceof FatalVectorError) throw err;
      return { status: "failed", error: err instanceof Error ? err : new Error(String(err)) };
    } finally {
      clearTimeout(timer);
    }
  }

  close(): void {
    this.warmDbPath = undefined;
    this.pool?.close();
    this.pool = null;
  }
}

export interface VecDeadlineStageOpts<E, R> {
  legs: VecDeadlineLegs;
  deadlineMs: number;
  /** The original query: its leg starts at once. */
  original: string;
  /** Query expansion in flight. Must not reject. */
  expansion: Promise<E>;
  legsOf: (expansion: E) => string[];
  /** Binds one leg to the pool. There is no in-process alternative here, on purpose. */
  legOn: (pool: VecDeadlinePool) => (query: string) => Promise<R>;
  /** Workers to warm beyond the one running the original leg. */
  warm: number;
  /** Runs after the legs settle on a pool that survived them. */
  after?: (pool: VecDeadlinePool) => void;
}

export interface VecDeadlineStageResult<E, R> {
  /** The expansion, when it landed in time: its lex legs stay usable even if the scans expired. */
  expansion: E | undefined;
  results: R[];
  degraded?: VecDegradedReason;
}

/** The vector stage under a deadline: the same legs in the same order as the unbounded path. */
export async function runVecStageWithDeadline<E, R>(
  opts: VecDeadlineStageOpts<E, R>,
): Promise<VecDeadlineStageResult<E, R>> {
  let landed: E | undefined;
  const expansion = opts.expansion.then((e) => {
    landed = e;
    return e;
  });
  // The original query's leg does not wait for the expansion (vn4rz.97), so it can finish while the
  // expansion is still in flight at the deadline. Keep that result: dropping it threw away a scan
  // that had completed (vn4rz.134, finding 4).
  let firstLeg: R | undefined;
  let firstLegDone = false;
  const outcome = await opts.legs.run(opts.deadlineMs, async (pool) => {
    const leg = opts.legOn(pool);
    let started = false;
    const out = await runVecLegsOverlapped({
      original: opts.original,
      expansion,
      legsOf: opts.legsOf,
      runLeg: (q) => {
        const p = leg(q);
        if (!started) {
          started = true;
          p.then(
            (r) => {
              firstLeg = r;
              firstLegDone = true;
            },
            () => {},
          );
        }
        return p;
      },
      pool,
      warm: opts.warm,
    });
    opts.after?.(pool);
    return out;
  });
  switch (outcome.status) {
    case "ok":
      return { expansion: outcome.value.expansion, results: outcome.value.results };
    case "expired":
      return {
        expansion: landed,
        results: firstLegDone ? [firstLeg as R] : [],
        degraded: "vec_deadline_expired",
      };
    case "shed":
      return { expansion: landed, results: [], degraded: "vec_deadline_shed" };
    case "failed":
      console.error(
        `[clawmem] vector leg worker failed, dropping vector results: ${outcome.error.message}`,
      );
      return { expansion: landed, results: [], degraded: "vec_leg_failed" };
  }
}
