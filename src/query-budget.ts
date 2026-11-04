/**
 * One whole-request budget for `clawmem serve` POST /query (master-harness-vn4rz.134).
 *
 * CLAWMEM_QUERY_BUDGET_MS is measured from the request's ARRIVAL, before the serial queue wait. It
 * bounds the queue wait, the vector stage and rerank. It extends CLAWMEM_QUERY_VEC_DEADLINE_MS
 * (vn4rz.131), which starts only when the vector stage does: behind the queue, four concurrent
 * POSTs at a 2000 ms deadline returned at 2055, 4073, 4084 and 4092 ms. Unset or 0 is off, and
 * every code path then runs exactly as it did before.
 *
 * FTS (vn4rz.135): with the budget on, the original query's FTS and any expansion lex leg run in a
 * read-only worker (src/fts-legs.ts) and are awaited against the budget left, so a slow FTS is cut
 * at the budget and the main thread keeps stamping arrivals meanwhile. A cut FTS yields no hits
 * (the query is then flagged degraded, later stages skipped), and the cut worker keeps its thread
 * until its SQLite call returns: terminate() does not free it. Those workers are capped: live plus
 * abandoned (up to 3 abandoned, with none live) stay under the cap, and past it a request is shed
 * without starting FTS. A shed with budget left is degraded but still runs the vector stage; only a
 * cut or a spent budget skips it.
 *
 * A worker is never terminated while it is still loading (src/worker-load-guard.ts). On Bun
 * 1.3.13 that panics the whole process (SIGILL, exit 132, in the worker's module transpile), so a
 * cut worker that is still booting is terminated only once it has posted its loaded signal.
 *
 * Cache writes (vn4rz.140): the rerank cache flush (store.ts flushCacheWrites) and the expansion
 * cache write are best-effort, so under a budget each waits QUERY_BUDGET_CACHE_WRITE_BUSY_MS for
 * the vault's write lock and is dropped past it (store.ts guardedCacheWrite), then the handle's
 * 5 s busy_timeout is put back. Before, a write lock held by another connection froze the main
 * thread for 5 s per write: a measured run answered at 5040 ms and 9093 ms under a 2000 ms budget,
 * bare arrays with the budget count at 0. A dropped write is counted in the timing row
 * (cache_write_dropped, cache_write_stalled, cache_write_ms) and logged ("cache write dropped"). It
 * does not degrade the reply: the answer is complete and on time, and only a later recompute is
 * lost. A write that carries the request past its budget does flag query_budget_expired. The
 * cache LOOKUPS never wait on that lock (the vault is WAL: a reader is not blocked by a writer).
 *
 * What it still does not bound: every other synchronous main-thread step. Named: opening the store
 * and resolving the collection scope, the expansion and rerank cache lookups (reads; unblocked by
 * a held write lock, but still synchronous), RRF fusion, result enrichment (one SQLite read per
 * row on the main handle), composite scoring and JSON building, disposing the local models, the
 * HTTP body parse, reply deserialization from the FTS worker, and a main-thread FTS fallback after
 * a worker failure. Each cache write can still hold the thread for up to the short wait above, and
 * a purge of the cache table at its row cap (about 200 ms, one write in a thousand) runs inside
 * the guarded write's transaction. Neither this budget's timer nor the next request's arrival clock
 * can run while one is going, so a request stuck in one answers only when it returns. The other
 * REST routes, which do their own FTS on the main thread, are outside this budget altogether. The
 * main-thread stages, over the 533 live timing rows (95.5 h), peaked at 415 ms; the "50 of 200 rows
 * at 1000 ms or more" figure sometimes quoted for this change is the FTS rows that this change moved
 * to the worker, not a remainder.
 *
 * Expiry is never an error: the response carries the best results so far, flagged degraded with
 * reason `query_budget_expired`. The pipeline checks the budget around the synchronous stages and
 * skips the later ones once it is spent. Rerank's fetch is aborted at the remaining budget instead
 * of riding its 60 s bound.
 */

export const QUERY_BUDGET_REASON = "query_budget_expired" as const;

/**
 * How long past the budget a request that already started may run before the server answers
 * anyway. The pipeline normally ends on its own at the budget; this covers a stage that ignores it.
 */
export const QUERY_BUDGET_GRACE_MS = 1000;

/**
 * How long a cache write under the budget waits for the vault's write lock before it is dropped
 * (vn4rz.140). A request makes at most two such writes (expansion, rerank scores), so a held lock
 * freezes the main thread for at most about twice this per request, against 5 s per write before.
 */
export const QUERY_BUDGET_CACHE_WRITE_BUSY_MS = 50;

/** setTimeout stores its delay in 32 bits: from 2^31 ms it fires at once instead of that late. */
export const QUERY_BUDGET_MAX_MS = 2 ** 31 - 1;

/**
 * CLAWMEM_QUERY_BUDGET_MS: a positive integer up to 2^31 - 1, or 0 (off) for anything else,
 * including a value setTimeout cannot hold.
 */
export function queryBudgetMs(env: Record<string, string | undefined> = process.env): number {
  const raw = env.CLAWMEM_QUERY_BUDGET_MS?.trim();
  if (!raw || !/^\d+$/.test(raw)) return 0;
  const ms = Number.parseInt(raw, 10);
  return ms > QUERY_BUDGET_MAX_MS ? 0 : ms;
}

/** Thrown inside the pipeline when a stage is cut at the budget. Always caught by its caller. */
export class QueryBudgetExpired extends Error {
  constructor(stage: string) {
    super(`query budget expired (${stage})`);
    this.name = "QueryBudgetExpired";
  }
}

export class QueryBudget {
  readonly totalMs: number;
  private readonly deadline: number;
  private readonly now: () => number;

  /** `arrivedAt` is a `now()` reading taken when the request arrived. */
  constructor(totalMs: number, arrivedAt?: number, now: () => number = () => performance.now()) {
    this.totalMs = totalMs;
    this.now = now;
    this.deadline = (arrivedAt ?? now()) + totalMs;
  }

  /** The budget CLAWMEM_QUERY_BUDGET_MS asks for, or null when it is off. */
  static fromEnv(
    arrivedAt?: number,
    env: Record<string, string | undefined> = process.env,
  ): QueryBudget | null {
    const ms = queryBudgetMs(env);
    return ms > 0 ? new QueryBudget(ms, arrivedAt) : null;
  }

  /** Milliseconds since arrival, unclamped: how much of the budget the queue wait already took. */
  elapsedMs(): number {
    return this.now() - (this.deadline - this.totalMs);
  }

  remainingMs(): number {
    return Math.max(0, this.deadline - this.now());
  }

  expired(): boolean {
    return this.remainingMs() <= 0;
  }
}

/**
 * Run `run` until it settles or the budget ends. On expiry the signal is aborted (so a fetch is
 * cancelled, not just abandoned) and this rejects with QueryBudgetExpired.
 */
export async function withinBudget<T>(
  budget: QueryBudget,
  stage: string,
  run: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  if (budget.expired()) throw new QueryBudgetExpired(stage);
  const ctrl = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expired = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      ctrl.abort();
      reject(new QueryBudgetExpired(stage));
    }, budget.remainingMs());
  });
  const work = run(ctrl.signal);
  // The loser of the race must not surface as an unhandled rejection.
  work.catch(() => {});
  try {
    return await Promise.race([work, expired]);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Test seam, a no-op unless set; nothing in production sets it.
 *
 *   CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS     block this long just before FTS, inside the FTS worker
 *   CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MATCH  only when the query text contains this string
 *
 * Bun.sleepSync on purpose: FTS is a synchronous SQLite call, and a native sleep blocks its thread
 * the same way. Only the FTS worker calls this, and only when a budget is set; the main thread
 * never does, so a test that passes with it set proves the stall is off the main thread.
 */
export function queryBudgetTestStall(
  query: string,
  env: Record<string, string | undefined> = process.env,
): void {
  const ms = Number(env.CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS);
  if (!(ms > 0)) return;
  const match = env.CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MATCH;
  if (match && !query.includes(match)) return;
  Bun.sleepSync(ms);
}

/**
 * Test seam, a no-op unless CLAWMEM_QUERY_BUDGET_TEST_CACHE_WRITE_STALL_MS is set: block this long
 * inside each guarded cache write (store.ts guardedCacheWrite), as a write that is slow in SQLite
 * itself would. Lets a test carry a request past its budget through a cache write, deterministically.
 */
export function queryBudgetTestCacheWriteStall(
  env: Record<string, string | undefined> = process.env,
): void {
  const ms = Number(env.CLAWMEM_QUERY_BUDGET_TEST_CACHE_WRITE_STALL_MS);
  if (ms > 0) Bun.sleepSync(ms);
}

/**
 * Test seam, a no-op unless CLAWMEM_QUERY_BUDGET_TEST_DISPOSE_STALL_MS is set: hold the request in
 * its dispose stage this long, the one stage the budget cannot cut. Carries a started request
 * past budget + grace with its hits already built.
 */
export async function queryBudgetTestDisposeStall(
  env: Record<string, string | undefined> = process.env,
): Promise<void> {
  const ms = Number(env.CLAWMEM_QUERY_BUDGET_TEST_DISPOSE_STALL_MS);
  if (ms > 0) await Bun.sleep(ms);
}

/**
 * Test seam, a no-op unless CLAWMEM_QUERY_BUDGET_TEST_FTS_FAIL_MATCH is set: the FTS worker throws
 * for a query containing that string, as a worker that cannot open the vault or crashes would.
 */
export function queryBudgetTestFtsFail(
  query: string,
  env: Record<string, string | undefined> = process.env,
): void {
  const match = env.CLAWMEM_QUERY_BUDGET_TEST_FTS_FAIL_MATCH;
  if (match && query.includes(match)) throw new Error("test: FTS worker failure");
}

export interface BudgetedSubmission<T> {
  /** Settles with the pipeline's answer, or with the spent answer once the budget has run out. */
  result: Promise<T>;
  /** The new tail of the serial queue: settles when the pipeline does, however it ends. */
  tail: Promise<void>;
}

/**
 * Queue one request behind `queue` under its budget (the budget is already counting from arrival).
 * A request still queued at its budget is answered then, without waiting for those ahead of it,
 * and its pipeline never starts. A started one ends on its own at the budget; the grace timer
 * answers anyway if a stage that cannot be cut holds it `graceMs` past that. The queue advances
 * only when the pipeline settles, so a pipeline that ran past its answer still finishes before
 * the next one starts.
 *
 *   run        the pipeline, called at this request's turn when budget remains
 *   unstarted  answers a request whose pipeline never started (emits its timing row)
 *   grace      answers a started request held past budget + grace (emits its timing row)
 *   empty      the same answer without side effects, for a queued turn that was answered early
 */
export function submitWithBudget<T>(o: {
  budget: QueryBudget;
  queue: Promise<unknown>;
  run: () => Promise<T>;
  unstarted: () => Promise<T>;
  grace: () => Promise<T>;
  empty: () => T;
  graceMs?: number;
}): BudgetedSubmission<T> {
  const graceMs = o.graceMs ?? QUERY_BUDGET_GRACE_MS;
  let started = false;
  let answered = false;
  const run = o.queue.then(async (): Promise<T> => {
    // The budget ran out while this request waited its turn: its answer has gone already.
    if (answered) return o.empty();
    if (o.budget.expired()) {
      answered = true;
      return o.unstarted();
    }
    started = true;
    return o.run();
  });
  const tail = run.then(
    () => undefined,
    () => undefined,
  );
  let timer: ReturnType<typeof setTimeout> | undefined;
  const early = new Promise<T>((resolve) => {
    const fire = (answer: () => Promise<T>) => {
      if (answered) return;
      answered = true;
      void answer().then(resolve);
    };
    timer = setTimeout(() => {
      if (!started) fire(o.unstarted);
      else timer = setTimeout(() => fire(o.grace), graceMs);
    }, o.budget.remainingMs());
  });
  const settle = (v: T): T => {
    clearTimeout(timer);
    return v;
  };
  const result = Promise.race([
    run.then(settle, (e) => {
      clearTimeout(timer);
      throw e;
    }),
    early,
  ]);
  return { result, tail };
}
