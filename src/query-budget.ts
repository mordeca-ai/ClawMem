/**
 * One whole-request budget for `clawmem serve` POST /query (master-harness-vn4rz.134).
 *
 * CLAWMEM_QUERY_BUDGET_MS is measured from the request's ARRIVAL, before the serial queue wait. It
 * bounds the queue wait, the vector stage and rerank. It extends CLAWMEM_QUERY_VEC_DEADLINE_MS
 * (vn4rz.131), which starts only when the vector stage does: behind the queue, four concurrent
 * POSTs at a 2000 ms deadline returned at 2055, 4073, 4084 and 4092 ms. Unset or 0 is off, and
 * every code path then runs exactly as it did before.
 *
 * What it does not bound: FTS and every other main-thread step. They are synchronous, so neither
 * this budget's timer nor the next request's arrival clock can run while one is going. A request
 * whose FTS runs past the budget answers only when FTS returns (degraded, FTS hits, vector stage
 * skipped). A request that arrives during such a stall is stamped when the server gets to it,
 * after the stall, so its budget starts late. That overrun is not in its own counts and so not in
 * the degraded rate that clawmem-recall-health reads. Moving FTS off the main thread is separate
 * work.
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
 *   CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MS     block the main thread this long just before FTS
 *   CLAWMEM_QUERY_BUDGET_TEST_FTS_STALL_MATCH  only when the query text contains this string
 *
 * Bun.sleepSync on purpose: FTS is a synchronous SQLite call, and a native sleep stalls the event
 * loop the same way, so neither the budget's timers nor a request's arrival clock can run.
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
 *   unstarted  answers a request the pipeline did not answer in time (emits its timing row)
 *   empty      the same answer without side effects, for a queued turn that was answered early
 */
export function submitWithBudget<T>(o: {
  budget: QueryBudget;
  queue: Promise<unknown>;
  run: () => Promise<T>;
  unstarted: () => Promise<T>;
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
    const fire = () => {
      if (answered) return;
      answered = true;
      void o.unstarted().then(resolve);
    };
    timer = setTimeout(() => {
      if (!started) fire();
      else timer = setTimeout(fire, graceMs);
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
