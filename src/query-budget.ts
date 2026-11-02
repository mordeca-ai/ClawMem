/**
 * One whole-request budget for `clawmem serve` POST /query (master-harness-vn4rz.134).
 *
 * CLAWMEM_QUERY_BUDGET_MS is measured from the request's ARRIVAL, before the serial queue wait, and
 * covers queue wait, FTS, the vector stage and rerank. It extends CLAWMEM_QUERY_VEC_DEADLINE_MS
 * (vn4rz.131), which starts only when the vector stage does: behind the queue, four concurrent
 * POSTs at a 2000 ms deadline returned at 2055, 4073, 4084 and 4092 ms. Unset or 0 is off, and
 * every code path then runs exactly as it did before.
 *
 * Expiry is never an error: the response carries the best results so far, flagged degraded with
 * reason `query_budget_expired`. FTS and the in-process stages are synchronous and cannot be
 * preempted, so the pipeline checks the budget around them and skips the later stages once it is
 * spent. Rerank's fetch is aborted at the remaining budget instead of riding its 60 s bound.
 */

export const QUERY_BUDGET_REASON = "query_budget_expired" as const;

/**
 * How long past the budget a request that already started may run before the server answers
 * anyway. The pipeline normally ends on its own at the budget; this covers a stage that ignores it.
 */
export const QUERY_BUDGET_GRACE_MS = 1000;

/** CLAWMEM_QUERY_BUDGET_MS: a positive integer, or 0 (off) for anything else. */
export function queryBudgetMs(env: Record<string, string | undefined> = process.env): number {
  const raw = env.CLAWMEM_QUERY_BUDGET_MS?.trim();
  if (!raw || !/^\d+$/.test(raw)) return 0;
  return Number.parseInt(raw, 10);
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
