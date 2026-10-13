/**
 * Opt-in per-stage timing for the hybrid `query` pipeline (master-harness-vn4rz.97).
 *
 * `CLAWMEM_QUERY_TIMING=1` (or `true`) makes the CLI `query` print one line to stderr when it
 * finishes:
 *
 *   [query-timing] {"startup_ms":612,"total_ms":3410,"stages":{"fts":{"start_ms":615,"ms":9},...}}
 *
 * `start_ms` is measured from process start (performance.timeOrigin), so overlapping stages
 * (expansion running alongside the vector legs) show their real wall-clock window, and the
 * stage durations do not have to add up to `total_ms`. Off by default: stdout and stderr stay
 * unchanged unless the variable is set.
 */

export type StageTiming = { start_ms: number; ms: number };

export type QueryTimingReport = {
  startup_ms: number;
  total_ms: number;
  stages: Record<string, StageTiming>;
  counts: Record<string, number>;
};

export function queryTimingEnabled(env: Record<string, string | undefined> = process.env): boolean {
  const raw = env.CLAWMEM_QUERY_TIMING?.trim().toLowerCase();
  return raw === "1" || raw === "true";
}

export class QueryTimer {
  readonly enabled: boolean;
  private readonly startupMs: number;
  private readonly stages: Record<string, StageTiming> = {};
  private readonly counts: Record<string, number> = {};
  private readonly now: () => number;

  constructor(enabled: boolean, now: () => number = () => performance.now()) {
    this.enabled = enabled;
    this.now = now;
    this.startupMs = now();
  }

  /** Start a stage; call the returned function when it ends. A repeated name accumulates. */
  start(name: string): () => void {
    if (!this.enabled) return () => {};
    const t0 = this.now();
    return () => {
      const ms = this.now() - t0;
      const prev = this.stages[name];
      if (prev) prev.ms += ms;
      else this.stages[name] = { start_ms: t0, ms };
    };
  }

  /** Record a count next to the timings (e.g. how many vector legs ran). */
  count(name: string, n: number): void {
    if (this.enabled) this.counts[name] = n;
  }

  /** Time an async (or sync) step under `name`, ending the stage even when it throws. */
  async time<T>(name: string, fn: () => T | Promise<T>): Promise<T> {
    const end = this.start(name);
    try {
      return await fn();
    } finally {
      end();
    }
  }

  report(): QueryTimingReport {
    const round = (n: number) => Math.round(n);
    const stages: Record<string, StageTiming> = {};
    for (const [name, s] of Object.entries(this.stages)) {
      stages[name] = { start_ms: round(s.start_ms), ms: round(s.ms) };
    }
    return { startup_ms: round(this.startupMs), total_ms: round(this.now()), stages, counts: { ...this.counts } };
  }

  /** Print the report to stderr when enabled; a no-op otherwise. */
  emit(write: (line: string) => void = line => console.error(line)): void {
    if (!this.enabled) return;
    write(`[query-timing] ${JSON.stringify(this.report())}`);
  }
}
