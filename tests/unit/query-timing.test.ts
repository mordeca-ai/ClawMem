import { describe, expect, it } from "bun:test";
import { QueryTimer, queryTimingEnabled } from "../../src/query-timing.ts";

/**
 * master-harness-vn4rz.97 — CLAWMEM_QUERY_TIMING=1 prints a per-stage breakdown of the CLI
 * hybrid `query` to stderr. Off by default, so the timer must be silent unless enabled.
 */

function fakeClock(ticks: number[]): () => number {
  let i = 0;
  return () => ticks[Math.min(i++, ticks.length - 1)]!;
}

describe("queryTimingEnabled", () => {
  it("is off unless the variable is 1 or true", () => {
    for (const v of [undefined, "", "0", "false", "yes", "2"]) {
      expect(queryTimingEnabled({ CLAWMEM_QUERY_TIMING: v })).toBe(false);
    }
    for (const v of ["1", "true", " TRUE "]) {
      expect(queryTimingEnabled({ CLAWMEM_QUERY_TIMING: v })).toBe(true);
    }
  });
});

describe("QueryTimer", () => {
  it("records start offset and duration per stage, accumulating repeated names", () => {
    // ctor=100, a.start=110, a.end=150, b.start=150, b.end=170, b.start=200, b.end=205, report=300
    const t = new QueryTimer(true, fakeClock([100, 110, 150, 150, 170, 200, 205, 300]));
    t.start("a")();
    t.start("b")();
    t.start("b")();
    t.count("legs", 3);
    expect(t.report()).toEqual({
      startup_ms: 100,
      total_ms: 300,
      stages: { a: { start_ms: 110, ms: 40 }, b: { start_ms: 150, ms: 25 } },
      counts: { legs: 3 },
    });
  });

  it("time() ends the stage even when the step throws", async () => {
    const t = new QueryTimer(true, fakeClock([0, 10, 30, 40]));
    await expect(t.time("boom", () => { throw new Error("x"); })).rejects.toThrow("x");
    expect(t.report().stages.boom).toEqual({ start_ms: 10, ms: 20 });
  });

  it("emits one [query-timing] JSON line when enabled and nothing when disabled", () => {
    const lines: string[] = [];
    const on = new QueryTimer(true, fakeClock([5, 6, 9, 20]));
    on.start("fts")();
    on.emit(l => lines.push(l));
    expect(lines).toHaveLength(1);
    expect(lines[0]!.startsWith("[query-timing] ")).toBe(true);
    expect(JSON.parse(lines[0]!.slice("[query-timing] ".length)).stages.fts).toEqual({ start_ms: 6, ms: 3 });

    const off = new QueryTimer(false);
    off.start("fts")();
    off.count("legs", 1);
    off.emit(l => lines.push(l));
    expect(lines).toHaveLength(1);
    expect(off.report().stages).toEqual({});
  });
});
