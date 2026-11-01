/**
 * Hook-level tests seed documents and assert they surface. Composite recency decays
 * from modified_at against the real clock, so a seed pinned to a literal date is a
 * time bomb: the score slides under the activation floor months later and the test
 * fails with no code change, which is how two suites went red on 2026-11-01
 * (master-harness-83qk0.309). Seed relative to Date.now(), computed once per file.
 */
import { describe, expect, it } from "bun:test";
import { Glob } from "bun";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..");
const HOOK_CALL = /\bcontextSurfacing\s*\(/;
const LITERAL_SEED = /^.*(?:const|let)\s+\w+\s*=\s*["'`]20\d\d-\d\d-\d\dT[^"'`]*["'`].*$/gm;

function pinnedSeeds(source: string): string[] {
  return HOOK_CALL.test(source) ? [...source.matchAll(LITERAL_SEED)].map((m) => m[0].trim()) : [];
}

describe("hook-level test seeds are not pinned to a calendar date", () => {
  it("detects a pinned seed in a hook-level test, and only there", () => {
    const pinned = 'const SEED_TS = "2026-01-01T00:00:00.000Z";\n';
    expect(pinnedSeeds(`${pinned}await contextSurfacing(store, input);`)).toHaveLength(1);
    expect(pinnedSeeds(pinned)).toHaveLength(0);
    expect(
      pinnedSeeds("const SEED_TS = new Date(Date.now() - 86400000).toISOString();\ncontextSurfacing(s, i);"),
    ).toHaveLength(0);
  });

  it("no test that calls contextSurfacing pins a literal timestamp", async () => {
    const offenders: string[] = [];
    let scanned = 0;
    for await (const rel of new Glob("**/*.test.ts").scan(ROOT)) {
      if (rel.endsWith("hook-seed-date-guard.test.ts")) continue;
      const source = await Bun.file(join(ROOT, rel)).text();
      if (HOOK_CALL.test(source)) scanned += 1;
      for (const line of pinnedSeeds(source)) offenders.push(`${rel}: ${line}`);
    }
    expect(scanned).toBeGreaterThan(1);
    expect(offenders).toEqual([]);
  });
});
