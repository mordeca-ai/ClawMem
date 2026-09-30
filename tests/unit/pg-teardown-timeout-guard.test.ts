/**
 * master-harness-83qk0.120: every afterAll hook that runs DROP DATABASE must pass an
 * explicit timeout.
 *
 * DROP DATABASE forces an immediate checkpoint on the shared PG server, so under the
 * nightly full sweep it took 9.6s against bun's 5s default hook timeout. The hook
 * timed out, the suite went RED, and the second throwaway vault database leaked. This
 * guard scans the PG integration suites so a new or edited teardown cannot drop the
 * timeout argument again.
 */

import { describe, it, expect } from "bun:test";
import { readFileSync, readdirSync } from "fs";
import { join } from "path";

const INTEGRATION_DIR = join(import.meta.dir, "..", "integration");
const MIN_TEARDOWN_TIMEOUT_MS = 60_000;

/** Return the full text of each `afterAll(...)` call, parens matched. */
export function afterAllCalls(src: string): string[] {
  const calls: string[] = [];
  let from = 0;
  for (;;) {
    const start = src.indexOf("afterAll(", from);
    if (start < 0) return calls;
    let depth = 0;
    let end = -1;
    for (let i = start + "afterAll".length; i < src.length; i++) {
      const ch = src[i];
      if (ch === "(") depth++;
      else if (ch === ")") {
        depth--;
        if (depth === 0) { end = i; break; }
      }
    }
    if (end < 0) throw new Error(`unbalanced afterAll( at offset ${start}`);
    calls.push(src.slice(start, end + 1));
    from = end + 1;
  }
}

/** The timeout a hook call passes after its callback, or null when it passes none. */
export function hookTimeoutMs(call: string, constants: Record<string, number>): number | null {
  const m = call.match(/\}\s*,\s*([A-Za-z_][A-Za-z0-9_]*|[0-9_]+)\s*\)$/);
  if (!m) return null;
  const token = m[1];
  if (/^[0-9_]+$/.test(token)) return Number(token.replace(/_/g, ""));
  return constants[token] ?? null;
}

function exportedConstants(): Record<string, number> {
  const src = readFileSync(join(INTEGRATION_DIR, "pg-test-schema.ts"), "utf-8");
  const out: Record<string, number> = {};
  for (const m of src.matchAll(/export const ([A-Z_]+) = ([0-9_]+);/g)) {
    out[m[1]] = Number(m[2].replace(/_/g, ""));
  }
  return out;
}

describe("PG teardown hooks that DROP DATABASE carry a timeout", () => {
  const constants = exportedConstants();

  it("the shared teardown timeout is well above bun's 5s default", () => {
    expect(constants.PG_TEST_TEARDOWN_TIMEOUT_MS).toBeGreaterThanOrEqual(MIN_TEARDOWN_TIMEOUT_MS);
  });

  it("the scanner flags a DROP DATABASE teardown with no timeout", () => {
    const bare = "afterAll(async () => {\n  await admin.query(`DROP DATABASE IF EXISTS ${db}`);\n});";
    const [call] = afterAllCalls(bare);
    expect(call).toContain("DROP DATABASE");
    expect(hookTimeoutMs(call, constants)).toBeNull();
    const timed = bare.replace(/\}\);$/, "}, PG_TEST_TEARDOWN_TIMEOUT_MS);");
    expect(hookTimeoutMs(afterAllCalls(timed)[0], constants)).toBe(constants.PG_TEST_TEARDOWN_TIMEOUT_MS);
  });

  const files = readdirSync(INTEGRATION_DIR).filter((f) => f.endsWith(".ts"));
  const dropping = files.flatMap((f) =>
    afterAllCalls(readFileSync(join(INTEGRATION_DIR, f), "utf-8"))
      .filter((c) => c.includes("DROP DATABASE"))
      .map((c) => ({ file: f, call: c })),
  );

  it("finds the DROP DATABASE teardowns (the scan is not vacuous)", () => {
    expect(dropping.length).toBeGreaterThanOrEqual(2);
  });

  for (const { file, call } of dropping) {
    it(`${file}: DROP DATABASE teardown passes a timeout >= ${MIN_TEARDOWN_TIMEOUT_MS}ms`, () => {
      const ms = hookTimeoutMs(call, constants);
      expect(ms).not.toBeNull();
      expect(ms!).toBeGreaterThanOrEqual(MIN_TEARDOWN_TIMEOUT_MS);
    });
  }
});
