/**
 * Session-id resolution for the CLI (master-harness-vn4rz.70).
 *
 * Claude Code injects CLAUDE_CODE_SESSION_ID into a session's environment; it
 * never sets CLAUDE_SESSION_ID. The only CLAUDE_SESSION_ID writer is the
 * master-harness recall hook, which copies the hook payload's session id into
 * its subprocess env. So a direct `clawmem focus` / `clawmem search` from a
 * Claude session must resolve the id from CLAUDE_CODE_SESSION_ID alone.
 *
 * Covers:
 *  - resolveEnvSessionId precedence: CLAUDE_CODE_SESSION_ID, then
 *    CLAUDE_SESSION_ID, then CLAWMEM_SESSION_ID; blanks are skipped
 *  - `clawmem focus show` (CLI, spawned) resolves from CLAUDE_CODE_SESSION_ID
 *    with CLAUDE_SESSION_ID unset, and still dies with no id at all
 *  - neither CLI call site reads the env vars directly any more
 */

import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { resolveEnvSessionId, writeSessionFocus } from "../../src/session-focus.ts";

const REPO_ROOT = resolve(import.meta.dir, "../..");
const SESSION_ENV_KEYS = ["CLAUDE_CODE_SESSION_ID", "CLAUDE_SESSION_ID", "CLAWMEM_SESSION_ID"];

let TMP_ROOT: string;

beforeEach(() => {
  TMP_ROOT = mkdtempSync(join(tmpdir(), "clawmem-sid-test-"));
});

afterEach(() => {
  rmSync(TMP_ROOT, { recursive: true, force: true });
});

/** Host env minus every session-id var, plus a hermetic focus root. */
function cliEnv(extra: Record<string, string>): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined && !SESSION_ENV_KEYS.includes(k)) env[k] = v;
  }
  delete env.CLAWMEM_SESSION_FOCUS;
  return { ...env, CLAWMEM_FOCUS_ROOT: TMP_ROOT, NO_COLOR: "1", ...extra };
}

function focusShow(extra: Record<string, string>) {
  const res = Bun.spawnSync({
    cmd: ["bun", "src/clawmem.ts", "focus", "show"],
    cwd: REPO_ROOT,
    env: cliEnv(extra),
    stdout: "pipe",
    stderr: "pipe",
  });
  return { code: res.exitCode, out: res.stdout.toString(), err: res.stderr.toString() };
}

describe("resolveEnvSessionId", () => {
  it("resolves from CLAUDE_CODE_SESSION_ID alone", () => {
    expect(resolveEnvSessionId({ CLAUDE_CODE_SESSION_ID: "cc-1" })).toBe("cc-1");
  });

  it("prefers CLAUDE_CODE_SESSION_ID over CLAUDE_SESSION_ID and CLAWMEM_SESSION_ID", () => {
    expect(
      resolveEnvSessionId({
        CLAUDE_CODE_SESSION_ID: "cc-1",
        CLAUDE_SESSION_ID: "hook-1",
        CLAWMEM_SESSION_ID: "cm-1",
      }),
    ).toBe("cc-1");
  });

  it("falls back to CLAUDE_SESSION_ID (the recall hook's contract), then CLAWMEM_SESSION_ID", () => {
    expect(resolveEnvSessionId({ CLAUDE_SESSION_ID: "hook-1", CLAWMEM_SESSION_ID: "cm-1" })).toBe("hook-1");
    expect(resolveEnvSessionId({ CLAWMEM_SESSION_ID: "cm-1" })).toBe("cm-1");
  });

  it("skips blank values and trims", () => {
    expect(resolveEnvSessionId({ CLAUDE_CODE_SESSION_ID: "  ", CLAUDE_SESSION_ID: " hook-1 " })).toBe("hook-1");
  });

  it("returns undefined when no id is set", () => {
    expect(resolveEnvSessionId({})).toBeUndefined();
    expect(resolveEnvSessionId({ CLAUDE_CODE_SESSION_ID: "" })).toBeUndefined();
  });
});

describe("clawmem focus show (CLI)", () => {
  it("resolves the session from CLAUDE_CODE_SESSION_ID with CLAUDE_SESSION_ID unset", () => {
    process.env.CLAWMEM_FOCUS_ROOT = TMP_ROOT;
    try {
      writeSessionFocus("cc-session-a", "vn4rz70 topic");
    } finally {
      delete process.env.CLAWMEM_FOCUS_ROOT;
    }
    const r = focusShow({ CLAUDE_CODE_SESSION_ID: "cc-session-a" });
    expect(r.err).not.toContain("No session id");
    expect(r.code).toBe(0);
    expect(r.out).toContain("cc-session-a");
    expect(r.out).toContain("vn4rz70 topic");
  });

  it("still dies with no session id when none of the vars is set", () => {
    const r = focusShow({});
    expect(r.code).not.toBe(0);
    expect(r.err).toContain("No session id");
    expect(r.err).toContain("CLAUDE_CODE_SESSION_ID");
  });
});

describe("call sites", () => {
  it("clawmem.ts reads session ids only through resolveEnvSessionId", () => {
    const src = readFileSync(join(REPO_ROOT, "src/clawmem.ts"), "utf8");
    expect(src).not.toMatch(/process\.env\.(CLAUDE_SESSION_ID|CLAUDE_CODE_SESSION_ID|CLAWMEM_SESSION_ID)/);
    expect(src.match(/resolveEnvSessionId\(\)/g)?.length ?? 0).toBeGreaterThanOrEqual(2);
  });
});
