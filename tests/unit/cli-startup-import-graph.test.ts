import { describe, it, expect, beforeAll, afterAll } from "bun:test";

/**
 * vn4rz.116 — CLI startup floor. Every `clawmem` invocation used to statically import the whole
 * tool (eval harness, indexer, consolidation, hooks, judge, MCP ...), costing ~0.4s of module
 * load before the verb ran; spawn-per-call consumers of vsearch/search paid it every time.
 * Heavy modules are now `await import()`ed inside the verbs that use them.
 *
 * This test records which src modules a process actually loads (tests/helpers/module-load-recorder.ts,
 * a Bun preload) and asserts the read-only verbs do NOT load the heavy ones. Deterministic: it
 * counts modules, never wall-clock time.
 */

import { mkdtempSync, rmSync, readFileSync, existsSync } from "fs";
import { join, relative } from "path";
import { tmpdir } from "os";

const REPO_ROOT = new URL("../..", import.meta.url).pathname;
const SRC_ROOT = join(REPO_ROOT, "src");
const PRELOAD = join(REPO_ROOT, "tests/helpers/module-load-recorder.ts");

/** src-relative paths that read-only verbs must never load. */
const HEAVY = [
  "eval/run.ts", "eval/gold.ts", "indexer.ts", "profile.ts", "directory-context.ts",
  "consolidation.ts", "maintenance.ts", "vector-daemon.ts", "canary.ts", "taint-clear.ts",
  "judge.ts", "judge-audit.ts", "merge-guards.ts", "backup.ts", "mcp.ts", "watcher.ts",
  "hooks/context-surfacing.ts", "hooks/session-bootstrap.ts", "hooks/decision-extractor.ts",
  "hooks/handoff-generator.ts", "hooks/feedback-loop.ts", "hooks/staleness-check.ts",
  "hooks/precompact-extract.ts", "hooks/postcompact-inject.ts", "hooks/pretool-inject.ts",
  "hooks/curator-nudge.ts",
];

let root: string;
let env: Record<string, string | undefined>;

function loaded(args: string[]): Set<string> {
  const rec = join(root, `rec-${Math.random().toString(36).slice(2)}.txt`);
  Bun.spawnSync({
    cmd: ["bun", "--preload", PRELOAD, "src/clawmem.ts", ...args],
    cwd: REPO_ROOT,
    env: { ...env, CLAWMEM_LOAD_RECORD: rec },
    stdout: "pipe",
    stderr: "pipe",
    timeout: 60_000,
  });
  expect(existsSync(rec)).toBe(true);
  return new Set(readFileSync(rec, "utf8").split("\n").filter(Boolean).map(p => relative(SRC_ROOT, p)));
}

beforeAll(() => {
  root = mkdtempSync(join(tmpdir(), "clawmem-import-graph-"));
  env = {
    ...Bun.env,
    INDEX_PATH: join(root, "test.sqlite"),
    CLAWMEM_NO_LOCAL_MODELS: "true",
    CLAWMEM_EMBED_URL: "http://127.0.0.1:1",
    CLAWMEM_LLM_URL: "http://127.0.0.1:1",
    CLAWMEM_RERANK_URL: "http://127.0.0.1:1",
    XDG_CONFIG_HOME: join(root, "config"),
  };
  // `status` creates the (empty) index so the read-only verbs below have one to open.
  Bun.spawnSync({ cmd: ["bun", "src/clawmem.ts", "status"], cwd: REPO_ROOT, env, stdout: "pipe", stderr: "pipe", timeout: 60_000 });
});

afterAll(() => rmSync(root, { recursive: true, force: true }));

describe("CLI import graph stays light for read-only verbs", () => {
  it("the recorder sees the entry module and store (control: it is actually recording)", () => {
    const mods = loaded(["--version"]);
    expect(mods.has("clawmem.ts")).toBe(true);
    expect(mods.has("store.ts")).toBe(true);
  });

  for (const [name, args] of [
    ["--version", ["--version"]],
    ["status", ["status"]],
    ["search", ["search", "kitchen", "timer", "-n", "3", "--json"]],
    ["vsearch", ["vsearch", "kitchen timer", "-n", "3", "--json"]],
    ["get", ["get", "nonexistent.md"]],
  ] as const) {
    it(`${name} does not load heavy modules`, () => {
      const mods = loaded([...args]);
      const offenders = HEAVY.filter(h => mods.has(h));
      expect(offenders).toEqual([]);
    });
  }

  it("lazy modules still load on the verb that needs them (eval pulls eval/run.ts)", () => {
    // `eval` with no args dies at usage; its lazy import runs first, so the module must appear.
    const mods = loaded(["eval"]);
    expect(mods.has("eval/run.ts")).toBe(true);
  });
});
