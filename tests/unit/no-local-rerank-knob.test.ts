/**
 * master-harness-zz3nx — CLAWMEM_NO_LOCAL_MODELS=true must cover the rerank leg.
 *
 * Measured root cause: with the knob set and CLAWMEM_RERANK_URL unset, bin/clawmem
 * injected its stock default http://localhost:8090. On a host with no server there
 * the rerank fetch hung for the full 60s REMOTE_RERANK_FETCH_TIMEOUT_MS before the
 * local fallback threw (65s vs 4s with a real endpoint). Two fixes, one test each:
 *   1. the wrapper leaves an unset rerank URL UNSET under the knob;
 *   2. LlamaCpp.rerank fails fast under the knob, before any node-llama-cpp import.
 */
import { describe, test, expect, afterEach } from "bun:test";
import { mkdtempSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { LlamaCpp } from "../../src/llm.ts";

const WRAPPER = resolve(import.meta.dir, "../../bin/clawmem");

/** Run bin/clawmem against a fake `bun` that prints its environment instead of clawmem. */
function wrapperEnv(extra: Record<string, string>, unset: string[] = []): Record<string, string> {
  const dir = mkdtempSync(join(tmpdir(), "zz3nx-"));
  const fake = join(dir, "bun");
  writeFileSync(fake, '#!/usr/bin/env bash\nif [ "$1" = --version ]; then echo 1.2.0; exit 0; fi\nenv\n');
  chmodSync(fake, 0o755);
  const env: Record<string, string> = { ...(process.env as Record<string, string>), ...extra };
  for (const k of unset) delete env[k];
  env.PATH = `${dir}:${env.PATH}`;
  const proc = Bun.spawnSync([WRAPPER, "status"], { env });
  expect(proc.exitCode).toBe(0);
  const out: Record<string, string> = {};
  for (const line of proc.stdout.toString().split("\n")) {
    const i = line.indexOf("=");
    if (i > 0) out[line.slice(0, i)] = line.slice(i + 1);
  }
  return out;
}

describe("bin/clawmem rerank URL default under CLAWMEM_NO_LOCAL_MODELS", () => {
  test("knob=true + unset URL => URL stays UNSET (no phantom localhost:8090)", () => {
    const env = wrapperEnv({ CLAWMEM_NO_LOCAL_MODELS: "true" }, ["CLAWMEM_RERANK_URL"]);
    expect(env.CLAWMEM_NO_LOCAL_MODELS).toBe("true");
    expect(env.CLAWMEM_RERANK_URL).toBeUndefined();
  });

  test("knob unset + unset URL => stock default preserved (local-first installs unchanged)", () => {
    const env = wrapperEnv({}, ["CLAWMEM_RERANK_URL", "CLAWMEM_NO_LOCAL_MODELS"]);
    expect(env.CLAWMEM_RERANK_URL).toBe("http://localhost:8090");
    expect(env.CLAWMEM_NO_LOCAL_MODELS).toBe("false");
  });

  test("an explicit URL always wins, knob or not", () => {
    const env = wrapperEnv({ CLAWMEM_NO_LOCAL_MODELS: "true", CLAWMEM_RERANK_URL: "http://gpu:8091" });
    expect(env.CLAWMEM_RERANK_URL).toBe("http://gpu:8091");
  });
});

describe("rerank leg fails fast under CLAWMEM_NO_LOCAL_MODELS=true", () => {
  const originalNoLocal = process.env.CLAWMEM_NO_LOCAL_MODELS;
  afterEach(() => {
    if (originalNoLocal === undefined) delete process.env.CLAWMEM_NO_LOCAL_MODELS;
    else process.env.CLAWMEM_NO_LOCAL_MODELS = originalNoLocal;
  });

  test("LlamaCpp.rerank rejects with the knob named, without loading a context", async () => {
    process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
    const llm = new LlamaCpp();
    let loaded = false;
    (llm as unknown as { ensureRerankContext: () => Promise<never> }).ensureRerankContext = async () => {
      loaded = true;
      throw new Error("must not be reached");
    };
    await expect(llm.rerank("q", [{ file: "a", text: "alpha" }])).rejects.toThrow(/CLAWMEM_NO_LOCAL_MODELS=true/);
    expect(loaded).toBe(false);
  });
});
