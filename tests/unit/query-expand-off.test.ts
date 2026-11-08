/**
 * master-harness-vn4rz.125 — CLAWMEM_QUERY_EXPAND=off is an explicit expansion off switch.
 *
 * Measured defect: callers turned expansion off by unsetting CLAWMEM_LLM_URL, but bin/clawmem
 * re-defaulted it to http://localhost:8089, so every recall POSTed its prompt there (HTTP 404)
 * and ran the fallback expansion. Two fixes, tested here:
 *   1. the wrapper leaves an unset LLM URL UNSET when expansion is off;
 *   2. store.expandQuery makes no LLM call and returns no expansion legs when it is off.
 */
import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setDefaultLlamaCpp, type Queryable } from "../../src/llm.ts";
import {
  createStore,
  expandQueryCacheKey,
  setCachedResult,
  DEFAULT_QUERY_MODEL,
  type Store,
} from "../../src/store.ts";

const WRAPPER = resolve(import.meta.dir, "../../bin/clawmem");

/** Run bin/clawmem against a fake `bun` that prints its environment instead of clawmem. */
function wrapperEnv(extra: Record<string, string>, unset: string[] = []): Record<string, string> {
  const dir = mkdtempSync(join(tmpdir(), "vn4rz125-"));
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

describe("bin/clawmem LLM URL default under CLAWMEM_QUERY_EXPAND", () => {
  const ALWAYS_UNSET = ["CLAWMEM_LLM_URL", "CLAWMEM_LLM_URL_SOURCE"];

  test("off + unset URL => URL stays UNSET and carries no stock-default marker", () => {
    const env = wrapperEnv({ CLAWMEM_QUERY_EXPAND: "off" }, ALWAYS_UNSET);
    expect(env.CLAWMEM_LLM_URL).toBeUndefined();
    expect(env.CLAWMEM_LLM_URL_SOURCE).toBeUndefined();
  });

  test("off + empty URL => no default is substituted", () => {
    const env = wrapperEnv({ CLAWMEM_QUERY_EXPAND: "off", CLAWMEM_LLM_URL: "" }, ["CLAWMEM_LLM_URL_SOURCE"]);
    expect(env.CLAWMEM_LLM_URL ?? "").toBe("");
  });

  test("off + explicit URL => the URL wins (other LLM features keep their endpoint)", () => {
    const env = wrapperEnv({ CLAWMEM_QUERY_EXPAND: "off", CLAWMEM_LLM_URL: "http://gpu:8093" });
    expect(env.CLAWMEM_LLM_URL).toBe("http://gpu:8093");
  });

  test("switch absent + unset URL => stock default and marker preserved", () => {
    const env = wrapperEnv({}, [...ALWAYS_UNSET, "CLAWMEM_QUERY_EXPAND"]);
    expect(env.CLAWMEM_LLM_URL).toBe("http://localhost:8089");
    expect(env.CLAWMEM_LLM_URL_SOURCE).toBe("default");
  });
});

describe("store.expandQuery under CLAWMEM_QUERY_EXPAND", () => {
  const QUERY = "integrator lease guard";
  const LIVE: Queryable[] = [{ type: "vec", text: "integrator lease guard expiry handling" }];
  const original = process.env.CLAWMEM_QUERY_EXPAND;
  let store: Store;
  let calls = 0;

  beforeEach(() => {
    calls = 0;
    setDefaultLlamaCpp({
      expandQuery: async () => {
        calls++;
        return LIVE;
      },
    } as never);
    store = createStore(":memory:");
  });

  afterEach(() => {
    store.close();
    setDefaultLlamaCpp(null);
    if (original === undefined) delete process.env.CLAWMEM_QUERY_EXPAND;
    else process.env.CLAWMEM_QUERY_EXPAND = original;
  });

  test("off => no LLM call and no expansion legs", async () => {
    process.env.CLAWMEM_QUERY_EXPAND = "off";
    expect(await store.expandQuery(QUERY)).toEqual([]);
    expect(calls).toBe(0);
  });

  test("off => a previously cached expansion is not served", async () => {
    setCachedResult(
      store.db,
      expandQueryCacheKey(QUERY, DEFAULT_QUERY_MODEL),
      JSON.stringify([{ type: "vec", query: "integrator lease guard expiry handling" }]),
    );
    process.env.CLAWMEM_QUERY_EXPAND = "off";
    expect(await store.expandQuery(QUERY)).toEqual([]);
    expect(calls).toBe(0);
  });

  test.each([[undefined], [""], ["on"]])("value %p => expansion still runs (positive control)", async (value) => {
    if (value === undefined) delete process.env.CLAWMEM_QUERY_EXPAND;
    else process.env.CLAWMEM_QUERY_EXPAND = value;
    expect(await store.expandQuery(QUERY)).toEqual([
      { type: "vec", query: "integrator lease guard expiry handling" },
    ]);
    expect(calls).toBe(1);
  });
});
