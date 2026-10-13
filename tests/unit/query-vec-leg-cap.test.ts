import { describe, expect, it } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, expandQueryCacheKey } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { capVectorExpansions, queryVecExpansionLegCap } from "../../src/search-utils.ts";
import { fakeEmbedVector, startFakeEmbedServer } from "../helpers/fake-embed-server.ts";

/**
 * master-harness-vn4rz.92 — CLAWMEM_QUERY_VEC_EXPANSION_LEGS caps the expansion vector legs
 * (vec/hyde) a hybrid query runs. Each vector leg pays a whole-table vec0 flat scan, so the
 * cap is the lever on hybrid query latency. Covered here:
 *   1. env parsing (unset/garbage = no cap)
 *   2. which legs survive (hyde first, then vec; lex untouched; order kept)
 *   3. CLI wiring: the embed endpoint sees exactly the surviving legs' texts
 */

const KNOB = "CLAWMEM_QUERY_VEC_EXPANSION_LEGS";
const CLI = new URL("../../src/clawmem.ts", import.meta.url).pathname;

type Leg = { type: "lex" | "vec" | "hyde"; query: string };
const LEGS: Leg[] = [
  { type: "lex", query: "lex one" },
  { type: "vec", query: "vec one" },
  { type: "vec", query: "vec two" },
  { type: "hyde", query: "hyde passage" },
  { type: "lex", query: "lex two" },
];

function withKnob<T>(value: string | undefined, fn: () => T): T {
  const saved = process.env[KNOB];
  if (value === undefined) delete process.env[KNOB];
  else process.env[KNOB] = value;
  try {
    return fn();
  } finally {
    if (saved === undefined) delete process.env[KNOB];
    else process.env[KNOB] = saved;
  }
}

describe("queryVecExpansionLegCap — env parsing", () => {
  it("no cap when unset, empty, negative, fractional or non-numeric", () => {
    for (const v of [undefined, "", "  ", "-1", "1.5", "two", "1x"]) {
      expect(withKnob(v, () => queryVecExpansionLegCap())).toBeUndefined();
    }
  });

  it("parses non-negative integers, including 0", () => {
    expect(withKnob("0", () => queryVecExpansionLegCap())).toBe(0);
    expect(withKnob("1", () => queryVecExpansionLegCap())).toBe(1);
    expect(withKnob(" 3 ", () => queryVecExpansionLegCap())).toBe(3);
  });
});

describe("capVectorExpansions — which legs survive", () => {
  it("undefined cap returns the list unchanged (default behaviour)", () => {
    expect(capVectorExpansions(LEGS, undefined)).toEqual(LEGS);
  });

  it("cap 1 keeps the hyde leg and every lex leg, in original order", () => {
    expect(capVectorExpansions(LEGS, 1).map((l) => l.query)).toEqual([
      "lex one",
      "hyde passage",
      "lex two",
    ]);
  });

  it("cap 2 keeps hyde, then the first vec leg", () => {
    expect(capVectorExpansions(LEGS, 2).map((l) => l.query)).toEqual([
      "lex one",
      "vec one",
      "hyde passage",
      "lex two",
    ]);
  });

  it("cap 0 drops every vector expansion and keeps lex", () => {
    expect(capVectorExpansions(LEGS, 0).map((l) => l.query)).toEqual(["lex one", "lex two"]);
  });

  it("without a hyde leg, the first vec legs survive", () => {
    const noHyde = LEGS.filter((l) => l.type !== "hyde");
    expect(capVectorExpansions(noHyde, 1).map((l) => l.query)).toEqual([
      "lex one",
      "vec one",
      "lex two",
    ]);
  });

  it("a cap at or above the vector-leg count is a no-op", () => {
    expect(capVectorExpansions(LEGS, 3)).toEqual(LEGS);
    expect(capVectorExpansions(LEGS, 9)).toEqual(LEGS);
  });
});

describe("CLI query — the cap reaches the vector legs (subprocess)", () => {
  it("the embed endpoint sees only the surviving expansion legs", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-vec-leg-cap-"));
    const docs = join(dir, "docs");
    mkdirSync(docs);
    writeFileSync(
      join(dir, "config.yaml"),
      `collections:\n  testcol:\n    path: ${docs}\n    pattern: "**/*.md"\n`,
    );

    const query = "orbital ferry schedule";
    const legs: Leg[] = [
      { type: "lex", query: "orbital ferry timetable" },
      { type: "vec", query: "orbital ferry departure times" },
      { type: "vec", query: "ferry schedule for orbital transfers" },
      {
        type: "hyde",
        query: "The orbital ferry departs the station every six hours on a fixed schedule.",
      },
    ];
    const db = join(dir, "index.sqlite");
    const store = createStore(db);
    try {
      const now = new Date().toISOString();
      const body = "# Orbital ferry schedule\n\nThe ferry departs every six hours.\n";
      const hash = hashContent(body + "ferry.md");
      store.insertContent(hash, body, now);
      store.insertDocument("testcol", "ferry.md", "Orbital ferry schedule", hash, now, now);
      store.ensureVecTable(512);
      store.insertEmbedding(
        hash,
        0,
        1,
        new Float32Array(fakeEmbedVector(body, 512)),
        "fake-embed-a",
        now,
        "full",
        undefined,
        "testcol/ferry.md",
      );
      // Pre-seed the expansion cache so the run is deterministic and needs no LLM.
      store.db
        .prepare(`INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)`)
        .run(expandQueryCacheKey(query), JSON.stringify(legs), now);
    } finally {
      store.db.close();
    }

    const server = startFakeEmbedServer({ dim: 512, model: "fake-embed-a" });
    async function run(cap?: string): Promise<{ embedded: string[]; stderr: string }> {
      const env: Record<string, string> = {};
      for (const [key, value] of Object.entries(process.env)) {
        if (value !== undefined && !key.startsWith("CLAWMEM_") && key !== "INDEX_PATH")
          env[key] = value;
      }
      Object.assign(env, {
        INDEX_PATH: db,
        CLAWMEM_CONFIG_DIR: dir,
        CLAWMEM_EMBED_URL: server.url,
        CLAWMEM_LLM_URL: server.url,
        CLAWMEM_RERANK_URL: server.url,
        CLAWMEM_NO_LOCAL_MODELS: "true",
        CLAWMEM_DISABLE_FTS_BYPASS: "true",
      });
      if (cap !== undefined) env[KNOB] = cap;
      const start = server.seen.length;
      const proc = Bun.spawn([process.execPath, CLI, "query", query, "--json"], {
        env,
        stdout: "pipe",
        stderr: "pipe",
      });
      const [, stderr, code] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
      ]);
      expect(code, `query failed: ${stderr}`).toBe(0);
      return { embedded: server.seen.slice(start), stderr };
    }
    const sawLeg = (embedded: string[], text: string) => embedded.some((e) => e.includes(text));
    const vecA = legs[1]!.query;
    const vecB = legs[2]!.query;
    const hyde = legs[3]!.query;

    try {
      const uncapped = await run();
      expect(sawLeg(uncapped.embedded, vecA)).toBe(true);
      expect(sawLeg(uncapped.embedded, vecB)).toBe(true);
      expect(sawLeg(uncapped.embedded, hyde)).toBe(true);
      expect(uncapped.stderr).not.toContain("vec-leg cap");

      const one = await run("1");
      expect(sawLeg(one.embedded, hyde)).toBe(true);
      expect(sawLeg(one.embedded, vecA)).toBe(false);
      expect(sawLeg(one.embedded, vecB)).toBe(false);
      expect(one.stderr).toContain("vec-leg cap 1: running 1/3 expansion vector legs");

      const zero = await run("0");
      expect(sawLeg(zero.embedded, hyde)).toBe(false);
      expect(sawLeg(zero.embedded, vecA)).toBe(false);
      expect(sawLeg(zero.embedded, vecB)).toBe(false);
      // The original-query leg always runs.
      expect(sawLeg(zero.embedded, query)).toBe(true);
    } finally {
      server.stop();
      rmSync(dir, { recursive: true, force: true });
    }
  }, 120_000);
});
