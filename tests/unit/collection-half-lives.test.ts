/**
 * master-harness-vn4rz.94: per-collection half-life override (`half_lives`).
 *
 * memory-topics needs its own decay balance (r65): re-typing feedback to
 * `preference` (global half-life Infinity) crowds still-decaying project /
 * reference docs out of the top 10. The override lets ONE collection change
 * its half-lives while every other collection keeps the global HALF_LIVES.
 *
 *   H1  the override applies only inside the configured collection,
 *   H2  an absent override is the pre-change behaviour, exactly,
 *   H3  invalid config is refused at load (unknown type, non-positive / NaN days),
 *   H4  the override reaches every scoring input: query-time recency, query-time
 *       confidence, and the stored confidence the indexer writes.
 */
import { describe, it, expect, beforeEach, afterEach, afterAll } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, rmSync, utimesSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  HALF_LIVES,
  NO_DECAY_TOKEN,
  applyCompositeScoring,
  baseHalfLife,
  confidenceScore,
  recencyScore,
  validateHalfLives,
  type EnrichedResult,
  type HalfLifeTable,
} from "../../src/memory.ts";
import {
  addCollection,
  clearConfigCache,
  collectionIndexOptions,
  getCollectionHalfLives,
  loadConfig,
} from "../../src/collections.ts";
import { indexCollection } from "../../src/indexer.ts";
import { createStore, type Store } from "../../src/store.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";

const DAY = 86_400_000;
const NOW = new Date("2026-10-13T00:00:00Z");
const daysAgo = (n: number) => new Date(NOW.getTime() - n * DAY);

function mkDoc(overrides: Partial<EnrichedResult> = {}): EnrichedResult {
  return {
    filepath: "clawmem://topics/a.md",
    displayPath: "topics/a.md",
    title: "A",
    score: 0.6,
    contentType: "project",
    modifiedAt: daysAgo(200).toISOString(),
    accessCount: 0,
    confidence: 0.5,
    qualityScore: 0.5,
    pinned: false,
    context: null,
    hash: "abc123",
    docid: "abc123",
    collectionName: "topics",
    bodyLength: 500,
    source: "vec",
    duplicateCount: 1,
    revisionCount: 1,
    ...overrides,
  };
}

const NO_DECAY: HalfLifeTable = { project: Infinity, note: Infinity, preference: Infinity };
const onlyTopics = (table: HalfLifeTable) => (c: string) => (c === "topics" ? table : undefined);
const none = () => undefined;

describe("H3: validateHalfLives refuses invalid config", () => {
  const where = 'collection "memory-topics"';

  it("accepts positive days, .inf and the no-decay token", () => {
    expect(
      validateHalfLives({ preference: 365, note: Infinity, project: NO_DECAY_TOKEN }, where),
    ).toEqual({
      preference: 365,
      note: Infinity,
      project: Infinity,
    });
    expect(validateHalfLives({}, where)).toEqual({});
  });

  it("refuses an unknown content_type, naming the collection and the known set", () => {
    expect(() => validateHalfLives({ feedback: 365 }, where)).toThrow(
      /collection "memory-topics": half_lives has unknown content_type "feedback"/,
    );
    expect(() => validateHalfLives({ feedback: 365 }, where)).toThrow(
      Object.keys(HALF_LIVES).join(", "),
    );
  });

  it("refuses zero, negative, NaN, -Infinity and non-numeric days", () => {
    for (const bad of [0, -30, NaN, -Infinity, "365", "never", true, null, [365]]) {
      expect(() => validateHalfLives({ preference: bad }, where)).toThrow(
        /half_lives\.preference = .* is not a positive number of days/,
      );
    }
  });

  it("refuses a non-mapping table", () => {
    for (const bad of [365, "none", null, [["preference", 365]]]) {
      expect(() => validateHalfLives(bad, where)).toThrow(/half_lives must be a mapping/);
    }
  });
});

describe("H2: absent override is the pre-change behaviour", () => {
  it("baseHalfLife falls back to HALF_LIVES, then 60", () => {
    for (const t of Object.keys(HALF_LIVES)) expect(baseHalfLife(t)).toBe(HALF_LIVES[t]!);
    expect(baseHalfLife("no-such-type")).toBe(60);
    // A table that does not name the type also falls through.
    expect(baseHalfLife("project", { preference: 365 })).toBe(HALF_LIVES.project!);
  });

  it("recency and confidence match the global formula when no table is passed", () => {
    const at = daysAgo(90);
    expect(recencyScore(at, "project", NOW)).toBe(Math.pow(0.5, 90 / 120));
    expect(recencyScore(at, "project", NOW, 0, null, undefined)).toBe(
      recencyScore(at, "project", NOW),
    );
    expect(confidenceScore("project", at, 0, NOW)).toBe(
      confidenceScore("project", at, 0, NOW, undefined, undefined, undefined),
    );
  });
});

describe("H1/H4: the override applies only inside its collection", () => {
  it("recencyScore / confidenceScore read the table", () => {
    const at = daysAgo(365);
    expect(recencyScore(at, "preference", NOW)).toBe(1);
    expect(recencyScore(at, "preference", NOW, 0, null, { preference: 365 })).toBeCloseTo(0.5, 10);
    expect(recencyScore(at, "project", NOW, 0, null, NO_DECAY)).toBe(1);
    expect(confidenceScore("project", at, 0, NOW, undefined, undefined, NO_DECAY)).toBeCloseTo(
      0.65,
      10,
    );
  });

  it("a no-decay override also skips attention decay; a finite one does not", () => {
    const modified = daysAgo(200);
    const lastAccess = daysAgo(100);
    const plain = confidenceScore("project", modified, 0, NOW, lastAccess, modified);
    const noDecay = confidenceScore("project", modified, 0, NOW, lastAccess, modified, NO_DECAY);
    expect(noDecay).toBeCloseTo(0.65, 10); // baseline x recency 1 x no attention decay
    const finite = confidenceScore("project", modified, 0, NOW, lastAccess, modified, {
      project: 120,
    });
    expect(finite).toBe(plain); // same half-life as the global table -> same score, attention decay kept
  });

  it("applyCompositeScoring: same doc, override collection vs other collection", () => {
    const inTopics = mkDoc();
    const elsewhere = mkDoc({
      collectionName: "other",
      displayPath: "other/a.md",
      filepath: "clawmem://other/a.md",
    });
    const opts = { now: NOW, halfLivesFor: onlyTopics(NO_DECAY) };
    const [t, o] = applyCompositeScoring([inTopics, elsewhere], "find the thing", undefined, opts);
    const [baseT, baseO] = applyCompositeScoring(
      [inTopics, elsewhere],
      "find the thing",
      undefined,
      {
        now: NOW,
        halfLivesFor: none,
      },
    );
    expect(t!.recencyScore).toBe(1);
    expect(t!.compositeScore).toBeGreaterThan(baseT!.compositeScore);
    // The other collection is untouched, to the bit.
    expect(o!.recencyScore).toBe(baseO!.recencyScore);
    expect(o!.compositeScore).toBe(baseO!.compositeScore);
    expect(baseT!.compositeScore).toBe(baseO!.compositeScore);
  });

  it("resolves each collection once per scoring pass", () => {
    const calls: string[] = [];
    const docs = [mkDoc(), mkDoc({ docid: "b" }), mkDoc({ collectionName: "other", docid: "c" })];
    applyCompositeScoring(docs, "find the thing", undefined, {
      now: NOW,
      halfLivesFor: (c) => {
        calls.push(c);
        return undefined;
      },
    });
    expect(calls.sort()).toEqual(["other", "topics"]);
  });
});

describe("H1/H3: from the collection config", () => {
  const DIR = mkdtempSync(join(tmpdir(), "clawmem-vn4rz.94-config-"));
  const saved = process.env.CLAWMEM_CONFIG_DIR;
  const write = (lines: string[]) =>
    writeFileSync(join(DIR, "config.yaml"), [...lines, ""].join("\n"));
  const BASE = ["collections:", "  topics:", "    path: /tmp/nowhere", '    pattern: "**/*.md"'];

  beforeEach(() => {
    rmSync(DIR, { recursive: true, force: true });
    mkdirSync(DIR, { recursive: true });
    process.env.CLAWMEM_CONFIG_DIR = DIR;
    clearConfigCache();
  });
  afterEach(() => {
    clearConfigCache();
    rmSync(DIR, { recursive: true, force: true });
  });
  afterAll(() => {
    if (saved === undefined) delete process.env.CLAWMEM_CONFIG_DIR;
    else process.env.CLAWMEM_CONFIG_DIR = saved;
    clearConfigCache();
  });

  it("loads a valid table (.inf and none = no decay) for that collection only", () => {
    write([
      ...BASE,
      "    half_lives: { preference: 365, project: .inf, note: none }",
      "  other:",
      "    path: /tmp/elsewhere",
      '    pattern: "**/*.md"',
    ]);
    expect(loadConfig().collections.topics!.half_lives).toEqual({
      preference: 365,
      project: Infinity,
      note: Infinity,
    });
    expect(getCollectionHalfLives("topics")).toEqual({
      preference: 365,
      project: Infinity,
      note: Infinity,
    });
    expect(getCollectionHalfLives("other")).toBeUndefined();
    expect(collectionIndexOptions(loadConfig().collections.topics!).halfLives).toEqual({
      preference: 365,
      project: Infinity,
      note: Infinity,
    });
  });

  it("the default scoring resolver reads the config", () => {
    write([...BASE, "    half_lives: { project: none }"]);
    const [viaConfig] = applyCompositeScoring([mkDoc()], "find the thing", undefined, { now: NOW });
    expect(viaConfig!.recencyScore).toBe(1);
    const [other] = applyCompositeScoring(
      [mkDoc({ collectionName: "other" })],
      "find the thing",
      undefined,
      {
        now: NOW,
      },
    );
    expect(other!.recencyScore).toBe(Math.pow(0.5, 200 / 120));
  });

  it("survives an idempotent addCollection, written back as the none token (83qk0.209)", () => {
    write([...BASE, "    half_lives: { preference: 365, project: none, note: .inf }"]);
    addCollection("topics", "/tmp/nowhere", "**/*.md"); // safe: writes only the temp config dir
    clearConfigCache();
    expect(getCollectionHalfLives("topics")).toEqual({
      preference: 365,
      project: Infinity,
      note: Infinity,
    });
    const raw = readFileSync(join(DIR, "config.yaml"), "utf-8");
    expect(raw).toContain("project: none");
    expect(raw).toContain("note: none");
    expect(raw).not.toContain(".inf");
  });

  it("an edited config invalidates the lookup", () => {
    write([...BASE, "    half_lives: { project: 365 }"]);
    expect(getCollectionHalfLives("topics")).toEqual({ project: 365 });
    write([...BASE, "    half_lives: { project: 30, note: 10 }"]);
    expect(getCollectionHalfLives("topics")).toEqual({ project: 30, note: 10 });
  });

  it("loadConfig refuses an unknown type and bad days, naming the collection", () => {
    write([...BASE, "    half_lives: { feedback: 365 }"]);
    expect(() => loadConfig()).toThrow(
      /collection "topics": half_lives has unknown content_type "feedback"/,
    );
    clearConfigCache();
    for (const bad of ["0", "-5", ".nan", "'365'", "never"]) {
      write([...BASE, `    half_lives: { preference: ${bad} }`]);
      clearConfigCache();
      expect(() => loadConfig()).toThrow(
        /collection "topics": half_lives\.preference = .* is not a positive number/,
      );
    }
  });
});

describe("H4: the indexer stores override-aware confidence", () => {
  const ROOT = mkdtempSync(join(tmpdir(), "clawmem-vn4rz.94-index-"));
  const CONTENT = `${ROOT}/content`;
  let store: Store;
  const deadLlm = {
    embed: async () => {
      throw new Error("no embedding endpoint in test");
    },
    query: async () => null,
    expandQuery: async () => [],
  } as any;

  beforeEach(() => {
    rmSync(ROOT, { recursive: true, force: true });
    mkdirSync(CONTENT, { recursive: true });
    setDefaultLlamaCpp(deadLlm);
    store = createStore(`${ROOT}/vault.sqlite`);
  });
  afterEach(() => {
    try {
      store.close();
    } catch {
      /* already closed */
    }
    setDefaultLlamaCpp(null);
    rmSync(ROOT, { recursive: true, force: true });
  });

  const storedConfidence = (coll: string) =>
    (
      store.db
        .prepare(
          "SELECT confidence FROM documents WHERE collection = ? AND path = ? AND active = 1",
        )
        .get(coll, "project_alpha.md") as { confidence: number }
    ).confidence;

  it("override collection stores the undecayed baseline; plain collection decays", async () => {
    const file = `${CONTENT}/project_alpha.md`;
    writeFileSync(file, "# Alpha\n\nA project memory body.\n");
    const old = new Date(Date.now() - 240 * DAY);
    utimesSync(file, old, old);

    await indexCollection(store, "topics", CONTENT, "**/*.md", {
      ...collectionIndexOptions({ half_lives: { project: Infinity } }),
    });
    await indexCollection(store, "plain", CONTENT, "**/*.md", { ...collectionIndexOptions({}) });

    expect(storedConfidence("topics")).toBeCloseTo(0.65, 10);
    expect(storedConfidence("plain")).toBeCloseTo(0.65 * Math.pow(0.5, 240 / 120), 3);
  });
});
