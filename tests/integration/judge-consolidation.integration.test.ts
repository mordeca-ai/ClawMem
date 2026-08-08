/**
 * Production-boundary integration for the judge-gated merge gate (v0.29.0):
 * drives `runHeavyMaintenanceTick` — the REAL Phase-2 path through
 * `synthesizeCluster` → `findSimilarConsolidation` → the contradiction gate →
 * audit-coupled mutation — with a real `openai`-lane judge (mock HTTP server or
 * a dead port), configured via env exactly as production reads it.
 *
 * The load-bearing pin (code-review t1 finding 1): a CONFIGURED judge that
 * fails at runtime hands the decision to the heuristic, and the heuristic must
 * NEVER authorize destructive `supersede` — regardless of the configured policy.
 */
import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { runHeavyMaintenanceTick } from "../../src/maintenance.ts";
import { createTestStore } from "../helpers/test-store.ts";
import { createMockLLM } from "../helpers/mock-llm.ts";
import type { Store } from "../../src/store.ts";

const TEST_COLLECTION = "_clawmem";

// The heavy lane enforces the name-aware merge-safety gate (strictest 3-gram
// cosine ≥ 0.98), so the pair differs ONLY by an inserted "no " inside a long
// identical sentence: trigram overlap stays ≥ 0.98 while the heuristic's
// negation asymmetry still fires.
const BASE =
  "after the full weekly release verification cycle across every staging and production " +
  "environment including the database migration rehearsal and the rollback drill the launch " +
  "review board evaluated the deployment readiness checklist and formally recorded ";
const EXISTING_TEXT = `${BASE}go for the scheduled deployment window`;
const PATTERN_TEXT = `${BASE}no go for the scheduled deployment window`;

function seedObservationDoc(store: Store, path: string, title: string, facts: string): number {
  const hash = `hash_${path}_${Math.random().toString(36).slice(2)}`;
  const modifiedAt = new Date().toISOString();
  store.db.prepare(`INSERT INTO content (hash, doc, created_at) VALUES (?, ?, ?)`)
    .run(hash, `# ${title}\n${facts}`, modifiedAt);
  store.db.prepare(
    `INSERT INTO documents
        (collection, path, title, hash, created_at, modified_at, active,
         content_type, observation_type, facts, narrative, last_accessed_at)
      VALUES (?, ?, ?, ?, ?, ?, 1, 'observation', 'decision', ?, ?, ?)`,
  ).run(TEST_COLLECTION, path, title, hash, modifiedAt, modifiedAt, facts, `${title} narrative`, modifiedAt);
  return (store.db.prepare(`SELECT id FROM documents WHERE collection = ? AND path = ?`)
    .get(TEST_COLLECTION, path) as { id: number }).id;
}

function seedExistingConsolidation(store: Store): number {
  store.db.prepare(
    `INSERT INTO consolidated_observations (observation, proof_count, source_doc_ids, trend, status, collection)
     VALUES (?, 2, '[900,901]', 'STABLE', 'active', ?)`,
  ).run(EXISTING_TEXT, TEST_COLLECTION);
  return Number(store.db.prepare(`SELECT last_insert_rowid() AS id`).get()!.id ?? 1);
}

/** MockLLM: synthesis prompt → the contradicting pattern; everything else → []. */
function mkLlm() {
  const llm = createMockLLM();
  llm.generate.mockImplementation(async (prompt: string) => ({
    text: prompt.includes("recurring patterns")
      ? JSON.stringify([{ observation: PATTERN_TEXT, proof_count: 2, source_indices: [1, 2] }])
      : "[]",
    model: "mock",
    done: true,
  }));
  return llm;
}

const JUDGE_ENV_KEYS = [
  "CLAWMEM_JUDGE_URL", "CLAWMEM_JUDGE_MODEL", "CLAWMEM_JUDGE_PROVIDER",
  "CLAWMEM_JUDGE_API_KEY", "CLAWMEM_CONTRADICTION_POLICY", "CLAWMEM_MERGE_GUARD_DRY_RUN",
];

describe("judge-gated Phase-2 merge gate — production boundary", () => {
  let store: Store;
  const saved: Record<string, string | undefined> = {};

  beforeEach(() => {
    store = createTestStore();
    for (const k of JUDGE_ENV_KEYS) saved[k] = process.env[k];
    // NOTE: the heavy lane passes guarded:true, which forces name-aware gate
    // ENFORCEMENT regardless of this flag — the fixture pair passes the real
    // 0.98 trigram floor by construction (see BASE above). The flag stays set
    // only to keep behavior stable if the guarded default ever changes.
    process.env.CLAWMEM_MERGE_GUARD_DRY_RUN = "true";
  });

  afterEach(() => {
    for (const k of JUDGE_ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  function seedScenario() {
    seedObservationDoc(store, "observations/a.md", "deploy pipeline A", PATTERN_TEXT);
    seedObservationDoc(store, "observations/b.md", "deploy pipeline B", PATTERN_TEXT);
    return seedExistingConsolidation(store);
  }

  it("F1 REGRESSION: a configured-but-DOWN judge (heuristic decides) never authorizes supersede", async () => {
    const existingId = seedScenario();
    // Configured judge at a dead port ⇒ typed transport failure ⇒ audited heuristic pair.
    process.env.CLAWMEM_JUDGE_URL = "http://127.0.0.1:1";
    process.env.CLAWMEM_JUDGE_MODEL = "any-model";
    process.env.CLAWMEM_CONTRADICTION_POLICY = "supersede";

    await runHeavyMaintenanceTick(store, mkLlm() as any, {});

    const old = store.db.prepare(`SELECT status, invalidated_by, invalidated_at FROM consolidated_observations WHERE id = ?`)
      .get(existingId) as { status: string; invalidated_by: number | null; invalidated_at: string | null };
    // Heuristic decided (negation asymmetry) → contradiction handled as LINK, not supersede.
    expect(old.status).toBe("active");
    expect(old.invalidated_at).toBeNull();
    expect(old.invalidated_by).not.toBeNull(); // link backlink to the new row

    const runs = store.db.prepare(
      `SELECT lane, outcome, fallback_from_run_id FROM judge_runs WHERE consumer = 'merge-phase2' ORDER BY id`,
    ).all() as { lane: string; outcome: string; fallback_from_run_id: number | null }[];
    expect(runs.length).toBe(2);
    expect(runs[0]!.outcome).toBe("unavailable"); // dead port = transport failure
    expect(runs[1]!.lane).toBe("heuristic");
    expect(runs[1]!.fallback_from_run_id).not.toBeNull();

    const actions = (store.db.prepare(
      `SELECT e.action FROM judge_events e JOIN judge_runs r ON r.id = e.run_id WHERE r.consumer = 'merge-phase2'`,
    ).all() as { action: string | null }[]).map(r => r.action);
    expect(actions).toContain("merge_link");
    expect(actions).toContain("merge_supersede_blocked");
    expect(actions).not.toContain("merge_supersede");
  // Fork note (9jyc0): explicit 45s budget — the judge's HTTP_TIMEOUT_MS is 15s, and on
  // hosts where a dead port hangs instead of refusing (WSL2), the "DOWN judge" leg takes
  // the full 15s before falling back to the heuristic. Default 5s budget fails there.
  }, 45_000);

  it("a WORKING judge verdict authorizes supersede, audit-coupled with the mutation", async () => {
    const existingId = seedScenario();
    const server = Bun.serve({
      port: 0,
      fetch: async () =>
        new Response(JSON.stringify({
          choices: [{
            message: {
              content: JSON.stringify([{ new_idx: 0, old_idx: 0, relation: "contradiction", confidence: 0.95, reasoning: "the new pattern negates the existing one" }]),
            },
            finish_reason: "stop",
          }],
          model: "mock-judge",
        }), { status: 200, headers: { "content-type": "application/json" } }),
    });
    try {
      process.env.CLAWMEM_JUDGE_URL = `http://127.0.0.1:${server.port}`;
      process.env.CLAWMEM_JUDGE_MODEL = "mock-judge";
      process.env.CLAWMEM_CONTRADICTION_POLICY = "supersede";

      await runHeavyMaintenanceTick(store, mkLlm() as any, {});

      const old = store.db.prepare(`SELECT status, invalidated_at FROM consolidated_observations WHERE id = ?`)
        .get(existingId) as { status: string; invalidated_at: string | null };
      expect(old.status).toBe("inactive"); // judge-authorized supersede applied
      expect(old.invalidated_at).not.toBeNull();

      const run = store.db.prepare(
        `SELECT lane, outcome, entries_admitted FROM judge_runs WHERE consumer = 'merge-phase2'`,
      ).get() as { lane: string; outcome: string; entries_admitted: number };
      expect(run.lane).toBe("openai");
      expect(run.outcome).toBe("ok");
      expect(run.entries_admitted).toBe(1);

      const actions = (store.db.prepare(
        `SELECT e.action FROM judge_events e JOIN judge_runs r ON r.id = e.run_id WHERE r.consumer = 'merge-phase2'`,
      ).all() as { action: string | null }[]).map(r => r.action);
      expect(actions).toContain("merge_supersede");
      expect(actions).not.toContain("merge_supersede_blocked");
    } finally {
      server.stop(true);
    }
  });

  it("no judge configured: heuristic-only single run, link-constrained", async () => {
    const existingId = seedScenario();
    for (const k of ["CLAWMEM_JUDGE_URL", "CLAWMEM_JUDGE_MODEL", "CLAWMEM_JUDGE_PROVIDER"]) delete process.env[k];
    process.env.CLAWMEM_CONTRADICTION_POLICY = "supersede";

    await runHeavyMaintenanceTick(store, mkLlm() as any, {});

    const old = store.db.prepare(`SELECT status FROM consolidated_observations WHERE id = ?`)
      .get(existingId) as { status: string };
    expect(old.status).toBe("active");

    const runs = store.db.prepare(
      `SELECT lane, fallback_from_run_id FROM judge_runs WHERE consumer = 'merge-phase2'`,
    ).all() as { lane: string; fallback_from_run_id: number | null }[];
    expect(runs.length).toBe(1);
    expect(runs[0]!.lane).toBe("heuristic");
    expect(runs[0]!.fallback_from_run_id).toBeNull();
  });
});

describe("judge-audit retention — production callers (code-review t3)", () => {
  it("the light consolidation tick prunes stale audit roots", async () => {
    const store = createTestStore();
    const { insertJudgeRun } = await import("../../src/judge-audit.ts");
    const { runConsolidationTick } = await import("../../src/consolidation.ts");
    const staleId = insertJudgeRun(store.db, { consumer: "merge-phase2", lane: "heuristic", outcome: "ok" });
    store.db.prepare(`UPDATE judge_runs SET ts = datetime('now', '-120 days') WHERE id = ?`).run(staleId);

    const llm = createMockLLM();
    llm.generate.mockImplementation(async () => ({ text: "[]", model: "mock", done: true }));
    await runConsolidationTick(store, llm as any, {});

    const n = (store.db.prepare(`SELECT COUNT(*) AS n FROM judge_runs`).get() as { n: number }).n;
    expect(n).toBe(0);
  });
});
