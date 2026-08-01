/**
 * Contradiction-judge audit primitives (v0.29.0, DESIGN §J7).
 *
 * The load-bearing properties: fallback pairs prune as a UNIT via the self-FK
 * cascade; the count cap counts ROOT runs (one logical evaluation each); the
 * current session's rows are never pruned; best-effort inserts never throw
 * (an audit failure on a non-mutating path must not take the hook down).
 */
import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { createStore, type Store } from "../../src/store.ts";
import {
  insertJudgeRun,
  insertJudgeEvent,
  insertJudgeRunBestEffort,
  pruneJudgeRuns,
  judgeAuditCounts,
  sha256Hex,
  JUDGE_HEAD_MAX,
} from "../../src/judge-audit.ts";

let store: Store;
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "judge-audit-"));
  store = createStore(join(dir, "test.sqlite"));
});

afterEach(() => {
  store.close();
  rmSync(dir, { recursive: true, force: true });
});

function mkRun(overrides: Partial<Parameters<typeof insertJudgeRun>[1]> = {}): number {
  return insertJudgeRun(store.db, {
    sessionId: "sess-a",
    consumer: "decision-extractor",
    lane: "anthropic",
    model: "claude-haiku-4-5",
    outcome: "ok",
    ...overrides,
  });
}

describe("judge_runs / judge_events schema", () => {
  test("run + event insert round-trips with counts", () => {
    const runId = mkRun({ newFactCount: 2, candidateCount: 5, responseSha256: sha256Hex("x") });
    const evId = insertJudgeEvent(store.db, {
      runId,
      eventType: "verdict",
      newIdx: 0,
      oldIdx: 0,
      newRef: "doc:12",
      oldRef: "doc:7",
      relation: "contradiction",
      confidence: 0.85,
      action: "eroded",
      scoreBefore: 0.8,
      scoreAfter: 0.55,
    });
    expect(runId).toBeGreaterThan(0);
    expect(evId).toBeGreaterThan(0);
    const counts = judgeAuditCounts(store.db);
    expect(counts.runs).toBe(1);
    expect(counts.events).toBe(1);
    expect(counts.oldestTs).not.toBeNull();
  });

  test("reasoning_head and evidence_head are bounded at JUDGE_HEAD_MAX", () => {
    const runId = mkRun();
    insertJudgeEvent(store.db, {
      runId,
      eventType: "reject",
      reasonCode: "invalid_confidence",
      reasoningHead: "r".repeat(JUDGE_HEAD_MAX + 500),
      evidenceHead: "e".repeat(JUDGE_HEAD_MAX + 500),
    });
    const row = store.db
      .prepare(`SELECT reasoning_head AS r, evidence_head AS e FROM judge_events WHERE run_id = ?`)
      .get(runId) as { r: string; e: string };
    expect(row.r.length).toBe(JUDGE_HEAD_MAX);
    expect(row.e.length).toBe(JUDGE_HEAD_MAX);
  });

  test("deleting a provider-failure root cascades its heuristic child and both runs' events", () => {
    const providerRun = mkRun({ consumer: "merge-phase2", lane: "openai", outcome: "timeout" });
    const heuristicRun = mkRun({
      consumer: "merge-phase2",
      lane: "heuristic",
      model: null,
      outcome: "ok",
      fallbackFromRunId: providerRun,
    });
    insertJudgeEvent(store.db, { runId: providerRun, eventType: "error", reasonCode: "write_error" });
    insertJudgeEvent(store.db, { runId: heuristicRun, eventType: "verdict", relation: "contradiction", confidence: 0.6, action: "merge_link" });

    store.db.prepare(`DELETE FROM judge_runs WHERE id = ?`).run(providerRun);

    const counts = judgeAuditCounts(store.db);
    expect(counts.runs).toBe(0);
    expect(counts.events).toBe(0);
  });
});

describe("pruneJudgeRuns", () => {
  test("prunes by age, taking the fallback pair as a unit", () => {
    const providerRun = mkRun({ lane: "openai", outcome: "http", sessionId: "old-sess" });
    const heuristicRun = mkRun({ lane: "heuristic", outcome: "ok", fallbackFromRunId: providerRun, sessionId: "old-sess" });
    insertJudgeEvent(store.db, { runId: heuristicRun, eventType: "verdict", action: "merge_allowed" });
    // Only the ROOT's age matters — backdate the root beyond the window.
    store.db.prepare(`UPDATE judge_runs SET ts = datetime('now', '-120 days') WHERE id = ?`).run(providerRun);

    const { pruned } = pruneJudgeRuns(store.db, { maxAgeDays: 90, excludeSessionId: "sess-current" });

    expect(pruned).toBe(1); // one ROOT pruned; the child went via cascade, not via selection
    expect(judgeAuditCounts(store.db).runs).toBe(0);
    expect(judgeAuditCounts(store.db).events).toBe(0);
  });

  test("count cap counts root runs only — a pair is one logical evaluation", () => {
    // Three logical evaluations: two standalone roots + one provider→heuristic pair.
    const r1 = mkRun({ sessionId: "s1" });
    const r2 = mkRun({ sessionId: "s1" });
    const provider = mkRun({ sessionId: "s1", lane: "openai", outcome: "unavailable" });
    mkRun({ sessionId: "s1", lane: "heuristic", outcome: "ok", fallbackFromRunId: provider });
    // Deterministic recency order: r1 oldest, then r2, then the pair root.
    store.db.prepare(`UPDATE judge_runs SET ts = datetime('now', '-3 hours') WHERE id = ?`).run(r1);
    store.db.prepare(`UPDATE judge_runs SET ts = datetime('now', '-2 hours') WHERE id = ?`).run(r2);

    const { pruned } = pruneJudgeRuns(store.db, { maxRuns: 2 });

    // Cap=2 keeps the two newest ROOTS (r2 + pair root); r1 is pruned. The heuristic
    // child survives with its root and never counted toward the cap.
    expect(pruned).toBe(1);
    const remaining = store.db.prepare(`SELECT id FROM judge_runs ORDER BY id`).all() as { id: number }[];
    expect(remaining.map(r => r.id)).not.toContain(r1);
    expect(remaining.length).toBe(3); // r2 + provider root + heuristic child
  });

  test("the excluded session's runs survive both age and count pruning", () => {
    const current = mkRun({ sessionId: "sess-current" });
    store.db.prepare(`UPDATE judge_runs SET ts = datetime('now', '-365 days') WHERE id = ?`).run(current);
    const other = mkRun({ sessionId: "sess-other" });
    store.db.prepare(`UPDATE judge_runs SET ts = datetime('now', '-365 days') WHERE id = ?`).run(other);

    const { pruned } = pruneJudgeRuns(store.db, { maxAgeDays: 90, maxRuns: 1, excludeSessionId: "sess-current" });

    const ids = (store.db.prepare(`SELECT id FROM judge_runs`).all() as { id: number }[]).map(r => r.id);
    expect(ids).toContain(current);
    expect(ids).not.toContain(other);
    expect(pruned).toBe(1);
  });
});

describe("fault injection", () => {
  test("insertJudgeRunBestEffort returns null instead of throwing when the table is gone", () => {
    store.db.exec(`DROP TABLE judge_events`);
    store.db.exec(`DROP TABLE judge_runs`);
    const id = insertJudgeRunBestEffort(store.db, {
      consumer: "decision-extractor",
      lane: "none",
      outcome: "no_judge_configured",
    });
    expect(id).toBeNull();
  });

  test("insertJudgeRun THROWS on the same failure — mutation-coupled paths must fail closed", () => {
    store.db.exec(`DROP TABLE judge_events`);
    store.db.exec(`DROP TABLE judge_runs`);
    expect(() =>
      insertJudgeRun(store.db, { consumer: "merge-phase2", lane: "heuristic", outcome: "ok" })
    ).toThrow();
  });
});
