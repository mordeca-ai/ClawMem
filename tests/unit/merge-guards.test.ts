import { describe, it, expect, afterEach, beforeEach } from "bun:test";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import {
  evaluateMergeContradiction,
  persistMergeEvaluation,
  resolveEffectiveContradictionPolicy,
  CONTRADICTION_MIN_CONFIDENCE,
  heuristicContradictionCheck,
  isActionableContradiction,
  resolveContradictionPolicy,
} from "../../src/merge-guards.ts";
import type { JudgeResolution, JudgeResult, JudgeRequest } from "../../src/judge.ts";
import { createStore, type Store } from "../../src/store.ts";

/**
 * Unit tests for Ext 2 — Contradiction-aware merge gate
 * (THOTH_EXTRACTION_PLAN.md Extraction 2; judge-gated since v0.29.0).
 *
 * The judge lane is stubbed at the JudgeResolution seam. The load-bearing
 * v0.29.0 properties: the legacy object contract is DEAD (an object response is
 * a parse reject, not a verdict), confidence-defaulting is GONE (a missing
 * confidence is an invalid entry, never 0.5), judge failure falls back to an
 * AUDITED heuristic pair, and `aborted` is terminal.
 */

const readyResolution = (fn: (req: JudgeRequest) => Promise<JudgeResult>): JudgeResolution => ({
  status: "ready",
  judge: {
    descriptor: {
      lane: "openai",
      model: "stub-judge",
      endpoint: "http://stub:1",
      supportsSystemRole: true,
      supportsJsonSchema: false,
      noThink: false,
      mayDownload: false,
    },
    judge: fn,
  },
});

const okText = (text: string) => async (): Promise<JudgeResult> =>
  ({ ok: true, text, model: "stub-judge", truncated: false });

// ─── heuristicContradictionCheck ───────────────────────────────────────

describe("heuristicContradictionCheck", () => {
  it("flags negation asymmetry (one side has 'not', the other doesn't)", () => {
    const result = heuristicContradictionCheck(
      "The migration completed on time",
      "The migration did not complete on time"
    );
    expect(result.contradictory).toBe(true);
    expect(result.source).toBe("heuristic");
    expect(result.confidence).toBeGreaterThan(0.5);
    expect(result.reason).toContain("negation");
  });

  it("flags common negation contractions (didn't, won't, cannot)", () => {
    expect(
      heuristicContradictionCheck(
        "Bob will ship this sprint",
        "Bob won't ship this sprint"
      ).contradictory
    ).toBe(true);
    expect(
      heuristicContradictionCheck(
        "The team finished on time",
        "The team didn't finish on time"
      ).contradictory
    ).toBe(true);
    expect(
      heuristicContradictionCheck(
        "The system can recover automatically",
        "The system cannot recover automatically"
      ).contradictory
    ).toBe(true);
  });

  it("flags number/date mismatches when both sides cite numbers", () => {
    const result = heuristicContradictionCheck(
      "Deploy count was 5 last week",
      "Deploy count was 7 last week"
    );
    expect(result.contradictory).toBe(true);
    expect(result.source).toBe("heuristic");
    expect(result.reason).toContain("mismatch");
  });

  it("does NOT flag when numbers overlap (same numeric anchor)", () => {
    const result = heuristicContradictionCheck(
      "Version 1.2 was released on 2026-04-10",
      "Version 1.2 shipped on 2026-04-10"
    );
    expect(result.contradictory).toBe(false);
  });

  it("does NOT flag two statements without any signal", () => {
    const result = heuristicContradictionCheck(
      "The team shipped the feature",
      "The team shipped the feature"
    );
    expect(result.contradictory).toBe(false);
    expect(result.confidence).toBe(0);
    expect(result.reason).toContain("no heuristic signal");
  });

  it("does NOT flag when both sides have matching negation", () => {
    const result = heuristicContradictionCheck(
      "The deployment did not succeed",
      "The rollout did not succeed"
    );
    // Both have negation; heuristic shouldn't mark a contradiction.
    expect(result.contradictory).toBe(false);
  });

  it("does NOT flag when only one side has numbers (no comparison possible)", () => {
    const result = heuristicContradictionCheck(
      "The feature shipped today",
      "The feature shipped 3 days after the milestone"
    );
    expect(result.contradictory).toBe(false);
  });

  it("always returns source='heuristic'", () => {
    expect(heuristicContradictionCheck("a", "b").source).toBe("heuristic");
    expect(
      heuristicContradictionCheck("not a", "b").source
    ).toBe("heuristic");
  });
});

// ─── evaluateMergeContradiction (judge-gated, §J5d/§J7 matrix) ─────────

describe("evaluateMergeContradiction", () => {
  it("a valid single-pair contradiction verdict decides, source='llm'", async () => {
    const resolution = readyResolution(okText(
      '[{"new_idx":0,"old_idx":0,"relation":"contradiction","confidence":0.92,"reasoning":"opposite outcomes"}]'
    ));
    const ev = await evaluateMergeContradiction(resolution, "Deploy succeeded", "Deploy failed");
    expect(ev.kind).toBe("decided");
    if (ev.kind !== "decided") return;
    expect(ev.result.source).toBe("llm");
    expect(ev.result.contradictory).toBe(true);
    expect(ev.result.confidence).toBe(0.92);
    expect(ev.runs).toHaveLength(1);
    expect(ev.runs[0]!.outcome).toBe("ok");
    expect(ev.events.some(e => e.eventType === "verdict" && e.relation === "contradiction")).toBe(true);
  });

  it("an empty array is a decisive 'no relationship' — not contradictory", async () => {
    const ev = await evaluateMergeContradiction(readyResolution(okText("[]")), "a", "b");
    if (ev.kind !== "decided") throw new Error("expected decided");
    expect(ev.result.contradictory).toBe(false);
    expect(ev.result.source).toBe("llm");
  });

  it("REGRESSION: the legacy object contract is DEAD — an object response falls back to the audited heuristic", async () => {
    // Pre-0.29.0 this parsed and its missing confidence defaulted to the
    // actionable 0.5 — the fail-open the redesign kills.
    const ev = await evaluateMergeContradiction(
      readyResolution(okText('{"contradictory": true, "confidence": 0.9, "reason": "legacy"}')),
      "The team shipped the feature",
      "The team shipped the feature",
    );
    if (ev.kind !== "decided") throw new Error("expected decided");
    expect(ev.result.source).toBe("heuristic");
    expect(ev.result.contradictory).toBe(false); // no heuristic signal on identical text
    expect(ev.runs).toHaveLength(2);
    expect(ev.runs[0]!.outcome).toBe("parse_reject");
    expect(ev.runs[1]!.lane).toBe("heuristic");
  });

  it("REGRESSION: a missing confidence is a SEMANTIC REJECT — audited heuristic decides, never a 0.5 default", async () => {
    // A non-empty response with zero admitted entries must NOT read as a decisive
    // "no relationship" — that recreated the parseable-response-bypasses-heuristic
    // fail-open for semantic admission errors (code-review t1 finding 3).
    const ev = await evaluateMergeContradiction(
      readyResolution(okText('[{"new_idx":0,"old_idx":0,"relation":"contradiction","reasoning":"no conf"}]')),
      "The deploy succeeded", "The deploy did not succeed",
    );
    if (ev.kind !== "decided") throw new Error("expected decided");
    expect(ev.result.source).toBe("heuristic");
    expect(ev.result.contradictory).toBe(true); // negation asymmetry decides, audited
    expect(ev.runs).toHaveLength(2);
    expect(ev.runs[0]!.outcome).toBe("parse_reject");
    expect(ev.runs[0]!.entriesRejected).toBe(1);
    expect(ev.runs[1]!.lane).toBe("heuristic");
  });

  it("judge unavailable → failed provider run + linked audited heuristic run; heuristic decides", async () => {
    const resolution = readyResolution(async () => ({ ok: false, reason: "unavailable", detail: "down" }));
    const ev = await evaluateMergeContradiction(resolution, "The deploy succeeded", "The deploy did not succeed");
    if (ev.kind !== "decided") throw new Error("expected decided");
    expect(ev.result.source).toBe("heuristic");
    expect(ev.result.contradictory).toBe(true); // negation asymmetry
    expect(ev.runs).toHaveLength(2);
    expect(ev.runs[0]!.outcome).toBe("unavailable");
    expect(ev.runs[1]!.lane).toBe("heuristic");
  });

  it("truncated responses fall back — never partially parsed", async () => {
    const resolution = readyResolution(async () => ({ ok: true, text: '[{"new_idx":0,', model: "stub", truncated: true }));
    const ev = await evaluateMergeContradiction(resolution, "Version 2 shipped", "Version 5 shipped");
    if (ev.kind !== "decided") throw new Error("expected decided");
    expect(ev.runs[0]!.outcome).toBe("truncated");
    expect(ev.result.source).toBe("heuristic");
    expect(ev.result.contradictory).toBe(true); // number mismatch
  });

  it("aborted is TERMINAL: no heuristic, no decision", async () => {
    const resolution = readyResolution(async () => ({ ok: false, reason: "aborted", detail: "caller cancelled" }));
    const ev = await evaluateMergeContradiction(resolution, "The deploy succeeded", "The deploy did not succeed");
    expect(ev.kind).toBe("aborted");
    if (ev.kind !== "aborted") return;
    expect(ev.runs).toHaveLength(1);
    expect(ev.runs[0]!.outcome).toBe("aborted");
  });

  it("unconfigured → heuristic-only single run", async () => {
    const ev = await evaluateMergeContradiction({ status: "unconfigured" }, "a", "b not a");
    if (ev.kind !== "decided") throw new Error("expected decided");
    expect(ev.result.source).toBe("heuristic");
    expect(ev.runs).toHaveLength(1);
    expect(ev.runs[0]!.lane).toBe("heuristic");
  });
});

// ─── persistMergeEvaluation (pair linkage) ─────────────────────────────

describe("persistMergeEvaluation", () => {
  let store: Store;
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "merge-guards-"));
    store = createStore(join(dir, "vault.sqlite"));
  });

  afterEach(() => {
    try { (store as any).close?.(); } catch { /* best-effort */ }
    rmSync(dir, { recursive: true, force: true });
  });

  it("a provider-failure→heuristic pair links via fallback_from_run_id; events bind to the DECIDING run", async () => {
    const resolution = readyResolution(async () => ({ ok: false, reason: "timeout", detail: "slow" }));
    const ev = await evaluateMergeContradiction(resolution, "The deploy succeeded", "The deploy did not succeed");
    if (ev.kind !== "decided") throw new Error("expected decided");

    const decidingRunId = persistMergeEvaluation(store.db, "merge-phase2", null, ev);

    const rows = store.db.prepare(
      `SELECT id, lane, outcome, fallback_from_run_id FROM judge_runs ORDER BY id`
    ).all() as { id: number; lane: string; outcome: string; fallback_from_run_id: number | null }[];
    expect(rows).toHaveLength(2);
    expect(rows[0]!.outcome).toBe("timeout");
    expect(rows[0]!.fallback_from_run_id).toBeNull();
    expect(rows[1]!.lane).toBe("heuristic");
    expect(rows[1]!.fallback_from_run_id).toBe(rows[0]!.id);
    expect(decidingRunId).toBe(rows[1]!.id);

    const ev0 = store.db.prepare(`SELECT run_id FROM judge_events`).get() as { run_id: number };
    expect(ev0.run_id).toBe(decidingRunId);
  });
});

// ─── resolveEffectiveContradictionPolicy (§J1 blocked supersede) ───────

describe("resolveEffectiveContradictionPolicy", () => {
  afterEach(() => {
    delete process.env.CLAWMEM_CONTRADICTION_POLICY;
  });

  it("configured supersede WITHOUT a judge is constrained to link, flagged blocked", () => {
    process.env.CLAWMEM_CONTRADICTION_POLICY = "supersede";
    const r = resolveEffectiveContradictionPolicy(false);
    expect(r.policy).toBe("link");
    expect(r.supersedeBlocked).toBe(true);
  });

  it("configured supersede WITH a judge stays supersede", () => {
    process.env.CLAWMEM_CONTRADICTION_POLICY = "supersede";
    const r = resolveEffectiveContradictionPolicy(true);
    expect(r.policy).toBe("supersede");
    expect(r.supersedeBlocked).toBe(false);
  });

  it("link is never blocked", () => {
    const r = resolveEffectiveContradictionPolicy(false);
    expect(r.policy).toBe("link");
    expect(r.supersedeBlocked).toBe(false);
  });
});

// ─── isActionableContradiction ─────────────────────────────────────────

describe("isActionableContradiction", () => {
  it("true when contradictory=true AND confidence >= threshold", () => {
    expect(
      isActionableContradiction({
        contradictory: true,
        confidence: 0.9,
        source: "llm",
      })
    ).toBe(true);
  });

  it("false when contradictory=false regardless of confidence", () => {
    expect(
      isActionableContradiction({
        contradictory: false,
        confidence: 1.0,
        source: "llm",
      })
    ).toBe(false);
  });

  it("false when contradictory=true but confidence below threshold", () => {
    expect(
      isActionableContradiction({
        contradictory: true,
        confidence: 0.2,
        source: "heuristic",
      })
    ).toBe(false);
  });

  it("boundary: confidence exactly at threshold is actionable", () => {
    expect(
      isActionableContradiction({
        contradictory: true,
        confidence: CONTRADICTION_MIN_CONFIDENCE,
        source: "heuristic",
      })
    ).toBe(true);
  });
});

// ─── resolveContradictionPolicy ────────────────────────────────────────

describe("resolveContradictionPolicy", () => {
  afterEach(() => {
    delete process.env.CLAWMEM_CONTRADICTION_POLICY;
  });

  it("defaults to 'link' when env unset", () => {
    expect(resolveContradictionPolicy()).toBe("link");
  });

  it("honors 'supersede' from env", () => {
    process.env.CLAWMEM_CONTRADICTION_POLICY = "supersede";
    expect(resolveContradictionPolicy()).toBe("supersede");
  });

  it("honors 'link' explicitly from env", () => {
    process.env.CLAWMEM_CONTRADICTION_POLICY = "link";
    expect(resolveContradictionPolicy()).toBe("link");
  });

  it("falls back to 'link' on invalid value", () => {
    process.env.CLAWMEM_CONTRADICTION_POLICY = "merge-everything";
    expect(resolveContradictionPolicy()).toBe("link");
  });
});

// ─── t3 regression pins ────────────────────────────────────────────────

describe("dirty-batch authorization + provider-event binding (code-review t3)", () => {
  it("REGRESSION: one VALID verdict wrapped in junk cannot decide — audited heuristic takes over", async () => {
    // Pre-fix, the valid (0,0) entry decided with source:"llm" and could
    // authorize supersede despite the rejected junk beside it.
    const dirty = '[{"new_idx":0,"old_idx":0,"relation":"contradiction","confidence":0.95,"reasoning":"real"},' +
                  '{"new_idx":0,"old_idx":9,"relation":"contradiction","confidence":0.9,"reasoning":"oob"}]';
    const ev = await evaluateMergeContradiction(
      readyResolution(okText(dirty)),
      "The team shipped the feature", "The team shipped the feature",
    );
    if (ev.kind !== "decided") throw new Error("expected decided");
    expect(ev.result.source).toBe("heuristic");
    expect(ev.runs[0]!.outcome).toBe("parse_reject");
    expect(ev.runs[0]!.entriesAdmitted).toBe(1);
    expect(ev.runs[0]!.entriesRejected).toBe(1);
    // The discarded-but-valid proposal AND the reject are provider-bound evidence.
    const pe = ev.providerEvents ?? [];
    expect(pe.some(e => e.eventType === "verdict" && e.relation === "contradiction" && e.confidence === 0.95)).toBe(true);
    expect(pe.some(e => e.eventType === "reject" && e.reasonCode === "index_oob")).toBe(true);
  });

  it("persistMergeEvaluation binds providerEvents to the PROVIDER run, verdict events to the deciding run", async () => {
    const dir2 = mkdtempSync(join(tmpdir(), "merge-guards-pe-"));
    const store2 = createStore(join(dir2, "vault.sqlite"));
    try {
      const dirty = '[{"new_idx":0,"old_idx":0,"relation":"contradiction","confidence":0.95,"reasoning":"real"},' +
                    '{"new_idx":0,"old_idx":9,"relation":"contradiction","confidence":0.9,"reasoning":"oob"}]';
      const ev = await evaluateMergeContradiction(readyResolution(okText(dirty)), "a", "b");
      if (ev.kind !== "decided") throw new Error("expected decided");
      const decidingId = persistMergeEvaluation(store2.db, "merge-phase2", null, ev);
      const rows = store2.db.prepare(
        `SELECT run_id, event_type, reason_code FROM judge_events ORDER BY id`,
      ).all() as { run_id: number; event_type: string; reason_code: string | null }[];
      const providerRunId = (store2.db.prepare(`SELECT MIN(id) AS id FROM judge_runs`).get() as { id: number }).id;
      expect(providerRunId).not.toBe(decidingId);
      expect(rows.filter(r => r.run_id === providerRunId).length).toBeGreaterThanOrEqual(2);
      expect(rows.filter(r => r.run_id === decidingId && r.event_type === "verdict").length).toBe(1);
    } finally {
      try { (store2 as any).close?.(); } catch { /* best-effort */ }
      rmSync(dir2, { recursive: true, force: true });
    }
  });
});
