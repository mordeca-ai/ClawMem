/**
 * Contradiction-aware merge gate (Ext 2 · judge-gated since v0.29.0).
 *
 * The LLM layer of this gate was fail-open: a parseable answer bypassed the
 * heuristic entirely, a missing `confidence` defaulted to the actionable
 * threshold, and — with the stock expansion model, which fabricates relations —
 * that could block valid merges or (under `supersede`) deactivate rows. Since
 * v0.29.0 the LLM check runs ONLY through the configured contradiction judge
 * (DESIGN §J5d), through the SAME strict relation-array contract, extractor,
 * and validator as the decision-extractor seam; without a judge the gate is
 * heuristic-only and constrained to non-deactivating `link`.
 *
 * Flow:
 *   1. `evaluateMergeContradiction` — judge-first per the exhaustive failure
 *      matrix (§J7): judge failure falls back to an AUDITED heuristic pair;
 *      `aborted` is terminal (no heuristic, no mutation).
 *   2. `heuristicContradictionCheck` — deterministic signal on negation
 *      asymmetry or number/date mismatch.
 *   3. `persistMergeEvaluation` — writes the evaluation's run/event rows;
 *      the CALLER owns transaction placement (§J7 taxonomy: Phase 2 couples
 *      them with the chosen mutation; Phase 3 commits them as a precondition).
 *
 * Adapted from Thoth `tools/memory_tool.py:111-184` contradiction-check
 * pattern (THOTH_EXTRACTION_PLAN.md Extraction 2).
 *
 * Reuses the A-MEM convention relation type `'contradicts'` (plural) —
 * see P0 taxonomy guard at `tests/unit/contradict-taxonomy.test.ts`.
 */

import type { Database } from "bun:sqlite";
import {
  buildContradictionPrompt,
  extractJudgeJson,
  JUDGE_VERDICT_SCHEMA,
  JUDGE_PROMPT_VERSION,
  type Judge,
  type JudgeResolution,
} from "./judge.ts";
import {
  insertJudgeRun,
  insertJudgeEvent,
  insertJudgeRunBestEffort,
  type JudgeConsumer,
  type JudgeRunInput,
  type JudgeEventInput,
} from "./judge-audit.ts";
import {
  unwrapContradictionArray,
  admitContradictionEntries,
} from "./hooks/decision-extractor.ts";
import { hashContent } from "./indexer.ts";

// =============================================================================
// Types
// =============================================================================

export type ContradictionSource = "llm" | "heuristic" | "unknown";

export interface ContradictionResult {
  contradictory: boolean;
  confidence: number; // 0.0 - 1.0
  reason?: string;
  source: ContradictionSource;
}

/**
 * Phase-2 contradiction handling policy. `link` (default) preserves
 * both rows as active and sets `invalidated_by` as a backlink for
 * operator queries. `supersede` additionally sets `invalidated_at` on
 * the old row so it stops surfacing in active recalls.
 */
export type ContradictionPolicy = "link" | "supersede";

export function resolveContradictionPolicy(): ContradictionPolicy {
  const raw = process.env.CLAWMEM_CONTRADICTION_POLICY;
  if (raw === "supersede") return "supersede";
  return "link"; // default
}

/**
 * v0.29.0 (§J1): the EFFECTIVE policy, resolved once UPSTREAM and passed into
 * mutation code as a parameter — the mutation helper never re-reads the env.
 * `supersede` is destructive and therefore requires a configured judge: an
 * unaudited heuristic whose number-mismatch score sits exactly at the default
 * action threshold must never select deactivation. Blocking is LOUD (once per
 * process here; a `merge_supersede_blocked` event per occurrence at the call
 * site; doctor reports the configured policy as presently inactive).
 */
let supersedeBlockedWarned = false;

export function resolveEffectiveContradictionPolicy(
  judgeReady: boolean
): { policy: ContradictionPolicy; supersedeBlocked: boolean } {
  const configured = resolveContradictionPolicy();
  if (configured === "supersede" && !judgeReady) {
    if (!supersedeBlockedWarned) {
      supersedeBlockedWarned = true;
      console.warn(
        `[merge-guards] CLAWMEM_CONTRADICTION_POLICY=supersede is configured but no ` +
        `contradiction judge is available — constraining to non-deactivating 'link' ` +
        `until CLAWMEM_JUDGE_* is configured (see docs/guides/inference-services.md). ` +
        `'clawmem doctor' reports this state.`,
      );
    }
    return { policy: "link", supersedeBlocked: true };
  }
  return { policy: configured, supersedeBlocked: false };
}

/**
 * Minimum LLM contradiction confidence to act on. Lower scores are
 * treated as inconclusive and the merge proceeds (conservative: only
 * block merges on clear contradictions). Overridable via
 * `CLAWMEM_CONTRADICTION_MIN_CONFIDENCE` env var (0.0 - 1.0).
 */
export const CONTRADICTION_MIN_CONFIDENCE = parseEnvFloat(
  "CLAWMEM_CONTRADICTION_MIN_CONFIDENCE",
  0.5
);

function parseEnvFloat(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const n = Number.parseFloat(raw);
  if (!Number.isFinite(n) || n < 0 || n > 1) return fallback;
  return n;
}

// =============================================================================
// Heuristic contradiction detection (deterministic, no LLM)
// =============================================================================

/**
 * Deterministic heuristic contradiction check.
 *
 * Signals:
 *  - **Negation asymmetry:** one side has an explicit negation token
 *    (`not`, `never`, `no`, `didn't`, etc.) and the other doesn't.
 *  - **Number/date mismatch:** both sides cite numbers or dates but the
 *    sets have no shared values.
 *
 * Intentionally conservative: returns `contradictory=false,
 * confidence=0` when no signal is found, leaving the decision to the
 * LLM or the caller's default.
 */
export function heuristicContradictionCheck(
  a: string,
  b: string
): ContradictionResult {
  const negA = hasNegation(a);
  const negB = hasNegation(b);

  // Negation asymmetry: one side explicitly negates, the other doesn't
  if (negA !== negB) {
    return {
      contradictory: true,
      confidence: 0.6,
      reason: "negation asymmetry — one statement has explicit negation",
      source: "heuristic",
    };
  }

  const numsA = extractNumbers(a);
  const numsB = extractNumbers(b);

  // Number/date mismatch: both cite numbers but no shared values
  if (numsA.length > 0 && numsB.length > 0) {
    const setA = new Set(numsA);
    const setB = new Set(numsB);
    const shared = [...setA].filter((n) => setB.has(n));
    if (shared.length === 0) {
      return {
        contradictory: true,
        confidence: 0.5,
        reason: `number/date mismatch (A=${numsA.join(",")} B=${numsB.join(",")})`,
        source: "heuristic",
      };
    }
  }

  // No heuristic signal
  return {
    contradictory: false,
    confidence: 0.0,
    reason: "no heuristic signal",
    source: "heuristic",
  };
}

/**
 * Extract standalone integers, decimals, and ISO-ish dates from a
 * string as a normalized set of numeric tokens.
 */
function extractNumbers(s: string): string[] {
  // Matches: integers, decimals (1.5, 1,000), ISO dates (2026-04-10),
  // US dates (04/10/2026), version strings (v0.7.1 → 0.7.1)
  const matches = s.match(/\b\d{1,5}(?:[.,/-]\d{1,5}){0,2}\b/g) || [];
  return matches.map((m) => m.replace(/,/g, ""));
}

/**
 * Return true if the string contains an explicit negation token.
 * Matches English contractions (didn't, won't, cannot, etc.) plus
 * bare negations (not, never, no).
 */
function hasNegation(s: string): boolean {
  return /\b(not|never|no|don['’]t|didn['’]t|won['’]t|cannot|can['’]t|wasn['’]t|isn['’]t|aren['’]t|weren['’]t|shouldn['’]t|couldn['’]t|wouldn['’]t)\b/i.test(
    s
  );
}

// =============================================================================
// Judge-based evaluation (v0.29.0 — the single-pair relation-array contract)
// =============================================================================

export type MergeRunSpec = Omit<JudgeRunInput, "consumer" | "sessionId" | "fallbackFromRunId">;
export type MergeEventSpec = Omit<JudgeEventInput, "runId">;

/**
 * One merge-gate evaluation, with its complete audit material. The caller owns
 * transaction placement (§J7): Phase 2 persists runs+events atomically WITH the
 * chosen mutation; Phase 3 persists them FIRST as a precondition. `aborted` is
 * terminal — standalone audit, no heuristic, no mutation, candidate deferred.
 */
export type MergeEvaluation =
  | { kind: "aborted"; runs: MergeRunSpec[] }
  | {
      kind: "decided";
      result: ContradictionResult;
      runs: MergeRunSpec[];
      events: MergeEventSpec[];
      /** Reject evidence bound to the FIRST (provider) run on fallback paths (t2 finding 2). */
      providerEvents?: MergeEventSpec[];
    };

const mkHeuristicRun = (): MergeRunSpec => ({
  lane: "heuristic",
  promptVersion: JUDGE_PROMPT_VERSION,
  newFactCount: 1,
  candidateCount: 1,
  outcome: "ok",
  entriesAdmitted: 1,
});

/**
 * Judge-first merge evaluation per the exhaustive failure matrix (§J7):
 *
 * - judge `ok` → the single-pair verdict decides (legacy object contract,
 *   permissive extraction, and confidence-defaulting are all GONE — a missing
 *   confidence is an invalid entry, not `0.5`).
 * - `unavailable|config|http|timeout|truncated|parse_reject` → failed provider
 *   run + linked audited heuristic run; the heuristic decides.
 * - `aborted` → terminal: no heuristic, no mutation.
 * - unconfigured → heuristic-only run (no provider run to pair).
 *
 * Argument order matches the legacy gate: `a` = existing text, `b` = new text.
 */
export async function evaluateMergeContradiction(
  resolution: JudgeResolution,
  a: string,
  b: string,
  minConfidence: number = CONTRADICTION_MIN_CONFIDENCE,
): Promise<MergeEvaluation> {
  const heuristicDecides = (providerRun: MergeRunSpec | null, providerEvents?: MergeEventSpec[]): MergeEvaluation => {
    const hres = heuristicContradictionCheck(a, b);
    const runs = providerRun ? [providerRun, mkHeuristicRun()] : [mkHeuristicRun()];
    return {
      kind: "decided",
      result: hres,
      runs,
      providerEvents,
      events: [{
        eventType: "verdict",
        newIdx: 0,
        oldIdx: 0,
        relation: hres.contradictory ? "contradiction" : null,
        confidence: hres.confidence,
        reasoningHead: hres.reason ?? null,
      }],
    };
  };

  if (resolution.status === "unconfigured") {
    return heuristicDecides(null);
  }
  if (resolution.status === "invalid") {
    return heuristicDecides({
      lane: "none",
      promptVersion: JUDGE_PROMPT_VERSION,
      newFactCount: 1,
      candidateCount: 1,
      outcome: "config",
    });
  }

  const judge: Judge = resolution.judge;
  const prompt = buildContradictionPrompt({
    newFacts: [b],
    existingSnippets: [a],
    minConfidence,
  });
  const providerBase = {
    lane: judge.descriptor.lane,
    model: judge.descriptor.model,
    endpoint: judge.descriptor.endpoint,
    promptVersion: prompt.promptVersion,
    newFactCount: 1,
    candidateCount: 1,
  } satisfies Partial<MergeRunSpec>;

  const res = await judge.judge({ system: prompt.system, user: prompt.user, schema: JUDGE_VERDICT_SCHEMA });

  if (!res.ok) {
    if (res.reason === "aborted") {
      return { kind: "aborted", runs: [{ ...providerBase, outcome: "aborted" }] };
    }
    return heuristicDecides({ ...providerBase, outcome: res.reason });
  }
  if (res.truncated) {
    return heuristicDecides({ ...providerBase, model: res.model, outcome: "truncated", responseSha256: hashContent(res.text) });
  }

  const sha = hashContent(res.text);
  const parsed = unwrapContradictionArray(extractJudgeJson(res.text));
  if (!Array.isArray(parsed)) {
    return heuristicDecides({ ...providerBase, model: res.model, outcome: "parse_reject", responseSha256: sha });
  }

  const batch = admitContradictionEntries(parsed, 1, 1);

  // Only a CLEAN single-pair response may decide (J5d; code-review t2 finding 1):
  // exactly one raw entry, one admitted, zero rejected/duplicate/inconsistent.
  // Any dirty non-empty response — a valid verdict wrapped in schema junk, repeats,
  // or inconsistent answers — is a parse_reject that hands the decision to the
  // AUDITED heuristic; treating it as decisive would let a polluted response
  // authorize destructive policy. Exact [] remains the legitimate no-relationship
  // result. The provider run keeps bounded per-entry reject evidence (t2 finding 2).
  const clean =
    parsed.length === 0 ||
    (parsed.length === 1 && batch.accepted.length === 1 &&
      batch.rejected === 0 && batch.duplicates === 0 && batch.inconsistent === 0);
  if (!clean) {
    const providerEvents: MergeEventSpec[] = [];
    // The admitted-but-DISCARDED proposals are audit facts too (t3 finding 2):
    // record each as a provider-bound verdict event — the run's parse_reject
    // outcome is what says none of them decided anything.
    for (const admitted of batch.accepted as { new_idx: number; old_idx: number; relation: string; confidence: number; reasoning: string }[]) {
      providerEvents.push({
        eventType: "verdict",
        newIdx: admitted.new_idx,
        oldIdx: admitted.old_idx,
        relation: admitted.relation,
        confidence: admitted.confidence,
        reasoningHead: admitted.reasoning,
      });
    }
    for (const { entry: rejectedEntry, reason } of batch.rejectedEntries) {
      providerEvents.push({
        eventType: "reject",
        reasonCode:
          reason === "invalid-relation" ? "invalid_relation"
          : reason === "invalid-reasoning" ? "invalid_reasoning"
          : reason === "placeholder-reasoning" ? "placeholder_reasoning"
          : reason === "invalid-confidence" ? "invalid_confidence"
          : "index_oob",
        evidenceHead: JSON.stringify(rejectedEntry),
      });
    }
    for (const { entry, repeats } of batch.duplicateEntries) {
      providerEvents.push({ eventType: "reject", reasonCode: "duplicate", evidenceHead: JSON.stringify({ repeats, entry }) });
    }
    for (const { entry, answers } of batch.inconsistentEntries) {
      providerEvents.push({ eventType: "reject", reasonCode: "inconsistent", evidenceHead: JSON.stringify({ answers, entry }) });
    }
    return heuristicDecides(
      {
        ...providerBase,
        model: res.model,
        outcome: "parse_reject",
        responseSha256: sha,
        entriesAdmitted: batch.accepted.length,
        entriesRejected: batch.rejected,
        entriesDuplicate: batch.duplicates,
        entriesInconsistent: batch.inconsistent,
      },
      providerEvents,
    );
  }

  const entry = batch.accepted[0] as
    | { relation: "same" | "update" | "contradiction"; confidence: number; reasoning: string }
    | undefined;

  const result: ContradictionResult = entry
    ? {
        contradictory: entry.relation === "contradiction",
        confidence: entry.confidence,
        reason: entry.reasoning,
        source: "llm",
      }
    : { contradictory: false, confidence: 0, reason: "no relationship found", source: "llm" };

  const events: MergeEventSpec[] = [];
  if (entry) {
    events.push({
      eventType: "verdict",
      newIdx: 0,
      oldIdx: 0,
      relation: entry.relation,
      confidence: entry.confidence,
      reasoningHead: entry.reasoning,
    });
  }
  for (const { entry: rejectedEntry, reason } of batch.rejectedEntries) {
    events.push({
      eventType: "reject",
      reasonCode:
        reason === "invalid-relation" ? "invalid_relation"
        : reason === "invalid-reasoning" ? "invalid_reasoning"
        : reason === "placeholder-reasoning" ? "placeholder_reasoning"
        : reason === "invalid-confidence" ? "invalid_confidence"
        : "index_oob",
      evidenceHead: JSON.stringify(rejectedEntry),
    });
  }

  return {
    kind: "decided",
    result,
    runs: [{
      ...providerBase,
      model: res.model,
      outcome: "ok",
      responseSha256: sha,
      entriesAdmitted: batch.accepted.length,
      entriesRejected: batch.rejected,
      entriesDuplicate: batch.duplicates,
      entriesInconsistent: batch.inconsistent,
    }],
    events,
  };
}

/**
 * Persist an evaluation's runs + events. THROWS on failure — the caller decides
 * what a failed audit means for its mutation (§J7: Phase 2 rolls the mutation
 * back; Phase 3 treats the classification as unavailable). A two-run evaluation
 * is a provider-failure→heuristic pair: the second run links to the first via
 * `fallback_from_run_id`, and events bind to the DECIDING (last) run.
 * Returns the deciding run's id.
 */
export function persistMergeEvaluation(
  db: Database,
  consumer: JudgeConsumer,
  sessionId: string | null,
  evaluation: { runs: MergeRunSpec[]; events?: MergeEventSpec[]; providerEvents?: MergeEventSpec[] },
): number {
  let previousRunId: number | null = null;
  let firstRunId: number | null = null;
  let decidingRunId = 0;
  for (const run of evaluation.runs) {
    decidingRunId = insertJudgeRun(db, {
      ...run,
      consumer,
      sessionId,
      fallbackFromRunId: previousRunId,
    });
    if (firstRunId === null) firstRunId = decidingRunId;
    previousRunId = decidingRunId;
  }
  // Reject evidence describes the PROVIDER response — bind it to the first run;
  // verdict events bind to the deciding (last) run.
  for (const ev of evaluation.providerEvents ?? []) {
    insertJudgeEvent(db, { runId: firstRunId ?? decidingRunId, ...ev });
  }
  for (const ev of evaluation.events ?? []) {
    insertJudgeEvent(db, { runId: decidingRunId, ...ev });
  }
  return decidingRunId;
}

/** Standalone best-effort persistence for terminal `aborted` evaluations. */
export function persistMergeEvaluationBestEffort(
  db: Database,
  consumer: JudgeConsumer,
  sessionId: string | null,
  evaluation: { runs: MergeRunSpec[] },
): void {
  let previousRunId: number | null = null;
  for (const run of evaluation.runs) {
    const id = insertJudgeRunBestEffort(db, {
      ...run,
      consumer,
      sessionId,
      fallbackFromRunId: previousRunId,
    });
    if (id === null) return;
    previousRunId = id;
  }
}

// =============================================================================
// Actionability
// =============================================================================

/**
 * Apply the `CONTRADICTION_MIN_CONFIDENCE` threshold to a
 * `ContradictionResult` — returns true iff the result claims a
 * contradiction AND meets the confidence floor.
 *
 * Callers use this to decide whether to block a merge. Keeping the
 * threshold check centralized means operators can tune via env var
 * without touching the merge code.
 */
export function isActionableContradiction(result: ContradictionResult): boolean {
  return (
    result.contradictory === true &&
    result.confidence >= CONTRADICTION_MIN_CONFIDENCE
  );
}
