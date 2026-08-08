import { describe, it, expect } from "bun:test";

/**
 * RankBreakdown contract (v0.36.0 memory_rank substrate): the explain option must
 * (1) change NOTHING about scoring — score-identical with and without it;
 * (2) capture factors from the SAME computation, so the documented identity
 *     (weightedBlend × quality × length × freq × canonical + pinBoost) × coActivation
 *     reproduces compositeScore exactly (not a re-derivation that could drift);
 * (3) reflect the applied semantics — the 0.3 length floor, the 1.0 pin cap, the
 *     recency-intent weight switch, and the co-activation stage's late update.
 */

import {
  applyCompositeScoring,
  RECENCY_WEIGHTS,
  type EnrichedResult,
  type CoActivationFn,
} from "../../src/memory.ts";

const NOW = new Date("2026-08-06T12:00:00.000Z");
const FRESH = "2026-08-06T11:00:00.000Z";
const OLD = "2026-02-01T00:00:00.000Z";

let seq = 0;
function mkResult(overrides: Partial<EnrichedResult> = {}): EnrichedResult {
  seq += 1;
  const path = overrides.displayPath ?? `user/doc-${seq}.md`;
  return {
    filepath: `clawmem://${path}`,
    displayPath: path,
    title: `Doc ${seq}`,
    score: 0.5,
    body: "body text",
    contentType: "note",
    modifiedAt: FRESH,
    authoredAt: null,
    accessCount: 0,
    confidence: 0.5,
    qualityScore: 0.5,
    pinned: false,
    context: null,
    hash: `hash-${seq}`,
    docid: `docid-${seq}`,
    collectionName: "user",
    bodyLength: 500,
    source: "fts",
    lastAccessedAt: null,
    duplicateCount: 1,
    revisionCount: 1,
    ...overrides,
  };
}

function reproduce(b: NonNullable<ReturnType<typeof applyCompositeScoring>[number]["rankBreakdown"]>): number {
  return (
    b.weightedBlend * b.qualityMultiplier * b.lengthFactor *
    b.frequencyBoostMultiplier * b.canonicalMultiplier + b.pinBoost
  ) * b.coActivationMultiplier;
}

describe("RankBreakdown (applyCompositeScoring explain)", () => {
  it("explain off attaches no breakdown", () => {
    const scored = applyCompositeScoring([mkResult()], "some query", undefined, { now: NOW });
    expect(scored[0]!.rankBreakdown).toBeUndefined();
  });

  it("explain changes no score and no ordering", () => {
    const fixtures = () => [
      mkResult({ score: 0.9, pinned: true, contentType: "decision" }),
      mkResult({ score: 0.7, bodyLength: 4000 }),
      mkResult({ score: 0.6, modifiedAt: OLD, accessCount: 12, lastAccessedAt: FRESH }),
      mkResult({ score: 0.4, revisionCount: 3, duplicateCount: 2 }),
    ];
    const coFn: CoActivationFn = (path) =>
      path.endsWith("doc-999.md") ? [] : [{ path: "user/never-matches.md", count: 3 }];
    const plain = applyCompositeScoring(fixtures(), "topic query", coFn, { now: NOW });
    seq -= 4; // regenerate identical paths/docids for the second pass
    const explained = applyCompositeScoring(fixtures(), "topic query", coFn, { now: NOW, explain: true });
    expect(explained.map(r => r.compositeScore)).toEqual(plain.map(r => r.compositeScore));
    expect(explained.map(r => r.displayPath)).toEqual(plain.map(r => r.displayPath));
    for (const r of explained) expect(r.rankBreakdown).toBeDefined();
  });

  it("captured factors reproduce compositeScore exactly, and finalComposite stays in sync", () => {
    const results = [
      mkResult({ score: 0.9, pinned: true, contentType: "decision", qualityScore: 1.0 }),
      mkResult({ score: 0.7, bodyLength: 6000 }),
      mkResult({ score: 0.5, modifiedAt: OLD, confidence: 0.9 }),
      mkResult({ score: 0.3, revisionCount: 4 }),
    ];
    const scored = applyCompositeScoring(results, "topic query", undefined, { now: NOW, explain: true });
    for (const r of scored) {
      const b = r.rankBreakdown!;
      expect(Math.abs(reproduce(b) - r.compositeScore)).toBeLessThan(1e-9);
      expect(b.finalComposite).toBe(r.compositeScore);
    }
  });

  it("length floor: an extreme body reports lengthFactor 0.3 with the floor flag", () => {
    const scored = applyCompositeScoring(
      [mkResult({ score: 0.8, bodyLength: 2_000_000 })],
      "topic query", undefined, { now: NOW, explain: true }
    );
    const b = scored[0]!.rankBreakdown!;
    expect(b.lengthFloorApplied).toBe(true);
    expect(b.lengthFactor).toBe(0.3);
    expect(Math.abs(reproduce(b) - scored[0]!.compositeScore)).toBeLessThan(1e-9);
  });

  it("pin cap: pinBoost is the post-cap delta — NEGATIVE when the cap clamps a >1.0 score down", () => {
    // Engineered above-1.0 pre-pin score: search 1.0, fresh, quality 1.0 (×1.3) pushes the
    // pre-pin composite past 1.0, and min(1.0, adjusted + 0.3) CLAMPS it to 1.0 — for such
    // docs the pin branch is clamp-to-1.0, not a boost (an unpinned twin would score
    // HIGHER). The breakdown must report that truthfully as a negative applied delta.
    // (Recorded as D-lane material in RANKING-DEFECT-HANDOFF-2026-08-06.md §Addendum.)
    const scored = applyCompositeScoring(
      [mkResult({ score: 1.0, pinned: true, contentType: "decision", qualityScore: 1.0, accessCount: 5, lastAccessedAt: FRESH })],
      "topic query", undefined, { now: NOW, explain: true }
    );
    const b = scored[0]!.rankBreakdown!;
    expect(scored[0]!.compositeScore).toBe(1.0);
    expect(b.pinBoost).toBeLessThan(0);
    expect(Math.abs(reproduce(b) - 1.0)).toBeLessThan(1e-9);
  });

  it("pin boost: the full +0.3 applies when the pre-pin score sits below the cap headroom", () => {
    const scored = applyCompositeScoring(
      [mkResult({ score: 0.2, pinned: true, modifiedAt: OLD })],
      "topic query", undefined, { now: NOW, explain: true }
    );
    const b = scored[0]!.rankBreakdown!;
    expect(b.pinBoost).toBeCloseTo(0.3, 10);
    expect(Math.abs(reproduce(b) - scored[0]!.compositeScore)).toBeLessThan(1e-9);
  });

  it("identity holds for a NEGATIVE composite (unconstrained stored confidence flips the length-branch selection)", () => {
    // Nothing constrains documents.confidence at the schema or writer; a strongly
    // negative stored value drives the blend — and the whole composite — negative.
    // With a huge body, max(a·0.3, a·lenFactor) then selects the SCALED branch even
    // though lenFactor < 0.3 (signs flip the comparison), so the recorded factor must
    // be the branch actually taken or the identity breaks.
    const scored = applyCompositeScoring(
      [mkResult({ score: 0, modifiedAt: OLD, confidence: -20, bodyLength: 2_000_000 })],
      "topic query", undefined, { now: NOW, explain: true }
    );
    const b = scored[0]!.rankBreakdown!;
    expect(scored[0]!.compositeScore).toBeLessThan(0);
    expect(b.lengthFloorApplied).toBe(false);
    expect(b.lengthFactor).toBeLessThan(0.3);
    expect(Math.abs(reproduce(b) - scored[0]!.compositeScore)).toBeLessThan(1e-12);
  });

  it("recency intent switches to RECENCY_WEIGHTS and flags the type-priority resort", () => {
    const scored = applyCompositeScoring(
      [mkResult({ score: 0.6 })],
      "what did we do last session", undefined, { now: NOW, explain: true }
    );
    const b = scored[0]!.rankBreakdown!;
    expect(b.recencyIntent).toBe(true);
    expect(b.typePriorityResort).toBe(true);
    expect(b.weights).toEqual(RECENCY_WEIGHTS);
  });

  it("co-activation stage updates the boosted doc's multiplier and finalComposite", () => {
    // Doc A dominates (top quartile of 4 = 1 doc); coFn names B as A's partner with
    // count 10 → multiplier 1 + min(10/10, 0.15) = 1.15 on B only.
    const a = mkResult({ score: 0.95, displayPath: "user/coact-a.md" });
    const b = mkResult({ score: 0.4, displayPath: "user/coact-b.md" });
    const c = mkResult({ score: 0.35, displayPath: "user/coact-c.md" });
    const d = mkResult({ score: 0.3, displayPath: "user/coact-d.md" });
    const coFn: CoActivationFn = (path) =>
      path === "user/coact-a.md" ? [{ path: "user/coact-b.md", count: 10 }] : [];
    const scored = applyCompositeScoring([a, b, c, d], "topic query", coFn, { now: NOW, explain: true });
    const boosted = scored.find(r => r.displayPath === "user/coact-b.md")!;
    const bb = boosted.rankBreakdown!;
    expect(bb.coActivationMultiplier).toBeCloseTo(1.15, 10);
    expect(bb.finalComposite).toBe(boosted.compositeScore);
    expect(Math.abs(reproduce(bb) - boosted.compositeScore)).toBeLessThan(1e-9);
    const unboosted = scored.find(r => r.displayPath === "user/coact-c.md")!;
    expect(unboosted.rankBreakdown!.coActivationMultiplier).toBe(1);
  });
});
