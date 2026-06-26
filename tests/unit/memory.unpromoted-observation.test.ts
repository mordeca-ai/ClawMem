import { describe, test, expect, afterEach } from "bun:test";
import { applyCompositeScoring, type EnrichedResult } from "../../src/memory.ts";

/**
 * Un-promoted-observation down-weight (ADR-0112 invariant 3, master-harness-4vhh).
 *
 * Asserts the rank PENALTY applied in applyCompositeScoring:
 *   - un-promoted agent-generated observations are down-ranked (× weight),
 *   - pinned (promoted) observations are UNAFFECTED (promotion wins),
 *   - non-observation docs are UNAFFECTED,
 *   - invalidated observations are left alone (handled by invalidation, not buried).
 *
 * The knob is read from CLAWMEM_UNPROMOTED_OBSERVATION_WEIGHT at scoring time; we
 * compare a baseline pass (weight=1.0, feature off) against a penalized pass
 * (weight=0.5). Single-element arrays keep co-activation/sort effects out so the
 * only delta between the two passes is the penalty multiplier itself.
 */

const ENV_KEY = "CLAWMEM_UNPROMOTED_OBSERVATION_WEIGHT";

function makeResult(overrides: Partial<EnrichedResult>): EnrichedResult {
  return {
    filepath: "clawmem://test/doc.md",
    displayPath: "test/doc.md",
    title: "Doc",
    score: 0.5,
    contentType: "note",
    modifiedAt: "2026-06-20T00:00:00.000Z",
    accessCount: 0,
    confidence: 0.5,
    qualityScore: 0.5,
    pinned: false,
    observationType: null,
    invalidatedAt: null,
    context: null,
    hash: "abc123",
    docid: "abc123",
    collectionName: "test",
    bodyLength: 500,
    source: "fts",
    duplicateCount: 1,
    revisionCount: 1,
    ...overrides,
  };
}

/** Score a single fixture under a given weight setting. */
function scoreUnder(weight: string | undefined, fixture: EnrichedResult): number {
  if (weight === undefined) delete process.env[ENV_KEY];
  else process.env[ENV_KEY] = weight;
  return applyCompositeScoring([fixture], "what is the deploy port")[0]!.compositeScore;
}

describe("un-promoted-observation down-weight", () => {
  afterEach(() => {
    delete process.env[ENV_KEY];
  });

  test("default knob is 0.5 (conservative penalty, not exclude)", () => {
    // Feature-off baseline vs default → observation halved.
    const obs = makeResult({ contentType: "observation" });
    const off = scoreUnder("1.0", obs);
    const def = scoreUnder(undefined, obs); // default 0.5
    expect(def).toBeCloseTo(off * 0.5, 5);
    expect(def).toBeGreaterThan(0); // penalty, never hard-zero
  });

  test("un-promoted observation via content_type is down-ranked", () => {
    const obs = makeResult({ contentType: "observation" });
    const baseline = scoreUnder("1.0", obs);
    const penalized = scoreUnder("0.5", obs);
    expect(penalized).toBeCloseTo(baseline * 0.5, 5);
    expect(penalized).toBeLessThan(baseline);
  });

  test("un-promoted observation via observation_type is down-ranked", () => {
    const obs = makeResult({ contentType: "note", observationType: "ambient" });
    const baseline = scoreUnder("1.0", obs);
    const penalized = scoreUnder("0.5", obs);
    expect(penalized).toBeCloseTo(baseline * 0.5, 5);
  });

  test("un-promoted observation via episodic collection is down-ranked", () => {
    const obs = makeResult({ contentType: "note", collectionName: "episodic-handoffs" });
    const baseline = scoreUnder("1.0", obs);
    const penalized = scoreUnder("0.5", obs);
    expect(penalized).toBeCloseTo(baseline * 0.5, 5);
  });

  test("PINNED (promoted) observation is UNAFFECTED — promotion wins", () => {
    const obs = makeResult({ contentType: "observation", pinned: true });
    const baseline = scoreUnder("1.0", obs);
    const penalized = scoreUnder("0.5", obs);
    expect(penalized).toBeCloseTo(baseline, 5);
  });

  test("normal note (non-observation) is UNAFFECTED", () => {
    const note = makeResult({ contentType: "note" });
    const baseline = scoreUnder("1.0", note);
    const penalized = scoreUnder("0.5", note);
    expect(penalized).toBeCloseTo(baseline, 5);
  });

  test("decision/research content is UNAFFECTED", () => {
    const decision = makeResult({ contentType: "decision" });
    const baseline = scoreUnder("1.0", decision);
    const penalized = scoreUnder("0.5", decision);
    expect(penalized).toBeCloseTo(baseline, 5);
  });

  test("INVALIDATED observation is left alone (not double-penalized)", () => {
    const obs = makeResult({ contentType: "observation", invalidatedAt: "2026-06-21T00:00:00.000Z" });
    const baseline = scoreUnder("1.0", obs);
    const penalized = scoreUnder("0.5", obs);
    expect(penalized).toBeCloseTo(baseline, 5);
  });

  test("weight=1.0 disables the penalty entirely", () => {
    const obs = makeResult({ contentType: "observation" });
    const off = scoreUnder("1.0", obs);
    const noEnvButDefault = scoreUnder(undefined, obs);
    // off (1.0) should be strictly greater than the default-penalized score.
    expect(off).toBeGreaterThan(noEnvButDefault);
  });
});
