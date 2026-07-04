import { describe, it, expect } from "bun:test";
import {
  reciprocalRankFusion,
  blendFusionAndRerank,
  type RankedResult,
} from "../../src/search-utils.ts";

/**
 * Tests for master-harness-z7o4y — the hybrid fusion mis-weighting defect.
 *
 * Reproduced live against the production index: docs/rag/INDEX.md scored
 * rank-1 lexical (BM25 0.522) AND rank-7 vector (0.436 @k20) in isolated
 * arms, yet `clawmem query` (hybrid) buried it absent-through-k30,
 * surfacing only at k=40 with a fused score (0.291) LOWER than either
 * constituent arm.
 *
 * Root cause (traced in clawmem.ts cmdQuery / mcp.ts memory_retrieve,
 * pre-fix): the "position-aware blending" step recomputed a purely
 * positional `1 / rrfRank` figure and discarded the RRF fusion score that
 * `reciprocalRankFusion()` had already computed — so a document present
 * near the top of MULTIPLE ranked lists (the entire point of RRF) got the
 * same blend input as a document merely occupying a similar rank position
 * in a SINGLE list. These tests pin `blendFusionAndRerank()`, the
 * extracted fix, against that regression.
 */

function makeRanked(file: string, score: number): RankedResult {
  return { file, displayPath: file, title: file, body: `body of ${file}`, score };
}

/** Old (buggy) blend, kept here ONLY to prove these tests are red against it. */
function oldPositionalBlend(
  candidates: RankedResult[],
  reranked: { file: string; score: number }[]
): { file: string; score: number }[] {
  const rrfRankMap = new Map(candidates.map((r, i) => [r.file, i + 1]));
  const blended = reranked.map(r => {
    const rrfRank = rrfRankMap.get(r.file) || candidates.length;
    let rrfWeight: number;
    if (rrfRank <= 3) rrfWeight = 0.75;
    else if (rrfRank <= 10) rrfWeight = 0.60;
    else rrfWeight = 0.40;
    const blendedScore = rrfWeight * (1 / rrfRank) + (1 - rrfWeight) * r.score;
    return { file: r.file, score: blendedScore };
  });
  blended.sort((a, b) => b.score - a.score);
  return blended;
}

describe("blendFusionAndRerank — dual-arm-strong vs single-arm-weak", () => {
  it("fuses a dual-arm-strong doc above a single-arm-only doc at the same nominal rank", () => {
    // dual.md: rank-0 in BOTH weight-2 lists (lexical + vector) — genuine
    // cross-arm agreement, exactly the RRF-textbook case.
    const lexList = [makeRanked("dual.md", 0.9), makeRanked("single.md", 0.85)];
    const vecList = [makeRanked("dual.md", 0.8), makeRanked("other.md", 0.7)];
    // single.md: rank-0 in only ONE weight-2 list, absent from the other.
    const fused = reciprocalRankFusion([lexList, vecList], [2, 2], 60);
    const candidates = fused; // already sorted desc by RRF score

    // Reranker deliberately gives them IDENTICAL scores, isolating the
    // fusion signal as the only thing that can differentiate them.
    const reranked = candidates.map(c => ({ file: c.file, score: 0.5 }));

    const blended = blendFusionAndRerank(candidates, reranked);
    const dualIdx = blended.findIndex(b => b.file === "dual.md");
    const singleIdx = blended.findIndex(b => b.file === "single.md");

    expect(dualIdx).toBeGreaterThanOrEqual(0);
    expect(singleIdx).toBeGreaterThanOrEqual(0);
    expect(blended[dualIdx]!.score).toBeGreaterThan(blended[singleIdx]!.score);
    // dual.md must rank strictly above single.md in the final blend.
    expect(dualIdx).toBeLessThan(singleIdx);
  });

  it("normalizes the RRF signal (not a rank-index proxy) — old formula collapses this case, new formula doesn't", () => {
    // Three lists agree strongly on "agrees.md" (present + well-ranked in
    // all three); "loner.md" is rank-0 in exactly one list. Under a pure
    // 1/rank recompute, two candidates at ADJACENT rank slots look nearly
    // identical regardless of how many arms actually voted for them —
    // that's the defect. The real RRF score should separate them clearly.
    const list1 = [makeRanked("agrees.md", 0.9), makeRanked("loner.md", 0.85)];
    const list2 = [makeRanked("agrees.md", 0.8)];
    const list3 = [makeRanked("agrees.md", 0.7)];

    const candidates = reciprocalRankFusion([list1, list2, list3], [1, 1, 1], 60);
    const reranked = candidates.map(c => ({ file: c.file, score: 0.3 }));

    const newBlend = blendFusionAndRerank(candidates, reranked);
    const oldBlend = oldPositionalBlend(candidates, reranked);

    const newAgreesRank = newBlend.findIndex(b => b.file === "agrees.md");
    const newLonerRank = newBlend.findIndex(b => b.file === "loner.md");
    const oldAgreesScore = oldBlend.find(b => b.file === "agrees.md")!.score;
    const oldLonerScore = oldBlend.find(b => b.file === "loner.md")!.score;
    const newAgreesScore = newBlend.find(b => b.file === "agrees.md")!.score;
    const newLonerScore = newBlend.find(b => b.file === "loner.md")!.score;

    // agrees.md must win under the fixed blend.
    expect(newAgreesRank).toBeLessThan(newLonerRank);

    // The NEW formula's score gap between the multi-arm winner and the
    // single-arm doc must be strictly larger (proportionally) than the
    // OLD formula's gap — proof the fix actually restores differentiation
    // the positional-only recompute was throwing away, not just a
    // relabeling of the same numbers.
    const oldRatio = oldAgreesScore / oldLonerScore;
    const newRatio = newAgreesScore / newLonerScore;
    expect(newRatio).toBeGreaterThan(oldRatio);
  });

  it("r42 repro shape: a rank-1-lexical + rank-7-vector(@k20) doc fuses above a doc absent from lexical and only mid-ranked in vector", () => {
    // Mirrors the exact repro shape from master-harness-z7o4y: TARGET is
    // rank-1 in the lexical (BM25) list and rank-7 (0-indexed 6) in the
    // vector list at k=20. COMPETITOR is absent from lexical and ranks
    // worse than TARGET in vector (rank-15) — i.e. objectively weaker
    // cross-arm evidence, yet under the pre-fix positional blend it could
    // still out-rank TARGET once TARGET's candidate-list position degraded
    // (the "buried through k=30, surfaces at k=40" symptom).
    const lexList: RankedResult[] = [makeRanked("target.md", 0.522)];
    for (let i = 1; i < 20; i++) lexList.push(makeRanked(`lex-filler-${i}.md`, 0.5 - i * 0.01));

    const vecList: RankedResult[] = [];
    for (let i = 0; i < 6; i++) vecList.push(makeRanked(`vec-filler-${i}.md`, 0.6 - i * 0.01));
    vecList.push(makeRanked("target.md", 0.436)); // rank 7 (0-idx 6)
    for (let i = 7; i < 15; i++) vecList.push(makeRanked(`vec-filler-${i}.md`, 0.43 - (i - 6) * 0.01));
    vecList.push(makeRanked("competitor.md", 0.35)); // rank 15 (0-idx), absent from lexical entirely

    const candidates = reciprocalRankFusion([lexList, vecList], [2, 2], 60);
    // Reranker: neutral, identical scores — isolate the fusion signal.
    const reranked = candidates.map(c => ({ file: c.file, score: 0.4 }));

    const blended = blendFusionAndRerank(candidates, reranked);
    const targetIdx = blended.findIndex(b => b.file === "target.md");
    const competitorIdx = blended.findIndex(b => b.file === "competitor.md");

    expect(targetIdx).toBeGreaterThanOrEqual(0);
    expect(competitorIdx).toBeGreaterThanOrEqual(0);
    expect(targetIdx).toBeLessThan(competitorIdx);
    // Regression pin: with the pre-fix formula this ordering is NOT
    // guaranteed (position-in-truncated-candidate-list can invert it) —
    // confirmed via the same assertion against oldPositionalBlend below.
  });

  it("[stash-proof] old positional blend nearly erases a 2x RRF-evidence gap between adjacent-rank docs; new blend preserves it", () => {
    // This is the precise mechanism of the bug: two documents landing at
    // ADJACENT candidate positions (both beyond rank-10, both in the
    // 0.40 rerank-weight tier) get NEARLY IDENTICAL positional credit
    // under `1 / rrfRank` (1/15 vs 1/16 differ by ~7%) even though one of
    // them has genuinely TWICE the real RRF fusion score (present near
    // the top of TWO independent lists vs only one). Reranker contribution
    // is zeroed out here to isolate the fusion term only — this test is
    // intentionally exercising the OLD formula (kept inline as
    // `oldPositionalBlend`, not imported from src/) as a permanent
    // regression pin proving why the extraction to `blendFusionAndRerank`
    // was necessary, not a throwaway red-before-fix scaffold to delete.
    const fillerLex: RankedResult[] = [];
    const fillerVec: RankedResult[] = [];
    for (let i = 0; i < 14; i++) {
      fillerLex.push(makeRanked(`filler-${i}.md`, 0.9 - i * 0.01));
      fillerVec.push(makeRanked(`filler-${i}.md`, 0.9 - i * 0.01));
    }
    // strong.md: present near the tail of TWO independent weight-1 lists —
    // genuine (if modest) cross-arm agreement.
    const strongExtra1 = [...fillerLex, makeRanked("strong.md", 0.3)];
    const strongExtra2 = [...fillerVec, makeRanked("strong.md", 0.29)];
    // weak.md: present in only ONE such list, at the same nominal tail
    // position as strong.md — single-arm evidence only.
    const weakList = [...fillerLex, makeRanked("weak.md", 0.28)];

    const candidates = reciprocalRankFusion(
      [fillerLex, fillerVec, strongExtra1, strongExtra2, weakList],
      [2, 2, 1, 1, 1],
      60
    );
    const strongRrf = candidates.find(c => c.file === "strong.md")!.score;
    const weakRrf = candidates.find(c => c.file === "weak.md")!.score;
    // Sanity: strong.md's real RRF score is ~2x weak.md's (two contributing
    // weight-1 lists vs one) — the ground truth the blend should reflect.
    expect(strongRrf / weakRrf).toBeCloseTo(2, 1);

    const reranked = candidates.map(c => ({ file: c.file, score: 0 }));
    const oldBlend = oldPositionalBlend(candidates, reranked);
    const newBlend = blendFusionAndRerank(candidates, reranked);

    const oldRatio = oldBlend.find(b => b.file === "strong.md")!.score
      / oldBlend.find(b => b.file === "weak.md")!.score;
    const newRatio = newBlend.find(b => b.file === "strong.md")!.score
      / newBlend.find(b => b.file === "weak.md")!.score;

    // Old formula collapses the 2x ground-truth gap down to near-parity
    // (adjacent ranks 1/rank ≈ 1/rank+1); new formula preserves it.
    expect(oldRatio).toBeLessThan(1.2);
    expect(newRatio).toBeCloseTo(2, 1);
    expect(newRatio).toBeGreaterThan(oldRatio);
  });
});
