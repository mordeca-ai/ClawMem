import { describe, test, expect } from "bun:test";
import { physicalPathKey, reciprocalRankFusion, uniquePhysicalResults, type RankedResult } from "../../src/search-utils.ts";

function makeResult(file: string, score: number): RankedResult {
  return { file, displayPath: file, title: file, body: "", score };
}

describe("reciprocalRankFusion", () => {
  const roots = { docs: "/vault/library/reference/documentation", canon: "/vault" };
  const duplicate = "clawmem://canon/library/reference/documentation/postgres/rrf.md";
  const original = "clawmem://docs/postgres/rrf.md";

  test("one source indexed in two collections gets one vote and leaves N distinct slots", () => {
    const rows = [
      makeResult(original, 1), makeResult(duplicate, 0.9),
      makeResult("clawmem://docs/second.md", 0.8), makeResult("clawmem://docs/third.md", 0.7),
    ];
    const fused = reciprocalRankFusion([rows], [1], 60, roots);
    expect(fused).toHaveLength(3);
    expect(fused.map(r => physicalPathKey(r.file, roots))).toEqual([
      "/vault/library/reference/documentation/postgres/rrf.md",
      "/vault/library/reference/documentation/second.md",
      "/vault/library/reference/documentation/third.md",
    ]);
    expect(fused[0]!.score).toBeCloseTo(0.05 + 1 / 61);
    expect(fused[1]!.score).toBeCloseTo(0.02 + 1 / 62);
    expect(uniquePhysicalResults(rows.map(r => ({ filepath: r.file })), 3, roots)).toHaveLength(3);
    expect(uniquePhysicalResults(rows.map(r => ({ filepath: r.file })), 0, roots)).toEqual([]);
  });

  test("single-collection ranking is unchanged", () => {
    const scoped = [makeResult(original, 1), makeResult("clawmem://docs/second.md", 0.8)];
    expect(reciprocalRankFusion([scoped], [1], 60, roots))
      .toEqual(reciprocalRankFusion([scoped], [1], 60, {}));
  });

  test("the best-ranked collection copy represents a cross-arm source", () => {
    const fused = reciprocalRankFusion([
      [makeResult("clawmem://docs/other.md", 1), makeResult(original, 0.8)],
      [makeResult(duplicate, 1)],
    ], [1, 1], 60, roots);
    expect(fused).toHaveLength(2);
    expect(fused.find(r => physicalPathKey(r.file, roots).endsWith("postgres/rrf.md"))?.file)
      .toBe(duplicate);
  });

  test("merges two ranked lists", () => {
    const list1 = [makeResult("a.md", 1.0), makeResult("b.md", 0.8)];
    const list2 = [makeResult("b.md", 1.0), makeResult("c.md", 0.6)];

    const result = reciprocalRankFusion([list1, list2], [1, 1]);
    expect(result.length).toBe(3);
    // b.md appears in both → highest score
    expect(result[0]!.file).toBe("b.md");
  });

  test("respects weights", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const list2 = [makeResult("b.md", 1.0)];

    const result = reciprocalRankFusion([list1, list2], [10, 1]);
    // a.md has weight 10, b.md has weight 1
    expect(result[0]!.file).toBe("a.md");
  });

  test("throws on weight/list length mismatch", () => {
    const list1 = [makeResult("a.md", 1.0)];
    expect(() => reciprocalRankFusion([list1], [1, 2])).toThrow("must match");
  });

  test("handles empty lists", () => {
    const result = reciprocalRankFusion([], []);
    expect(result).toHaveLength(0);
  });

  test("handles empty weights (defaults to 1)", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const result = reciprocalRankFusion([list1], []);
    expect(result).toHaveLength(1);
  });

  test("single list passthrough", () => {
    const list1 = [makeResult("a.md", 1.0), makeResult("b.md", 0.5)];
    const result = reciprocalRankFusion([list1], [1]);
    expect(result).toHaveLength(2);
    expect(result[0]!.file).toBe("a.md");
  });

  test("sanitizes NaN weights to 1", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const list2 = [makeResult("b.md", 1.0)];
    const result = reciprocalRankFusion([list1, list2], [NaN, 1]);
    expect(result).toHaveLength(2);
    // NaN weight becomes 1, so both lists have equal weight
    expect(result.every(r => Number.isFinite(r.score))).toBe(true);
  });

  test("sanitizes negative weights to 1", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const result = reciprocalRankFusion([list1], [-5]);
    expect(result).toHaveLength(1);
    expect(result[0]!.score).toBeGreaterThan(0);
  });

  test("skips zero-weight lists", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const list2 = [makeResult("b.md", 1.0)];
    const result = reciprocalRankFusion([list1, list2], [1, 0]);
    // b.md from zero-weight list should not appear
    expect(result).toHaveLength(1);
    expect(result[0]!.file).toBe("a.md");
  });

  test("sanitizes invalid k to default 60", () => {
    const list1 = [makeResult("a.md", 1.0)];
    const result = reciprocalRankFusion([list1], [1], NaN);
    expect(result).toHaveLength(1);
    expect(Number.isFinite(result[0]!.score)).toBe(true);
  });
});
