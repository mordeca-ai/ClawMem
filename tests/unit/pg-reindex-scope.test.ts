/**
 * Regression guard for master-harness-vn4rz.7 pass D.
 *
 * src/pg/reindex.ts's own header promises that it reuses src/indexer.ts's helpers
 * verbatim "so a PG-indexed document and a sqlite-indexed document are parsed
 * identically and any parity gap is a WRITE-path gap, not a parse gap." The PARSE
 * helpers were reused. The SCOPE predicate — shouldExclude() — was not: reindex
 * globbed and wrote everything the pattern matched.
 *
 * The consequence was not abstract. The PG vault held 54 documents the sqlite
 * vault had always correctly refused, among them 36 SUPERSEDED ADRs out of
 * intelligence/decisions/_superseded/**, plus _reviews/**, the _cracks-*.md
 * scratch files across four game wikis, _VENDORED.md and _vitals-table.md — live,
 * retrievable, and indistinguishable at query time from current content. A reader
 * asking the vault about a decision could be handed the version that was
 * explicitly superseded.
 *
 * Two things are therefore under test, and the SECOND is the load-bearing one:
 *
 *   1. reindex skips the three exclusion classes (leading "_", dotted segment,
 *      EXCLUDED_DIRS member) and COUNTS what it skipped into the summary — a
 *      silently-dropped file being the same class of defect as the silently-
 *      swallowed parse error of master-harness-vn4rz.34;
 *
 *   2. the PG path's decision is shouldExclude's decision on EVERY input, not
 *      merely on the cases someone thought to enumerate. That is asserted
 *      differentially — the same fixture set driven through both — rather than by
 *      restating the rule, because a test that restates the rule drifts in
 *      lockstep with the copy it is supposed to be guarding against.
 */
import { describe, it, expect } from "bun:test";
import { EXCLUDED_DIRS, shouldExclude } from "../../src/indexer.ts";
import { noteOutOfScope, type ReindexStats } from "../../src/pg/reindex.ts";

const freshStats = (): Pick<ReindexStats, "skippedOutOfScope"> => ({ skippedOutOfScope: {} });

/**
 * Paths that MUST be skipped, each naming the real row it stands for. These are
 * not invented shapes: every one was pulled from the enumeration of the 54
 * polluting rows in the live `clawmem` database.
 */
const REAL_POLLUTION = [
  "intelligence/decisions/_superseded/0055-beads-jsonl-not-tracked.md",
  "intelligence/decisions/_reviews/0162-review.md",
  "library/fighting/sf6/wiki/_cracks-sf6.md",
  "library/fighting/vf5-revo/wiki/_cracks-vf5.md",
  "library/crpg/rdr2/wiki/_cracks-rdr2.md",
  "library/jrpg/ff7-rebirth/wiki/_cracks-ff7.md",
  "blazor/examples/_VENDORED.md",
  "notes/_vitals-table.md",
];

describe("reindexCollection's scope gate skips out-of-default-scope files", () => {
  it("skips a leading-underscore segment (ADR-0071 _superseded/ archive)", () => {
    const stats = freshStats();
    expect(noteOutOfScope(stats, "intelligence/decisions/_superseded/0055-old.md")).toBe(true);
    expect(Object.keys(stats.skippedOutOfScope)).toEqual([
      "intelligence/decisions/_superseded/0055-old.md",
    ]);
    expect(stats.skippedOutOfScope["intelligence/decisions/_superseded/0055-old.md"]).toBe(
      'out-of-default-scope segment: "_superseded"',
    );
  });

  it("skips a leading-underscore FILENAME, not only a directory", () => {
    // _cracks-sf6.md and _VENDORED.md are files, not dirs. A predicate that only
    // walked directory segments would have let both through.
    const stats = freshStats();
    expect(noteOutOfScope(stats, "library/fighting/sf6/wiki/_cracks-sf6.md")).toBe(true);
    expect(stats.skippedOutOfScope["library/fighting/sf6/wiki/_cracks-sf6.md"]).toBe(
      'out-of-default-scope segment: "_cracks-sf6.md"',
    );
  });

  it("skips a dotted segment", () => {
    const stats = freshStats();
    expect(noteOutOfScope(stats, ".obsidian/workspace.md")).toBe(true);
    expect(noteOutOfScope(stats, "notes/.hidden/draft.md")).toBe(true);
    expect(Object.keys(stats.skippedOutOfScope).length).toBe(2);
    expect(stats.skippedOutOfScope["notes/.hidden/draft.md"]).toBe(
      'out-of-default-scope segment: ".hidden"',
    );
  });

  it("skips an EXCLUDED_DIRS segment — every member, not a sampled few", () => {
    for (const dir of EXCLUDED_DIRS) {
      const stats = freshStats();
      const rel = `a/${dir}/b.md`;
      expect(noteOutOfScope(stats, rel)).toBe(true);
      expect(Object.keys(stats.skippedOutOfScope)).toEqual([rel]);
    }
  });

  it("does NOT skip an ordinary in-scope path, and records nothing for it", () => {
    const stats = freshStats();
    for (const rel of [
      "intelligence/decisions/0162-knowledge-substrate.md",
      "library/fighting/sf6/wiki/characters/ryu.md",
      "notes/README.md",
      "a/b/c/under_score/mid_file.md", // underscore NOT leading — in scope
    ]) {
      expect(noteOutOfScope(stats, rel)).toBe(false);
    }
    expect(stats.skippedOutOfScope).toEqual({});
  });

  it("COUNTS every skip into the summary, keyed by path, with a reason", () => {
    const stats = freshStats();
    const mixed = [...REAL_POLLUTION, "notes/keep-me.md", "intelligence/decisions/0162-keep.md"];
    let skipped = 0;
    for (const rel of mixed) if (noteOutOfScope(stats, rel)) skipped++;

    expect(skipped).toBe(REAL_POLLUTION.length);
    expect(Object.keys(stats.skippedOutOfScope).sort()).toEqual([...REAL_POLLUTION].sort());
    for (const why of Object.values(stats.skippedOutOfScope)) {
      expect(why).toMatch(/^out-of-default-scope segment: ".+"$/);
    }
  });
});

describe("the PG scope decision cannot drift from shouldExclude", () => {
  /**
   * THE ANTI-DRIFT ASSERTION. shouldExclude is the oracle; noteOutOfScope's
   * boolean must agree with it on every input. Deliberately expressed as
   * agreement-with-the-oracle rather than as expected literals, so that a future
   * change to the exclusion rule in indexer.ts either propagates to both paths or
   * fails HERE — which is precisely what did not happen when the parse helpers
   * were reused and this predicate was not.
   */
  const FIXTURES = [
    ...REAL_POLLUTION,
    ...[...EXCLUDED_DIRS].map(d => `a/${d}/b.md`),
    ...[...EXCLUDED_DIRS].map(d => `${d}.md`),
    ".git/config.md",
    "a/.git/config.md",
    "_superseded/x.md",
    "x/_superseded/y/z.md",
    "_.md",
    "._.md",
    "a/./b.md",
    "notes/README.md",
    "notes/under_score.md",
    "a/b/c.md",
    "a/b/c/d/e/f/deeply-nested.md",
    "Vendor/b.md", // case-sensitive: NOT the excluded "vendor"
    "a/vendors/b.md", // substring, not a whole segment
    "a/my_vendor/b.md",
    "PRIVATE/b.md",
    "_PRIVATE/b.md",
    "instructional-media/notes/sound-design.md",
    "memory/2026-09-02.md",
  ];

  it("agrees with shouldExclude on every fixture", () => {
    for (const rel of FIXTURES) {
      const stats = freshStats();
      expect({ rel, skipped: noteOutOfScope(stats, rel) }).toEqual({
        rel,
        skipped: shouldExclude(rel),
      });
      // and the summary is written if and only if the file was skipped
      expect(Object.keys(stats.skippedOutOfScope).length).toBe(shouldExclude(rel) ? 1 : 0);
    }
  });

  it("agrees with shouldExclude on generated segment combinations", () => {
    // Cross-product of a segment alphabet against three positions. Cheap, and it
    // covers orderings nobody enumerated by hand.
    const alphabet = ["a", "_x", ".x", "vendor", "dist", "_PRIVATE", "x_y", "x.y", ""];
    for (const p of alphabet) {
      for (const q of alphabet) {
        for (const r of ["file.md", "_file.md", ".file.md"]) {
          const rel = `${p}/${q}/${r}`;
          const stats = freshStats();
          expect({ rel, skipped: noteOutOfScope(stats, rel) }).toEqual({
            rel,
            skipped: shouldExclude(rel),
          });
        }
      }
    }
  });
});
