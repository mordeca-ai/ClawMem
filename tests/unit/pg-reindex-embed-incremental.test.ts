/**
 * EMBED-INCREMENTAL decision for the PG documents tier (master-harness-vn4rz.86).
 *
 * THE DEFECT. reindexCollection queued every fragment of every document whose
 * hash was new to the in-process `hashesDone` set and never asked the vault
 * whether it already held vectors for that hash, so every agents-skills tick
 * re-embedded the whole collection (355 docs / ~16.4k fragments) on a one-file
 * delta. The fix skips a hash only when the stored set is COMPLETE; this file
 * pins that rule without a database. The end-to-end behaviour is in
 * tests/integration/pg-reindex-embed-incremental.test.ts.
 */
import { describe, it, expect } from "bun:test";
import { formatReindexSummaryLine, hashFullyEmbedded } from "../../src/pg/reindex.ts";

describe("hashFullyEmbedded — skip only a complete stored vector set", () => {
  it("skips when rows === fragment count and seqs are 0..n-1", () => {
    expect(hashFullyEmbedded({ rows: 5, maxSeq: 4 }, 5)).toBe(true);
    expect(hashFullyEmbedded({ rows: 1, maxSeq: 0 }, 1)).toBe(true);
  });

  it("re-embeds a hash the vault holds no vectors for", () => {
    expect(hashFullyEmbedded(undefined, 3)).toBe(false);
  });

  it("re-embeds a PARTIAL set (an earlier embed failed mid-document)", () => {
    expect(hashFullyEmbedded({ rows: 2, maxSeq: 1 }, 5)).toBe(false);
    // Same count, but a hole at seq 0 and a stray row past the end.
    expect(hashFullyEmbedded({ rows: 5, maxSeq: 5 }, 5)).toBe(false);
  });

  it("re-embeds when the splitter now cuts the document into a different count", () => {
    expect(hashFullyEmbedded({ rows: 6, maxSeq: 5 }, 5)).toBe(false);
  });

  it("never counts a zero-fragment document as already embedded", () => {
    expect(hashFullyEmbedded({ rows: 0, maxSeq: -1 }, 0)).toBe(false);
    expect(hashFullyEmbedded(undefined, 0)).toBe(false);
  });
});

describe("formatReindexSummaryLine — the wrapper-parsed summary contract", () => {
  const stats = {
    collection: "agents-skills", documentsWritten: 355, fragmentsEmbedded: 12,
    embedFailures: 0, documentsDeactivated: 1, hashesAlreadyEmbedded: 354, wallClockMs: 4321,
  };

  it("keeps the prefix the master-harness wrapper matches, unchanged", () => {
    expect(formatReindexSummaryLine(stats)).toStartWith(
      "agents-skills: 355 docs written, 12 embedded, 0 embed failures, 1 deactivated, ",
    );
  });

  it("APPENDS `N already embedded` after deactivated and before the wall-clock", () => {
    expect(formatReindexSummaryLine(stats)).toBe(
      "agents-skills: 355 docs written, 12 embedded, 0 embed failures, 1 deactivated, " +
      "354 already embedded, 4.3s",
    );
  });

  it("matches the wrapper's regex shape, including the new optional group", () => {
    // Mirror of master-harness tools/clawmem-pg-reindex _SUMMARY_RE (vn4rz.86).
    const re = /^(?<col>[\w.-]+): (?<written>\d+) docs written, (?<embedded>\d+) embedded, (?<failures>\d+) embed failures(?:, (?<deactivated>\d+) deactivated)?(?:, (?<already>\d+) already embedded)?/m;
    const m = re.exec(formatReindexSummaryLine(stats));
    expect(m?.groups?.already).toBe("354");
    expect(m?.groups?.deactivated).toBe("1");
  });
});
