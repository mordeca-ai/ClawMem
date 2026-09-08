/**
 * Guards for the absent-from-walk sweep (master-harness-vn4rz.41).
 *
 * THE DEFECT. src/pg/reindex.ts walked the files that exist and upserted them,
 * and nothing ever retired a PG row whose source file had been DELETED. Reindex
 * therefore converged in ONE direction only and Postgres monotonically
 * accumulated rows for files that no longer exist. Measured on the live vault
 * before the fix: `episodic-handoffs` held 123 active rows against 55 files on
 * disk, and every one of the 68 surplus paths was confirmed absent — pruned
 * handoffs, live and retrievable and indistinguishable at query time from
 * content that still exists.
 *
 * The sweep is DESTRUCTIVE in effect (it removes content from retrieval), so
 * what is under test here is mostly the set of circumstances in which it must
 * REFUSE to run. Three of the five cases below are refusals, and each asserts
 * the REASON STRING, not merely the boolean: "did not sweep" and "did not sweep
 * and said why" are different products, and only the second one is debuggable
 * from a summary line six weeks later.
 *
 * No database. sweepDecision is pure and deactivateAbsentDocuments's empty
 * keep-set guard throws before it ever resolves a vault or opens a transaction,
 * which is what makes the mass-deactivation refusal provable here rather than
 * only in the integration suite.
 */
import { describe, it, expect } from "bun:test";
import { sweepDecision } from "../../src/pg/reindex.ts";
import { deactivateAbsentDocuments } from "../../src/pg/write.ts";

describe("sweepDecision — the sweep runs by default", () => {
  it("sweeps on a plain full walk (no options at all)", () => {
    // DEFAULT TRUE is the load-bearing half: one-directional convergence was
    // the defect, so two-directional must be what you get without asking.
    expect(sweepDecision({}, 55)).toEqual({ sweep: true, reason: null });
  });

  it("sweeps when sweep:true is passed explicitly", () => {
    expect(sweepDecision({ sweep: true }, 1)).toEqual({ sweep: true, reason: null });
  });

  it("sweeps on a single in-scope file — one file is a real walk", () => {
    expect(sweepDecision({}, 1).sweep).toBe(true);
  });
});

describe("sweepDecision — REFUSALS, each with a specific reason", () => {
  it("does NOT sweep when --no-sweep was passed", () => {
    const d = sweepDecision({ sweep: false }, 55);
    expect(d.sweep).toBe(false);
    expect(d.reason).toBe("disabled by --no-sweep");
  });

  it("does NOT sweep on a --limit run: a partial walk may not advance a destructive conclusion", () => {
    // The master-harness-vn4rz.40 precedent. A truncated file set means
    // absent-from-walk does not imply absent-from-disk, so every unwalked
    // document would be retired by a run that never even looked at it.
    const d = sweepDecision({ limit: 5 }, 5);
    expect(d.sweep).toBe(false);
    expect(d.reason).toMatch(/partial walk/);
    expect(d.reason).toMatch(/absent-from-walk does not imply absent-from-disk/);
  });

  it("does NOT sweep on --limit 0 — zero is a deliberate walk of nothing, not 'unlimited'", () => {
    // The falsy-zero trap: `if (opts.limit)` would have swept here.
    const d = sweepDecision({ limit: 0 }, 0);
    expect(d.sweep).toBe(false);
    expect(d.reason).toMatch(/partial walk/);
  });

  it("does NOT sweep on --limit larger than the corpus", () => {
    // The walk happens to be complete, but the RUN still declared itself
    // partial. The decision keys on the declaration, not on a coincidence.
    const d = sweepDecision({ limit: 100_000 }, 55);
    expect(d.sweep).toBe(false);
    expect(d.reason).toMatch(/partial walk/);
  });

  it("does NOT sweep an EMPTY in-scope walk — that is a missing root, not an empty collection", () => {
    const d = sweepDecision({}, 0);
    expect(d.sweep).toBe(false);
    expect(d.reason).toMatch(/empty walk/);
    expect(d.reason).toMatch(/missing\/unreadable collection root/);
  });

  it("refuses an empty walk even when the sweep was explicitly requested", () => {
    // sweep:true is a REQUEST, never a bypass.
    expect(sweepDecision({ sweep: true }, 0).sweep).toBe(false);
  });

  it("--no-sweep outranks the other refusals, and every refusal names itself", () => {
    expect(sweepDecision({ sweep: false, limit: 3 }, 0).reason).toBe("disabled by --no-sweep");
    for (const opts of [{ sweep: false }, { limit: 1 }, {}]) {
      const d = sweepDecision(opts, 0);
      expect(d.sweep).toBe(false);
      // Never silently skip: a refusal with a null or empty reason is a
      // one-directional run wearing a green summary as camouflage.
      expect(typeof d.reason).toBe("string");
      expect((d.reason ?? "").length).toBeGreaterThan(20);
    }
  });

  it("a refusal never carries null, and a sweep never carries a reason", () => {
    const cases: Array<[{ sweep?: boolean; limit?: number }, number]> = [
      [{}, 0], [{}, 7], [{ sweep: false }, 7], [{ limit: 2 }, 7], [{ sweep: true }, 0],
    ];
    for (const [opts, n] of cases) {
      const d = sweepDecision(opts, n);
      expect(d.sweep ? d.reason === null : typeof d.reason === "string").toBe(true);
    }
  });
});

describe("deactivateAbsentDocuments REFUSES an empty keep-set at the primitive", () => {
  /**
   * THE GUARD MUST LIVE AT THE PRIMITIVE. reindexCollection already refuses an
   * empty walk via sweepDecision, but a guard that exists only in one caller is
   * a guard the second caller bypasses by accident. This asserts the throw
   * happens before any vault resolution or transaction — the collection name
   * below does not exist in any config, so reaching resolveVault would fail
   * with a different message entirely.
   */
  it("throws, naming the mass-deactivation hazard", async () => {
    await expect(
      deactivateAbsentDocuments("__vn4rz41_no_such_collection", [], "reason"),
    ).rejects.toThrow(/mass-deactivation hazard/);
  });

  it("names the collection and says an empty walk is indistinguishable from a missing root", async () => {
    await expect(
      deactivateAbsentDocuments("__vn4rz41_no_such_collection", [], "reason"),
    ).rejects.toThrow(/__vn4rz41_no_such_collection/);
    await expect(
      deactivateAbsentDocuments("__vn4rz41_no_such_collection", [], "reason"),
    ).rejects.toThrow(/ENTIRE collection/);
  });
});
