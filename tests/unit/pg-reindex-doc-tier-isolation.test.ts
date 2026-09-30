/**
 * One vault's doc_tier reconcile fault must not fail the reindex pass
 * (master-harness-vn4rz.77 skeptic fix). NO DATABASE: the reconcile function is
 * injected; the real reconcileDocTierAllVaults loop is what is under test.
 */
import { describe, it, expect } from "bun:test";
import { reconcileDocTierAllVaults } from "../../src/pg/reindex.ts";
import type { Vault } from "../../src/pg/vaults.ts";

type Reported = { vault: Vault; skipped?: string; markedTrue?: number };

describe("reconcileDocTierAllVaults", () => {
  it("reports a throwing vault as skipped and STILL reconciles the other; never rejects", async () => {
    const seen: Reported[] = [];
    const reconcile = (async (v: Vault) => {
      if (v === "sfw") throw new Error("connection refused");
      return { vault: v, markedTrue: 0, markedFalse: 0, wallClockMs: 1 };
    }) as never;
    await reconcileDocTierAllVaults(r => seen.push(r as Reported), reconcile, () => true);
    expect(seen).toHaveLength(2);
    expect(seen.find(r => r.vault === "sfw")!.skipped).toBe("reconcile failed: connection refused");
    expect(seen.find(r => r.vault === "nsfw")).toMatchObject({ markedTrue: 0 });
  });

  it("a fault in the LATER vault does not lose the earlier vault's result", async () => {
    const seen: Reported[] = [];
    const reconcile = (async (v: Vault) => {
      if (v === "nsfw") throw new Error("boom");
      return { vault: v, markedTrue: 3, markedFalse: 0, wallClockMs: 1 };
    }) as never;
    await reconcileDocTierAllVaults(r => seen.push(r as Reported), reconcile, () => true);
    expect(seen.find(r => r.vault === "sfw")).toMatchObject({ markedTrue: 3 });
    expect(seen.find(r => r.vault === "nsfw")!.skipped).toContain("boom");
  });

  it("skips an unconfigured vault without calling reconcile", async () => {
    const called: Vault[] = [];
    const reconcile = (async (v: Vault) => { called.push(v); return { vault: v }; }) as never;
    await reconcileDocTierAllVaults(undefined, reconcile, v => v === "sfw");
    expect(called).toEqual(["sfw"]);
  });
});
