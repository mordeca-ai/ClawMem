import { describe, it, expect } from "bun:test";
import { explainVecLoadError } from "../../src/store.ts";

// Issue #20 (Yoloshii/ClawMem): on macOS the default system SQLite is compiled
// without extension support, so sqlite-vec fails to load at `clawmem bootstrap`
// with a cryptic "does not support dynamic extension loading". explainVecLoadError
// must turn that into actionable guidance (brew install sqlite) on macOS, and
// must leave every other platform / unrelated error untouched.
//
// These assert the CORRECT behavior (not the old behavior) — they fail against
// the pre-fix source, which had no such mapping, and pass after the fix.
describe("explainVecLoadError (Issue #20 — macOS sqlite-vec)", () => {
  const extErr = new Error("This build of sqlite3 does not support dynamic extension loading");

  it("macOS + no Homebrew SQLite found → actionable 'brew install sqlite' guidance", () => {
    const out = explainVecLoadError(extErr, "darwin", null);
    expect(out.message).toContain("brew install sqlite");
    expect(out.message).toContain("macOS");
    expect(out.message).toContain("No Homebrew SQLite was found");
    expect(out.message).toContain("ClawMem#20");
    // original error text preserved for debugging
    expect(out.message).toContain("does not support dynamic extension loading");
  });

  it("macOS + a custom SQLite that still failed → suggests reinstall and names the path", () => {
    const path = "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib";
    const out = explainVecLoadError(extErr, "darwin", path);
    expect(out.message).toContain("brew reinstall sqlite");
    expect(out.message).toContain(path);
  });

  it("non-macOS → returns the original error untouched (no macOS guidance)", () => {
    const out = explainVecLoadError(extErr, "linux", null);
    expect(out).toBe(extErr);
    expect(out.message).not.toContain("brew install");
  });

  it("macOS but an unrelated error → returned untouched", () => {
    const other = new Error("disk I/O error");
    const out = explainVecLoadError(other, "darwin", null);
    expect(out).toBe(other);
    expect(out.message).not.toContain("brew install");
  });

  it("wraps a non-Error thrown value into an Error and still maps it on macOS", () => {
    const out = explainVecLoadError("does not support dynamic extension loading", "darwin", null);
    expect(out).toBeInstanceOf(Error);
    expect(out.message).toContain("brew install sqlite");
  });
});
