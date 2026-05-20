/**
 * Unit tests for per-collection scope resolution (computeCollectionScope).
 *
 * Covers the precedence contract used by `clawmem search/vsearch/query`:
 *   - explicit -c flag: single, comma-multi, case-insensitive canonicalization,
 *     de-dupe, whitespace tolerance, unknown-name → { error }
 *   - flag overrides session focus
 *   - focus auto-scope ONLY when the topic names a real collection
 *     (case-insensitive); free-text focus → null (fail-open to unscoped)
 *   - no flag + no focus → null (baseline / unscoped)
 */

import { describe, it, expect } from "bun:test";
import { computeCollectionScope } from "../../src/collection-scope.ts";

const KNOWN = ["library", "reference", "openclaw-workspace", "stormlight"];

describe("computeCollectionScope — explicit -c flag", () => {
  it("scopes to a single named collection", () => {
    expect(computeCollectionScope("stormlight", undefined, KNOWN)).toEqual({
      collections: ["stormlight"],
      source: "flag",
    });
  });

  it("scopes to multiple comma-separated collections", () => {
    expect(computeCollectionScope("stormlight,library", undefined, KNOWN)).toEqual({
      collections: ["stormlight", "library"],
      source: "flag",
    });
  });

  it("is case-insensitive and returns the canonical configured name", () => {
    expect(computeCollectionScope("STORMLIGHT", undefined, KNOWN)).toEqual({
      collections: ["stormlight"],
      source: "flag",
    });
  });

  it("tolerates surrounding whitespace in a comma list", () => {
    expect(computeCollectionScope(" stormlight , library ", undefined, KNOWN)).toEqual({
      collections: ["stormlight", "library"],
      source: "flag",
    });
  });

  it("de-dupes repeated names", () => {
    expect(computeCollectionScope("stormlight,stormlight", undefined, KNOWN)).toEqual({
      collections: ["stormlight"],
      source: "flag",
    });
  });

  it("returns an error for an unknown collection (no silent empty result)", () => {
    const r = computeCollectionScope("bogus", undefined, KNOWN);
    expect(r).not.toBeNull();
    expect(r && "error" in r).toBe(true);
    expect((r as { error: string }).error).toContain("Unknown collection 'bogus'");
    expect((r as { error: string }).error).toContain("stormlight");
  });

  it("rejects the whole list if any one name is unknown", () => {
    const r = computeCollectionScope("stormlight,bogus", undefined, KNOWN);
    expect(r && "error" in r).toBe(true);
  });

  it("treats an empty/whitespace flag as no flag (falls through)", () => {
    expect(computeCollectionScope("   ", undefined, KNOWN)).toBeNull();
    expect(computeCollectionScope(",,", undefined, KNOWN)).toBeNull();
  });
});

describe("computeCollectionScope — focus auto-scope", () => {
  it("scopes when the focus topic names a real collection", () => {
    expect(computeCollectionScope(undefined, "stormlight", KNOWN)).toEqual({
      collections: ["stormlight"],
      source: "focus",
    });
  });

  it("matches focus case-insensitively to the canonical name", () => {
    expect(computeCollectionScope(undefined, "Stormlight", KNOWN)).toEqual({
      collections: ["stormlight"],
      source: "focus",
    });
  });

  it("does NOT scope when the focus is free-text (not a collection)", () => {
    expect(computeCollectionScope(undefined, "combat mechanics", KNOWN)).toBeNull();
  });

  it("does NOT scope on empty/undefined focus", () => {
    expect(computeCollectionScope(undefined, undefined, KNOWN)).toBeNull();
    expect(computeCollectionScope(undefined, "  ", KNOWN)).toBeNull();
  });
});

describe("computeCollectionScope — precedence", () => {
  it("explicit flag overrides session focus", () => {
    expect(computeCollectionScope("library", "stormlight", KNOWN)).toEqual({
      collections: ["library"],
      source: "flag",
    });
  });

  it("an unknown flag errors even when a valid focus is present", () => {
    const r = computeCollectionScope("bogus", "stormlight", KNOWN);
    expect(r && "error" in r).toBe(true);
  });
});
