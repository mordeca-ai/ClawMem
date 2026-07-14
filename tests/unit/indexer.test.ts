import { describe, it, expect, beforeEach, afterEach, spyOn } from "bun:test";
import {
  shouldExclude,
  hashContent,
  extractTitle,
  parseDocument,
  computeQualityScore,
  indexCollection,
} from "../../src/indexer.ts";
import { createStore, type Store } from "../../src/store.ts";
import { mkdtempSync, writeFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

// ─── shouldExclude ──────────────────────────────────────────────────

describe("shouldExclude", () => {
  it("excludes .git directories", () => {
    expect(shouldExclude(".git/config")).toBe(true);
    expect(shouldExclude("repo/.git/HEAD")).toBe(true);
  });

  it("excludes node_modules", () => {
    expect(shouldExclude("node_modules/pkg/index.js")).toBe(true);
  });

  it("excludes gits directories", () => {
    expect(shouldExclude("gits/repo/file.md")).toBe(true);
  });

  it("excludes _PRIVATE", () => {
    expect(shouldExclude("_PRIVATE/notes.md")).toBe(true);
  });

  it("excludes underscore-prefixed dirs at any depth (ADR-0071 out-of-scope convention)", () => {
    expect(shouldExclude("_superseded/0024-old-decision.md")).toBe(true);
    expect(shouldExclude("synthesis/_superseded/voice-plan.md")).toBe(true);
    expect(shouldExclude("recipes/_quarantine/bad-doc.md")).toBe(true);
    expect(shouldExclude("_inbox/README.md")).toBe(true);
  });

  it("excludes hidden directories (dot-prefixed)", () => {
    expect(shouldExclude(".hidden/file.md")).toBe(true);
    expect(shouldExclude("path/.secret/file.md")).toBe(true);
  });

  it("does NOT exclude normal paths", () => {
    expect(shouldExclude("docs/notes.md")).toBe(false);
    expect(shouldExclude("research/analysis.md")).toBe(false);
    expect(shouldExclude("MEMORY.md")).toBe(false);
  });

  it("excludes nested excluded dirs", () => {
    expect(shouldExclude("project/gits/repo/file.md")).toBe(true);
    expect(shouldExclude("deep/path/node_modules/pkg.md")).toBe(true);
  });

  it("excludes scraped directory", () => {
    expect(shouldExclude("scraped/page.md")).toBe(true);
  });

  it("excludes dist and build", () => {
    expect(shouldExclude("dist/bundle.js")).toBe(true);
    expect(shouldExclude("build/output.md")).toBe(true);
  });
});

// ─── hashContent ────────────────────────────────────────────────────

describe("hashContent", () => {
  it("produces consistent SHA-256 hex digest", () => {
    const h1 = hashContent("hello world");
    const h2 = hashContent("hello world");
    expect(h1).toBe(h2);
    expect(h1).toHaveLength(64); // SHA-256 hex
  });

  it("produces different hashes for different content", () => {
    const h1 = hashContent("content A");
    const h2 = hashContent("content B");
    expect(h1).not.toBe(h2);
  });
});

// ─── extractTitle ───────────────────────────────────────────────────

describe("extractTitle", () => {
  it("extracts first heading from markdown", () => {
    expect(extractTitle("# My Title\n\nBody text", "file.md")).toBe("My Title");
  });

  it("extracts h2 heading", () => {
    expect(extractTitle("## Section Title\nContent", "file.md")).toBe("Section Title");
  });

  it("falls back to filename without extension", () => {
    expect(extractTitle("No heading here", "my-notes.md")).toBe("my-notes");
  });

  it("handles files with no headings", () => {
    expect(extractTitle("Just plain text", "readme.txt")).toBe("readme");
  });

  it("trims whitespace from heading", () => {
    expect(extractTitle("#   Spaced Title  \n", "file.md")).toBe("Spaced Title");
  });
});

// ─── parseDocument ──────────────────────────────────────────────────

describe("parseDocument", () => {
  it("extracts frontmatter fields from YAML", () => {
    const content = `---
content_type: decision
tags: [api, rest]
domain: backend
---

# Decision

Body here.`;
    const { body, meta } = parseDocument(content, "file.md");
    expect(meta.content_type).toBe("decision");
    expect(meta.tags).toEqual(["api", "rest"]);
    expect(meta.domain).toBe("backend");
    expect(body.trim()).toContain("# Decision");
  });

  it("infers content_type from path when not in frontmatter", () => {
    const { meta } = parseDocument("# Notes\nSome text", "sessions/2026-03-01.md");
    expect(meta.content_type).toBe("handoff");
  });

  it("handles documents without frontmatter gracefully", () => {
    const { body, meta } = parseDocument("# Just a title\nBody", "random.md");
    expect(body).toContain("# Just a title");
    expect(meta.content_type).toBe("note");
  });

  it("returns body without frontmatter delimiters", () => {
    const content = `---
title: Test
---

Body content here.`;
    const { body } = parseDocument(content, "file.md");
    expect(body.trim()).toBe("Body content here.");
    expect(body).not.toContain("---");
  });
});

// ─── computeQualityScore ────────────────────────────────────────────

describe("computeQualityScore", () => {
  it("returns base 0.3 for empty doc with no meta", () => {
    const score = computeQualityScore("", { content_type: "note" });
    // base 0.3, stub penalty -0.1 (length < 50) = 0.2
    expect(score).toBeCloseTo(0.2, 2);
  });

  it("boosts for document length > 200", () => {
    const short = computeQualityScore("x".repeat(100), { content_type: "note" });
    const medium = computeQualityScore("x".repeat(300), { content_type: "note" });
    expect(medium).toBeGreaterThan(short);
  });

  it("boosts for document length > 500", () => {
    const medium = computeQualityScore("x".repeat(300), { content_type: "note" });
    const long = computeQualityScore("x".repeat(600), { content_type: "note" });
    expect(long).toBeGreaterThan(medium);
  });

  it("boosts for headings", () => {
    const noHeading = computeQualityScore("x".repeat(300), { content_type: "note" });
    const heading = computeQualityScore("## Section\n" + "x".repeat(300), { content_type: "note" });
    expect(heading).toBeGreaterThan(noHeading);
  });

  it("boosts for bullet lists", () => {
    const noList = computeQualityScore("x".repeat(300), { content_type: "note" });
    const list = computeQualityScore("- item one\n" + "x".repeat(300), { content_type: "note" });
    expect(list).toBeGreaterThan(noList);
  });

  it("boosts for decision keywords", () => {
    const plain = computeQualityScore("x".repeat(300), { content_type: "note" });
    const decision = computeQualityScore("We decided to use REST " + "x".repeat(300), { content_type: "note" });
    expect(decision).toBeGreaterThan(plain);
  });

  it("boosts for fix/bug keywords", () => {
    const plain = computeQualityScore("x".repeat(300), { content_type: "note" });
    const fix = computeQualityScore("Fixed the authentication bug " + "x".repeat(300), { content_type: "note" });
    expect(fix).toBeGreaterThan(plain);
  });

  it("boosts for rich frontmatter (up to +0.15)", () => {
    const sparse = computeQualityScore("x".repeat(300), { content_type: "note" });
    const rich = computeQualityScore("x".repeat(300), {
      content_type: "note",
      tags: ["a", "b"],
      domain: "backend",
      workstream: "auth",
    });
    expect(rich).toBeGreaterThan(sparse);
    expect(rich - sparse).toBeLessThanOrEqual(0.15 + 0.001);
  });

  it("penalizes trivial stubs (<50 chars)", () => {
    const stub = computeQualityScore("Short.", { content_type: "note" });
    expect(stub).toBeLessThan(0.3); // base minus penalty
  });

  it("clamps between 0 and 1.0", () => {
    // Max possible: base 0.3 + length 0.2 + heading 0.1 + list 0.05 + decision 0.15 + fix 0.1 + meta 0.15 = 1.05 → clamped to 1.0
    const maxDoc = "## Heading\n- item\nWe decided to fix the bug\n" + "x".repeat(600);
    const score = computeQualityScore(maxDoc, {
      content_type: "decision",
      tags: ["a"],
      domain: "d",
      workstream: "w",
    });
    expect(score).toBeLessThanOrEqual(1.0);
    expect(score).toBeGreaterThanOrEqual(0);
  });
});

// ─── per-collection default content_type (rvzn8.2) ─────────────────

describe("parseDocument defaultContentType", () => {
  it("uses the collection default for a frontmatter-less doc (kills inference)", () => {
    // "0123-....md" infers `note` (no filename signal); the default must win.
    const { meta } = parseDocument("# ADR 123\nBody", "0123-some-decision-topic.md", "decision");
    expect(meta.content_type).toBe("decision");
  });

  it("explicit frontmatter content_type beats the collection default", () => {
    const content = "---\ncontent_type: hub\n---\n# Index";
    const { meta } = parseDocument(content, "0042-x.md", "decision");
    expect(meta.content_type).toBe("hub");
  });

  it("falls back to inference when no default is configured (unchanged behavior)", () => {
    const { meta } = parseDocument("# Notes", "research/topic.md");
    expect(meta.content_type).toBe("research");
  });

  it("applies the default even when frontmatter parsing fails", () => {
    const broken = "---\n: not: [valid yaml\n---\nbody";
    const { meta } = parseDocument(broken, "0007-y.md", "decision");
    expect(meta.content_type).toBe("decision");
  });
});

// ─── indexCollection: missing collection root (master-harness-x564y) ─
//
// A pruned/stale collection root used to hard-fail the WHOLE `clawmem update`
// run — Bun.Glob.scanSync throws ENOENT on a nonexistent cwd, and that error
// escaped indexCollection uncaught, aborting the loop over every OTHER
// configured collection too (the jioi8 incident shape; see l18bm). The fix
// is a warn-and-skip: log a loud warning naming the collection + missing
// path, return zeroed stats, and let the caller's loop continue to the next
// collection.

describe("indexCollection — missing collection root", () => {
  let store: Store;
  let warnSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    process.env.CLAWMEM_ENABLE_AMEM = "false";
    process.env.CLAWMEM_NO_LOCAL_MODELS = "true";
    store = createStore(":memory:");
    warnSpy = spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it("warns and returns zeroed stats instead of throwing when the root doesn't exist", async () => {
    const missingPath = join(tmpdir(), `clawmem-x564y-missing-${Date.now()}`);

    const stats = await indexCollection(store, "missing-col", missingPath, "**/*.md");

    expect(stats).toEqual({ added: 0, updated: 0, unchanged: 0, removed: 0 });
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const [message] = warnSpy.mock.calls[0]!;
    expect(String(message)).toContain("missing-col");
    expect(String(message)).toContain(missingPath);
    expect(String(message)).toContain("skipping");
  });

  it("does not touch the database for the missing collection", async () => {
    const missingPath = join(tmpdir(), `clawmem-x564y-missing-${Date.now()}`);

    await indexCollection(store, "missing-col", missingPath, "**/*.md");

    const doc = store.findActiveDocument("missing-col", "anything.md");
    expect(doc).toBeNull();
  });

  it("a sibling healthy collection still indexes when a prior collection's root is missing", async () => {
    const missingPath = join(tmpdir(), `clawmem-x564y-missing-${Date.now()}`);
    const healthyDir = mkdtempSync(join(tmpdir(), "clawmem-x564y-healthy-"));
    writeFileSync(join(healthyDir, "doc.md"), "# Healthy doc\n\nStill indexes fine.");

    try {
      // Missing collection first (matches the incident shape: an earlier
      // stale/pruned root must not prevent a LATER healthy collection from
      // being reached in the same `clawmem update` loop).
      const missingStats = await indexCollection(store, "missing-col", missingPath, "**/*.md");
      const healthyStats = await indexCollection(store, "healthy-col", healthyDir, "**/*.md");

      expect(missingStats).toEqual({ added: 0, updated: 0, unchanged: 0, removed: 0 });
      expect(healthyStats.added).toBe(1);

      const doc = store.findActiveDocument("healthy-col", "doc.md");
      expect(doc).not.toBeNull();
    } finally {
      rmSync(healthyDir, { recursive: true, force: true });
    }
  });

  it("healthy-only run is unchanged (no missing collection involved)", async () => {
    const healthyDir = mkdtempSync(join(tmpdir(), "clawmem-x564y-healthy-only-"));
    writeFileSync(join(healthyDir, "doc.md"), "# Healthy doc\n\nUnaffected by the fix.");

    try {
      const stats = await indexCollection(store, "healthy-col", healthyDir, "**/*.md");

      expect(stats.added).toBe(1);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      rmSync(healthyDir, { recursive: true, force: true });
    }
  });
});
