/**
 * Regression guard for master-harness-vn4rz.34.
 *
 * parseDocument used to wrap its gray-matter call in a BARE `catch {}`. A
 * frontmatter block the YAML parser refused — canonically an unquoted `': '`
 * inside a scalar — cost the document its metadata with nobody ever told.
 *
 * Three distinct consequences, all covered below, because fixing only the
 * obvious one (lost title) would leave the other two silently in place:
 *
 *   1. title / description / tags / domain / workstream are DROPPED;
 *   2. content_type is NOT dropped — it falls through to filename inference,
 *      so the document gets a CONFIDENTLY WRONG type rather than an absent
 *      one, and with it the wrong decay curve (ADR-0143 amendment: filename
 *      inference mis-typed 100 of 118 ADRs as `note`, 60d decay);
 *   3. the raw YAML stays in the BODY and is embedded as prose, polluting the
 *      vector layer with metadata text that was never content.
 *
 * The change under test is VISIBILITY, not parsing semantics: the fallback
 * behaviour is asserted to be unchanged, while the failure is now (a) warned
 * about by path with the parser's own message, (b) reported on the return
 * value so a caller can tell a FAILED parse from a declared absence, and
 * (c) counted into the reindex summary.
 */
import { describe, it, expect, spyOn } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import { parseDocument } from "../../src/indexer.ts";
import { noteFrontmatterFailure, type ReindexStats } from "../../src/pg/reindex.ts";

/**
 * Fixtures are INLINE rather than files under tests/fixtures/ because this
 * repo's .gitignore ignores `*.md` behind an explicit allowlist — a fixture
 * file would be silently untracked and the test would fail on a fresh clone.
 *
 * MALFORMED and WELLFORMED differ in exactly one respect: whether the `': '`
 * inside the description scalar is quoted. That is the whole defect.
 */
const MALFORMED = `---
name: project-malformed-fixture
description: 2026-05-16 signal from BJ — the arc is wrong. RESOLVED by ADR-0088 (two-repo topology): the answer was simplify, not re-platform
tags:
  - alpha
  - beta
domain: harness
workstream: memory
---

Body prose that a reader would expect to be the only thing embedded.
`;

const WELLFORMED = `---
name: project-wellformed-fixture
description: "2026-05-16 signal from BJ: quoting the scalar is the whole repair"
tags:
  - alpha
  - beta
domain: harness
workstream: memory
---

Body prose that a reader would expect to be the only thing embedded.
`;

const NO_FRONTMATTER = `Just a body. This document declares no frontmatter block at all.
`;

const FIXTURES: Record<string, string> = {
  "malformed-unquoted-colon.md": MALFORMED,
  "wellformed.md": WELLFORMED,
  "no-frontmatter.md": NO_FRONTMATTER,
};
const read = (n: string) => FIXTURES[n]!;

/** Run `fn` with console.warn captured rather than printed. */
function captureWarnings<T>(fn: () => T): { result: T; warnings: string[] } {
  const warnings: string[] = [];
  const spy = spyOn(console, "warn").mockImplementation((...args: unknown[]) => {
    warnings.push(args.map(String).join(" "));
  });
  try {
    return { result: fn(), warnings };
  } finally {
    spy.mockRestore();
  }
}

describe("parseDocument: unparseable frontmatter is loud, counted and distinguishable", () => {
  it("consequence 1 — metadata is dropped, and that drop is now REPORTED not swallowed", () => {
    const { result: bad } = captureWarnings(() =>
      parseDocument(read("malformed-unquoted-colon.md"), "notes/malformed-unquoted-colon.md"));
    const good = parseDocument(read("wellformed.md"), "notes/wellformed.md");

    // The control: an identical document differing only in QUOTING parses.
    expect(good.meta.description).toBe("2026-05-16 signal from BJ: quoting the scalar is the whole repair");
    expect(good.meta.tags).toEqual(["alpha", "beta"]);
    expect(good.meta.domain).toBe("harness");
    expect(good.meta.workstream).toBe("memory");
    expect(good.frontmatterError).toBeUndefined();

    // The defect: every one of those is gone.
    expect(bad.meta.description).toBeUndefined();
    expect(bad.meta.tags).toBeUndefined();
    expect(bad.meta.domain).toBeUndefined();
    expect(bad.meta.workstream).toBeUndefined();
    // ...but it is no longer SILENT.
    expect(bad.frontmatterError).toBeDefined();
    expect(bad.frontmatterError!.path).toBe("notes/malformed-unquoted-colon.md");
    expect(bad.frontmatterError!.message.length).toBeGreaterThan(0);
  });

  it("consequence 2 — content_type is INFERRED, not absent: a wrong type, surfaced", () => {
    const { result: bad } = captureWarnings(() =>
      parseDocument(read("malformed-unquoted-colon.md"), "notes/malformed-unquoted-colon.md"));
    // Fallback semantics deliberately UNCHANGED — inference still runs...
    expect(bad.meta.content_type).toBeDefined();
    // ...which is precisely why the failure has to be reported: a caller reading
    // meta alone sees a confident content_type and no hint that the declaration
    // it should have come from was rejected.
    expect(bad.frontmatterError).toBeDefined();

    // A per-collection default still wins over inference on the failure path.
    const { result: withDefault } = captureWarnings(() =>
      parseDocument(read("malformed-unquoted-colon.md"), "notes/malformed-unquoted-colon.md", "decision"));
    expect(withDefault.meta.content_type).toBe("decision");
    expect(withDefault.frontmatterError).toBeDefined();
  });

  it("consequence 3 — the raw YAML is retained in the body and would be embedded as prose", () => {
    const { result: bad } = captureWarnings(() =>
      parseDocument(read("malformed-unquoted-colon.md"), "notes/malformed-unquoted-colon.md"));
    const good = parseDocument(read("wellformed.md"), "notes/wellformed.md");

    expect(good.body.startsWith("---")).toBe(false);
    expect(good.body).not.toContain("project-wellformed-fixture");

    expect(bad.body.startsWith("---")).toBe(true);
    expect(bad.body).toContain("project-malformed-fixture"); // metadata text, in the vector layer
    expect(bad.frontmatterError).toBeDefined();
  });

  it("warns with the offending PATH and the parser's OWN message (positive shape)", () => {
    const { warnings } = captureWarnings(() =>
      parseDocument(read("malformed-unquoted-colon.md"), "notes/malformed-unquoted-colon.md"));

    expect(warnings.length).toBe(1);
    const w = warnings[0]!;
    expect(w).toContain("frontmatter parse FAILED");
    expect(w).toContain("notes/malformed-unquoted-colon.md");
    // The parser's own diagnostic, not a message we invented — it must carry
    // enough to FIND the bad line.
    expect(w).toMatch(/line \d+, column \d+/);
  });

  it("stays silent for a document that parses, and for one with no frontmatter (logical shape)", () => {
    const a = captureWarnings(() => parseDocument(read("wellformed.md"), "notes/wellformed.md"));
    const b = captureWarnings(() => parseDocument(read("no-frontmatter.md"), "notes/no-frontmatter.md"));
    expect(a.warnings).toEqual([]);
    expect(b.warnings).toEqual([]);
  });

  it("a caller can tell 'declared NO frontmatter' from 'frontmatter FAILED to parse'", () => {
    const absent = parseDocument(read("no-frontmatter.md"), "notes/no-frontmatter.md");
    const { result: failed } = captureWarnings(() =>
      parseDocument(read("malformed-unquoted-colon.md"), "notes/malformed-unquoted-colon.md"));

    // Both lose their metadata and keep the whole content as body — which is
    // exactly why `meta` alone can never distinguish them.
    expect(absent.meta.title).toBeUndefined();
    expect(failed.meta.title).toBeUndefined();

    // The discriminator:
    expect(absent.frontmatterError).toBeUndefined();
    expect(failed.frontmatterError).toBeDefined();
  });

  it("reports the failure on EVERY call, not just the first (gray-matter cache bypass)", () => {
    // gray-matter memoizes by content string on its no-options call path, and
    // writes the cache entry BEFORE parsing — so a throw leaves the empty,
    // half-built result cached. Without the `{}` in parseDocument, calls 2..N
    // on identical content return that empty parse and never throw, meaning a
    // malformed document would be warned about and counted exactly ONCE per
    // process and would afterwards masquerade as a clean parse.
    const raw = read("malformed-unquoted-colon.md");
    for (let i = 0; i < 3; i++) {
      const { result, warnings } = captureWarnings(() => parseDocument(raw, `notes/dup-${i}.md`));
      expect(result.frontmatterError).toBeDefined();
      expect(result.frontmatterError!.path).toBe(`notes/dup-${i}.md`);
      expect(warnings.length).toBe(1);
    }
  });

  it("the failure is COUNTED into the reindex summary, keyed by path", () => {
    const stats: Pick<ReindexStats, "frontmatterParseFailures"> = { frontmatterParseFailures: {} };

    for (const [file, rel] of [
      ["wellformed.md", "notes/wellformed.md"],
      ["no-frontmatter.md", "notes/no-frontmatter.md"],
      ["malformed-unquoted-colon.md", "notes/malformed-unquoted-colon.md"],
    ] as const) {
      const { result } = captureWarnings(() => parseDocument(read(file), rel));
      noteFrontmatterFailure(stats, rel, result.frontmatterError);
    }

    const entries = Object.entries(stats.frontmatterParseFailures);
    expect(entries.length).toBe(1);
    expect(entries[0]![0]).toBe("notes/malformed-unquoted-colon.md");
    expect(entries[0]![1]).toMatch(/line \d+, column \d+/);
  });
});

describe("the two known real memory-topics documents parse after repair", () => {
  const MEM = "/home/bj/.claude/projects/-home-bj-claude-master-harness/memory";
  for (const name of ["project_deploy_tooling_pivot.md", "project_clawmem_trial_and_deployment.md"]) {
    it(`${name} has parseable frontmatter`, () => {
      let raw: string;
      try {
        raw = readFileSync(join(MEM, name), "utf8");
      } catch {
        return; // machine-local corpus absent — nothing to assert
      }
      const { result, warnings } = captureWarnings(() => parseDocument(raw, name));
      expect(warnings).toEqual([]);
      expect(result.frontmatterError).toBeUndefined();
      expect(result.meta.description).toBeTruthy();
      expect(result.body.startsWith("---")).toBe(false);
    });
  }
});
