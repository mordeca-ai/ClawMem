/**
 * Regression guard for master-harness-wzwh8 (phase 1: the counter only).
 *
 * parseDocument reads frontmatter `title:` and `content_type:`. Much of the
 * corpus declares perfectly valid YAML under OTHER keys — `name:` for the
 * title, `type:` / `metadata.type` for the content type — so the declared value
 * is silently never seen: the title falls back to the first heading and
 * content_type to the per-collection default or filename inference. Unlike the
 * vn4rz.34 parse failure there is no error to swallow, so nothing counted it.
 *
 * The change under test is VISIBILITY ONLY. Which vocabulary is canonical is a
 * separate decision (bead wzwh8 phase 2); here every existing field value is
 * asserted UNCHANGED while the gap becomes reportable, per collection.
 *
 * Fixtures are inline for the same reason as frontmatter-parse-failure.test.ts:
 * this repo's .gitignore ignores `*.md` behind an allowlist.
 */
import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { mkdirSync, writeFileSync, rmSync } from "fs";
import {
  parseDocument,
  indexCollection,
  emptyFrontmatterVocabCounts,
  noteFrontmatterVocab,
  formatFrontmatterVocab,
} from "../../src/indexer.ts";
import { inferContentType } from "../../src/memory.ts";
import { createStore, type Store } from "../../src/store.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";

const NAME_TYPE = `---
name: feedback-fixture-one
type: feedback
description: a memory-topics style document
---

# Heading One

Body.
`;

const METADATA_TYPE = `---
name: skill-fixture
metadata:
  type: feedback
---

Body without a heading.
`;

const CANONICAL = `---
title: Canonical Fixture
content_type: decision
---

# Some Heading

Body.
`;

const NO_FRONTMATTER = `# Plain Heading

No frontmatter at all.
`;

const MALFORMED = `---
name: x
description: unquoted: colon breaks yaml: here
---
Body.
`;

describe("parseDocument reports the frontmatter vocabulary gap (A1)", () => {
  it("name:/type: → title-less + content_type-less, declared [name, type]; values unchanged", () => {
    const rel = "notes/feedback_fixture_one.md";
    const doc = parseDocument(NAME_TYPE, rel);
    expect(doc.frontmatter).toBe("parsed");
    expect(doc.vocabGap).toEqual({
      titleMissing: true,
      contentTypeMissing: true,
      declaredKeys: ["name", "type"],
    });
    // Behaviour UNCHANGED: title stays absent, content_type is today's fallback.
    expect(doc.meta.title).toBeUndefined();
    expect(doc.meta.content_type).toBe(inferContentType(rel));
    expect(doc.meta.description).toBe("a memory-topics style document");
  });

  it("per-collection default still wins exactly as before", () => {
    const doc = parseDocument(NAME_TYPE, "notes/x.md", "note");
    expect(doc.meta.content_type).toBe("note");
    expect(doc.vocabGap?.contentTypeMissing).toBe(true);
  });

  it("metadata: {type: feedback} → declared [name, metadata.type]", () => {
    const rel = "skills/foo/SKILL.md";
    const doc = parseDocument(METADATA_TYPE, rel);
    expect(doc.frontmatter).toBe("parsed");
    expect(doc.vocabGap).toEqual({
      titleMissing: true,
      contentTypeMissing: true,
      declaredKeys: ["name", "metadata.type"],
    });
    expect(doc.meta.title).toBeUndefined();
    expect(doc.meta.content_type).toBe(inferContentType(rel));
  });
});

describe("clean and absent frontmatter are distinguishable (A2)", () => {
  it("title: + content_type: → parsed, no gap", () => {
    const doc = parseDocument(CANONICAL, "notes/c.md");
    expect(doc.frontmatter).toBe("parsed");
    expect(doc.vocabGap).toBeUndefined();
    expect(doc.meta.title).toBe("Canonical Fixture");
    expect(doc.meta.content_type).toBe("decision");
  });

  it("no frontmatter → 'absent', NOT a title-less gap", () => {
    const doc = parseDocument(NO_FRONTMATTER, "notes/plain.md");
    expect(doc.frontmatter).toBe("absent");
    expect(doc.vocabGap).toBeUndefined();
  });

  it("an empty block is present-and-parsed, and title-less", () => {
    const doc = parseDocument("---\n---\nbody\n", "notes/empty.md");
    expect(doc.frontmatter).toBe("parsed");
    expect(doc.vocabGap).toEqual({
      titleMissing: true,
      contentTypeMissing: true,
      declaredKeys: [],
    });
  });

  it("an unparseable block is 'failed' (vn4rz.34), never a vocabulary gap", () => {
    const orig = console.warn;
    console.warn = () => {};
    try {
      const doc = parseDocument(MALFORMED, "notes/bad.md");
      expect(doc.frontmatter).toBe("failed");
      expect(doc.frontmatterError).toBeDefined();
      expect(doc.vocabGap).toBeUndefined();
    } finally {
      console.warn = orig;
    }
  });

  it("the counter folds each shape into the right bucket", () => {
    const counts = emptyFrontmatterVocabCounts();
    const orig = console.warn;
    console.warn = () => {};
    try {
      for (const [c, rel] of [
        [NAME_TYPE, "a.md"],
        [METADATA_TYPE, "b.md"],
        [CANONICAL, "c.md"],
        [NO_FRONTMATTER, "d.md"],
        [MALFORMED, "e.md"],
      ] as const) {
        noteFrontmatterVocab(counts, parseDocument(c, rel));
      }
    } finally {
      console.warn = orig;
    }
    expect(counts).toEqual({
      examined: 5,
      noFrontmatter: 1,
      unparseable: 1,
      titleless: 2,
      titlelessDeclaringName: 2,
      contentTypeless: 2,
      contentTypelessDeclaringType: 1,
      contentTypelessDeclaringMetadataType: 1,
      // wzwh8.1: no frontmatter_map was passed, so the map counters stay zero.
      titleMapped: 0,
      contentTypeMapped: 0,
      contentTypeMapRejected: 0,
    });
    expect(formatFrontmatterVocab(counts)).toBe(
      "frontmatter: 2 title-less (2 declare name:), 2 content_type-less " +
        "(1 declare type:, 1 declare metadata.type); 1 without frontmatter, 1 unparseable (of 5 examined)",
    );
  });
});

describe("the per-collection indexing summary carries the counter (A3)", () => {
  const ROOT = `/tmp/clawmem-wzwh8-vocab-${process.pid}`;
  const CONTENT = `${ROOT}/content`;
  let store: Store;
  const deadLlm = {
    embed: async () => {
      throw new Error("no embedding endpoint in test");
    },
    query: async () => null,
    expandQuery: async () => [],
  } as any;

  beforeEach(() => {
    rmSync(ROOT, { recursive: true, force: true });
    mkdirSync(CONTENT, { recursive: true });
    setDefaultLlamaCpp(deadLlm);
    store = createStore(`${ROOT}/vault.sqlite`);
  });
  afterEach(() => {
    try {
      store.close();
    } catch {
      /* already closed */
    }
    setDefaultLlamaCpp(null);
    rmSync(ROOT, { recursive: true, force: true });
  });

  it("is non-zero on a vault holding name:/type: documents", async () => {
    writeFileSync(`${CONTENT}/feedback_one.md`, NAME_TYPE);
    writeFileSync(`${CONTENT}/skill.md`, METADATA_TYPE);
    writeFileSync(`${CONTENT}/canonical.md`, CANONICAL);
    const stats = await indexCollection(store, "docs", CONTENT, "**/*.md");
    expect(stats.added).toBe(3);
    expect(stats.frontmatterVocab?.titleless).toBe(2);
    expect(stats.frontmatterVocab?.titlelessDeclaringName).toBe(2);
    expect(stats.frontmatterVocab?.contentTypeless).toBe(2);
    expect(stats.frontmatterVocab?.contentTypelessDeclaringType).toBe(1);
    expect(stats.frontmatterVocab?.contentTypelessDeclaringMetadataType).toBe(1);
    expect(formatFrontmatterVocab(stats.frontmatterVocab!)).toContain(
      "2 title-less (2 declare name:)",
    );

    // ZERO behaviour change: what got stored is exactly the pre-wzwh8 fallback.
    const row = store.db
      .prepare(
        "SELECT title, content_type FROM documents WHERE collection = 'docs' AND path = 'feedback_one.md'",
      )
      .get() as { title: string; content_type: string };
    expect(row.title).toBe("Heading One");
    expect(row.content_type).toBe(inferContentType("feedback_one.md"));
  });

  it("is zero on a clean vault", async () => {
    writeFileSync(`${CONTENT}/canonical.md`, CANONICAL);
    writeFileSync(`${CONTENT}/plain.md`, NO_FRONTMATTER);
    const stats = await indexCollection(store, "docs", CONTENT, "**/*.md");
    expect(stats.frontmatterVocab?.examined).toBe(2);
    expect(stats.frontmatterVocab?.titleless).toBe(0);
    expect(stats.frontmatterVocab?.contentTypeless).toBe(0);
    expect(stats.frontmatterVocab?.noFrontmatter).toBe(1);
    expect(formatFrontmatterVocab(stats.frontmatterVocab!)).toBeNull();
  });
});
