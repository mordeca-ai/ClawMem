/**
 * master-harness-wzwh8.1 (wzwh8 phase 2): the per-collection, OPT-IN
 * frontmatter key map.
 *
 * Phase 1 (frontmatter-vocab-counter.test.ts) made the gap VISIBLE: documents
 * authored against external specs declare `name:` / `metadata.type` instead of
 * `title:` / `content_type:`. This file pins the fix: a collection may declare
 * where its title / content_type live, and
 *   A1  the map fills title + a translated content_type,
 *   A2  a collection WITHOUT a map is byte-for-byte phase-1 behaviour,
 *   A3  the canonical key always wins,
 *   A4  an unmappable value is NOT stored — default chain + a rejected count,
 *   A5  a malformed map fails loud at config load,
 *   A6  every indexing path (sqlite indexCollection, pg reindex, pg origin-load)
 *       honours the map through its one parse call site.
 *
 * Fixtures are inline: this repo's .gitignore ignores `*.md` behind an allowlist.
 */
import { describe, it, expect, beforeEach, afterEach, afterAll } from "bun:test";
import { mkdirSync, writeFileSync, rmSync, readFileSync, readdirSync, statSync } from "fs";
import { join } from "path";
import {
  parseDocument,
  indexCollection,
  emptyFrontmatterVocabCounts,
  noteFrontmatterVocab,
  formatFrontmatterVocab,
} from "../../src/indexer.ts";
import { validateFrontmatterMap, type FrontmatterMap } from "../../src/frontmatter-map.ts";
import {
  loadConfig,
  clearConfigCache,
  addCollection,
  collectionIndexOptions,
} from "../../src/collections.ts";
import { inferContentType, CONTENT_TYPE_VALUES } from "../../src/memory.ts";
import { parseForReindex } from "../../src/pg/reindex.ts";
import { parseOriginDocument } from "../../src/pg/origin.ts";
import { createStore, type Store } from "../../src/store.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";

/** Claude Code auto-memory shape: name + description + metadata.type. */
const MEMORY_TOPIC = `---
name: Foo
description: an auto-memory style document
metadata:
  type: feedback
---

# Heading One

Body.
`;

const CANONICAL_AND_MAPPED = `---
title: Bar
name: Foo
content_type: decision
metadata:
  type: feedback
---

# Heading

Body.
`;

const BOGUS_TYPE = `---
name: Baz
metadata:
  type: bogus
---

Body.
`;

const IDENTITY_TYPE = `---
name: Qux
metadata:
  type: project
---

Body.
`;

const MAP: FrontmatterMap = {
  title: "name",
  content_type: "metadata.type",
  content_type_values: { feedback: "preference" },
};

const REL = "topics/feedback_foo.md";

describe("A1: a mapped collection fills title and a translated content_type", () => {
  it("name: Foo + metadata.type feedback -> title Foo, content_type preference", () => {
    const doc = parseDocument(MEMORY_TOPIC, REL, undefined, MAP);
    expect(doc.frontmatter).toBe("parsed");
    expect(doc.meta.title).toBe("Foo");
    expect(doc.meta.content_type).toBe("preference");
    expect(doc.frontmatterMap).toEqual({ titleMapped: true, contentTypeMapped: true });
    // Both fields now resolved, so there is no remaining vocabulary gap.
    expect(doc.vocabGap).toBeUndefined();
    // Untouched fields still read from their canonical keys.
    expect(doc.meta.description).toBe("an auto-memory style document");
  });

  it("a mapped value that is already a ContentType passes without a table entry", () => {
    const doc = parseDocument(IDENTITY_TYPE, "x/qux.md", undefined, MAP);
    expect(doc.meta.title).toBe("Qux");
    expect(doc.meta.content_type).toBe("project");
    expect(doc.frontmatterMap).toEqual({ titleMapped: true, contentTypeMapped: true });
  });

  it("the mapped content_type beats the collection default (it is a declared value)", () => {
    const doc = parseDocument(MEMORY_TOPIC, REL, "hub", MAP);
    expect(doc.meta.content_type).toBe("preference");
  });
});

describe("A2: a collection WITHOUT a map is unchanged from phase 1", () => {
  it("title undefined, content_type = today's fallback, vocabGap = phase-1 record", () => {
    const doc = parseDocument(MEMORY_TOPIC, REL);
    expect(doc.meta.title).toBeUndefined();
    expect(doc.meta.content_type).toBe(inferContentType(REL));
    // The exact phase-1 (v0.36.34) vocabGap for this shape.
    expect(doc.vocabGap).toEqual({
      titleMissing: true,
      contentTypeMissing: true,
      declaredKeys: ["name", "metadata.type"],
    });
    expect(doc.frontmatterMap).toBeUndefined();
  });

  it("the collection default still wins exactly as before", () => {
    const doc = parseDocument(MEMORY_TOPIC, REL, "note");
    expect(doc.meta.content_type).toBe("note");
    expect(doc.vocabGap?.contentTypeMissing).toBe(true);
  });

  it("a 3-arg call and a 4-arg undefined-map call are identical (default off)", () => {
    for (const c of [MEMORY_TOPIC, CANONICAL_AND_MAPPED, BOGUS_TYPE, IDENTITY_TYPE]) {
      expect(parseDocument(c, REL, "hub", undefined)).toEqual(parseDocument(c, REL, "hub"));
    }
  });

  it("the summary line of an unmapped collection has no frontmatter_map tail", () => {
    const counts = emptyFrontmatterVocabCounts();
    noteFrontmatterVocab(counts, parseDocument(MEMORY_TOPIC, REL));
    expect(counts.titleMapped + counts.contentTypeMapped + counts.contentTypeMapRejected).toBe(0);
    expect(formatFrontmatterVocab(counts)).toBe(
      "frontmatter: 1 title-less (1 declare name:), 1 content_type-less " +
        "(0 declare type:, 1 declare metadata.type); 0 without frontmatter, 0 unparseable (of 1 examined)",
    );
  });
});

describe("A3: the canonical key always wins", () => {
  it("title: Bar beats name: Foo; content_type: decision beats metadata.type", () => {
    const doc = parseDocument(CANONICAL_AND_MAPPED, REL, undefined, MAP);
    expect(doc.meta.title).toBe("Bar");
    expect(doc.meta.content_type).toBe("decision");
    expect(doc.frontmatterMap).toEqual({ titleMapped: false, contentTypeMapped: false });
  });
});

describe("A4: an unmappable value is not stored and is counted", () => {
  it("metadata.type: bogus -> collection default, rejected recorded", () => {
    const doc = parseDocument(BOGUS_TYPE, "x/baz.md", "hub", MAP);
    expect(doc.meta.content_type).toBe("hub");
    expect(doc.meta.title).toBe("Baz");
    expect(doc.frontmatterMap).toEqual({
      titleMapped: true,
      contentTypeMapped: false,
      contentTypeRejected: "bogus",
    });
    expect(doc.vocabGap?.contentTypeMissing).toBe(true);
  });

  it("with no collection default it falls to filename inference", () => {
    const rel = "research/baz.md";
    const doc = parseDocument(BOGUS_TYPE, rel, undefined, MAP);
    expect(doc.meta.content_type).toBe(inferContentType(rel));
  });

  it("the rejected counter increments and the summary shows the map tail", () => {
    const counts = emptyFrontmatterVocabCounts();
    noteFrontmatterVocab(counts, parseDocument(MEMORY_TOPIC, REL, undefined, MAP));
    noteFrontmatterVocab(counts, parseDocument(BOGUS_TYPE, "x/baz.md", undefined, MAP));
    noteFrontmatterVocab(counts, parseDocument(CANONICAL_AND_MAPPED, "x/bar.md", undefined, MAP));
    expect(counts.titleMapped).toBe(2);
    expect(counts.contentTypeMapped).toBe(1);
    expect(counts.contentTypeMapRejected).toBe(1);
    expect(counts.titleless).toBe(0);
    expect(counts.contentTypeless).toBe(1);
    expect(formatFrontmatterVocab(counts)).toBe(
      "frontmatter: 0 title-less (0 declare name:), 1 content_type-less " +
        "(0 declare type:, 1 declare metadata.type); 0 without frontmatter, 0 unparseable (of 3 examined); " +
        "frontmatter_map filled 2 title, 1 content_type, rejected 1 content_type value(s)",
    );
  });

  it("a fully-mapped collection still prints its line (mapped fills are visible)", () => {
    const counts = emptyFrontmatterVocabCounts();
    noteFrontmatterVocab(counts, parseDocument(MEMORY_TOPIC, REL, undefined, MAP));
    expect(counts.titleless + counts.contentTypeless).toBe(0);
    expect(formatFrontmatterVocab(counts)).toContain(
      "frontmatter_map filled 1 title, 1 content_type",
    );
  });
});

describe("A5: a malformed frontmatter_map fails loud", () => {
  const where = 'collection "memory-topics"';
  const cases: Array<[string, unknown, RegExp]> = [
    ["not a mapping", "name", /must be a mapping/],
    ["null (empty key)", null, /must be a mapping/],
    ["unknown key", { titel: "name" }, /unknown key "titel"/],
    ["non-string title", { title: 5 }, /frontmatter_map\.title must be a non-empty string/],
    [
      "empty content_type",
      { content_type: "" },
      /frontmatter_map\.content_type must be a non-empty string/,
    ],
    ["bad dotted path", { title: "metadata..name" }, /empty path segment/],
    [
      "invalid content_type value",
      { content_type: "metadata.type", content_type_values: { feedback: "bogus" } },
      /content_type_values\.feedback = "bogus" is not a valid content_type/,
    ],
    [
      "values without a source key",
      { content_type_values: { feedback: "note" } },
      /no effect without frontmatter_map\.content_type/,
    ],
    [
      "values not a mapping",
      { content_type: "type", content_type_values: ["note"] },
      /must be a mapping of/,
    ],
  ];
  for (const [label, raw, re] of cases) {
    it(`rejects ${label}`, () => {
      expect(() => validateFrontmatterMap(raw, where)).toThrow(re);
      expect(() => validateFrontmatterMap(raw, where)).toThrow(/collection "memory-topics"/);
    });
  }

  it("names the valid content types in the message", () => {
    expect(() =>
      validateFrontmatterMap({ content_type: "type", content_type_values: { a: "x" } }, where),
    ).toThrow(CONTENT_TYPE_VALUES.join(", "));
  });

  it("accepts a well-formed map", () => {
    expect(validateFrontmatterMap(MAP, where)).toEqual(MAP);
    expect(validateFrontmatterMap({ title: "name" }, where)).toEqual({ title: "name" });
  });

  describe("at config load", () => {
    const DIR = `/tmp/clawmem-wzwh8.1-config-${process.pid}`;
    const saved = process.env.CLAWMEM_CONFIG_DIR;
    beforeEach(() => {
      rmSync(DIR, { recursive: true, force: true });
      mkdirSync(DIR, { recursive: true });
      process.env.CLAWMEM_CONFIG_DIR = DIR;
      clearConfigCache();
    });
    afterEach(() => {
      clearConfigCache();
      rmSync(DIR, { recursive: true, force: true });
    });
    afterAll(() => {
      if (saved === undefined) delete process.env.CLAWMEM_CONFIG_DIR;
      else process.env.CLAWMEM_CONFIG_DIR = saved;
      clearConfigCache();
    });

    it("loadConfig throws, naming the collection and the field", () => {
      writeFileSync(
        join(DIR, "config.yaml"),
        [
          "collections:",
          "  memory-topics:",
          "    path: /tmp/nowhere",
          '    pattern: "**/*.md"',
          "    frontmatter_map:",
          "      content_type: metadata.type",
          "      content_type_values:",
          "        feedback: bogus",
          "",
        ].join("\n"),
      );
      expect(() => loadConfig()).toThrow(
        /collection "memory-topics": frontmatter_map\.content_type_values\.feedback = "bogus"/,
      );
    });

    it("a valid map loads, reaches collectionIndexOptions, and survives addCollection", () => {
      writeFileSync(
        join(DIR, "config.yaml"),
        [
          "collections:",
          "  memory-topics:",
          "    path: /tmp/nowhere",
          '    pattern: "**/*.md"',
          "    content_type: note",
          "    frontmatter_map:",
          "      title: name",
          "      content_type: metadata.type",
          "      content_type_values: { feedback: preference }",
          "  plain:",
          "    path: /tmp/elsewhere",
          '    pattern: "**/*.md"',
          "",
        ].join("\n"),
      );
      const cfg = loadConfig();
      expect(cfg.collections["memory-topics"]!.frontmatter_map).toEqual(MAP);
      expect(collectionIndexOptions(cfg.collections["memory-topics"]!)).toEqual({
        defaultContentType: "note",
        frontmatterMap: MAP,
      });
      // No cross-collection bleed: the other collection has no map.
      expect(collectionIndexOptions(cfg.collections.plain!).frontmatterMap).toBeUndefined();

      addCollection("memory-topics", "/tmp/nowhere", "**/*.md");
      clearConfigCache();
      expect(loadConfig().collections["memory-topics"]!.frontmatter_map).toEqual(MAP);
    });
  });
});

describe("A6: every indexing path honours the map", () => {
  describe("sqlite indexCollection (insert and changed-content update paths)", () => {
    const ROOT = `/tmp/clawmem-wzwh8.1-index-${process.pid}`;
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

    const row = (coll: string, path: string) =>
      store.db
        .prepare(
          "SELECT title, content_type FROM documents WHERE collection = ? AND path = ? AND active = 1",
        )
        .get(coll, path) as { title: string; content_type: string };

    it("stores the mapped title/content_type and counts them; unmapped collection unchanged", async () => {
      writeFileSync(`${CONTENT}/feedback_foo.md`, MEMORY_TOPIC);
      writeFileSync(`${CONTENT}/baz.md`, BOGUS_TYPE);

      const mapped = await indexCollection(store, "topics", CONTENT, "**/*.md", {
        ...collectionIndexOptions({ content_type: "hub", frontmatter_map: MAP }),
      });
      expect(mapped.added).toBe(2);
      expect(row("topics", "feedback_foo.md")).toEqual({
        title: "Foo",
        content_type: "preference",
      });
      // Rejected value -> collection default, never "bogus".
      expect(row("topics", "baz.md")).toEqual({ title: "Baz", content_type: "hub" });
      expect(mapped.frontmatterVocab?.titleMapped).toBe(2);
      expect(mapped.frontmatterVocab?.contentTypeMapped).toBe(1);
      expect(mapped.frontmatterVocab?.contentTypeMapRejected).toBe(1);

      // Same files, a collection with no map: pre-change behaviour.
      const plain = await indexCollection(store, "plain", CONTENT, "**/*.md");
      expect(plain.added).toBe(2);
      expect(row("plain", "feedback_foo.md")).toEqual({
        title: "Heading One",
        content_type: inferContentType("feedback_foo.md"),
      });
      expect(plain.frontmatterVocab?.titleMapped).toBe(0);
      expect(plain.frontmatterVocab?.contentTypeMapRejected).toBe(0);
    });

    it("the changed-content update path maps too", async () => {
      const opts = collectionIndexOptions({ frontmatter_map: MAP });
      writeFileSync(`${CONTENT}/feedback_foo.md`, MEMORY_TOPIC);
      await indexCollection(store, "topics", CONTENT, "**/*.md", opts);
      writeFileSync(
        `${CONTENT}/feedback_foo.md`,
        MEMORY_TOPIC.replace("name: Foo", "name: Foo Two"),
      );
      const again = await indexCollection(store, "topics", CONTENT, "**/*.md", opts);
      expect(again.updated).toBe(1);
      expect(row("topics", "feedback_foo.md")).toEqual({
        title: "Foo Two",
        content_type: "preference",
      });
      expect(again.frontmatterVocab?.titleMapped).toBe(1);
    });
  });

  it("pg reindex: parseForReindex (its sole parse call) maps and counts", () => {
    const stats = { frontmatterParseFailures: {}, frontmatterVocab: emptyFrontmatterVocabCounts() };
    const mapped = parseForReindex(stats, MEMORY_TOPIC, REL, MAP);
    expect(mapped.title).toBe("Foo");
    expect(mapped.meta.content_type).toBe("preference");
    expect(stats.frontmatterVocab.titleMapped).toBe(1);
    expect(stats.frontmatterVocab.contentTypeMapped).toBe(1);

    const unmapped = parseForReindex(stats, MEMORY_TOPIC, REL);
    expect(unmapped.title).toBe("Heading One");
    expect(unmapped.meta.content_type).toBe(inferContentType(REL));
  });

  describe("pg reindex honours the collection default content_type (vn4rz.76)", () => {
    const mk = () => ({ frontmatterParseFailures: {}, frontmatterVocab: emptyFrontmatterVocabCounts() });
    const PLAIN = "---\ntitle: T\n---\n\nBody.\n";
    const OWN = "---\ntitle: T\ncontent_type: decision\n---\n\nBody.\n";
    const BAD_MAPPED = "---\nname: Foo\nmetadata:\n  type: bogus-type\n---\n\nBody.\n";
    // Pick a default that filename inference would NOT produce for REL.
    const DEF = inferContentType(REL) === "reference" ? "note" : "reference";

    it("applies the default when the doc has no content_type", () => {
      expect(parseForReindex(mk(), PLAIN, REL, undefined, DEF).meta.content_type).toBe(DEF);
    });
    it("keeps the doc's own content_type over the default", () => {
      expect(parseForReindex(mk(), OWN, REL, undefined, DEF).meta.content_type).toBe("decision");
    });
    it("no default -> filename inference unchanged", () => {
      expect(parseForReindex(mk(), PLAIN, REL).meta.content_type).toBe(inferContentType(REL));
    });
    it("a rejected mapped value falls back to the collection default", () => {
      const r = parseForReindex(mk(), BAD_MAPPED, REL, MAP, DEF);
      expect(r.meta.content_type).toBe(DEF);
      expect(parseForReindex(mk(), BAD_MAPPED, REL, MAP).meta.content_type).toBe(inferContentType(REL));
    });
    it("parity: pg parseForReindex content_type === sqlite's resolution", () => {
      for (const raw of [PLAIN, OWN, BAD_MAPPED]) {
        for (const def of [undefined, DEF]) {
          const pg = parseForReindex(mk(), raw, REL, MAP, def).meta.content_type;
          const sqlite = parseDocument(raw, REL, def, MAP).meta.content_type || inferContentType(REL);
          expect(pg).toBe(sqlite);
        }
      }
    });
  });

  it("pg origin-load: parseOriginDocument (its sole parse call) maps the title", () => {
    expect(parseOriginDocument(MEMORY_TOPIC, REL, MAP).meta.title).toBe("Foo");
    expect(parseOriginDocument(MEMORY_TOPIC, REL).meta.title).toBeUndefined();
  });

  describe("call-site wiring (source scan)", () => {
    const SRC = join(import.meta.dir, "../../src");
    const files: string[] = [];
    const walk = (d: string) => {
      for (const e of readdirSync(d)) {
        const p = join(d, e);
        if (statSync(p).isDirectory()) walk(p);
        else if (p.endsWith(".ts")) files.push(p);
      }
    };
    walk(SRC);
    const read = (rel: string) => readFileSync(join(SRC, rel), "utf-8");

    /**
     * indexCollection calls that deliberately take NO collection config: the
     * `clawmem mine` import into a transient staging root, and the MCP vault
     * sync of an arbitrary caller-supplied root. Neither passes a collection
     * default content_type today either.
     */
    const UNCONFIGURED = [
      /indexCollection\(s, collectionName, stagingDir, "\*\*\/\*\.md", \{ importMode: true \}\)/,
      /indexCollection\(s, collName, root, pattern \|\| "\*\*\/\*\.md"\)/,
    ];

    it("every configured-collection indexCollection() call goes through collectionIndexOptions", () => {
      let checked = 0;
      for (const f of files) {
        const src = readFileSync(f, "utf-8");
        for (const m of src.matchAll(/await indexCollection\([^;]*?\);/g)) {
          checked++;
          const call = m[0];
          if (UNCONFIGURED.some((re) => re.test(call))) continue;
          expect({ file: f, call, viaHelper: call.includes("collectionIndexOptions(") }).toEqual({
            file: f,
            call,
            viaHelper: true,
          });
        }
      }
      expect(checked).toBeGreaterThanOrEqual(8);
    });

    it("no call site threads defaultContentType by hand (it would skip the map)", () => {
      // collections.ts is the helper itself — the one sanctioned place.
      for (const f of files.filter((x) => !x.endsWith("/collections.ts"))) {
        expect({
          file: f,
          hit: /defaultContentType:\s*\w+\.content_type/.test(readFileSync(f, "utf-8")),
        }).toEqual({
          file: f,
          hit: false,
        });
      }
    });

    it("indexCollection passes the map to all three of its parseDocument calls", () => {
      const src = read("indexer.ts");
      const calls = [...src.matchAll(/parseDocument\(content, relativePath[^)]*\)/g)].map(
        (m) => m[0],
      );
      expect(calls.length).toBe(3);
      for (const c of calls) expect(c).toContain("options?.frontmatterMap");
    });

    it("pg reindex and origin-load each have exactly one parseDocument call, fed the map", () => {
      const reindex = read("pg/reindex.ts");
      expect([...reindex.matchAll(/\bparseDocument\(/g)].length).toBe(1);
      expect(reindex).toContain("parseDocument(raw, rel, defaultContentType, frontmatterMap)");
      expect(reindex).toContain("parseForReindex(stats, raw, rel, frontmatterMap, defaultContentType)");
      expect(reindex).toContain(
        "reindexCollection(c.name, c.path, c.pattern, opts, c.frontmatter_map, c.content_type)",
      );

      const origin = read("pg/origin.ts");
      expect([...origin.matchAll(/\bparseDocument\(/g)].length).toBe(1);
      expect(origin).toContain("parseOriginDocument(raw, rel, opts.frontmatterMap)");
      expect(read("pg/cli.ts")).toContain("frontmatterMap: c.frontmatter_map");
    });
  });
});
