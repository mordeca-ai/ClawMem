/**
 * master-harness-vn4rz.84 (ruling B): frontmatter-driven supersession.
 *
 * A collection's frontmatter_map may declare `invalidated_at` / `superseded_by`
 * source keys. A document that fills one is written with invalidated_at set
 * (so every search path, which filters `invalidated_at IS NULL`, drops it) and
 * `invalidation_source = 'frontmatter'`. This file pins:
 *   S1  the validator accepts the new keys and rejects malformed ones;
 *   S2  parse: tombstone shapes, an unparseable invalidated_at is rejected and
 *       counted (never thrown), the bare canonical keys are NOT read;
 *   S3  index: the tombstone lands, superseded_by resolves (path or unique
 *       title, any file order), the doc drops out of FTS;
 *   S4  removing the tombstone + re-index restores the document;
 *   S5  provenance: a re-index never clears another writer's invalidation;
 *   S6  map absent = no behaviour change.
 * Vector and hybrid (`vsearch` / `query`) exclusion is end-to-end in
 * frontmatter-supersession-search.test.ts.
 */
import { describe, it, expect, beforeEach, afterEach, afterAll } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  parseDocument,
  indexCollection,
  emptyFrontmatterVocabCounts,
  noteFrontmatterVocab,
  formatFrontmatterVocab,
  FRONTMATTER_INVALIDATION_SOURCE,
} from "../../src/indexer.ts";
import {
  validateFrontmatterMap,
  readFrontmatterInvalidation,
  type FrontmatterMap,
} from "../../src/frontmatter-map.ts";
import { loadConfig, clearConfigCache } from "../../src/collections.ts";
import { createStore, type Store } from "../../src/store.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";

const MAP: FrontmatterMap = {
  title: "name",
  invalidated_at: "invalidated_at",
  superseded_by: "superseded_by",
};

const where = 'collection "memory-topics"';

// ---------------------------------------------------------------------------
// S1 — validator
// ---------------------------------------------------------------------------
describe("S1: validateFrontmatterMap accepts the supersession keys and rejects malformed ones", () => {
  it("accepts both keys, alone or together, dotted paths included", () => {
    expect(validateFrontmatterMap(MAP, where)).toEqual(MAP);
    expect(validateFrontmatterMap({ superseded_by: "meta.replaced_by" }, where)).toEqual({
      superseded_by: "meta.replaced_by",
    });
    expect(validateFrontmatterMap({ invalidated_at: "lifecycle.retired" }, where)).toEqual({
      invalidated_at: "lifecycle.retired",
    });
  });

  for (const [field, bad] of [
    ["invalidated_at", ""],
    ["invalidated_at", 7],
    ["invalidated_at", "a..b"],
    ["superseded_by", "   "],
    ["superseded_by", ["x"]],
    ["superseded_by", ".lead"],
  ] as const) {
    it(`rejects ${field} = ${JSON.stringify(bad)} naming the field`, () => {
      expect(() => validateFrontmatterMap({ [field]: bad }, where)).toThrow(
        new RegExp(`frontmatter_map\\.${field}`),
      );
    });
  }

  it("still rejects unknown keys and lists the new ones as allowed", () => {
    expect(() => validateFrontmatterMap({ supersedes: "x" }, where)).toThrow(
      /unknown key "supersedes" \(allowed: .*invalidated_at, superseded_by\)/,
    );
  });

  describe("at config load", () => {
    const DIR = mkdtempSync(join(tmpdir(), "clawmem-vn4rz84-config-"));
    const saved = process.env.CLAWMEM_CONFIG_DIR;
    beforeEach(() => {
      rmSync(DIR, { recursive: true, force: true });
      mkdirSync(DIR, { recursive: true });
      process.env.CLAWMEM_CONFIG_DIR = DIR;
      clearConfigCache();
    });
    afterAll(() => {
      if (saved === undefined) delete process.env.CLAWMEM_CONFIG_DIR;
      else process.env.CLAWMEM_CONFIG_DIR = saved;
      clearConfigCache();
      rmSync(DIR, { recursive: true, force: true });
    });
    const cfg = (mapLines: string[]) =>
      writeFileSync(
        join(DIR, "config.yaml"),
        [
          "collections:",
          "  memory-topics:",
          "    path: /tmp/nowhere",
          '    pattern: "**/*.md"',
          "    frontmatter_map:",
          ...mapLines.map((l) => `      ${l}`),
          "",
        ].join("\n"),
      );

    it("a valid supersession map loads", () => {
      cfg(["invalidated_at: invalidated_at", "superseded_by: superseded_by"]);
      expect(loadConfig().collections["memory-topics"]!.frontmatter_map).toEqual({
        invalidated_at: "invalidated_at",
        superseded_by: "superseded_by",
      });
    });

    it("a malformed one fails loud, naming the collection and field", () => {
      cfg(["superseded_by: 42"]);
      expect(() => loadConfig()).toThrow(
        /collection "memory-topics": frontmatter_map\.superseded_by/,
      );
    });
  });
});

// ---------------------------------------------------------------------------
// S2 — parse
// ---------------------------------------------------------------------------
const fm = (lines: string[], body = "Body text.") => `---\n${lines.join("\n")}\n---\n\n${body}\n`;

describe("S2: parseDocument reads the tombstone only through the map", () => {
  it("superseded_by alone = tombstone at index time with the reference", () => {
    const d = parseDocument(
      fm(["name: Old", "superseded_by: topics/new.md"]),
      "old.md",
      undefined,
      MAP,
    );
    expect(d.meta.invalidation).toEqual({ at: null, supersededBy: "topics/new.md" });
  });

  it("an unquoted date-only invalidated_at (gray-matter Date) normalizes to UTC midnight", () => {
    const d = parseDocument(
      fm(["name: Old", "invalidated_at: 2026-10-01"]),
      "old.md",
      undefined,
      MAP,
    );
    expect(d.meta.invalidation).toEqual({ at: "2026-10-01T00:00:00.000Z", supersededBy: null });
  });

  it("a quoted RFC 3339 datetime is honoured", () => {
    const d = parseDocument(
      fm(["name: Old", 'invalidated_at: "2026-10-01T09:30:00+02:00"', "superseded_by: New"]),
      "old.md",
      undefined,
      MAP,
    );
    expect(d.meta.invalidation).toEqual({ at: "2026-10-01T07:30:00.000Z", supersededBy: "New" });
  });

  it("an unparseable invalidated_at is rejected and counted, never thrown", () => {
    const content = fm(["name: Old", "invalidated_at: next tuesday"]);
    let d: ReturnType<typeof parseDocument> | undefined;
    expect(() => {
      d = parseDocument(content, "old.md", undefined, MAP);
    }).not.toThrow();
    expect(d!.frontmatter).toBe("parsed");
    expect(d!.meta.invalidation).toBeNull();
    expect(d!.frontmatterMap?.invalidatedAtRejected).toBe("next tuesday");
    const counts = emptyFrontmatterVocabCounts();
    noteFrontmatterVocab(counts, d!);
    expect(counts.invalidatedAtMapRejected).toBe(1);
    expect(counts.invalidatedByFrontmatter).toBe(0);
    expect(formatFrontmatterVocab(counts)).toContain("rejected 1 invalidated_at value(s)");
  });

  it("an unparseable invalidated_at beside superseded_by still tombstones, at index time", () => {
    const d = parseDocument(
      // Quoted: js-yaml turns an UNQUOTED 2026-13-45 into a rolled-over Date
      // (2027-02-14) before frontmatter-map ever sees it, the same as authored_at.
      fm(["name: Old", 'invalidated_at: "2026-13-45"', "superseded_by: new.md"]),
      "old.md",
      undefined,
      MAP,
    );
    expect(d.meta.invalidation).toEqual({ at: null, supersededBy: "new.md" });
    expect(d.frontmatterMap?.invalidatedAtRejected).toBe("2026-13-45");
  });

  it("empty / null / false values declare nothing", () => {
    for (const v of [
      "superseded_by:",
      'superseded_by: ""',
      "superseded_by: false",
      "invalidated_at:",
    ]) {
      const d = parseDocument(fm(["name: X", v]), "x.md", undefined, MAP);
      expect({ v, inv: d.meta.invalidation }).toEqual({ v, inv: null });
    }
  });

  it("a dotted source key reaches into a nested mapping", () => {
    const map = { superseded_by: "lifecycle.replaced_by" };
    const d = parseDocument(fm(["lifecycle:", "  replaced_by: next.md"]), "x.md", undefined, map);
    expect(d.meta.invalidation).toEqual({ at: null, supersededBy: "next.md" });
  });

  it("DECISION: the bare canonical keys are NOT read without a map, nor with a map lacking the keys", () => {
    const content = fm(["title: Old", "invalidated_at: 2026-10-01", "superseded_by: new.md"]);
    expect(parseDocument(content, "old.md").meta.invalidation).toBeNull();
    expect(
      parseDocument(content, "old.md", undefined, { title: "name" }).meta.invalidation,
    ).toBeNull();
    expect(readFrontmatterInvalidation({ superseded_by: "x" }, undefined)).toEqual({
      invalidation: null,
    });
  });

  it("a frontmatter parse failure leaves invalidation undefined (stored state untouched)", () => {
    const orig = console.warn;
    console.warn = () => {};
    try {
      const d = parseDocument(
        "---\nname: [unclosed\nsuperseded_by: x\n---\nbody\n",
        "bad.md",
        undefined,
        MAP,
      );
      expect(d.frontmatter).toBe("failed");
      expect(d.meta.invalidation).toBeUndefined();
    } finally {
      console.warn = orig;
    }
  });
});

// ---------------------------------------------------------------------------
// S3–S6 — the sqlite write path
// ---------------------------------------------------------------------------
const deadLlm = {
  embed: async () => {
    throw new Error("no embedding endpoint in test");
  },
  query: async () => null,
  expandQuery: async () => [],
} as any;

type Row = {
  id: number;
  invalidated_at: string | null;
  invalidation_source: string | null;
  superseded_by: number | null;
  invalidated_by: number | null;
};

describe("S3-S6: indexCollection applies, resolves, clears and protects", () => {
  let ROOT: string;
  let CONTENT: string;
  let store: Store;
  const COL = "memory-topics";

  beforeEach(() => {
    ROOT = mkdtempSync(join(tmpdir(), "clawmem-vn4rz84-index-"));
    CONTENT = join(ROOT, "content");
    mkdirSync(join(CONTENT, "topics"), { recursive: true });
    setDefaultLlamaCpp(deadLlm);
    store = createStore(join(ROOT, "vault.sqlite"));
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

  const write = (rel: string, content: string) => writeFileSync(join(CONTENT, rel), content);
  const index = (opts: { map?: FrontmatterMap; force?: boolean } = {}) =>
    indexCollection(store, COL, CONTENT, "**/*.md", {
      ...(opts.map === undefined ? { frontmatterMap: MAP } : { frontmatterMap: opts.map }),
      force: opts.force,
    });
  const indexNoMap = () => indexCollection(store, COL, CONTENT, "**/*.md");
  const row = (path: string): Row =>
    store.db
      .prepare(
        "SELECT id, invalidated_at, invalidation_source, superseded_by, invalidated_by FROM documents WHERE collection = ? AND path = ?",
      )
      .get(COL, path) as Row;
  const ftsPaths = (q: string) => store.searchFTS(q, 20).map((r) => r.filepath);

  // `a-old.md` sorts BEFORE its replacement `topics/z-new.md`: resolution must not depend on walk order.
  const OLD = (extra: string[] = []) =>
    fm(["name: Old quokka deploy notes", ...extra], "The quokka deploy runbook, first edition.");
  const NEW = fm(["name: New quokka deploy notes"], "The quokka deploy runbook, second edition.");

  it("S3: the tombstone sets invalidated_at + source, resolves superseded_by by path, and leaves FTS", async () => {
    write("a-old.md", OLD(["superseded_by: ./topics/z-new.md"]));
    write("topics/z-new.md", NEW);
    const stats = await index();
    const old = row("a-old.md");
    const neu = row("topics/z-new.md");
    expect(old.invalidated_at).not.toBeNull();
    expect(old.invalidation_source).toBe(FRONTMATTER_INVALIDATION_SOURCE);
    expect(old.superseded_by).toBe(neu.id);
    expect(neu.invalidated_at).toBeNull();
    // Soft: still active, still in the table.
    expect(
      (
        store.db.prepare("SELECT active FROM documents WHERE id = ?").get(old.id) as {
          active: number;
        }
      ).active,
    ).toBe(1);
    const hits = ftsPaths("quokka runbook");
    expect(hits.some((p) => p.endsWith("topics/z-new.md"))).toBe(true);
    expect(hits.some((p) => p.endsWith("a-old.md"))).toBe(false);
    expect(stats.frontmatterVocab?.invalidatedByFrontmatter).toBe(1);
    expect(stats.frontmatterVocab?.supersededByResolved).toBe(1);
    expect(formatFrontmatterVocab(stats.frontmatterVocab!)).toContain(
      "supersession 1 tombstoned, superseded_by 1 resolved / 0 unresolved",
    );
  });

  it("S3: an explicit invalidated_at is the stored instant", async () => {
    write("a-old.md", OLD(["invalidated_at: 2026-09-15"]));
    await index();
    expect(row("a-old.md").invalidated_at).toBe("2026-09-15T00:00:00.000Z");
    expect(row("a-old.md").superseded_by).toBeNull();
  });

  it("S3: superseded_by resolves by a unique title; an ambiguous or missing title stays NULL but still tombstones", async () => {
    write("a-old.md", OLD(["superseded_by: New quokka deploy notes"]));
    write("topics/z-new.md", NEW);
    write("b-amb.md", fm(["name: B", "superseded_by: Twin"], "b body"));
    write("t1.md", fm(["name: Twin"], "twin one"));
    write("t2.md", fm(["name: Twin"], "twin two"));
    write("c-missing.md", fm(["name: C", "superseded_by: nowhere.md"], "c body"));
    const stats = await index();
    expect(row("a-old.md").superseded_by).toBe(row("topics/z-new.md").id);
    expect(row("b-amb.md").superseded_by).toBeNull();
    expect(row("b-amb.md").invalidated_at).not.toBeNull();
    expect(row("c-missing.md").superseded_by).toBeNull();
    expect(row("c-missing.md").invalidated_at).not.toBeNull();
    expect(stats.frontmatterVocab?.supersededByResolved).toBe(1);
    expect(stats.frontmatterVocab?.supersededByUnresolved).toBe(2);
  });

  it("S3: an unresolved superseded_by is linked on a later pass once the replacement exists (old file unchanged)", async () => {
    write("a-old.md", OLD(["superseded_by: topics/z-new.md"]));
    await index();
    const first = row("a-old.md");
    expect(first.superseded_by).toBeNull();
    write("topics/z-new.md", NEW);
    const stats = await index();
    expect(stats.unchanged).toBe(1);
    const after = row("a-old.md");
    expect(after.superseded_by).toBe(row("topics/z-new.md").id);
    // The index-time stamp is not bumped by the retry.
    expect(after.invalidated_at).toBe(first.invalidated_at);
  });

  it("S3: a tombstone added in the same edit as an authored_at change still lands (dated-only short-circuit)", async () => {
    write("a-old.md", OLD(["authored_at: 2026-01-01"]));
    await index();
    write("a-old.md", OLD(["authored_at: 2026-02-01", "invalidated_at: 2026-09-15"]));
    const stats = await index();
    expect(stats.dated).toBe(1);
    expect(row("a-old.md").invalidated_at).toBe("2026-09-15T00:00:00.000Z");
  });

  it("S4: removing the tombstone and re-indexing restores the document to search", async () => {
    write("a-old.md", OLD(["superseded_by: topics/z-new.md"]));
    write("topics/z-new.md", NEW);
    await index();
    expect(ftsPaths("first edition").some((p) => p.endsWith("a-old.md"))).toBe(false);
    write("a-old.md", OLD());
    const stats = await index();
    expect(stats.updated).toBe(1);
    expect(row("a-old.md")).toMatchObject({
      invalidated_at: null,
      invalidation_source: null,
      superseded_by: null,
    });
    expect(ftsPaths("first edition").some((p) => p.endsWith("a-old.md"))).toBe(true);
  });

  it("S4: a frontmatter parse failure on a changed file leaves the tombstone in place", async () => {
    write("a-old.md", OLD(["superseded_by: gone.md"]));
    await index();
    const before = row("a-old.md");
    const orig = console.warn;
    console.warn = () => {};
    try {
      write("a-old.md", "---\nname: [unclosed\n---\nThe quokka deploy runbook, first edition.\n");
      await index();
    } finally {
      console.warn = orig;
    }
    expect(row("a-old.md").invalidated_at).toBe(before.invalidated_at);
    expect(row("a-old.md").invalidation_source).toBe(FRONTMATTER_INVALIDATION_SOURCE);
  });

  for (const source of ["contradiction", null] as const) {
    it(`S5: an invalidation from source ${JSON.stringify(source)} survives re-index (changed file, --force, no tombstone)`, async () => {
      write("a-old.md", OLD());
      await index();
      const id = row("a-old.md").id;
      store.db
        .prepare(
          "UPDATE documents SET invalidated_at = '2026-05-05 05:05:05', invalidated_by = 999, invalidation_source = ? WHERE id = ?",
        )
        .run(source, id);
      write("a-old.md", OLD(["description: edited"]));
      expect((await index()).updated).toBe(1);
      await index({ force: true });
      await indexNoMap();
      expect(row("a-old.md")).toMatchObject({
        invalidated_at: "2026-05-05 05:05:05",
        invalidated_by: 999,
        invalidation_source: source,
      });
      expect(ftsPaths("first edition").some((p) => p.endsWith("a-old.md"))).toBe(false);
    });
  }

  it("S5: a tombstone never overwrites another source's invalidation", async () => {
    write("a-old.md", OLD());
    await index();
    const id = row("a-old.md").id;
    store.db
      .prepare(
        "UPDATE documents SET invalidated_at = '2026-05-05 05:05:05', invalidated_by = 7, invalidation_source = 'contradiction' WHERE id = ?",
      )
      .run(id);
    write("a-old.md", OLD(["invalidated_at: 2026-09-15", "superseded_by: topics/z-new.md"]));
    write("topics/z-new.md", NEW);
    await index();
    expect(row("a-old.md")).toMatchObject({
      invalidated_at: "2026-05-05 05:05:05",
      invalidated_by: 7,
      invalidation_source: "contradiction",
      superseded_by: null,
    });
  });

  it("S6: map absent = the bare keys do nothing; no invalidation, no summary tail", async () => {
    write(
      "a-old.md",
      fm(
        ["title: Old", "invalidated_at: 2026-10-01", "superseded_by: topics/z-new.md"],
        "quokka first edition",
      ),
    );
    write("topics/z-new.md", NEW);
    const stats = await indexNoMap();
    for (const p of ["a-old.md", "topics/z-new.md"]) {
      expect(row(p)).toMatchObject({
        invalidated_at: null,
        invalidation_source: null,
        superseded_by: null,
      });
    }
    expect(ftsPaths("first edition").some((p) => p.endsWith("a-old.md"))).toBe(true);
    const v = stats.frontmatterVocab!;
    expect([
      v.invalidatedByFrontmatter,
      v.invalidatedAtMapRejected,
      v.supersededByResolved,
      v.supersededByUnresolved,
    ]).toEqual([0, 0, 0, 0]);
    const line = formatFrontmatterVocab(v);
    expect(line === null || !line.includes("supersession")).toBe(true);
  });
});
