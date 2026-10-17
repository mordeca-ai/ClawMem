/**
 * master-harness-65amk — `keywords` frontmatter is parsed, normalized and embeddable.
 *
 * Mirrors the s1lli description fix: parseDocument extracts `keywords` (YAML list or
 * string, normalized to one comma-joined string), buildDocEmbedTask adds it to the
 * synthesized frontmatter so splitDocument emits a `frontmatter` fragment labeled
 * `keywords`, and the pg reindex synthesizes the same object. A document WITHOUT
 * keywords must cut exactly the fragments it cut before (no re-embed churn).
 */
import { describe, it, expect } from "bun:test";
import { buildDocEmbedTask } from "../../src/clawmem.ts";
import { buildEmbedFrontmatter } from "../../src/embed-input.ts";
import { normalizeKeywords, parseDocument } from "../../src/indexer.ts";
import { reindexEmbedFrontmatter } from "../../src/pg/reindex.ts";
import { splitDocument } from "../../src/splitter.ts";

const BODY = "# Some doc\n\n## Section\n\n" + "Body text long enough to split. ".repeat(12);

const doc = (frontmatter: string) => `---\ntitle: Some Doc\n${frontmatter}---\n\n${BODY}`;

describe("parseDocument — keywords (A1)", () => {
  it("joins a YAML list with ', '", () => {
    const { meta } = parseDocument(
      doc("keywords:\n  - retrieval\n  - hybrid search\n  - rerank\n"),
      "a.md",
    );
    expect(meta.keywords).toBe("retrieval, hybrid search, rerank");
  });

  it("accepts a flow-style list and stringifies scalar elements", () => {
    const { meta } = parseDocument(doc("keywords: [sqlite, 2026, true]\n"), "a.md");
    expect(meta.keywords).toBe("sqlite, 2026, true");
  });

  it("keeps a string verbatim (trimmed)", () => {
    const { meta } = parseDocument(doc('keywords: "  vector, lexical  "\n'), "a.md");
    expect(meta.keywords).toBe("vector, lexical");
  });

  it("is undefined for a non-string / non-list value", () => {
    for (const fm of [
      "keywords: 42\n",
      "keywords: true\n",
      "keywords:\n  nested: map\n",
      "keywords: 2026-01-02\n",
      "keywords:\n",
    ]) {
      expect(parseDocument(doc(fm), "a.md").meta.keywords).toBeUndefined();
    }
  });

  it("is undefined for empty forms and drops empty / object list elements", () => {
    expect(parseDocument(doc('keywords: ""\n'), "a.md").meta.keywords).toBeUndefined();
    expect(parseDocument(doc("keywords: []\n"), "a.md").meta.keywords).toBeUndefined();
    expect(parseDocument(doc('keywords: ["", "  "]\n'), "a.md").meta.keywords).toBeUndefined();
    expect(
      parseDocument(doc('keywords:\n  - a\n  - {x: 1}\n  - ""\n  - b\n'), "a.md").meta.keywords,
    ).toBe("a, b");
  });

  it("is undefined when the frontmatter has no keywords key", () => {
    expect(parseDocument(doc("description: d\n"), "a.md").meta.keywords).toBeUndefined();
  });

  it("normalizeKeywords is the same rule, directly", () => {
    expect(normalizeKeywords(["x", " y ", 3])).toBe("x, y, 3");
    expect(normalizeKeywords(null)).toBeUndefined();
    expect(normalizeKeywords(new Date())).toBeUndefined();
    expect(normalizeKeywords({ a: 1 })).toBeUndefined();
  });
});

describe("buildDocEmbedTask — keywords fragment (A2)", () => {
  it("emits a frontmatter fragment labeled `keywords` carrying the normalized value", () => {
    const task = buildDocEmbedTask(
      "h1",
      BODY,
      "docs/a.md",
      "Some Doc",
      "A summary.",
      "docs",
      "retrieval, rerank",
    );
    const kw = task.fragments.filter((f) => f.type === "frontmatter" && f.label === "keywords");
    expect(kw).toHaveLength(1);
    expect(kw[0]!.content).toBe("keywords: retrieval, rerank");
    // title + description stay, in key order before keywords (seq layout is load-bearing).
    const labels = task.fragments.filter((f) => f.type === "frontmatter").map((f) => f.label);
    expect(labels).toEqual(["title", "description", "keywords"]);
  });

  it("without keywords, the fragment set is identical to the pre-change output", () => {
    const withUndefined = buildDocEmbedTask(
      "h2",
      BODY,
      "docs/a.md",
      "Some Doc",
      "A summary.",
      "docs",
      undefined,
    );
    const preChangeCall = buildDocEmbedTask(
      "h2",
      BODY,
      "docs/a.md",
      "Some Doc",
      "A summary.",
      "docs",
    );
    const withNull = buildDocEmbedTask(
      "h2",
      BODY,
      "docs/a.md",
      "Some Doc",
      "A summary.",
      "docs",
      null,
    );
    const withEmpty = buildDocEmbedTask(
      "h2",
      BODY,
      "docs/a.md",
      "Some Doc",
      "A summary.",
      "docs",
      "",
    );
    // The literal pre-65amk computation: splitDocument over { title, description }.
    const preChangeSplit = splitDocument(BODY, { title: "Some Doc", description: "A summary." });
    expect(withUndefined.fragments).toEqual(preChangeSplit);
    expect(preChangeCall.fragments).toEqual(preChangeSplit);
    expect(withNull.fragments).toEqual(preChangeSplit);
    expect(withEmpty.fragments).toEqual(preChangeSplit);
    expect(withUndefined.fragments.some((f) => f.label === "keywords")).toBe(false);
  });

  it("buildEmbedFrontmatter omits the key when keywords is empty", () => {
    expect(buildEmbedFrontmatter("T", "a.md", null, "")).toEqual({ title: "T" });
    expect(buildEmbedFrontmatter("T", "a.md", null, "k")).toEqual({ title: "T", keywords: "k" });
  });
});

describe("pg reindex frontmatter synthesis — keywords (A5)", () => {
  it("adds keywords exactly as description is added, matching the sqlite embed input", () => {
    const meta = { description: "A summary.", keywords: "retrieval, rerank" };
    expect(reindexEmbedFrontmatter("Some Doc", meta)).toEqual({
      title: "Some Doc",
      description: "A summary.",
      keywords: "retrieval, rerank",
    });
    expect(reindexEmbedFrontmatter("Some Doc", meta)).toEqual(
      buildEmbedFrontmatter("Some Doc", "docs/a.md", meta.description, meta.keywords),
    );
  });

  it("adds nothing for absent keywords (no fragment churn on the pg path either)", () => {
    expect(reindexEmbedFrontmatter("Some Doc", { description: "A summary." })).toEqual({
      title: "Some Doc",
      description: "A summary.",
    });
    expect(reindexEmbedFrontmatter("Some Doc", {})).toEqual({ title: "Some Doc" });
  });

  it("drives a `keywords` fragment through the shared splitter from parsed source meta", () => {
    const { body, meta } = parseDocument(doc("keywords: [alpha, beta]\n"), "a.md");
    const frags = splitDocument(body, reindexEmbedFrontmatter("Some Doc", meta));
    expect(frags.find((f) => f.label === "keywords")?.content).toBe("keywords: alpha, beta");
  });
});
