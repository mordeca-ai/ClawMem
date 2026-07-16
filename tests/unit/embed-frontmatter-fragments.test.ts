import { describe, it, expect } from "bun:test";
import { buildDocEmbedTask } from "../../src/clawmem.ts";
import { parseDocument } from "../../src/indexer.ts";
import { splitDocument } from "../../src/splitter.ts";

/**
 * Tests for master-harness-z7o4y — frontmatter unembeddable by construction.
 *
 * Mechanism (traced in clawmem.ts cmdEmbed, pre-fix): `content.doc` — what
 * `getHashesNeedingFragments()` hands to the embed pipeline as `body` — is
 * the FRONTMATTER-STRIPPED body (parseDocument's gray-matter call strips
 * frontmatter at INDEX time, before `insertContent` persists it). The old
 * `cmdEmbed` loop re-ran `parseDocument(body, path)` on that already-stripped
 * body hoping to recover title/description as a `frontmatter`-type
 * `splitDocument` fragment — but there is no frontmatter left in `body` to
 * find, so `extractFrontmatter` never fires. This explains ADR-0060's
 * "zero embedding-rank effect" null: the shaped title/description fields
 * were never structurally reachable by the embedder.
 *
 * `buildDocEmbedTask` is the extracted, unit-testable fix.
 */

describe("buildDocEmbedTask — frontmatter fragment generation at embed time", () => {
  it("generates a frontmatter-type fragment carrying the title, from stripped body alone", () => {
    // Simulate real production shape: `body` is what's ACTUALLY stored in
    // content.doc — frontmatter already stripped by the indexer. A
    // pre-fix re-parse of this exact string finds nothing.
    const rawSource = `---
title: "RAG & hybrid retrieval — how vector + lexical search, fusion, and reranking work end-to-end"
description: "Hub index for retrieval-augmented generation reference docs."
---

# rag/

Some body content that is long enough to clear the minimum document length
threshold for fragment splitting, well past two hundred characters so the
splitter actually attempts to extract sections, lists, and frontmatter
fragments instead of short-circuiting on the "too short to split" path.
`;
    const { body } = parseDocument(rawSource, "docs/rag/INDEX.md");
    // Sanity: confirms body is genuinely stripped, matching what
    // getHashesNeedingFragments() would hand the embed pipeline as `body`.
    expect(body).not.toContain("description:");
    expect(body.startsWith("---")).toBe(false);

    const task = buildDocEmbedTask(
      "somehash",
      body,
      "docs/rag/INDEX.md",
      "RAG & hybrid retrieval — how vector + lexical search, fusion, and reranking work end-to-end",
      null,
      "docs"
    );

    const fmFrags = task.fragments.filter(f => f.type === "frontmatter");
    expect(fmFrags.length).toBeGreaterThanOrEqual(1);
    const titleFrag = fmFrags.find(f => f.label === "title");
    expect(titleFrag).toBeDefined();
    expect(titleFrag!.content).toContain("hybrid retrieval");
  });

  it("[stash-proof] the pre-fix re-parse-the-stripped-body approach generates ZERO frontmatter fragments for the same input", () => {
    // Reconstructs the OLD cmdEmbed behavior inline (parseDocument run
    // AGAIN over the already-stripped body) to pin the regression — the
    // mechanism-level proof that this was structurally impossible before
    // the fix, not merely untested.
    const rawSource = `---
title: "RAG & hybrid retrieval — how vector + lexical search, fusion, and reranking work end-to-end"
description: "Hub index for retrieval-augmented generation reference docs."
---

# rag/

Some body content that is long enough to clear the minimum document length
threshold for fragment splitting, well past two hundred characters so the
splitter actually attempts to extract sections, lists, and frontmatter
fragments instead of short-circuiting on the "too short to split" path.
`;
    const { body } = parseDocument(rawSource, "docs/rag/INDEX.md");

    // OLD behavior: re-parse the already-stripped body for frontmatter.
    let oldFrontmatter: Record<string, any> | undefined;
    try {
      const reparsed = parseDocument(body, "docs/rag/INDEX.md");
      oldFrontmatter = reparsed.meta as any;
    } catch { /* no-op */ }
    const oldFragments = splitDocument(body, oldFrontmatter);
    const oldFmFrags = oldFragments.filter(f => f.type === "frontmatter");
    expect(oldFmFrags.length).toBe(0);
  });

  it("falls back to a filename-derived title when the documents row has none", () => {
    const body = "# Some doc\n\n" + "x".repeat(250);
    const task = buildDocEmbedTask("h2", body, "docs/some-doc.md", null, null, "docs");
    expect(task.title).toBe("some-doc");
    const fmFrags = task.fragments.filter(f => f.type === "frontmatter");
    expect(fmFrags.find(f => f.label === "title")?.content).toContain("some-doc");
  });

  it("does not choke on a short document (below the split threshold) — still returns the full fragment", () => {
    const task = buildDocEmbedTask("h3", "short body", "docs/short.md", "Short Title", null, "docs");
    expect(task.fragments.length).toBeGreaterThanOrEqual(1);
    expect(task.fragments[0]!.type).toBe("full");
  });

  // master-harness-s1lli: description is now persisted (documents.description)
  // and recovered at embed time via `docDescription`, mirroring title.
  it("generates a frontmatter-type fragment carrying the description, when docDescription is provided", () => {
    const body = "# Some doc\n\n" + "x".repeat(250);
    const task = buildDocEmbedTask(
      "h4",
      body,
      "docs/some-doc.md",
      "Some Doc",
      "A short summary of what this document covers.",
      "docs"
    );
    const fmFrags = task.fragments.filter(f => f.type === "frontmatter");
    const descFrag = fmFrags.find(f => f.label === "description");
    expect(descFrag).toBeDefined();
    expect(descFrag!.content).toContain("A short summary of what this document covers.");
    // Title fragment should still be present alongside it.
    expect(fmFrags.find(f => f.label === "title")).toBeDefined();
  });

  it("does not generate a description fragment when docDescription is null or undefined", () => {
    const body = "# Some doc\n\n" + "x".repeat(250);
    const taskNull = buildDocEmbedTask("h5", body, "docs/some-doc.md", "Some Doc", null, "docs");
    const taskUndefined = buildDocEmbedTask("h6", body, "docs/some-doc.md", "Some Doc", undefined, "docs");

    for (const task of [taskNull, taskUndefined]) {
      const fmFrags = task.fragments.filter(f => f.type === "frontmatter");
      expect(fmFrags.find(f => f.label === "description")).toBeUndefined();
      // Title fragment is unaffected by the absence of a description.
      expect(fmFrags.find(f => f.label === "title")).toBeDefined();
    }
  });
});
