/**
 * master-harness-65amk — documents.keywords persistence (A3) and the
 * `clawmem backfill-meta` verb (A4).
 *
 * Fixture vault: real source files indexed through indexCollection, then the
 * description/keywords columns nulled to reproduce rows indexed before s1lli/65amk,
 * plus fake fragment vectors so "re-embed is queued" is observable as the hash
 * re-entering getHashesNeedingFragments. Assertions are on rows, never on source shape.
 */
import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import {
  applyDocMetaBackfill,
  canonicalDocId,
  createStore,
  getHashesNeedingFragments,
  type Store,
} from "../../src/store.ts";
import { indexCollection } from "../../src/indexer.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";
import { acquireWorkerLease } from "../../src/worker-lease.ts";
import { formatBackfillMetaSummary, runBackfillMeta } from "../../src/backfill-meta.ts";
import type { NamedCollection } from "../../src/collections.ts";

const ROOT = mkdtempSync(join(tmpdir(), "clawmem-backfill-meta-"));
const DOCS = `${ROOT}/docs`;
const NOTES = `${ROOT}/notes`;
const DB = `${ROOT}/vault.sqlite`;

/** Enrichment is unavailable in every case below — the realistic no-GPU condition. */
const deadLlm = {
  embed: async () => {
    throw new Error("no embedding endpoint in test");
  },
  query: async () => null,
  expandQuery: async () => [],
} as any;

const COLLECTIONS: NamedCollection[] = [
  { name: "docs", path: DOCS, pattern: "**/*.md" },
  { name: "notes", path: NOTES, pattern: "**/*.md" },
];

const body = (name: string) =>
  `# ${name}\n\n## Details\n\n${`Fixture body for ${name}, long enough to split. `.repeat(8)}\n`;
const withFm = (fm: string, name: string) => `---\ntitle: ${name}\n${fm}---\n${body(name)}`;

let store: Store;

const write = (dir: string, rel: string, content: string) => {
  mkdirSync(join(dir, rel, ".."), { recursive: true });
  writeFileSync(join(dir, rel), content);
};

/** Index both collections, null the new columns (pre-v0.10.13 rows), and fake-embed every hash. */
async function seedVault(): Promise<void> {
  await indexCollection(store, "docs", DOCS, "**/*.md");
  await indexCollection(store, "notes", NOTES, "**/*.md");
  store.db.exec(`UPDATE documents SET description = NULL, keywords = NULL`);
  store.ensureVecTable(4);
  const rows = store.db
    .prepare(`SELECT DISTINCT hash, collection, path FROM documents WHERE active = 1 GROUP BY hash`)
    .all() as { hash: string; collection: string; path: string }[];
  const now = new Date().toISOString();
  for (const r of rows) {
    for (let seq = 0; seq < 3; seq++) {
      store.insertEmbedding(
        r.hash,
        seq,
        seq,
        new Float32Array([1, seq, 0, 0]),
        "fake",
        now,
        seq === 0 ? "full" : "section",
        null as any,
        canonicalDocId(r.collection, r.path),
      );
    }
  }
  store.db.exec(`UPDATE documents SET embed_state = 'synced', embed_attempts = 0`);
}

const snapshot = () => ({
  docs: store.db
    .prepare(
      `SELECT id, collection, path, hash, description, keywords, embed_state, embed_attempts, active FROM documents ORDER BY id`,
    )
    .all(),
  cv: (store.db.prepare(`SELECT count(*) AS n FROM content_vectors`).get() as { n: number }).n,
  vv: (store.db.prepare(`SELECT count(*) AS n FROM vectors_vec`).get() as { n: number }).n,
});

const rowOf = (collection: string, path: string) =>
  store.db
    .prepare(
      `SELECT id, hash, description, keywords FROM documents WHERE collection = ? AND path = ? AND active = 1`,
    )
    .get(collection, path) as {
    id: number;
    hash: string;
    description: string | null;
    keywords: string | null;
  };

const needing = () => new Set(getHashesNeedingFragments(store.db).map((r) => r.hash));

beforeEach(() => {
  rmSync(ROOT, { recursive: true, force: true });
  mkdirSync(DOCS, { recursive: true });
  mkdirSync(NOTES, { recursive: true });
  setDefaultLlamaCpp(deadLlm);
  store = createStore(DB);
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

// ---------------------------------------------------------------------------
// A3 — schema + persistence
// ---------------------------------------------------------------------------

describe("documents.keywords persistence (A3)", () => {
  it("the migration adds `keywords` to a pre-migration DB, existing rows NULL, no data loss", () => {
    const now = new Date().toISOString();
    store.insertContent("h-old", "old body", now);
    store.insertDocument("docs", "old.md", "Old", "h-old", now, now, "old description", "fs");
    store.close();

    // Reproduce the pre-65amk schema: the same DB without the column.
    const raw = new Database(DB);
    raw.exec(`ALTER TABLE documents DROP COLUMN keywords`);
    expect(
      (raw.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[]).some(
        (c) => c.name === "keywords",
      ),
    ).toBe(false);
    raw.close();

    store = createStore(DB);
    const cols = (store.db.prepare(`PRAGMA table_info(documents)`).all() as { name: string }[]).map(
      (c) => c.name,
    );
    expect(cols).toContain("keywords");
    const row = store.db
      .prepare(
        `SELECT title, hash, description, keywords, origin FROM documents WHERE path = 'old.md'`,
      )
      .get() as Record<string, unknown>;
    expect(row).toEqual({
      title: "Old",
      hash: "h-old",
      description: "old description",
      keywords: null,
      origin: "fs",
    });
    expect(
      (store.db.prepare(`SELECT doc FROM content WHERE hash = 'h-old'`).get() as { doc: string })
        .doc,
    ).toBe("old body");
  });

  it("insertDocument / updateDocumentMeta / saveMemory round-trip keywords; getHashesNeedingFragments returns it", () => {
    const now = new Date().toISOString();
    store.insertContent("h-a", "body a", now);
    store.insertDocument("docs", "a.md", "A", "h-a", now, now, "desc a", "fs", "alpha, beta");
    expect(rowOf("docs", "a.md").keywords).toBe("alpha, beta");

    store.updateDocumentMeta(rowOf("docs", "a.md").id, { keywords: "gamma" });
    expect(rowOf("docs", "a.md").keywords).toBe("gamma");
    // undefined leaves it untouched
    store.updateDocumentMeta(rowOf("docs", "a.md").id, { domain: "x" });
    expect(rowOf("docs", "a.md").keywords).toBe("gamma");

    store.saveMemory({
      collection: "_clawmem",
      path: "m.md",
      title: "M",
      body: "memory body",
      contentType: "note",
      keywords: "mem, kw",
    });
    expect(rowOf("_clawmem", "m.md").keywords).toBe("mem, kw");

    const work = store.getHashesNeedingFragments();
    expect(work.find((r) => r.hash === "h-a")).toMatchObject({
      description: "desc a",
      keywords: "gamma",
    });
    expect(work.find((r) => r.path === "m.md")?.keywords).toBe("mem, kw");
  });

  it("indexCollection persists normalized keywords from source frontmatter", async () => {
    write(DOCS, "k.md", withFm("description: kd\nkeywords:\n  - one\n  - two\n", "K"));
    write(DOCS, "s.md", withFm("keywords: solo\n", "S"));
    write(DOCS, "n.md", body("N"));
    await indexCollection(store, "docs", DOCS, "**/*.md");
    expect(rowOf("docs", "k.md")).toMatchObject({ description: "kd", keywords: "one, two" });
    expect(rowOf("docs", "s.md").keywords).toBe("solo");
    expect(rowOf("docs", "n.md").keywords).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// A4 — backfill-meta
// ---------------------------------------------------------------------------

describe("backfill-meta (A4)", () => {
  beforeEach(async () => {
    write(DOCS, "both.md", withFm("description: Both summary\nkeywords: [alpha, beta]\n", "Both"));
    write(DOCS, "desc-only.md", withFm("description: Desc summary\n", "DescOnly"));
    write(DOCS, "plain.md", body("Plain"));
    write(DOCS, "gone.md", withFm("description: Gone summary\n", "Gone"));
    write(NOTES, "kw-only.md", withFm("keywords: solo\n", "KwOnly"));
    await seedVault();
    unlinkSync(join(DOCS, "gone.md")); // row stays active; source missing
  });

  it("dry run reports correct counts and mutates NOTHING", () => {
    const before = snapshot();
    expect(before.cv).toBeGreaterThan(0);
    const out = runBackfillMeta(store, { apply: false, collections: COLLECTIONS });
    expect(out.exitCode).toBe(0);
    expect(out.mode).toBe("dry-run");
    expect(out.counts).toMatchObject({
      scanned: 5,
      missing_source: 1,
      would_update_description: 2, // both.md, desc-only.md
      would_update_keywords: 2, // both.md, kw-only.md
      rows_to_update: 3,
      hashes_to_invalidate: 3,
      vectors_to_delete: 9,
      source_changed: 0,
    });
    expect(snapshot()).toEqual(before);
    expect(needing().size).toBe(0);
    expect(formatBackfillMetaSummary(out)).toMatch(
      /^backfill-meta: mode=dry-run scanned=5 missing_source=1 .* exit=0$/,
    );
  });

  it("dry run works on a READ-ONLY handle (what the CLI opens)", () => {
    const ro = createStore(DB, { readonly: true });
    try {
      const out = runBackfillMeta(ro, { apply: false, collections: COLLECTIONS });
      expect(out.counts.rows_to_update).toBe(3);
    } finally {
      ro.close();
    }
  });

  it("--apply populates description + keywords and queues exactly the touched hashes", () => {
    const untouched = rowOf("docs", "plain.md").hash;
    const out = runBackfillMeta(store, { apply: true, collections: COLLECTIONS });
    expect(out.exitCode).toBe(0);
    expect(out.applied).toEqual({
      applied: 3,
      raced: 0,
      invalidated_hashes: 3,
      deleted_vectors: 9,
    });

    expect(rowOf("docs", "both.md")).toMatchObject({
      description: "Both summary",
      keywords: "alpha, beta",
    });
    expect(rowOf("docs", "desc-only.md")).toMatchObject({
      description: "Desc summary",
      keywords: null,
    });
    expect(rowOf("notes", "kw-only.md")).toMatchObject({ description: null, keywords: "solo" });
    expect(rowOf("docs", "plain.md")).toMatchObject({ description: null, keywords: null });

    const work = getHashesNeedingFragments(store.db);
    const touched = ["both.md", "desc-only.md"]
      .map((p) => rowOf("docs", p).hash)
      .concat(rowOf("notes", "kw-only.md").hash);
    expect(new Set(work.map((r) => r.hash))).toEqual(new Set(touched));
    expect(work.find((r) => r.hash === rowOf("docs", "both.md").hash)).toMatchObject({
      description: "Both summary",
      keywords: "alpha, beta",
    });
    // ALL fragment rows of a touched hash are gone, from both vector tables.
    for (const h of touched) {
      expect(
        (
          store.db.prepare(`SELECT count(*) AS n FROM content_vectors WHERE hash = ?`).get(h) as {
            n: number;
          }
        ).n,
      ).toBe(0);
      expect(
        (
          store.db
            .prepare(`SELECT count(*) AS n FROM vectors_vec WHERE hash_seq LIKE ?`)
            .get(`${h}_%`) as { n: number }
        ).n,
      ).toBe(0);
    }
    // Untouched docs keep their vectors and stay off the worklist.
    expect(work.some((r) => r.hash === untouched)).toBe(false);
    expect(
      (
        store.db
          .prepare(`SELECT count(*) AS n FROM content_vectors WHERE hash = ?`)
          .get(untouched) as { n: number }
      ).n,
    ).toBe(3);

    // Idempotent: a second run plans nothing.
    expect(
      runBackfillMeta(store, { apply: false, collections: COLLECTIONS }).counts.rows_to_update,
    ).toBe(0);
  });

  it("--collection and --limit scope the scan", () => {
    const notesOnly = runBackfillMeta(store, {
      apply: true,
      collection: "notes",
      collections: COLLECTIONS,
    });
    expect(notesOnly.counts).toMatchObject({
      scanned: 1,
      rows_to_update: 1,
      would_update_keywords: 1,
      would_update_description: 0,
    });
    expect(rowOf("docs", "both.md").description).toBeNull(); // other collection untouched
    expect(needing()).toEqual(new Set([rowOf("notes", "kw-only.md").hash]));

    // ORDER BY collection, path: docs/both.md first, then docs/desc-only.md.
    const limited = runBackfillMeta(store, { apply: true, limit: 1, collections: COLLECTIONS });
    expect(limited.counts.scanned).toBe(1);
    expect(rowOf("docs", "both.md").keywords).toBe("alpha, beta");
    expect(rowOf("docs", "desc-only.md").description).toBeNull();
  });

  it("missing source is counted, not fatal, and an unconfigured collection is counted too", () => {
    const out = runBackfillMeta(store, { apply: true, collections: [COLLECTIONS[0]!] });
    expect(out.exitCode).toBe(0);
    expect(out.counts).toMatchObject({
      missing_source: 1,
      unconfigured_collection: 1,
      rows_to_update: 2,
    });
  });

  it("a source whose body changed since indexing is skipped (the indexer owns it)", () => {
    write(
      DOCS,
      "both.md",
      withFm("description: Both summary\nkeywords: [alpha]\n", "Both") + "\nnew paragraph\n",
    );
    const out = runBackfillMeta(store, { apply: false, collections: COLLECTIONS });
    expect(out.counts.source_changed).toBe(1);
    expect(out.counts.rows_to_update).toBe(2);
  });

  it("--apply refuses (exit 1, nothing written) while an embed holds the lease", () => {
    const before = snapshot();
    const held = acquireWorkerLease(store, "embedding", 60_000);
    expect(held.acquired).toBe(true);
    const out = runBackfillMeta(store, { apply: true, collections: COLLECTIONS });
    expect(out.exitCode).toBe(1);
    expect(out.refused).toContain("lease");
    expect(snapshot()).toEqual(before);
    expect(formatBackfillMetaSummary(out)).toContain("refused=1 exit=1");
  });

  it("the writer is lease-fenced: a stale token writes nothing", () => {
    const before = snapshot();
    acquireWorkerLease(store, "embedding", 60_000);
    expect(() =>
      applyDocMetaBackfill(
        store.db,
        [{ id: rowOf("docs", "both.md").id, hash: rowOf("docs", "both.md").hash, keywords: "x" }],
        { workerName: "embedding", token: "not-the-holder" },
      ),
    ).toThrow();
    expect(snapshot()).toEqual(before);
  });
});

describe("backfill-meta shared hashes", () => {
  it("writes each alias, but invalidates a hash only when its canonical alias changed", async () => {
    // Same body (same content hash) under two paths with different frontmatter.
    write(DOCS, "a/shared.md", `---\ntitle: Shared\n---\n${body("Shared")}`);
    write(DOCS, "b/shared.md", `---\ntitle: Shared\nkeywords: only-b\n---\n${body("Shared")}`);
    await seedVault();
    const a = rowOf("docs", "a/shared.md");
    const b = rowOf("docs", "b/shared.md");
    expect(a.hash).toBe(b.hash);

    const out = runBackfillMeta(store, { apply: true, collections: COLLECTIONS });
    expect(out.counts).toMatchObject({
      rows_to_update: 1,
      hashes_to_invalidate: 0,
      noncanonical_only_hashes: 1,
    });
    expect(rowOf("docs", "b/shared.md").keywords).toBe("only-b");
    expect(needing().has(a.hash)).toBe(false); // canonical docs/a/shared.md unchanged → vectors kept

    // Now the canonical alias gains keywords → the hash IS invalidated.
    write(DOCS, "a/shared.md", `---\ntitle: Shared\nkeywords: canon-a\n---\n${body("Shared")}`);
    const out2 = runBackfillMeta(store, { apply: true, collections: COLLECTIONS });
    expect(out2.counts).toMatchObject({ rows_to_update: 1, hashes_to_invalidate: 1 });
    const work = getHashesNeedingFragments(store.db).find((r) => r.hash === a.hash);
    expect(work).toMatchObject({ path: "a/shared.md", keywords: "canon-a" });
  });
});
