import { afterEach, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, insertContent, insertDocument, type SearchResult, type Store } from "../../src/store.ts";
import { enrichResults } from "../../src/search-utils.ts";

/**
 * master-harness-vn4rz.97 — two fixed costs in the CLI hybrid `query`, both kept exact:
 *   1. enrichResults looked each result up with `(collection || '/' || path) = ?`, which no
 *      index can serve, so every result scanned the documents table.
 *   2. rerank wrote each score to llm_cache as its own autocommit INSERT (one WAL commit per
 *      candidate, 30 per query).
 */

function seed(store: Store, collection: string, path: string, opts: { active?: boolean; contentType?: string } = {}) {
  const now = new Date().toISOString();
  const hash = `h_${collection}_${path}_${Math.random().toString(36).slice(2)}`;
  insertContent(store.db, hash, `# ${path}\n\nbody of ${path}\n`, now);
  insertDocument(store.db, collection, path, path, hash, now, now);
  if (opts.contentType) {
    store.db.prepare(`UPDATE documents SET content_type = ? WHERE collection = ? AND path = ?`).run(opts.contentType, collection, path);
  }
  if (opts.active === false) {
    store.db.prepare(`UPDATE documents SET active = 0 WHERE collection = ? AND path = ?`).run(collection, path);
  }
}

function result(displayPath: string): SearchResult {
  return {
    filepath: `clawmem://${displayPath}`,
    displayPath,
    title: displayPath,
    context: null,
    hash: "x",
    docid: "x",
    collectionName: displayPath.split("/")[0]!,
    modifiedAt: "2020-01-01T00:00:00.000Z",
    bodyLength: 1,
    score: 0.5,
    source: "fts",
  };
}

/** Capture the SQL text of every statement prepared on `db` while `fn` runs. */
function capturePrepared<T>(store: Store, fn: () => T): { value: T; sql: string[] } {
  const sql: string[] = [];
  const db = store.db as any;
  const original = db.prepare;
  db.prepare = function (this: unknown, text: string, ...rest: unknown[]) {
    sql.push(text);
    return original.call(db, text, ...rest);
  };
  try {
    return { value: fn(), sql };
  } finally {
    db.prepare = original;
  }
}

describe("enrichResults — indexed documents lookup (vn4rz.97)", () => {
  it("the lookup for an active document is served by the (collection, path) index, not a table scan", () => {
    const store = createStore(":memory:");
    for (let i = 0; i < 20; i++) seed(store, "docs", `note-${i}.md`);
    seed(store, "docs", "target.md", { contentType: "decision" });

    const { value, sql } = capturePrepared(store, () => enrichResults(store, [result("docs/target.md")], "q"));
    expect(value[0]!.contentType).toBe("decision");
    const lookups = sql.filter((s) => /FROM documents/.test(s));
    expect(lookups.length).toBeGreaterThan(0);
    // Bind whatever the first lookup takes: (collection, path) or the joined display path.
    const params = (lookups[0]!.match(/\?/g) ?? []).length === 2 ? ["docs", "target.md"] : ["docs/target.md"];
    const plan = (store.db.prepare(`EXPLAIN QUERY PLAN ${lookups[0]}`).all(...params) as { detail: string }[])
      .map((r) => r.detail)
      .join(" | ");
    expect(plan).not.toMatch(/SCAN documents/);
    expect(plan).toMatch(/USING INDEX sqlite_autoindex_documents_1 \(collection=\? AND path=\?\)/);
  });

  it("returns the same metadata as the full-scan expression for active, inactive and missing documents", () => {
    const store = createStore(":memory:");
    seed(store, "docs", "a/deep/path.md", { contentType: "hub" });
    seed(store, "docs", "gone.md", { contentType: "decision", active: false });
    const enriched = enrichResults(store, [result("docs/a/deep/path.md"), result("docs/gone.md"), result("docs/missing.md")], "q");
    expect(enriched.map((r) => r.contentType)).toEqual(["hub", "note", "note"]);
    // Defaults for misses are unchanged.
    expect(enriched[1]!.confidence).toBe(0.5);
    expect(enriched[2]!.modifiedAt).toBe("2020-01-01T00:00:00.000Z");
  });

  it("a hand-edited collection name containing '/' still resolves (fallback to the expression)", () => {
    const store = createStore(":memory:");
    seed(store, "team/a", "doc.md", { contentType: "decision" });
    const enriched = enrichResults(store, [result("team/a/doc.md")], "q");
    expect(enriched[0]!.contentType).toBe("decision");
  });
});

describe("rerank — llm_cache score writes share one transaction (vn4rz.97)", () => {
  const saved = process.env.CLAWMEM_RERANK_URL;
  const cleanups: (() => void)[] = [];
  afterEach(() => {
    if (saved === undefined) delete process.env.CLAWMEM_RERANK_URL;
    else process.env.CLAWMEM_RERANK_URL = saved;
    for (const c of cleanups.splice(0)) c();
  });

  it("every rerank score INSERT runs inside a transaction, and the cached scores replay", async () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-rerank-tx-"));
    let rerankCalls = 0;
    const server = Bun.serve({
      port: 0,
      async fetch(req) {
        rerankCalls++;
        const body = (await req.json()) as { documents: string[] };
        return Response.json({
          results: body.documents.map((text, index) => ({ index, relevance_score: (text.length % 7) / 10 })),
        });
      },
    });
    cleanups.push(() => server.stop(true), () => rmSync(dir, { recursive: true, force: true }));
    process.env.CLAWMEM_RERANK_URL = `http://127.0.0.1:${server.port}`;
    const store = createStore(join(dir, "index.sqlite"));
    cleanups.push(() => store.db.close());

    const docs = Array.from({ length: 10 }, (_, i) => ({ file: `docs/d${i}.md`, text: `document number ${i}` }));
    const inTx: boolean[] = [];
    const db = store.db as any;
    const original = db.prepare;
    db.prepare = function (this: unknown, text: string, ...rest: unknown[]) {
      const stmt = original.call(db, text, ...rest);
      if (/INSERT OR REPLACE INTO llm_cache/.test(text)) {
        const run = stmt.run.bind(stmt);
        stmt.run = (...args: unknown[]) => {
          inTx.push(db.inTransaction);
          return run(...args);
        };
      }
      return stmt;
    };
    let first: { file: string; score: number }[];
    try {
      first = await store.rerank("which document", docs);
    } finally {
      db.prepare = original;
    }
    expect(inTx).toHaveLength(10);
    expect(inTx.every(Boolean)).toBe(true);
    expect(store.db.inTransaction).toBe(false);

    // All ten rows are committed, and a second call replays them with identical scores.
    const rows = store.db.prepare(`SELECT count(*) AS n FROM llm_cache`).get() as { n: number };
    expect(rows.n).toBe(10);
    const callsAfterFirst = rerankCalls;
    expect(callsAfterFirst).toBe(3); // 10 docs in batches of 4
    const second = await store.rerank("which document", docs);
    expect(rerankCalls).toBe(callsAfterFirst);
    expect(second).toEqual(first);
  });
});
