/**
 * PG LEXICAL (FTS) read path — the cheap tier (master-harness-2wx75 slice 4).
 *
 * NO DATABASE. Every case drives the REAL exported functions from
 * src/pg/search-fts.ts; the DB is replaced by a RECORDING FAKE that captures
 * the SQL text and the bind values, which is what makes the query CONTRACT
 * (the `@@` match against the bare `d.fts` column that `documents_fts_idx` can
 * serve, the weighted `ts_rank_cd`, the parameterized prefix constructor, the
 * active/invalidated fence, the in-SQL collection filter, the total order)
 * assertable at all — a live plan proves it for one corpus on one day; these
 * assertions hold for every caller.
 *
 * WHAT THIS TIER CANNOT PROVE, stated plainly rather than implied: that
 * PostgreSQL honours those predicates, that the weight array really ranks a
 * title match above a body match, or that the tsvector contains body terms at
 * all. All three are the engine's behaviour and belong to
 * tests/integration/pg-search-fts.test.ts. Here it is proven only that the
 * right SQL is what the engine is asked to run — the half that can regress in
 * this file.
 */

import { describe, it, expect } from "bun:test";
import {
  DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS,
  buildFtsSearchQuery,
  pgSearchFts,
  pgSearchFtsDetailed,
  toSearchResults,
  type PgFtsRow,
} from "../../src/pg/search-fts.ts";
import {
  PG_FTS_STOPWORDS,
  buildPgFtsQuery,
  tokenizeForPgFts,
} from "../../src/pg/fts-query.ts";
import { buildFTS5Query } from "../../src/store.ts";
import type { PgQueryable } from "../../src/pg/search.ts";

/** A `pg` client stand-in that records what it was asked and replays fixtures. */
function fakeClient(rowsFor: (sql: string, values: unknown[]) => unknown[]): PgQueryable & {
  calls: { text: string; values: unknown[] }[];
} {
  const calls: { text: string; values: unknown[] }[] = [];
  return {
    calls,
    async query(text: string, values: unknown[] = []) {
      calls.push({ text, values });
      return { rows: rowsFor(text, values) as never[] };
    },
  };
}

/**
 * The fake's row router. `numnode` is the emptiness pre-check; anything with
 * `ts_rank_cd` is the ranked scan. `lexemes` decides which side of the
 * degraded/genuine split the fake puts us on, so a test can pick either
 * WITHOUT the other assertions changing.
 */
function ftsClient(opts: { lexemes?: number; rows?: PgFtsRow[] } = {}) {
  const lexemes = opts.lexemes ?? 2;
  return fakeClient(sql => {
    if (sql.includes("numnode")) return [{ n: lexemes }];
    if (sql.includes("ts_rank_cd")) return opts.rows ?? [];
    return [];
  });
}

function row(over: Partial<PgFtsRow> = {}): PgFtsRow {
  return {
    hash: "a".repeat(64),
    collection: "research",
    path: "a/b.md",
    title: "B",
    modified_at: "2026-09-01T00:00:00.000Z",
    body: "hello world",
    rank: 0.5,
    ...over,
  };
}

describe("buildPgFtsQuery — sqlite FTS5 parity", () => {
  it("X10 builds server-stemmed prefix nodes for qualit/bi/encod without a phrase operator", () => {
    const query = buildPgFtsQuery(
      "Why does a cross-encoder reranker provide qualitatively different signal than a bi-encoder vector search?",
    )!;
    expect(query.values).toContain("qualitatively");
    expect(query.values).toContain("bi");
    expect(query.values.filter(value => value === "encoder")).toHaveLength(2);
    expect(query.text.match(/to_tsquery\('english', \$\d+::text \|\| ':\*'\)/g)?.length)
      .toBe(query.values.length);
    expect(query.text).not.toContain("<->");
    expect(query.text).not.toContain("cross-encoder");
  });

  it("treats uppercase OR between groups as OR and lowercase or as a term", () => {
    const upper = buildPgFtsQuery("cocoa OR frosting")!;
    expect(upper.values).toEqual(["cocoa", "frosting"]);
    expect(upper.text).toContain(" || ");

    const lower = buildPgFtsQuery("or")!;
    expect(lower.values).toEqual(["or"]);
    expect(lower.text).not.toContain(") || to_tsquery(");
  });

  it("drops the copied curated stopwords exactly where buildFTS5Query drops them", () => {
    for (const stopword of PG_FTS_STOPWORDS) {
      expect(buildFTS5Query(`${stopword} sentinel`)).toBe('"sentinel"*');
      expect(buildPgFtsQuery(`${stopword} sentinel`)!.values).toEqual(["sentinel"]);
    }
  });

  it("falls back to all terms when a group contains only curated stopwords", () => {
    const query = buildPgFtsQuery("the and of")!;
    expect(query.values).toEqual(["the", "and", "of"]);
    expect(query.text).toContain(" && ");
  });

  it("tokenizes lowercase Unicode words on every non-letter/non-number separator", () => {
    expect(tokenizeForPgFts("ÉTÉ_embedding/inference-東京.42"))
      .toEqual(["été", "embedding", "inference", "東京", "42"]);
  });

  it("keeps adversarial user text in bind values and never interpolates it into SQL", () => {
    const inputs = [
      `quality'quoted`, "quality & vector", "quality | vector", "quality ! vector",
      "quality:vector", "quality(vector)", "quality\\vector", "品質 vector",
    ];
    for (const input of inputs) {
      const query = buildPgFtsQuery(input)!;
      expect(query.values.length).toBeGreaterThan(0);
      expect(query.text).not.toContain(input);
      expect(query.text).toMatch(/^\(?to_tsquery\('english', \$1::text \|\| ':\*'\)/);
    }
    for (const input of ["", "   ", `' & | ! : ( ) \\`]) {
      expect(buildPgFtsQuery(input)).toBeNull();
    }
  });
});

describe("buildFtsSearchQuery — the SQL contract", () => {
  it("matches with `@@` against the BARE d.fts column (the only GIN-servable shape)", () => {
    const q = buildFtsSearchQuery("fox", null, 20)!;
    // Not `to_tsvector(...) @@ ...`: computing the vector at query time throws
    // away documents_fts_idx AND loses the setweight() weights the trigger
    // stored, so the rank weights below would have nothing to act on.
    expect(q.text).toMatch(/d\.fts @@ to_tsquery\('english', \$1::text \|\| ':\*'\)/);
    expect(q.text).not.toContain("to_tsvector(");
  });

  it("uses the same parameterized prefix expression for match and rank", () => {
    const q = buildFtsSearchQuery("quality", null, 20)!;
    const node = "to_tsquery('english', $1::text || ':*')";
    expect(q.text.split(node)).toHaveLength(3);
    expect(q.values).toEqual(["quality", 20]);
    expect(q.text).not.toContain("websearch_to_tsquery");
  });

  it("names the text-search configuration explicitly at the query end", () => {
    // migrations/001 §2: the trigger writes with to_tsvector('english', ...).
    // A bare two-arg query form would depend on the session's
    // default_text_search_config and could stem differently — silently.
    const q = buildFtsSearchQuery("fox", null, 20)!;
    expect(q.text.match(/'english'/g)?.length).toBe(2); // rank + match
  });

  it("ranks with ts_rank_cd and PostgreSQL's DEFAULT weights, giving 10:1 title:body", () => {
    const q = buildFtsSearchQuery("fox", null, 20)!;
    // {D, C, B, A} = {0.1, 0.2, 0.4, 1.0}. A:D = 10:1, the SAME title:body
    // ratio as the sqlite arm's bm25(documents_fts, 10.0, 1.0). Cover density
    // (`_cd`) rather than plain ts_rank: it accounts for term proximity.
    expect(q.text).toContain("ts_rank_cd('{0.1, 0.2, 0.4, 1.0}', d.fts,");
    expect(q.text).not.toMatch(/ts_rank\(/);
  });

  it("fences to active, non-invalidated documents", () => {
    const q = buildFtsSearchQuery("fox", null, 20)!;
    expect(q.text).toContain("d.active = true");
    expect(q.text).toContain("d.invalidated_at IS NULL");
  });

  it("joins content for the body, as the sqlite arm does", () => {
    expect(buildFtsSearchQuery("fox", null, 20)!.text)
      .toContain("JOIN content ON content.hash = d.hash");
  });

  it("orders TOTALLY — rank first, then the (collection, path) unique key", () => {
    // Rank alone leaves ties to whatever order the plan emitted, and with a
    // weighted tsvector over short documents ties are common.
    expect(buildFtsSearchQuery("fox", null, 20)!.text)
      .toMatch(/ORDER BY rank DESC, d\.collection ASC, d\.path ASC/);
  });

  it("omits the collection filter when none is requested, binding only query + limit", () => {
    const q = buildFtsSearchQuery("fox", null, 20)!;
    // `d.collection` is in the SELECT list and the ORDER BY either way; what
    // must be absent is the FILTER.
    expect(q.text).not.toContain("ANY(");
    expect(q.values).toEqual(["fox", 20]);
    expect(q.text).toContain("LIMIT $2");
  });

  it("applies the collection filter IN SQL, parameterized, when collections are requested", () => {
    // In SQL and not as a post-filter: a post-filter over a fixed overfetch can
    // be STARVED by higher-ranked ineligible documents. Same reasoning the
    // sqlite arm's own comment gives.
    const q = buildFtsSearchQuery("fox", ["research", "decisions"], 5)!;
    expect(q.text).toContain("d.collection = ANY($2::text[])");
    expect(q.values).toEqual(["fox", ["research", "decisions"], 5]);
    expect(q.text).toContain("LIMIT $3");
    // Nothing user-supplied is ever interpolated into SQL text.
    expect(q.text).not.toContain("research");
  });

  it("binds the limit as a parameter, not literal text", () => {
    expect(buildFtsSearchQuery("fox", null, 7)!.values.at(-1)).toBe(7);
    expect(buildFtsSearchQuery("fox", null, 7)!.text).not.toContain("LIMIT 7");
  });
});

describe("pgSearchFtsDetailed — the query text reaches $1", () => {
  it("substitutes the raw query into $1 and never interpolates it", async () => {
    const c = ftsClient({ rows: [row()] });
    await pgSearchFtsDetailed(c, "unbalanced & ( input", { collections: "research" });
    const scan = c.calls.find(k => k.text.includes("ts_rank_cd"))!;
    expect(scan.values).toEqual(["unbalanced", "input", ["research"], 20]);
    expect(scan.text).not.toContain("unbalanced");
  });

  it("bounds BOTH statements in ONE transaction, with SET LOCAL and no hnsw GUC", async () => {
    const c = ftsClient({ rows: [row()] });
    await pgSearchFtsDetailed(c, "fox", {});
    const texts = c.calls.map(k => k.text);
    expect(texts[0]).toBe("BEGIN");
    expect(texts[1]).toBe(`SET LOCAL statement_timeout = ${DEFAULT_PG_SEARCH_STATEMENT_TIMEOUT_MS}`);
    expect(texts.at(-1)).toBe("COMMIT");
    // SET LOCAL, not SET: a session SET on a POOLED client leaks the bound onto
    // the next checkout the first time anything skips a finally.
    expect(texts.some(t => /^SET statement_timeout/.test(t))).toBe(false);
    // The pgvector recall knob has no meaning on a GIN/tsvector path.
    expect(texts.some(t => t.includes("hnsw"))).toBe(false);
    // And the pre-check and the scan really are inside the SAME transaction.
    expect(texts.filter(t => t === "BEGIN").length).toBe(1);
    expect(texts.some(t => t.includes("numnode"))).toBe(true);
  });

  it("honours statementTimeoutMs, including 0 = no bound", async () => {
    const c = ftsClient({ rows: [] });
    await pgSearchFtsDetailed(c, "fox", { statementTimeoutMs: 250 });
    expect(c.calls[1]!.text).toBe("SET LOCAL statement_timeout = 250");
    const c0 = ftsClient({ rows: [] });
    await pgSearchFtsDetailed(c0, "fox", { statementTimeoutMs: 0 });
    expect(c0.calls[1]!.text).toBe("SET LOCAL statement_timeout = 0");
  });
});

describe("pgSearchFtsDetailed — the degraded channel", () => {
  it("DEGRADED empty-tsquery: punctuation-only input normalizes to no tokens", async () => {
    const c = ftsClient({ rows: [row()] });
    const out = await pgSearchFtsDetailed(c, `' & | ! : ( ) \\`, {});
    expect(out).toEqual({
      results: [], degraded: true, degradedReason: "empty-tsquery", scannedRows: 0,
    });
    expect(c.calls).toEqual([]);
  });

  it("DEGRADED empty-tsquery: the query normalized to zero lexemes", async () => {
    // All stopwords. The scan would match nothing, but reporting that as "no
    // matches" is a lie — nothing was searched for.
    const c = ftsClient({ lexemes: 0, rows: [row()] });
    const out = await pgSearchFtsDetailed(c, "the and of", {});
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("empty-tsquery");
    expect(out.results).toEqual([]);
    expect(out.scannedRows).toBe(0);
    // PROOF IT SHORT-CIRCUITED: the fake was primed with a matching row and the
    // ranked scan was never issued. Without this the case would pass even if
    // the reason were attached to an ordinary empty result.
    expect(c.calls.some(k => k.text.includes("ts_rank_cd"))).toBe(false);
  });

  it("DEGRADED budget-exhausted: the wall-clock budget was spent before the SQL leg", async () => {
    const c = ftsClient({ rows: [row()] });
    const out = await pgSearchFtsDetailed(c, "fox", { timeoutMs: 0 });
    expect(out.degraded).toBe(true);
    expect(out.degradedReason).toBe("budget-exhausted");
    expect(out.results).toEqual([]);
    expect(out.scannedRows).toBe(0);
    // A budget already spent buys no round trips at all — not even BEGIN.
    expect(c.calls).toEqual([]);
  });

  it("CONTROL — healthy multi-row: degraded false, rows counted, source is fts", async () => {
    const rows = [
      row({ path: "one.md", hash: "1".repeat(64), rank: 0.9 }),
      row({ path: "two.md", hash: "2".repeat(64), rank: 0.4 }),
      row({ path: "three.md", collection: "decisions", hash: "3".repeat(64), rank: 0.1 }),
    ];
    const out = await pgSearchFtsDetailed(ftsClient({ rows }), "fox", {});
    expect(out.degraded).toBe(false);
    expect(out.degradedReason).toBeUndefined();
    expect(out.scannedRows).toBe(3);
    expect(out.results.map(r => r.displayPath)).toEqual([
      "research/one.md", "research/two.md", "decisions/three.md",
    ]);
    expect(out.results.map(r => r.source)).toEqual(["fts", "fts", "fts"]);
    expect(out.results[0]!.score).toBe(0.9);
  });

  it("CONTROL — GENUINE EMPTY: the scan ran, zero rows matched, degraded is FALSE", async () => {
    const c = ftsClient({ lexemes: 3, rows: [] });
    const out = await pgSearchFtsDetailed(c, "unmatchable-term", {});
    expect(out.degraded).toBe(false);
    expect(out.degradedReason).toBeUndefined();
    expect(out.results).toEqual([]);
    expect(out.scannedRows).toBe(0);
    // The difference from empty-tsquery is not the result — it is THIS: the
    // ranked scan was actually issued.
    expect(c.calls.some(k => k.text.includes("ts_rank_cd"))).toBe(true);
  });

  it("GENUINE-EMPTY and empty-tsquery are NOT the same value — the whole point", async () => {
    // Slice 3's convention, inherited. Both have `results: []`; a caller that
    // switches on the RESULT cannot tell them apart, and a caller that switches
    // on `degraded`/`degradedReason` always can. If these two ever compare
    // equal, the degraded channel has stopped carrying information.
    const genuine = await pgSearchFtsDetailed(ftsClient({ lexemes: 3, rows: [] }), "nope", {});
    const empty = await pgSearchFtsDetailed(ftsClient({ lexemes: 0 }), "the and of", {});
    expect(genuine.results).toEqual(empty.results);   // indistinguishable HERE...
    expect(genuine).not.toEqual(empty);               // ...and distinguishable there.
    expect(genuine.degraded).toBe(false);
    expect(empty.degraded).toBe(true);
    expect(genuine.degradedReason).not.toBe(empty.degradedReason);
  });
});

describe("pgSearchFts — the back-compat wrapper", () => {
  it("returns exactly `.results` as a bare array", async () => {
    const rows = [row({ path: "one.md", rank: 0.9 }), row({ path: "two.md", rank: 0.2 })];
    const bare = await pgSearchFts(ftsClient({ rows }), "fox", { collections: "research" });
    const detailed = await pgSearchFtsDetailed(ftsClient({ rows }), "fox", { collections: "research" });
    expect(bare).toEqual(detailed.results);
    expect(Array.isArray(bare)).toBe(true);
  });

  it("returns a bare [] on a degraded outcome — which is why the detailed form exists", async () => {
    expect(await pgSearchFts(ftsClient({ lexemes: 0 }), "the and of", {})).toEqual([]);
  });
});

describe("toSearchResults — the mapping", () => {
  it("builds the clawmem:// filepath, the docid, and a null context", () => {
    const [r] = toSearchResults([row({ hash: "abcdef" + "0".repeat(58) })], 20);
    expect(r!.filepath).toBe("clawmem://research/a/b.md");
    expect(r!.displayPath).toBe("research/a/b.md");
    expect(r!.docid).toBe("abcdef");
    // No folder-context table in PG yet: null, not "" pretending a lookup ran.
    expect(r!.context).toBeNull();
    expect(r!.bodyLength).toBe("hello world".length);
  });

  it("falls back to the path when a document has no title", () => {
    expect(toSearchResults([row({ title: null })], 20)[0]!.title).toBe("a/b.md");
  });

  it("coerces a numeric-string rank and a Date modified_at", () => {
    // `pg` hands back float4/float8 as strings under some type configurations,
    // and timestamptz as a Date. Both must land as the SearchResult's types.
    const [r] = toSearchResults(
      [row({ rank: "0.75", modified_at: new Date("2026-01-02T03:04:05.000Z") })], 20);
    expect(r!.score).toBe(0.75);
    expect(r!.modifiedAt).toBe("2026-01-02T03:04:05.000Z");
  });

  it("applies the limit itself, so the mapping is honest without the SQL LIMIT", () => {
    const rows = [row({ path: "a.md" }), row({ path: "b.md" }), row({ path: "c.md" })];
    expect(toSearchResults(rows, 2).map(r => r.displayPath))
      .toEqual(["research/a.md", "research/b.md"]);
  });
});
