/**
 * PG read path honours ADR-0085 session focus (master-harness-vn4rz.111).
 *
 * Precedence parity with the sqlite CLI: explicit -c > session focus naming a
 * real collection > unscoped. Fake search backends filter a two-collection
 * corpus by the `collections` option they receive, so a hit from the wrong
 * collection means the scope never reached the query.
 */

import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { runRetrieve, type RetrieveArgs, type RetrieveDeps, type RetrieveMode } from "../../src/pg/retrieve.ts";
import { resolvePgCollectionScope } from "../../src/pg/focus-scope.ts";
import { computeCollectionScope } from "../../src/collection-scope.ts";
import { writeSessionFocus } from "../../src/session-focus.ts";
import type { PgQueryable } from "../../src/pg/search.ts";
import type { SearchResult } from "../../src/store.ts";

const KNOWN = ["alpha", "beta"];
const SID = "sess-vn4rz111";

function sr(collection: string, p: string): SearchResult {
  return {
    filepath: `clawmem://${collection}/${p}`, displayPath: `${collection}/${p}`, title: p,
    context: null, hash: "a".repeat(64), docid: "aaaaaa", collectionName: collection,
    modifiedAt: "", bodyLength: 4, body: "body", score: 0.5, source: "fts",
  };
}
const CORPUS = [sr("alpha", "a.md"), sr("beta", "b.md")];
const pick = (cols?: string[]) => CORPUS.filter(r => !cols || cols.includes(r.collectionName));

function deps(): RetrieveDeps {
  const hybrid = (rs: SearchResult[]) =>
    ({ results: rs, candidates: rs, arms: "vec+fts", degraded: false, armFailures: [], rrfK: 60 });
  return {
    knownCollections: () => KNOWN,
    async withClient(_v, fn) { return fn({} as PgQueryable); },
    embedder: () => ({ async embed() { return { embedding: [1], model: "m" }; } }),
    async reranker() { return async (_q, docs) => docs.map(d => ({ file: d.file, score: 1 })); },
    searchFts: (async (_c, _q, o) => ({ results: pick(o?.collections), degraded: false, scannedRows: 2 })) as RetrieveDeps["searchFts"],
    searchVec: (async (_c, _q, o) => ({ results: pick(o?.collections), degraded: false, storedModels: 1, scannedFragments: 2 })) as RetrieveDeps["searchVec"],
    searchReranked: (async (_c, _q, o) => {
      const rs = pick(o?.collections);
      return { results: rs, hybrid: hybrid(rs), rerank: "applied", candidateCount: rs.length,
        timings: { hybridMs: 1, rerankMs: 1, totalMs: 2 }, deadlineMs: 1500 };
    }) as RetrieveDeps["searchReranked"],
    async dispose() {},
  };
}

const args = (mode: RetrieveMode, collections?: string[]): RetrieveArgs => ({
  mode, query: "q", limit: 10, collections, vault: "sfw", deadlineMs: undefined, noRerank: false,
});
async function files(mode: RetrieveMode, collections?: string[]) {
  const { output } = await runRetrieve(args(mode, collections), deps());
  return output.results.map(r => r.file).sort();
}

let root: string;
const saved: Record<string, string | undefined> = {};
const ENV = ["CLAWMEM_FOCUS_ROOT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_SESSION_ID", "CLAWMEM_SESSION_ID", "CLAWMEM_SESSION_FOCUS"];
beforeEach(() => {
  for (const k of ENV) { saved[k] = process.env[k]; delete process.env[k]; }
  root = fs.mkdtempSync(path.join(os.tmpdir(), "pg-focus-"));
  process.env.CLAWMEM_FOCUS_ROOT = root;
  process.env.CLAUDE_CODE_SESSION_ID = SID;
});
afterEach(() => {
  for (const k of ENV) { if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k]; }
  fs.rmSync(root, { recursive: true, force: true });
});

const MODES: RetrieveMode[] = ["search", "vsearch", "query"];

describe("PG retrieve honours session focus", () => {
  for (const mode of MODES) {
    it(`${mode}: focus names a collection -> only that collection`, async () => {
      writeSessionFocus(SID, "alpha");
      expect(await files(mode)).toEqual(["alpha/a.md"]);
    });
    it(`${mode}: explicit collection overrides focus`, async () => {
      writeSessionFocus(SID, "alpha");
      expect(await files(mode, ["beta"])).toEqual(["beta/b.md"]);
    });
    it(`${mode}: no focus file -> unscoped`, async () => {
      expect(await files(mode)).toEqual(["alpha/a.md", "beta/b.md"]);
    });
  }
  it("free-text focus that is not a collection stays unscoped", async () => {
    writeSessionFocus(SID, "some topic");
    expect(await files("vsearch")).toEqual(["alpha/a.md", "beta/b.md"]);
  });
  it("deps without knownCollections never consult focus", async () => {
    writeSessionFocus(SID, "alpha");
    const d = deps(); delete d.knownCollections;
    const { output } = await runRetrieve(args("vsearch"), d);
    expect(output.results.length).toBe(2);
  });
});

describe("parity with the sqlite resolver", () => {
  it("resolves the same collection set for the same session + focus", () => {
    writeSessionFocus(SID, "BETA");
    const sqlite = computeCollectionScope(undefined, "BETA", KNOWN);
    expect(sqlite && !("error" in sqlite) ? sqlite.collections : null).toEqual(["beta"]);
    expect(resolvePgCollectionScope(undefined, KNOWN)).toEqual(["beta"]);
  });
  it("explicit passes through; no focus -> undefined", () => {
    expect(resolvePgCollectionScope(["alpha"], KNOWN)).toEqual(["alpha"]);
    expect(resolvePgCollectionScope(undefined, KNOWN)).toBeUndefined();
  });
});
