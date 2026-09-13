/**
 * `bun src/pg/cli.ts retrieve` — the MACHINE-READABLE PG read path
 * (master-harness-2wx75 slice 7).
 *
 * Slices 1-6 built four read functions and nothing called them. This module is
 * the first caller: one verb, three modes, and EXACTLY ONE JSON object on
 * stdout, so the master-harness parity eval (r47) can run the PG arm next to
 * the sqlite `clawmem search|vsearch|query` CLI and normalize both identically.
 *
 * THE OUTPUT CONTRACT IS PINNED (schema `clawmem-pg-retrieve/v1`). A consumer is
 * coded against these field names; renaming one is a v2, never an edit.
 *
 *   mode      search  -> pgSearchFtsDetailed      (lexical arm)
 *             vsearch -> pgSearchVecDetailed      (vector arm)
 *             query   -> pgSearchRerankedDetailed (RRF hybrid + cross-encoder)
 *
 *   exit 0    the call COMPLETED — including degraded (single-arm, degraded-empty)
 *   exit 2    the call THREW, or the flags did not parse. Still one JSON object,
 *             with `results: []` and `error: "<type>: <message>"`.
 *
 * THE RERANKER WIRING. `query` mode is the first place the injection documented
 * in search-reranked.ts ruling 5 actually executes:
 *   `(q, docs, o) => store.rerank(q, docs, DEFAULT_RERANK_MODEL, undefined, o)`
 * `store.rerank` reads the CLAWMEM_RERANK_URL / CLAWMEM_RERANK_API_KEY contract
 * itself — nothing here re-reads it. `intent` is `undefined` because the sqlite
 * `clawmem query` CLI passes none, and parity means the same reranker input.
 *
 * WHY THE RERANK CACHE IS AN IN-MEMORY SQLITE. `store.rerank` needs a db handle
 * for its llm_cache. Handing it the operator's sqlite index would make a PG read
 * path WRITE to the store it is replacing, and would replay sqlite-era cached
 * scores into a PG measurement. `:memory:` means every call scores live: slower
 * than a warm sqlite CLI, but what it measures is the reranker, not the cache.
 *
 * STDOUT DISCIPLINE. Anything in the call chain that `console.log`s would corrupt
 * the one-object contract, so console.log/info are redirected to stderr for the
 * duration of the run and the JSON is written with process.stdout.write.
 */

import type { SearchResult } from "../store.ts";
import type { PgQueryable, PgVecEmbedder } from "./search.ts";
import { pgSearchVecDetailed } from "./search.ts";
import { pgSearchFtsDetailed } from "./search-fts.ts";
import { pgSearchRerankedDetailed, type PgReranker, type PgRerankStatus } from "./search-reranked.ts";
import type { PgHybridArmFailure } from "./search-hybrid.ts";
import type { Vault } from "./vaults.ts";

export const RETRIEVE_SCHEMA = "clawmem-pg-retrieve/v1" as const;
export const RETRIEVE_DEFAULT_LIMIT = 10;
export const EXIT_OK = 0;
export const EXIT_THREW = 2;

export type RetrieveMode = "search" | "vsearch" | "query";
const MODES: readonly RetrieveMode[] = ["search", "vsearch", "query"];

export interface RetrieveArgs {
  mode: RetrieveMode;
  query: string;
  limit: number;
  collections: string[] | undefined;
  vault: Vault;
  deadlineMs: number | undefined;
  noRerank: boolean;
}

/** The pinned wire shape. Field names are the contract — see the header. */
export interface RetrieveOutput {
  schema: typeof RETRIEVE_SCHEMA;
  mode: RetrieveMode | null;
  query: string | null;
  limit: number | null;
  results: { file: string; score: number }[];
  degraded: boolean;
  degradedReason: string | null;
  rerankStatus: PgRerankStatus | null;
  timings: {
    totalMs: number;
    hybridMs: number | null;
    rerankMs: number | null;
    embedMs: number | null;
  };
  error: string | null;
}

/** A flag that did not parse. Reported as `UsageError: …`, exit 2. */
export class RetrieveUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UsageError";
  }
}

const KNOWN_VALUE_FLAGS = new Set(["--mode", "--query", "--limit", "--collection", "--vault", "--deadline-ms"]);
const KNOWN_BOOL_FLAGS = new Set(["--no-rerank"]);

function positiveInt(name: string, raw: string): number {
  if (!/^\d+$/.test(raw) || Number(raw) <= 0) {
    throw new RetrieveUsageError(`${name} must be a positive integer, got ${JSON.stringify(raw)}`);
  }
  return Number(raw);
}

/**
 * Parse `retrieve`'s flags. Pure; throws RetrieveUsageError on anything it does
 * not understand — an unknown flag is refused rather than ignored, because a
 * typo'd `--colection` that silently searched every collection would be a green
 * run over the wrong scope.
 *
 * `--vault public` is the operator-facing name for the `sfw` vault; both spell it.
 */
export function parseRetrieveArgs(argv: string[]): RetrieveArgs {
  const values = new Map<string, string>();
  const bools = new Set<string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (KNOWN_BOOL_FLAGS.has(a)) { bools.add(a); continue; }
    if (KNOWN_VALUE_FLAGS.has(a)) {
      const v = argv[i + 1];
      if (v === undefined || v.startsWith("--")) throw new RetrieveUsageError(`${a} requires a value`);
      values.set(a, v);
      i++;
      continue;
    }
    throw new RetrieveUsageError(`unknown argument ${JSON.stringify(a)}`);
  }

  const mode = values.get("--mode");
  if (mode === undefined) throw new RetrieveUsageError("--mode <search|vsearch|query> is required");
  if (!(MODES as readonly string[]).includes(mode)) {
    throw new RetrieveUsageError(`--mode must be one of ${MODES.join("|")}, got ${JSON.stringify(mode)}`);
  }
  const query = values.get("--query");
  if (query === undefined || query.trim() === "") throw new RetrieveUsageError("--query <text> is required");

  const limitRaw = values.get("--limit");
  const deadlineRaw = values.get("--deadline-ms");
  const colRaw = values.get("--collection");
  const collections = colRaw === undefined
    ? undefined
    : colRaw.split(",").map(s => s.trim()).filter(s => s.length > 0);
  if (collections !== undefined && collections.length === 0) {
    throw new RetrieveUsageError("--collection names no collection");
  }

  const vaultRaw = values.get("--vault") ?? "public";
  let vault: Vault;
  if (vaultRaw === "public" || vaultRaw === "sfw") vault = "sfw";
  else if (vaultRaw === "nsfw") vault = "nsfw";
  else throw new RetrieveUsageError(`--vault must be public or nsfw, got ${JSON.stringify(vaultRaw)}`);

  return {
    mode: mode as RetrieveMode,
    query,
    limit: limitRaw === undefined ? RETRIEVE_DEFAULT_LIMIT : positiveInt("--limit", limitRaw),
    collections,
    vault,
    deadlineMs: deadlineRaw === undefined ? undefined : positiveInt("--deadline-ms", deadlineRaw),
    noRerank: bools.has("--no-rerank"),
  };
}

/**
 * Everything `runRetrieve` touches outside itself. Injected so the unit tier
 * drives the REAL dispatch + mapping with no database, no embed endpoint and no
 * reranker; production gets `productionRetrieveDeps()`.
 */
export interface RetrieveDeps {
  withClient<T>(vault: Vault, fn: (c: PgQueryable) => Promise<T>): Promise<T>;
  /** The embedding backend for vsearch/query. */
  embedder(): PgVecEmbedder;
  /** The reranker for query mode; called only when --no-rerank is absent. */
  reranker(): Promise<PgReranker>;
  searchFts: typeof pgSearchFtsDetailed;
  searchVec: typeof pgSearchVecDetailed;
  searchReranked: typeof pgSearchRerankedDetailed;
  /** Release anything the deps opened (pools, the rerank cache store, the LLM). */
  dispose(): Promise<void>;
}

const round1 = (n: number) => Math.round(n * 10) / 10;

/** The `file` key both parity arms normalize on: `<collection>/<relative path>`. */
function toWire(results: SearchResult[]): { file: string; score: number }[] {
  return results.map(r => ({ file: r.displayPath, score: r.score }));
}

function describeArmFailure(f: PgHybridArmFailure): string {
  if (f.kind === "degraded") return `${f.arm}:${f.reason}`;
  const e = f.error;
  return `${f.arm}:threw:${e instanceof Error ? `${e.name}: ${e.message}` : String(e)}`;
}

export function errorString(e: unknown): string {
  if (e instanceof Error) return `${e.name || "Error"}: ${e.message}`;
  return `Error: ${String(e)}`;
}

/** Wrap an embedder so the embed leg's wall clock is READ, not inferred. */
function timedEmbedder(inner: PgVecEmbedder, sink: { ms: number | null }): PgVecEmbedder {
  return {
    async embed(text, options) {
      const t = performance.now();
      try {
        return await inner.embed(text, options);
      } finally {
        sink.ms = (sink.ms ?? 0) + (performance.now() - t);
      }
    },
  };
}

function emptyOutput(partial: Partial<Pick<RetrieveOutput, "mode" | "query" | "limit">>): RetrieveOutput {
  return {
    schema: RETRIEVE_SCHEMA,
    mode: partial.mode ?? null,
    query: partial.query ?? null,
    limit: partial.limit ?? null,
    results: [],
    degraded: false,
    degradedReason: null,
    rerankStatus: null,
    timings: { totalMs: 0, hybridMs: null, rerankMs: null, embedMs: null },
    error: null,
  };
}

/**
 * Run one retrieval and map it onto the pinned contract. Never throws: a thrown
 * failure becomes `error` + exit 2.
 */
export async function runRetrieve(
  args: RetrieveArgs,
  deps: RetrieveDeps,
): Promise<{ output: RetrieveOutput; exitCode: number }> {
  const t0 = performance.now();
  const out = emptyOutput(args);
  const embed = { ms: null as number | null };
  const finish = (exitCode: number) => {
    out.timings.totalMs = round1(performance.now() - t0);
    if (out.timings.embedMs === null && embed.ms !== null) out.timings.embedMs = round1(embed.ms);
    return { output: out, exitCode };
  };

  try {
    const collections = args.collections;
    switch (args.mode) {
      case "search": {
        const r = await deps.withClient(args.vault, c => deps.searchFts(c, args.query, {
          limit: args.limit,
          ...(collections === undefined ? {} : { collections }),
          ...(args.deadlineMs === undefined ? {} : { timeoutMs: args.deadlineMs }),
        }));
        out.results = toWire(r.results);
        out.degraded = r.degraded;
        out.degradedReason = r.degradedReason ?? null;
        break;
      }
      case "vsearch": {
        const embedder = timedEmbedder(deps.embedder(), embed);
        const r = await deps.withClient(args.vault, c => deps.searchVec(c, args.query, {
          limit: args.limit,
          embedder,
          ...(collections === undefined ? {} : { collections }),
          ...(args.deadlineMs === undefined ? {} : { timeoutMs: args.deadlineMs }),
        }));
        out.results = toWire(r.results);
        out.degraded = r.degraded;
        out.degradedReason = r.degradedReason ?? null;
        break;
      }
      case "query": {
        const embedder = timedEmbedder(deps.embedder(), embed);
        // Built BEFORE the connection is checked out, so reranker construction
        // cost never counts against a PG statement or the rerank deadline.
        const reranker = args.noRerank ? undefined : await deps.reranker();
        const r = await deps.withClient(args.vault, c => deps.searchReranked(c, args.query, {
          limit: args.limit,
          embedder,
          ...(collections === undefined ? {} : { collections }),
          ...(args.deadlineMs === undefined ? {} : { deadlineMs: args.deadlineMs }),
          ...(reranker === undefined ? {} : { reranker }),
        }));
        out.results = toWire(r.results);
        out.degraded = r.hybrid.degraded;
        out.degradedReason = r.hybrid.armFailures.length > 0
          ? r.hybrid.armFailures.map(describeArmFailure).join("; ")
          : null;
        out.rerankStatus = r.rerank;
        out.timings.hybridMs = round1(r.timings.hybridMs);
        out.timings.rerankMs = round1(r.timings.rerankMs);
        if (r.rerankReason !== undefined) {
          // Not a contract field: the reason is operator evidence, so it goes to
          // stderr where a human reading a parity run will see it.
          console.error(`[pg-retrieve] rerank ${r.rerank}: ${r.rerankReason}`);
        }
        break;
      }
    }
    return finish(EXIT_OK);
  } catch (e) {
    out.results = [];
    out.degraded = false;
    out.degradedReason = null;
    out.error = errorString(e);
    return finish(EXIT_THREW);
  }
}

/**
 * The production dependencies. Imports are DYNAMIC so a unit test that injects
 * its own deps never loads the pool, the LLM client or the sqlite store.
 */
export function productionRetrieveDeps(): RetrieveDeps {
  let closeStore: (() => void) | undefined;
  let embedderLoaded = false;
  return {
    async withClient(vault, fn) {
      const { withClient } = await import("./client.ts");
      return withClient(vault, fn);
    },
    embedder() {
      embedderLoaded = true;
      // Lazy: the vec arm itself defaults to getDefaultLlamaCpp(); resolving it
      // on first embed keeps the import off the `search` path entirely.
      return {
        async embed(text, options) {
          const { getDefaultLlamaCpp } = await import("../llm.ts");
          return getDefaultLlamaCpp().embed(text, options);
        },
      };
    },
    async reranker() {
      const { createStore, DEFAULT_RERANK_MODEL } = await import("../store.ts");
      const store = createStore(":memory:");
      closeStore = () => store.close();
      return (q, docs, o) => store.rerank(q, docs, DEFAULT_RERANK_MODEL, undefined, o);
    },
    searchFts: pgSearchFtsDetailed,
    searchVec: pgSearchVecDetailed,
    searchReranked: pgSearchRerankedDetailed,
    async dispose() {
      const { closePool } = await import("./client.ts");
      await closePool().catch(() => {});
      try { closeStore?.(); } catch { /* best effort */ }
      if (embedderLoaded) {
        const { disposeDefaultLlamaCpp } = await import("../llm.ts");
        await disposeDefaultLlamaCpp().catch(() => {});
      }
    },
  };
}

export interface RetrieveIo {
  stdout(line: string): void;
}

/**
 * The verb: parse, run, print ONE JSON object, return the exit code.
 * A parse failure is still one JSON object (`UsageError: …`, exit 2).
 */
export async function retrieveCli(
  argv: string[],
  deps: RetrieveDeps = productionRetrieveDeps(),
  io: RetrieveIo = { stdout: s => { process.stdout.write(s); } },
): Promise<number> {
  const origLog = console.log;
  const origInfo = console.info;
  console.log = (...a: unknown[]) => console.error(...a);
  console.info = (...a: unknown[]) => console.error(...a);
  try {
    let result: { output: RetrieveOutput; exitCode: number };
    try {
      const args = parseRetrieveArgs(argv);
      result = await runRetrieve(args, deps);
    } catch (e) {
      const output = emptyOutput({});
      output.error = errorString(e);
      result = { output, exitCode: EXIT_THREW };
    } finally {
      await deps.dispose().catch(() => {});
    }
    io.stdout(`${JSON.stringify(result.output)}\n`);
    return result.exitCode;
  } finally {
    console.log = origLog;
    console.info = origInfo;
  }
}
