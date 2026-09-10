/**
 * PostgreSQL RERANKED READ PATH — the cross-encoder stage on top of RRF fusion
 * (master-harness-2wx75 slice 6).
 *
 * Slices 1-3 built the vector arm (src/pg/search.ts), slice 4 the lexical arm
 * (src/pg/search-fts.ts), slice 5 the RRF fusion of the two
 * (src/pg/search-hybrid.ts). This module is the rerank stage and nothing else.
 * It COMPOSES the hybrid call; it does not modify it.
 *
 * SIX RULINGS ARE ENCODED HERE. Each is a decision, and each has a unit case
 * that goes red if it is undone (see tests/unit/pg-search-reranked.test.ts).
 *
 * 1. COMPOSITION, NOT MUTATION. `pgSearchRerankedDetailed` CALLS
 *    `pgSearchHybridDetailed` and EMBEDS its result verbatim under `hybrid`.
 *    Slice 5 asserts a three-way invariant over that object — `degraded ===
 *    false` ⇔ `arms === "vec+fts"` ⇔ `armFailures.length === 0` — and folding
 *    rerank state into `degraded` or `arms` would break it. A rerank problem is
 *    NOT an arm problem: retrieval succeeded either way.
 *
 * 2. THE RERANK STATUS IS AN ALWAYS-PRESENT DISCRIMINATOR, mirroring slice 5's
 *    `arms`. `rerank: PgRerankStatus` says exactly what happened, and the
 *    contract for every value other than `"applied"` is that `results` is the
 *    hybrid's fused ordering BYTE-IDENTICALLY. A rerank problem degrades to
 *    fusion — never to garbage, never to empty, never to a reordering nobody
 *    chose.
 *
 * 3. A RERANK PROBLEM NEVER THROWS THE SEARCH. Both arms failing throws (slice
 *    5, deliberately: "we could not look" must stay distinguishable from
 *    "nothing matched"). Reranking is categorically different — the documents
 *    are already retrieved and already usefully ordered. So every reranker
 *    rejection, timeout and degenerate response is caught here and reported in
 *    `rerank`; nothing propagates to the caller.
 *
 * 4. ONE DEADLINE THAT DECREMENTS — NOT A THIRD PER-STAGE TIMEOUT. This is the
 *    load-bearing budget decision. The hybrid's two arms run SEQUENTIALLY (one
 *    PG connection hosts one transaction; concurrent BEGIN / SET LOCAL / COMMIT
 *    blocks earn SQLSTATE 25P01), so `timeoutMs` is already a PER-ARM bound and
 *    the hybrid's worst case is vec + fts, not max(vec, fts). Adding an
 *    independent `rerankTimeoutMs` would make the worst case vec + fts + rerank
 *    and put this bead's `hook p95 <= 1800 ms` clause out of reach BY
 *    CONSTRUCTION. Instead `deadlineMs` is an OVERALL wall-clock budget for the
 *    whole call: t0 is taken once, the hybrid runs, and the reranker is invoked
 *    with whatever is LEFT. If less than PG_RERANK_MIN_BUDGET_MS remains we do
 *    not start it at all (`"skipped-budget"`). Measured stage timings are
 *    returned so a parity run can READ the additive budget instead of
 *    inferring it.
 *
 *    RELATIONSHIP TO `timeoutMs`: `timeoutMs` bounds EACH ARM inside the
 *    hybrid; `deadlineMs` bounds THE WHOLE CALL including both arms and the
 *    rerank. They are not redundant and neither derives the other — a caller
 *    that sets `timeoutMs` larger than `deadlineMs` has simply guaranteed the
 *    rerank stage will be skipped for budget, which is a visible status rather
 *    than a hidden overrun. `deadlineMs` defaults to
 *    DEFAULT_PG_RERANK_DEADLINE_MS (1500), chosen to sit under the 1800 ms
 *    hook p95 clause with headroom for the caller's own work; it is NOT a
 *    measured value and vn4rz's parity run is what will price it.
 *
 * 5. THE RERANKER IS INJECTED, exactly as the embedder already is. `src/pg/`
 *    does NOT import `src/store.ts`'s `rerank()`. The `CLAWMEM_RERANK_URL` /
 *    `CLAWMEM_RERANK_API_KEY` contract, the remote-GPU-then-local fallback
 *    chain, the batch-of-4 VRAM cap, the REMOTE_RERANK_FETCH_TIMEOUT_MS
 *    deadline and the d0hz per-backend cache-key namespacing (the URL is IN the
 *    cache key, so a yoshiee score never replays as a cloud score) ALL live in
 *    `store.ts`'s `rerank()` and are INHERITED THROUGH INJECTION, never re-read
 *    here. A second read of that env inside `src/pg/` is precisely the drift
 *    slice 5's "reuse the helper, don't restate it" ruling exists to prevent.
 *    Callers wire it as
 *      `(q, docs, o) => store.rerank(q, docs, DEFAULT_RERANK_MODEL, intent, o)`.
 *
 * 6. THE BLEND IS `blendRerank`, NOT `blendFusionAndRerank`, AND THERE IS NO
 *    THIRD BLENDER. Two CORRECTNESS differences, not preferences:
 *      (i) `blendRerank` maps over CANDIDATES, so partial rerank coverage can
 *          never DROP a document. `blendFusionAndRerank` maps over the RERANK
 *          OUTPUT — a reranker that returns fewer rows than it was handed
 *          silently loses the rest.
 *     (ii) `blendRerank` has the degenerate-floor check and the `onFallback`
 *          hook, which is exactly the typed-degrade instrumentation slices 3-5
 *          established as this bead's house style. `onFallback` is wired into
 *          the `"degenerate"` status below — not swallowed.
 *    `blendFusionAndRerank` additionally still applies `rrfWeight = 0.75` at
 *    rrfRank <= 3, the very defect `blendRerank`'s own docstring names ("made
 *    RRF rank-1 mathematically immovable by the reranker"). `blendRerank` is
 *    harness-validated (2026-06-25, NL+KW known-item recall: lifts recall@1-5
 *    and MRR@10).
 *
 * KNOWN DUPLICATION, REPORTED RATHER THAN REFACTORED: the candidate cap
 * expression `Math.max(limit, 30)` is now written in two places — here and
 * src/clawmem.ts's `RERANK_CAP` (~line 1549), which is the sqlite reference
 * implementation of this same stage. Unifying it means touching the sqlite read
 * path, which is out of this slice. Same for the 4000-char text truncation
 * (src/clawmem.ts ~line 1608).
 *
 * Nothing in src/ calls this yet — caller migration is a separate step, exactly
 * as it is for the hybrid path.
 */

import type { SearchResult } from "../store.ts";
import { blendRerank, RERANK_DEGENERATE_FLOOR } from "../search-utils.ts";
import type { PgQueryable, PgVecEmbedder } from "./search.ts";
import {
  pgSearchHybridDetailed,
  type PgHybridSearchResult,
  type PgSearchHybridOptions,
} from "./search-hybrid.ts";

/**
 * The floor below which we decline to START the rerank stage.
 *
 * Named and exported so it is greppable and so the unit tier can pin the
 * budget arithmetic rather than infer it. 250 ms is a policy choice, not a
 * measurement: below it a cross-encoder call is very unlikely to return, and
 * spending the remaining budget failing is strictly worse than serving the
 * fused order we already have.
 */
export const PG_RERANK_MIN_BUDGET_MS = 250;

/**
 * Default OVERALL wall-clock budget for a reranked search, in ms.
 *
 * Sits under this bead's `hook p95 <= 1800 ms` clause with headroom for the
 * caller. NOT measured — see ruling 4.
 */
export const DEFAULT_PG_RERANK_DEADLINE_MS = 1500;

/** Default candidate-pool cap floor; mirrors src/clawmem.ts's RERANK_CAP. */
export const PG_RERANK_CAP_FLOOR = 30;

/** Max characters of body text handed to the reranker per candidate. */
export const PG_RERANK_TEXT_CHARS = 4000;

/**
 * What the rerank stage actually did. ALWAYS present (ruling 2).
 *
 * For EVERY value other than `"applied"`, `results` is the embedded hybrid's
 * fused ordering byte-identically.
 */
export type PgRerankStatus =
  /** Ran, returned usable scores, and the blend used them. */
  | "applied"
  /** No reranker was injected. A CALLER CHOICE, not a failure. */
  | "skipped-no-reranker"
  /** The deadline left less than PG_RERANK_MIN_BUDGET_MS; we chose not to start. */
  | "skipped-budget"
  /**
   * There was nothing to rank: every candidate body was empty, or fusion
   * returned no candidates at all. A degenerate INPUT, not a reranker fault.
   */
  | "skipped-no-text"
  /**
   * The reranker responded, but no score cleared the degenerate floor (this
   * includes an empty response). Surfaced via `blendRerank`'s `onFallback`.
   */
  | "degenerate"
  /** The reranker threw, or blew the remaining budget. */
  | "failed";

/**
 * The injected reranker. Structurally identical to how `PgVecEmbedder` is
 * injected into the vector arm, and for the same reason: `src/pg/` must not
 * reach into `src/store.ts`.
 *
 * `opts.timeoutMs` is the REMAINING budget, already decremented (ruling 4).
 * `opts.signal` aborts at the same instant; a reranker that honours neither is
 * still bounded, because this module races it against the deadline and reports
 * `"failed"`.
 */
export type PgReranker = (
  query: string,
  documents: { file: string; text: string }[],
  opts?: { timeoutMs?: number; signal?: AbortSignal },
) => Promise<{ file: string; score: number }[]>;

/** Measured stage timings, so the additive budget is READ rather than inferred. */
export interface PgRerankTimings {
  /** Wall clock spent inside pgSearchHybridDetailed (both arms, sequentially). */
  hybridMs: number;
  /** Wall clock spent inside the injected reranker. 0 when it did not run. */
  rerankMs: number;
  /** Wall clock for the whole call, from t0 to return. */
  totalMs: number;
}

/**
 * Options for a reranked search. Every hybrid option is forwarded unchanged, so
 * migrating a caller from the hybrid path to this one is a call-site change.
 *
 * There is deliberately NO `rerankTimeoutMs`. See ruling 4.
 */
export interface PgSearchRerankedOptions extends PgSearchHybridOptions {
  /** The injected reranker. Omitted ⇒ `"skipped-no-reranker"` (a caller choice). */
  reranker?: PgReranker;
  /**
   * OVERALL wall-clock budget for the WHOLE call — both arms plus the rerank.
   * Defaults to DEFAULT_PG_RERANK_DEADLINE_MS.
   */
  deadlineMs?: number;
  /**
   * How many fused candidates to hand the reranker. Defaults to
   * `Math.max(limit, PG_RERANK_CAP_FLOOR)` — mirrors src/clawmem.ts:1549 so the
   * two read paths pool the same way. Candidates beyond the cap are NEVER
   * dropped: they keep their fused order after the reranked block.
   */
  rerankCap?: number;
  /** Reranker dominance in the blend. Defaults to `blendRerank`'s own 0.9. */
  rerankWeight?: number;
  /** Overrides `blendRerank`'s degenerate floor (RERANK_DEGENERATE_FLOOR). */
  degenerateFloor?: number;
}

/** The typed result of a reranked search. */
export interface PgRerankedSearchResult {
  /**
   * The final ordering, capped at `limit`. `rerank === "applied"` ⇒ blended
   * scores on `blendRerank`'s [0,1] scale; ANY OTHER STATUS ⇒ the embedded
   * hybrid's `results` array, unchanged (same objects, same RRF scores, same
   * order).
   */
  results: SearchResult[];
  /**
   * The hybrid call's own result, VERBATIM. Slice 5's `degraded` / `arms` /
   * `armFailures` invariant holds over this object and is re-asserted against
   * it in both test tiers — composition must not disturb it.
   */
  hybrid: PgHybridSearchResult;
  /** What the rerank stage did. Always present (ruling 2). */
  rerank: PgRerankStatus;
  /**
   * `blendRerank`'s own fallback reason, when `rerank === "degenerate"`, or the
   * reranker's error message when `rerank === "failed"`. Never a bare boolean:
   * "the reranker was useless" and "the reranker exploded" are different facts.
   */
  rerankReason?: string;
  /** The error the reranker raised, when `rerank === "failed"`. Carried, never swallowed. */
  rerankError?: unknown;
  /** How many fused candidates were handed to the reranker (0 when it did not run). */
  candidateCount: number;
  /** How many rows the reranker returned. Undefined when it did not run. */
  rerankedCount?: number;
  /** Measured stage timings. */
  timings: PgRerankTimings;
  /** The overall budget actually applied. */
  deadlineMs: number;
}

/**
 * Run `fn` under an abortable deadline.
 *
 * The reranker is handed both `timeoutMs` and a `signal`, but a reranker that
 * honours neither must still not blow the call's budget — so the promise is
 * raced against a timer. The timer is always cleared, including on the happy
 * path, so a resolved call never keeps the event loop alive.
 */
async function withDeadline<T>(
  budgetMs: number,
  fn: (opts: { timeoutMs: number; signal: AbortSignal }) => Promise<T>,
): Promise<T> {
  const ac = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expiry = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      ac.abort();
      reject(new Error(`pg rerank exceeded its remaining budget of ${Math.round(budgetMs)}ms`));
    }, budgetMs);
  });
  try {
    return await Promise.race([fn({ timeoutMs: budgetMs, signal: ac.signal }), expiry]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/**
 * Hybrid search plus a cross-encoder rerank stage, under ONE overall deadline.
 *
 * Reads as: fuse, then re-order the top of the fusion with a model that sees
 * the query and the document text together — and if that second step cannot be
 * done well, say so and serve the fusion.
 */
export async function pgSearchRerankedDetailed(
  c: PgQueryable,
  query: string,
  opts: PgSearchRerankedOptions = {},
): Promise<PgRerankedSearchResult> {
  // ONE t0 FOR THE WHOLE CALL. Every budget decision below is derived from this
  // single reading; taking a second clock reading per stage is how a
  // "per-stage timeout" creeps back in (ruling 4).
  const t0 = performance.now();
  const deadlineMs = opts.deadlineMs ?? DEFAULT_PG_RERANK_DEADLINE_MS;
  const limit = opts.limit ?? 20;
  const rerankCap = opts.rerankCap ?? Math.max(limit, PG_RERANK_CAP_FLOOR);

  const hybridOpts: PgSearchHybridOptions = {
    ...(opts.collections === undefined ? {} : { collections: opts.collections }),
    limit,
    ...(opts.timeoutMs === undefined ? {} : { timeoutMs: opts.timeoutMs }),
    ...(opts.statementTimeoutMs === undefined ? {} : { statementTimeoutMs: opts.statementTimeoutMs }),
    ...(opts.embedder === undefined ? {} : { embedder: opts.embedder as PgVecEmbedder }),
    ...(opts.overfetch === undefined ? {} : { overfetch: opts.overfetch }),
    ...(opts.rrfK === undefined ? {} : { rrfK: opts.rrfK }),
  };

  // The hybrid is NOT wrapped in a try/catch. Both arms failing must still
  // reject (slice 5, ruling 4 of that module) — this slice degrades RERANK
  // problems, not RETRIEVAL problems, and folding the two together would make
  // "we could not look" unreachable again.
  const hybrid = await pgSearchHybridDetailed(c, query, hybridOpts);
  const hybridMs = performance.now() - t0;

  /** Every non-"applied" exit: the fused order, byte-identically (ruling 2). */
  const degradeToFusion = (
    rerank: Exclude<PgRerankStatus, "applied">,
    extra: { rerankReason?: string; rerankError?: unknown; candidateCount?: number; rerankedCount?: number } = {},
  ): PgRerankedSearchResult => ({
    results: hybrid.results,
    hybrid,
    rerank,
    ...(extra.rerankReason === undefined ? {} : { rerankReason: extra.rerankReason }),
    ...("rerankError" in extra ? { rerankError: extra.rerankError } : {}),
    candidateCount: extra.candidateCount ?? 0,
    ...(extra.rerankedCount === undefined ? {} : { rerankedCount: extra.rerankedCount }),
    timings: { hybridMs, rerankMs: 0, totalMs: performance.now() - t0 },
    deadlineMs,
  });

  // PRECEDENCE, stated because more than one condition can hold at once:
  // no-reranker (a caller choice) > no-text (a degenerate INPUT, cheap to see
  // and needing no budget at all) > budget. Only the last one is about time.
  if (!opts.reranker) return degradeToFusion("skipped-no-reranker");

  const candidates = hybrid.results.slice(0, rerankCap);
  const documents = candidates.map(r => ({
    file: r.filepath,
    // `body` is optional on SearchResult and the vec arm can return a row with
    // an empty one. An empty string must not crash the reranker and must not be
    // scored as if it had been legitimately ranked — hence the all-empty check
    // immediately below.
    text: (r.body ?? "").slice(0, PG_RERANK_TEXT_CHARS),
  }));
  if (documents.length === 0 || documents.every(d => d.text.length === 0)) {
    return degradeToFusion("skipped-no-text", { candidateCount: documents.length });
  }

  const remaining = deadlineMs - (performance.now() - t0);
  if (remaining < PG_RERANK_MIN_BUDGET_MS) {
    return degradeToFusion("skipped-budget", {
      candidateCount: documents.length,
      rerankReason:
        `${Math.round(remaining)}ms left of a ${deadlineMs}ms deadline, ` +
        `below the ${PG_RERANK_MIN_BUDGET_MS}ms floor`,
    });
  }

  const rerankStart = performance.now();
  let reranked: { file: string; score: number }[];
  try {
    // `remaining`, NOT `deadlineMs`. Handing the reranker the full deadline is
    // what makes the worst case additive (ruling 4).
    reranked = await withDeadline(remaining, o => opts.reranker!(query, documents, o));
  } catch (error) {
    return degradeToFusion("failed", {
      candidateCount: documents.length,
      rerankError: error,
      rerankReason: error instanceof Error ? error.message : String(error),
    });
  }
  const rerankMs = performance.now() - rerankStart;

  // `onFallback` fires iff blendRerank found no usable signal — an empty
  // response or every score at/below the floor. That IS the "degenerate"
  // status; wiring it here is what keeps the degrade visible instead of a
  // silent collapse back to RRF order (ruling 6).
  let fallbackReason: string | undefined;
  const blended = blendRerank(
    candidates.map(r => ({ file: r.filepath, score: r.score })),
    reranked,
    {
      ...(opts.rerankWeight === undefined ? {} : { rerankWeight: opts.rerankWeight }),
      degenerateFloor: opts.degenerateFloor ?? RERANK_DEGENERATE_FLOOR,
      onFallback: (reason: string) => { fallbackReason = reason; },
    },
  );
  if (fallbackReason !== undefined) {
    return {
      ...degradeToFusion("degenerate", {
        candidateCount: documents.length,
        rerankedCount: reranked.length,
        rerankReason: fallbackReason,
      }),
      timings: { hybridMs, rerankMs, totalMs: performance.now() - t0 },
    };
  }

  // Map the blended order back onto the full SearchResults, carrying the
  // BLENDED score. Candidates beyond `rerankCap` were never seen by the
  // reranker and keep their fused order behind the reranked block — the same
  // "never drop a document" property blendRerank has within the pool.
  const byPath = new Map(candidates.map(r => [r.filepath, r]));
  const head = blended
    .map(b => { const orig = byPath.get(b.file); return orig ? { ...orig, score: b.score } : null; })
    .filter((r): r is SearchResult => r !== null);
  const tail = hybrid.results.slice(rerankCap);

  return {
    results: [...head, ...tail].slice(0, limit),
    hybrid,
    rerank: "applied",
    candidateCount: documents.length,
    rerankedCount: reranked.length,
    timings: { hybridMs, rerankMs, totalMs: performance.now() - t0 },
    deadlineMs,
  };
}

/**
 * Reranked search returning bare `SearchResult[]`.
 *
 * BACK-COMPAT-SHAPED WRAPPER, matching pgSearchVec / pgSearchFts /
 * pgSearchHybrid. Note what this signature CANNOT express: whether the rerank
 * stage ran at all, and whether the answer was one-arm or two. Prefer
 * pgSearchRerankedDetailed for anything user-facing.
 */
export async function pgSearchReranked(
  c: PgQueryable,
  query: string,
  opts: PgSearchRerankedOptions = {},
): Promise<SearchResult[]> {
  return (await pgSearchRerankedDetailed(c, query, opts)).results;
}
