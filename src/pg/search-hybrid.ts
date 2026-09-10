/**
 * PostgreSQL HYBRID READ PATH — RRF fusion of the vector and lexical arms
 * (master-harness-2wx75 slice 5).
 *
 * Slices 1-3 built the vector arm (src/pg/search.ts), slice 4 the lexical arm
 * (src/pg/search-fts.ts). Both were proven in ISOLATION and unproven in
 * COMPOSITION. This module is the composition and nothing else.
 *
 * THREE DESIGN RULINGS ARE ENCODED HERE. Each one is a decision, not an
 * accident, and each has unit cases that go red if it is undone.
 *
 * 1. FUSION CONSUMES RANKS, NOT SCORES. There is deliberately NO cross-backend
 *    score normalization, no blend weight and no scaling between the arms. A
 *    cosine-derived `1 - distance` and a `ts_rank_cd` value are not on a shared
 *    scale, and no comparability between them has been MEASURED on this corpus
 *    — so inventing one would encode a relationship nobody has evidence for.
 *    Reciprocal Rank Fusion needs only the ORDER each arm returned, which is
 *    exactly the part of each arm's output we have grounds to trust.
 *
 * 2. THE ARMS ARE EQUALLY WEIGHTED, AND THAT IS NOT CONFIGURABLE. `[1, 1]`
 *    below is hard-coded on purpose: a per-arm weight is a tuning knob, and a
 *    tuning knob with no eval behind it is the same unmeasured claim as a
 *    normalization. When the retrieval eval exists, the weights become a
 *    measured parameter; until then there is nothing to tune against.
 *
 * 3. A FUSED RESULT NEVER SILENTLY PRESENTS ITSELF AS TWO-ARM. `arms` is a
 *    typed, always-present discriminator ("vec+fts" | "vec-only" | "fts-only" |
 *    "none"). One degraded arm is NOT an error — the healthy arm's ranking is
 *    returned — but the caller can always see it was a single-arm answer, and
 *    a caller that renders "hybrid search" over a `"fts-only"` result is
 *    telling its user something false. See PgHybridSearchResult.
 *
 * REUSE, NOT REIMPLEMENTATION. The RRF itself is src/search-utils.ts's
 * `reciprocalRankFusion` — the same helper the sqlite hybrid path has used all
 * along, with its own unit suite at tests/unit/search-utils.rrf.test.ts. A
 * second RRF in this file would be a second place for the constant k=60 and the
 * rank-bonus policy to drift. `toRanked` / `attachRrfScores` are reused for the
 * same reason.
 */

import type { SearchResult } from "../store.ts";
import { reciprocalRankFusion, toRanked, attachRrfScores } from "../search-utils.ts";
import {
  pgSearchVecDetailed,
  type PgQueryable,
  type PgVecEmbedder,
  type PgVecDegradedReason,
  type PgVecSearchDetailedResult,
} from "./search.ts";
import {
  pgSearchFtsDetailed,
  type PgFtsDegradedReason,
  type PgFtsSearchDetailedResult,
} from "./search-fts.ts";

/**
 * The standard RRF constant. 60 is `reciprocalRankFusion`'s own default and the
 * value the sqlite hybrid path uses; it is named here so the hybrid arm's k is
 * greppable and so the unit tier can assert that changing it changes the
 * fusion (a k that no case can move is a knob that is not wired).
 */
export const DEFAULT_PG_HYBRID_RRF_K = 60;

/** Which arms actually CONTRIBUTED rows to `results`. */
export type PgHybridArms = "vec+fts" | "vec-only" | "fts-only" | "none";

/**
 * Why an arm did not contribute.
 *
 * `kind: "degraded"` carries the arm's OWN reason union verbatim — the vec and
 * FTS reason vocabularies are deliberately NOT merged into one flat union,
 * because "embed-unavailable" and "empty-tsquery" are not the same kind of
 * fact and a caller that wants to act on one needs to know which arm it came
 * from.
 *
 * `kind: "threw"` is the composition-only case. Each arm throws (rather than
 * degrades) when its SQL leg was cancelled by its server-side bound — correct
 * in isolation, but under fusion a cancelled vector scan must not destroy a
 * perfectly good lexical answer. So a rejected arm is caught HERE, at the
 * composition boundary, and demoted to a non-contributing arm. The error is
 * carried, never swallowed.
 */
export type PgHybridArmFailure =
  | { arm: "vec"; kind: "degraded"; reason: PgVecDegradedReason }
  | { arm: "vec"; kind: "threw"; error: unknown }
  | { arm: "fts"; kind: "degraded"; reason: PgFtsDegradedReason }
  | { arm: "fts"; kind: "threw"; error: unknown };

/**
 * The typed result of a hybrid search.
 *
 * FOUR OUTCOMES a caller must handle, and `arms` alone distinguishes all four:
 *
 *  1. TWO-ARM — `arms: "vec+fts"`, `degraded: false`, `armFailures: []`. Both
 *     arms ran and both are trustworthy. `results` is the RRF of their two
 *     orderings. This is the ONLY value on which a caller may claim "hybrid".
 *  2. SINGLE-ARM — `arms: "vec-only" | "fts-only"`, `degraded: true`, one entry
 *     in `armFailures`. The healthy arm's ranking, in its own order. NOT an
 *     error and NOT empty — but also not hybrid, and ruling #3 above is that
 *     the caller can always tell.
 *  3. NO-ARM — `arms: "none"`, `degraded: true`, `results: []`, both arms in
 *     `armFailures`. Nothing was searched. Presenting this as "no matches" is
 *     the lie both arms' degraded channels exist to prevent.
 *  4. THROWN — both arms rejected. The call rejects with the VEC arm's error
 *     (the FTS error is attached as `cause`). A total failure is never folded
 *     into an empty answer.
 *
 * `degraded: false` ⇔ `arms === "vec+fts"` ⇔ `armFailures.length === 0`. That
 * three-way equivalence is asserted directly in the unit tier.
 *
 * GENUINE-EMPTY SURVIVES FUSION. Two healthy arms that both matched zero
 * documents give `arms: "vec+fts"`, `degraded: false`, `results: []` — the
 * trustworthy "nothing matched" each arm already distinguished from "we could
 * not look", preserved rather than collapsed.
 */
export interface PgHybridSearchResult {
  /**
   * The fused ranking, capped at `limit`. Scores are RRF scores — a FUSION
   * scale, deliberately not either arm's native scale (ruling #1). Do not
   * compare a value here against a `pgSearchVec` or `pgSearchFts` score.
   */
  results: SearchResult[];
  /** Which arms contributed. Always present; never claims more than ran. */
  arms: PgHybridArms;
  /** true iff at least one arm failed to contribute. */
  degraded: boolean;
  /** Every non-contributing arm, with its arm-native cause. */
  armFailures: PgHybridArmFailure[];
  /** The vec arm's own detailed result, when it did not throw. */
  vec?: PgVecSearchDetailedResult;
  /** The FTS arm's own detailed result, when it did not throw. */
  fts?: PgFtsSearchDetailedResult;
  /** The RRF constant actually applied. */
  rrfK: number;
}

/**
 * Options for a hybrid search. Every field is forwarded to BOTH arms so the two
 * legs cannot silently disagree about scope or budget — there is deliberately
 * no per-arm limit, per-arm timeout or per-arm collection filter.
 *
 * There is also deliberately no `weights` field. See ruling #2.
 */
export interface PgSearchHybridOptions {
  /** One collection name, a list of them, or nothing = every collection. */
  collections?: string | string[];
  /**
   * Max documents returned AFTER fusion. Default 20 — same as both arms and
   * sqlite. Each arm is asked for this many, so fusion sees up to 2x`limit`
   * candidates and the union is then capped back to `limit`.
   */
  limit?: number;
  /** Wall-clock budget in ms, applied to each arm independently. */
  timeoutMs?: number;
  /** Server-side bound on each arm's SQL legs, in ms. */
  statementTimeoutMs?: number;
  /** Embedding backend for the vec arm. Defaults to the vec arm's default. */
  embedder?: PgVecEmbedder;
  /** Fragment overfetch for the vec arm's ANN pass. */
  overfetch?: number;
  /** RRF constant. Defaults to DEFAULT_PG_HYBRID_RRF_K. */
  rrfK?: number;
}

/**
 * Fuse a set of already-ranked, already-trustworthy arm outputs.
 *
 * Pure and exported so the fusion policy — ranks only, equal weights, union
 * capped to `limit`, single-list passthrough — is assertable without a
 * database or an embedder.
 *
 * `lists` MUST contain only CONTRIBUTING arms. Handing it a degraded arm's
 * empty list would be harmless arithmetically and wrong semantically: the
 * `arms` discriminator is computed from the same set, so an empty list in here
 * is how a one-arm answer starts claiming to be two-arm.
 */
export function fuseRankedArms(
  lists: SearchResult[][],
  limit: number,
  k: number = DEFAULT_PG_HYBRID_RRF_K,
): SearchResult[] {
  if (lists.length === 0) return [];
  // A FRESH weights literal per call: reciprocalRankFusion sanitizes weights
  // IN PLACE, so a shared array would be mutated under us.
  const weights = lists.map(() => 1);
  const fused = reciprocalRankFusion(lists.map(l => l.map(toRanked)), weights, k);
  // Flatten in arm order; attachRrfScores keeps the FIRST SearchResult it sees
  // per filepath, so a document both arms returned is rendered from the first
  // arm's row. Only the row's presentation fields come from there — the score
  // is the fused one.
  return attachRrfScores(fused, lists.flat()).slice(0, limit);
}

/**
 * Hybrid search over the PG vault: run both arms, fuse by rank.
 *
 * The arms run CONCURRENTLY (`allSettled`, so one rejection cannot cancel the
 * other's answer — see PgHybridArmFailure `kind: "threw"`). Both receive the
 * same scope and the same budget.
 *
 * This sits behind the same `(client, query, options)` shape as
 * pgSearchVecDetailed / pgSearchFtsDetailed, so migrating a caller onto the
 * hybrid path is a call-site change rather than a re-architecture. Caller
 * migration is NOT done here; nothing in src/ calls this yet.
 */
export async function pgSearchHybridDetailed(
  c: PgQueryable,
  query: string,
  opts: PgSearchHybridOptions = {},
): Promise<PgHybridSearchResult> {
  const limit = opts.limit ?? 20;
  const rrfK = opts.rrfK ?? DEFAULT_PG_HYBRID_RRF_K;
  const shared = {
    ...(opts.collections === undefined ? {} : { collections: opts.collections }),
    limit,
    ...(opts.timeoutMs === undefined ? {} : { timeoutMs: opts.timeoutMs }),
    ...(opts.statementTimeoutMs === undefined ? {} : { statementTimeoutMs: opts.statementTimeoutMs }),
  };

  const [vecOutcome, ftsOutcome] = await Promise.allSettled([
    pgSearchVecDetailed(c, query, {
      ...shared,
      ...(opts.embedder === undefined ? {} : { embedder: opts.embedder }),
      ...(opts.overfetch === undefined ? {} : { overfetch: opts.overfetch }),
    }),
    pgSearchFtsDetailed(c, query, shared),
  ]);

  // BOTH REJECTED: outcome 4. Never demote a total failure to an empty answer.
  if (vecOutcome.status === "rejected" && ftsOutcome.status === "rejected") {
    const err = vecOutcome.reason;
    if (err instanceof Error && err.cause === undefined) err.cause = ftsOutcome.reason;
    throw err;
  }

  const armFailures: PgHybridArmFailure[] = [];
  const lists: SearchResult[][] = [];

  const vec = vecOutcome.status === "fulfilled" ? vecOutcome.value : undefined;
  if (vecOutcome.status === "rejected") {
    armFailures.push({ arm: "vec", kind: "threw", error: vecOutcome.reason });
  } else if (vec!.degraded) {
    armFailures.push({ arm: "vec", kind: "degraded", reason: vec!.degradedReason! });
  } else {
    lists.push(vec!.results);
  }

  const fts = ftsOutcome.status === "fulfilled" ? ftsOutcome.value : undefined;
  if (ftsOutcome.status === "rejected") {
    armFailures.push({ arm: "fts", kind: "threw", error: ftsOutcome.reason });
  } else if (fts!.degraded) {
    armFailures.push({ arm: "fts", kind: "degraded", reason: fts!.degradedReason! });
  } else {
    lists.push(fts!.results);
  }

  const vecContributed = vec !== undefined && !vec.degraded;
  const ftsContributed = fts !== undefined && !fts.degraded;
  const arms: PgHybridArms = vecContributed && ftsContributed
    ? "vec+fts"
    : vecContributed
      ? "vec-only"
      : ftsContributed
        ? "fts-only"
        : "none";

  return {
    results: fuseRankedArms(lists, limit, rrfK),
    arms,
    degraded: armFailures.length > 0,
    armFailures,
    ...(vec === undefined ? {} : { vec }),
    ...(fts === undefined ? {} : { fts }),
    rrfK,
  };
}

/**
 * Hybrid search returning bare `SearchResult[]`.
 *
 * BACK-COMPAT-SHAPED WRAPPER, matching pgSearchVec / pgSearchFts. Note what
 * this signature CANNOT express: whether the answer was one-arm or two. Prefer
 * pgSearchHybridDetailed for anything user-facing — a UI that says "hybrid"
 * over this return value is guessing.
 */
export async function pgSearchHybrid(
  c: PgQueryable,
  query: string,
  opts: PgSearchHybridOptions = {},
): Promise<SearchResult[]> {
  return (await pgSearchHybridDetailed(c, query, opts)).results;
}
