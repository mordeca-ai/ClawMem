/**
 * Reranker health probe — asserts the reranker DISCRIMINATES, not just that it responds.
 *
 * Background: the deployed zerank-2 GGUF was mis-converted (no score head), so it returned
 * HTTP 200 + valid JSON + finite positive scores (~1e-11) yet ranked near-randomly and silently
 * collapsed the final ranking to RRF. Liveness checks all passed. This probe instead runs a small
 * golden set of same-topic (query, relevant, hard-negative) triples through the LIVE reranker
 * (cache-bypassed) and asserts three things:
 *
 *   1. coverage     — the reranker scored every probe doc (store.rerank throws RerankCoverageError
 *                     otherwise, before its zero-fill would hide an omitted score);
 *   2. calibration  — the best relevant-doc score lands in a sane band (>= CALIB_FLOOR);
 *   3. ordering     — EVERY pair ranks the relevant doc ABOVE the hard negative (scale-free);
 *   4. discrimination — EVERY pair clears a MARGIN measured in LOGIT space (see below).
 *
 * Why logit space (master-harness-1nvlz): the endpoint returns sigmoid-squashed relevance in
 * [0,1], and the sigmoid SATURATES. A pair the model separates by 4.06 logits — an enormous,
 * correct separation — compresses to 0.035 in sigmoid space once both docs sit in the tail, so a
 * fixed ADDITIVE sigmoid-space threshold is not scale-invariant: it called a healthy, correctly
 * ordering reranker "degenerate" (8/8 coverage, 0/8 inversions) purely because the model's scores
 * had moved out of the sigmoid's linear region. Logit space undoes the squash exactly
 * (ln(s/(1-s)) reproduces the server's pre-sigmoid logits to full precision), so the margin is
 * again a difference of model outputs rather than a difference of saturated probabilities.
 *
 * See RERANKER-HEALTH-GUARD-DESIGN.md.
 */
import { readFileSync } from "fs";
import { join } from "path";
import { DEFAULT_RERANK_MODEL, RerankCoverageError, RerankMalformedResponseError, type Store } from "../store.ts";

// Calibration band — LOCKED from a live zerank-2-seq baseline (2026-06-26, 8-pair golden set):
//   relevant scores 0.9233-0.9700, hard-neg max 0.3120, 0/8 inverted.
//   broken zerank-2 GGUF regime: every score <= 8.03e-7 (32-query probe). 5-6 OOM of separation.
// CALIB_FLOOR is the arm that catches the ~0 collapse, and the ONLY arm whose failure implicates a
// score-head-less GGUF. It stays in SIGMOID space and is deliberately conservative — the margin
// does the discrimination, so the band must not false-fail a working-but-lower reranker.
export const RERANK_CALIB_FLOOR = 0.05; // band: best relevant-doc score across pairs must clear this

// Discrimination margin — LOGIT space, recalibrated 2026-09-01 against the live
// bge-reranker-v2-m3-q4_k_m behind the sigmoid proxy on 127.0.0.1:8091, replaying the shipped
// 8-pair golden set through the production path (400-char truncation), 3 repeat runs:
//   per-pair logit margins 1.17 / 9.98 / 5.11 / 11.80 / 11.56 / 7.66 / 4.06 / 7.06 — 0/8 inverted.
//   run-to-run spread on the tightest pair (pair 0): 1.167 .. 1.305 (~10%).
// Threshold 0.5 sits ~2.3x below that live minimum (and ~8x below the 2nd-tightest pair, 4.06),
// while a degenerate reranker — constant or collapsed scores — yields a margin of exactly 0 and an
// inverted one yields a negative margin, so the healthy/degenerate gap the guard must straddle is
// [0, 1.17] and 0.5 is near its geometric middle. In plain terms 0.5 logits ≈ 1.65:1 odds
// separation: the least separation still worth calling "discriminating".
// NOTE the deliberate rename from the old sigmoid-space RERANK_DISCRIM_MARGIN (0.25): the units
// changed, so a stale import must break loudly rather than silently compare logits to 0.25.
export const RERANK_DISCRIM_LOGIT_MARGIN = 0.5; // per-pair: logit(rel) - logit(neg) >= this

// Clamp epsilon for the sigmoid→logit inverse. The endpoint routinely returns scores that round to
// 1.0 (observed 0.999370 and up); an unclamped logit(1) is +Infinity, which would poison every
// comparison downstream (Infinity - Infinity = NaN, and NaN < threshold is false — a check that
// silently cannot fail). 1e-9 caps |logit| at ~20.7, an order of magnitude above the largest real
// logit this model produces (7.4), so clamping can never distort a genuine measurement — it only
// bounds the degenerate ones.
export const RERANK_LOGIT_CLAMP_EPS = 1e-9;
// Default per-probe-request timeout (the production remote fetch is otherwise untimed; a hung
// reranker must not hang the healthcheck).
export const RERANK_PROBE_TIMEOUT_MS = 10_000;

/**
 * Inverse sigmoid, clamped. Maps a relevance score in [0,1] back to the model's pre-sigmoid logit.
 * Scores OUTSIDE [0,1] cannot be probabilities, so they are already logit-space and pass through
 * unchanged — that keeps the probe correct if it is ever pointed at a raw-logit endpoint instead of
 * the sigmoid proxy (pointing it at the wrong space is precisely the failure mode this guard was
 * misdiagnosed as having).
 */
export function toLogit(score: number, eps: number = RERANK_LOGIT_CLAMP_EPS): number {
  if (!Number.isFinite(score)) return NaN;
  if (score < 0 || score > 1) return score; // already logit space — identity
  const s = Math.min(1 - eps, Math.max(eps, score));
  return Math.log(s / (1 - s));
}

/** True iff BOTH scores of a pair look like probabilities (so the pair needs the sigmoid inverse). */
function pairIsProbabilitySpace(a: number, b: number): boolean {
  return a >= 0 && a <= 1 && b >= 0 && b <= 1;
}

export interface GoldenTriple {
  query: string;
  relevant: string;
  hardNegative: string;
  note?: string;
}

export interface RerankHealthResult {
  ok: boolean;
  coverageOk: boolean;
  maxScore: number; // best relevant-doc score across pairs, SIGMOID space (calibration-band input)
  /** Smallest logit(relevant) - logit(hardNegative) across pairs. Renamed from the old sigmoid-space
   *  `minMargin` so a stale reader cannot silently compare logits against a sigmoid threshold. */
  minLogitMargin: number;
  inversions: number; // pairs where the hard negative scored >= the relevant doc (scale-free signal)
  pairsTotal: number;
  pairsScored: number; // pairs where both docs were scored (full coverage)
  failures: string[]; // human-readable failure reasons (empty iff ok)
  /** True iff the SIGMOID calibration band failed — the only arm that implicates a score-head-less
   *  GGUF, and therefore the only one that may carry the "re-deploy the seq-cls sidecar" advice. */
  calibrationFailed: boolean;
  thresholds: { calibFloor: number; discrimLogitMargin: number };
}

/**
 * Remediation advice for a FAILED probe, routed by WHICH arm failed.
 *
 * The zerank-2 / seq-cls-sidecar prescription is emitted ONLY for a calibration-floor failure. That
 * collapse to ~0 is the score-head-less GGUF's signature and the calibration arm is what detects
 * it; every other arm (margin, ordering, coverage) describes a reranker that is demonstrably
 * producing real, separated scores, for which "re-deploy the sidecar" is not merely unhelpful but
 * actively wrong — it was told to an operator mid-incident while doctor printed `max score 1.0e+0`
 * one line above (master-harness-1nvlz).
 */
export function rerankFailureAdvice(health: Pick<RerankHealthResult, "calibrationFailed" | "inversions" | "pairsScored" | "pairsTotal">): string {
  if (health.calibrationFailed) {
    return `Scores collapsed to ~0 — likely the deprecated zerank-2 GGUF (no score head). Re-deploy the seq-cls sidecar. See CLAUDE.md "SOTA upgrade".`;
  }
  if (health.pairsScored < health.pairsTotal) {
    return `The reranker did not score every probe doc — check the endpoint's top_n / batch handling and the request log. Do NOT re-deploy on this signal alone; the scores it did return may be fine.`;
  }
  if (health.inversions > 0) {
    return `The reranker ranked a hard negative at or above its relevant doc — a real ordering failure. Verify the deployed model and that query/document fields are not swapped. Compare against a known-good model before replacing the service.`;
  }
  return `The reranker orders every pair correctly but one or more pairs separate by less than the calibrated logit margin. This is NOT the score-collapse signature — do not re-deploy the sidecar. Either the serving model changed (compare against the calibration recorded in src/health/rerank-health.ts) or the threshold needs re-deriving for the current model.`;
}

/** Load the shipped golden set (or an explicit path, for tests). */
export function loadGoldenSet(path?: string): GoldenTriple[] {
  const p = path ?? join(import.meta.dir, "rerank-golden.json");
  const parsed = JSON.parse(readFileSync(p, "utf-8")) as { triples: GoldenTriple[] };
  return parsed.triples;
}

/**
 * Probe the reranker behind `store` for discrimination + calibration. Routes through store.rerank
 * with { noCache, requireLiveCoverage, timeoutMs } so it exercises the FULL production path
 * (remote → local fallback, dedup, intent, 400-char truncation) while bypassing the cache and
 * enforcing live coverage. `store` is structurally typed so tests can pass a fake reranker.
 */
export async function probeRerankHealth(
  store: Pick<Store, "rerank">,
  opts: {
    thresholds?: { calibFloor?: number; discrimLogitMargin?: number };
    timeoutMs?: number;
    model?: string;
    triples?: GoldenTriple[];
  } = {},
): Promise<RerankHealthResult> {
  const calibFloor = opts.thresholds?.calibFloor ?? RERANK_CALIB_FLOOR;
  const discrimLogitMargin = opts.thresholds?.discrimLogitMargin ?? RERANK_DISCRIM_LOGIT_MARGIN;
  const timeoutMs = opts.timeoutMs ?? RERANK_PROBE_TIMEOUT_MS;
  const model = opts.model ?? DEFAULT_RERANK_MODEL;
  const triples = opts.triples ?? loadGoldenSet();

  const failures: string[] = [];
  let maxScore = 0;
  let minLogitMargin = Infinity;
  let inversions = 0;
  let pairsScored = 0;

  for (let i = 0; i < triples.length; i++) {
    const t = triples[i]!;
    const relFile = `golden-${i}-rel`;
    const negFile = `golden-${i}-neg`;
    const docs = [
      { file: relFile, text: t.relevant },
      { file: negFile, text: t.hardNegative },
    ];
    const label = `pair ${i} ("${t.query.slice(0, 40)}")`;

    let scored: { file: string; score: number }[];
    try {
      // intent omitted (4th arg undefined); options force a live, coverage-checked call.
      scored = await store.rerank(t.query, docs, model, undefined, {
        noCache: true,
        requireLiveCoverage: true,
        timeoutMs,
      });
    } catch (err) {
      if (err instanceof RerankCoverageError) {
        failures.push(`${label}: coverage — reranker did not score ${err.missing.length} doc(s)`);
      } else if (err instanceof RerankMalformedResponseError) {
        failures.push(`${label}: malformed response — ${err.problems.join("; ")}`);
      } else {
        failures.push(`${label}: probe error — ${(err as Error).message}`);
      }
      continue;
    }

    const scoreMap = new Map(scored.map((s) => [s.file, s.score]));
    const relScore = scoreMap.get(relFile);
    const negScore = scoreMap.get(negFile);
    if (relScore === undefined || negScore === undefined || !Number.isFinite(relScore) || !Number.isFinite(negScore)) {
      // requireLiveCoverage should have thrown already; defensive.
      failures.push(`${label}: missing or non-finite score after rerank`);
      continue;
    }

    pairsScored++;
    maxScore = Math.max(maxScore, relScore);

    // ORDERING is the primary, scale-free assertion: a reranker that ranks the hard negative at or
    // above the relevant doc is broken no matter what space the scores live in, and no threshold
    // choice can make that verdict wrong.
    if (!(relScore > negScore)) {
      inversions++;
      failures.push(
        `${label}: INVERTED — hard negative scored >= relevant (rel ${relScore.toFixed(4)} vs neg ${negScore.toFixed(4)})`,
      );
    }

    // MARGIN in logit space (see RERANK_DISCRIM_LOGIT_MARGIN). Pairs already in logit space pass
    // through unconverted; probability pairs get the clamped sigmoid inverse.
    const inProb = pairIsProbabilitySpace(relScore, negScore);
    const relL = inProb ? toLogit(relScore) : relScore;
    const negL = inProb ? toLogit(negScore) : negScore;
    const logitMargin = relL - negL;
    minLogitMargin = Math.min(minLogitMargin, logitMargin);
    if (logitMargin < discrimLogitMargin) {
      failures.push(
        `${label}: logit margin ${logitMargin.toFixed(3)} < ${discrimLogitMargin} ` +
          `(rel ${relScore.toFixed(4)}→${relL.toFixed(3)} vs neg ${negScore.toFixed(4)}→${negL.toFixed(3)})`,
      );
    }
  }

  let calibrationFailed = false;
  if (maxScore < calibFloor) {
    calibrationFailed = true;
    failures.push(
      `calibration: max relevant-doc score ${maxScore.toExponential(2)} < floor ${calibFloor} — reranker is inert/degenerate (likely the deprecated zerank-2 GGUF; re-deploy the seq-cls sidecar)`,
    );
  }
  if (minLogitMargin === Infinity) minLogitMargin = 0; // no pair scored

  return {
    ok: failures.length === 0,
    coverageOk: pairsScored === triples.length,
    maxScore,
    minLogitMargin,
    inversions,
    pairsTotal: triples.length,
    pairsScored,
    failures,
    calibrationFailed,
    thresholds: { calibFloor, discrimLogitMargin },
  };
}
