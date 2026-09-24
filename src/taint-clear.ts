/**
 * Evidence-gated clear of the embed-geometry taint (master-harness-vn4rz.59).
 *
 * `embed_geometry_taint` is set when an embed run could not verify its geometry (no
 * preflight, unverified end, or mid-run drift). Until now only a verified full
 * `clawmem embed --force` cleared it. After a targeted remediation (e.g.
 * `embed --requeue-hashes` of the affected window) the vault can be proven healthy
 * without a whole-vault rebuild. This module clears the flag ONLY on that proof:
 *
 *   1. the embed single-writer lease is acquired (never clear while an embed runs);
 *   2. the geometry canary battery PASSES, including drift vs the stored baseline;
 *   3. the doctor sampled-vector validator reaches the target with zero definitive
 *      failures (stale-policy / arm-mismatch rows are skipped and do not count);
 *   4. the clear is lease-fenced and leaves an audit record in
 *      `embed_geometry_taint_cleared`.
 *
 * Any failed gate refuses (exit 1) and leaves the taint untouched.
 */
import { runCanaryBattery, runSampledVectorValidation, SAMPLE_REPLACEMENT_BUDGET } from "./canary.ts";
import { EmbedLeaseLostError, type Store } from "./store.ts";
import { acquireWorkerLease, releaseWorkerLease, renewWorkerLease } from "./worker-lease.ts";

export const TAINT_FLAG = "embed_geometry_taint";
export const TAINT_CLEARED_FLAG = "embed_geometry_taint_cleared";
export const DEFAULT_CLEAR_TAINT_SAMPLE = 200;
/** Same lease name and TTL as cmdEmbed — the clear serializes against every embed. */
export const EMBED_LEASE_NAME = "embedding";
export const EMBED_LEASE_TTL_MS = 60_000;

export type ClearTaintEmbed = (
  text: string,
) => Promise<{ embedding: number[] | Float32Array; model?: string; endpoint?: string; input?: string } | null>;

export type ClearTaintGate = "lease" | "canary" | "sampler" | "taint-changed";

export interface TaintClearAudit {
  prior_reason: string;
  cleared_at: string;
  method: "doctor --clear-taint";
  sample_target: number;
  validated: number;
  validated_seq0: number;
  eligible: number;
  attempts: number;
  stale_policy_skipped: number;
  canary: { pass: true; drift_checked: true; profile_key: string; margins: Record<string, number> };
}

export type ClearTaintOutcome =
  | { status: "no-taint"; exitCode: 0; message: string }
  | { status: "cleared"; exitCode: 0; message: string; audit: TaintClearAudit }
  | { status: "refused"; exitCode: 1; gate: ClearTaintGate; message: string; details: string[] };

/** Replacement budget for the clear: bounded, but wide enough that skipped stale-policy
 *  rows do not starve a large target (the doctor default of 8 is sized for target 16). */
export function clearTaintReplacementBudget(target: number): number {
  return Math.max(SAMPLE_REPLACEMENT_BUDGET, Math.ceil(target / 4));
}

export async function clearGeometryTaint(
  s: Store,
  embed: ClearTaintEmbed,
  opts: { sample: number; now?: () => Date },
): Promise<ClearTaintOutcome> {
  const now = opts.now ?? (() => new Date());
  const prior = s.getVaultFlag(TAINT_FLAG);
  if (prior === null) {
    return { status: "no-taint", exitCode: 0, message: "No geometry taint present — nothing to clear." };
  }

  const refuse = (gate: ClearTaintGate, message: string, details: string[] = []): ClearTaintOutcome =>
    ({ status: "refused", exitCode: 1, gate, message: `Refused (${gate} gate): ${message}. Taint left in place.`, details });

  const lease = acquireWorkerLease(s, EMBED_LEASE_NAME, EMBED_LEASE_TTL_MS);
  if (!lease.acquired || !lease.token) {
    return refuse("lease", "the embed lease is held by another writer (an embed is running)");
  }
  const leaseGuard = { workerName: EMBED_LEASE_NAME, token: lease.token };
  let leaseLost = false;
  const heartbeat = setInterval(() => {
    if (!renewWorkerLease(s, EMBED_LEASE_NAME, leaseGuard.token, EMBED_LEASE_TTL_MS)) leaseLost = true;
  }, Math.floor(EMBED_LEASE_TTL_MS / 2));

  try {
    const canary = await runCanaryBattery(embed, key => s.getCanaryBaseline(key));
    if ("unavailable" in canary) return refuse("canary", `geometry canary unavailable (${canary.reason})`);
    if (!canary.pass) return refuse("canary", "geometry canary FAILED", canary.failures);
    if (!canary.driftChecked) {
      return refuse("canary", `no stored canary baseline for ${canary.profileKey} — drift vs the vault's geometry cannot be checked`);
    }

    const target = opts.sample;
    const summary = await runSampledVectorValidation(s, embed, {
      target,
      replacementBudget: clearTaintReplacementBudget(target),
    });
    const counts = `${summary.validated}/${target} validated, ${summary.eligible} eligible, ${summary.attempts} attempts, ${summary.stalePolicy} stale-policy skipped`;
    if (summary.definitiveFailures.length > 0) {
      return refuse("sampler", `sampled vectors DEFINITIVE failure (${counts})`, summary.definitiveFailures.slice(0, 4));
    }
    if (summary.validated < target) {
      return refuse("sampler", `sampled vectors INCOMPLETE — target not reached (${counts})`, summary.stalePolicyRows.slice(0, 4));
    }
    if (summary.validatedSeq0 < summary.seq0Target) {
      return refuse("sampler", `sampled vectors DEGRADED — seq-0 quota unmet (${summary.validatedSeq0}/${summary.seq0Target}; ${counts})`);
    }
    if (leaseLost) return refuse("lease", "the embed lease was lost during validation");
    const current = s.getVaultFlag(TAINT_FLAG);
    if (current !== prior) return refuse("taint-changed", "the taint changed while validating — re-run to evaluate the new state");

    const audit: TaintClearAudit = {
      prior_reason: prior,
      cleared_at: now().toISOString(),
      method: "doctor --clear-taint",
      sample_target: target,
      validated: summary.validated,
      validated_seq0: summary.validatedSeq0,
      eligible: summary.eligible,
      attempts: summary.attempts,
      stale_policy_skipped: summary.stalePolicy,
      canary: { pass: true, drift_checked: true, profile_key: canary.profileKey, margins: canary.margins },
    };
    // Audit first, then clear: a crash between the two leaves the taint set (doctor stays
    // red) — never a cleared taint without its record. Both writes are lease-fenced.
    s.setVaultFlag(TAINT_CLEARED_FLAG, JSON.stringify(audit), leaseGuard);
    s.clearVaultFlag(TAINT_FLAG, leaseGuard);
    return {
      status: "cleared",
      exitCode: 0,
      message: `Geometry taint cleared (${summary.validated}/${target} sampled vectors validated, canary pass + drift OK). Prior reason: ${prior}`,
      audit,
    };
  } catch (err) {
    if (err instanceof EmbedLeaseLostError) return refuse("lease", "the embed lease was lost before the clear");
    throw err;
  } finally {
    clearInterval(heartbeat);
    releaseWorkerLease(s, EMBED_LEASE_NAME, leaseGuard.token);
  }
}

/** One-line doctor rendering of the last evidence-gated clear, or null. */
export function describeLastTaintClear(raw: string | null): string | null {
  if (!raw) return null;
  try {
    const a = JSON.parse(raw) as Partial<TaintClearAudit>;
    return `last cleared ${a.cleared_at ?? "?"} via ${a.method ?? "?"} (${a.validated ?? "?"}/${a.sample_target ?? "?"} sampled vectors validated; prior: ${a.prior_reason ?? "?"})`;
  } catch {
    return `last clear record unparseable: ${raw.slice(0, 120)}`;
  }
}
