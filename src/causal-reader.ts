/**
 * s342 causal reader — wire shaping shared by the MCP tool and the REST route.
 *
 * `CAUSAL_READER_MAX_BYTES` caps the COMPLETE serialized response: for MCP, the
 * UTF-8 bytes of the full result value (text AND structured copies together,
 * envelope-inclusive accounting); for REST, the complete JSON body. Truncation
 * removes whole edges from BOTH representations symmetrically — one retained
 * edge set — following the reader's deterministic total order
 * (depth, weight DESC, sourceDocId, targetDocId, direction), and the
 * `truncated` flag is reported in both.
 */

import type { CausalEdgeRecord } from "./store.ts";

export const CAUSAL_READER_MAX_BYTES = 65_536;

/** Display/snapshot cap for fact text (prompt input, stored snapshots, and
 *  synthesized legacy witnesses all share it). */
export const CAUSAL_FACT_CHAR_CAP = 300;

export type LegacyEdgeSource = { weight: number | null; metadata: string | null };

/**
 * THE shared validity rule for pre-cut edge evidence — used identically by the
 * writer's materialize-vs-refuse split, the census's materializable
 * classification, and the reader's lazy read-through, so no surface can call an
 * edge valid that another calls unresolved. Old-writer metadata qualifies only
 * when it parses to an object with nonempty reasoning + both fact snapshots AND
 * the edge weight is a finite confidence in [0,1]. Anything else (Beads
 * `{origin:'beads'}`, manual rows, corruption, out-of-range weights) yields
 * NOTHING.
 */
export function parseLegacyEdgeWitness(row: LegacyEdgeSource): {
  reasoning: string;
  sourceFact: string;
  targetFact: string;
  confidence: number;
} | null {
  if (row.metadata == null) return null;
  if (typeof row.weight !== "number" || !Number.isFinite(row.weight) || row.weight < 0 || row.weight > 1) {
    return null;
  }
  let meta: unknown;
  try {
    meta = JSON.parse(row.metadata);
  } catch {
    return null;
  }
  if (!meta || typeof meta !== "object" || Array.isArray(meta)) return null;
  const m = meta as Record<string, unknown>;
  if (typeof m.reasoning !== "string" || m.reasoning.trim().length === 0) return null;
  if (typeof m.source_fact !== "string" || m.source_fact.trim().length === 0) return null;
  if (typeof m.target_fact !== "string" || m.target_fact.trim().length === 0) return null;
  return {
    reasoning: m.reasoning,
    sourceFact: m.source_fact.slice(0, CAUSAL_FACT_CHAR_CAP),
    targetFact: m.target_fact.slice(0, CAUSAL_FACT_CHAR_CAP),
    confidence: row.weight,
  };
}

/**
 * Build the smallest-suffix-dropped response that fits the byte ceiling.
 *
 * `build` must construct the COMPLETE response value (the whole MCP result
 * object or the whole REST body) from a retained edge prefix and the current
 * truncated flag; it is re-invoked after each drop so both representations
 * always reflect the same retained set. Edges must already be in the reader's
 * deterministic total order — the drop order IS that order, from the tail.
 *
 * The ceiling is UNCONDITIONAL: if even the zero-edge base envelope exceeds it
 * (an unbounded base field slipped past the callers' display bounds),
 * `overflowBuild` — whose output must be a small static value — replaces the
 * response entirely rather than shipping an oversized one.
 */
export function capCausalWire<T>(
  edges: CausalEdgeRecord[],
  truncatedByBudget: boolean,
  build: (kept: CausalEdgeRecord[], truncated: boolean) => T,
  overflowBuild: () => T,
  maxBytes: number = CAUSAL_READER_MAX_BYTES,
): T {
  let kept = edges;
  let truncated = truncatedByBudget;
  let result = build(kept, truncated);
  while (kept.length > 0 && Buffer.byteLength(JSON.stringify(result), "utf8") > maxBytes) {
    kept = kept.slice(0, kept.length - 1);
    truncated = true;
    result = build(kept, truncated);
  }
  if (Buffer.byteLength(JSON.stringify(result), "utf8") > maxBytes) {
    console.warn(`[causal-reader] base response exceeded the ${maxBytes}-byte ceiling — returning the overflow envelope`);
    const fallback = overflowBuild();
    // The fallback's size is VERIFIED, not assumed: an oversized fallback is a
    // caller bug (it must be a small static value) and fails loudly rather
    // than shipping an oversized response under a "capped" contract.
    if (Buffer.byteLength(JSON.stringify(fallback), "utf8") > maxBytes) {
      throw new Error(`capCausalWire: overflowBuild() exceeded the ${maxBytes}-byte ceiling — it must produce a small static value`);
    }
    return fallback;
  }
  return result;
}
