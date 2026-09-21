/**
 * Embed-input fingerprints (master-harness-5n0ew).
 *
 * A fingerprint is what `clawmem doctor` uses to decide whether a stored vector can be
 * compared against a fresh embed at all. It must therefore attest the text the MODEL
 * actually received — not the text clawmem assembled before the embed arm truncated it —
 * and the arm that received it, because the remote endpoint and the in-process
 * node-llama-cpp fallback are not interchangeable geometries (measured on the live vault:
 * even with identical input and the same EmbeddingGemma weights, the Q8 in-process arm and
 * the ollama arm agree only at cos 0.968–0.9995 on >512-token inputs).
 *
 * Two schemes are in the wild:
 *
 *   v1 (legacy, bare 64-hex)  sha256(formatted fragment text BEFORE truncation). Says
 *                             nothing about what was sent when the input was oversized, and
 *                             nothing about which arm embedded it.
 *   v2 `v2:<arm>:<64-hex>`     sha256(text actually handed to the model, AFTER the embed
 *                             arm's own truncation), plus the producing arm.
 *
 * The v2 hash is computed from `EmbeddingResult.input`, which the embed arm reports from
 * the very string it put on the wire / into the context. There is deliberately NO second
 * copy of the truncation policy anywhere else: embed time and validate time both read the
 * sent text back from the same embedder, so they cannot disagree about what "the input"
 * was without the fingerprint showing it.
 *
 * Self-describing on purpose: no schema migration, and every v1 row stays readable.
 */
import { createHash } from "crypto";
import { LOCAL_EMBED_ARM_LABEL, type EmbeddingResult } from "./llm.ts";

export type EmbedArm = "remote" | "local" | "unreported";

export type ParsedEmbedInputFp =
  | { version: 1; hex: string }
  | { version: 2; arm: EmbedArm; hex: string };

export function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/** Which arm produced a result, from the arm label the embedder reported. */
export function embedArmOf(result: Pick<EmbeddingResult, "endpoint"> | null | undefined): EmbedArm {
  const endpoint = result?.endpoint;
  if (!endpoint) return "unreported";
  if (endpoint === LOCAL_EMBED_ARM_LABEL) return "local";
  return "remote";
}

/**
 * The exact text a result was produced from. `formattedText` is what clawmem ASKED to
 * embed; `result.input` is what the arm reports it actually sent. An arm that does not
 * report (a test double, an older LLM implementation) is taken at its word that it sent
 * the request unchanged — the same assumption v1 made, now explicit and local.
 */
export function sentEmbedInput(formattedText: string, result: Pick<EmbeddingResult, "input"> | null | undefined): string {
  return typeof result?.input === "string" ? result.input : formattedText;
}

/** v2 fingerprint over the text actually embedded, bound to the producing arm. */
export function embedInputFingerprint(
  formattedText: string,
  result: Pick<EmbeddingResult, "input" | "endpoint"> | null | undefined
): string {
  return `v2:${embedArmOf(result)}:${sha256Hex(sentEmbedInput(formattedText, result))}`;
}

/**
 * Parse a stored fingerprint. Absent → null (the no-fingerprint legacy structural tier).
 * Anything that is not a well-formed v2 value is read as v1 — an opaque pre-truncation
 * digest compared verbatim, so a garbled value can only ever MISMATCH (stale-input), never
 * silently validate.
 */
export function parseEmbedInputFp(fp: string | null | undefined): ParsedEmbedInputFp | null {
  if (!fp) return null;
  const m = /^v2:(remote|local|unreported):([0-9a-f]{64})$/.exec(fp);
  if (m) return { version: 2, arm: m[1] as EmbedArm, hex: m[2]! };
  return { version: 1, hex: fp };
}
