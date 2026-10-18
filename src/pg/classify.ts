/**
 * Index-time content classification for ORIGIN collections
 * (master-harness-vn4rz.66; operator ruling Review Hub a7981735, 2026-09-02).
 *
 * ---------------------------------------------------------------------------
 * WHY THIS MODULE EXISTS
 *
 * The vault tripwire in src/pg/vaults.ts routes by COLLECTION and PATH. That is
 * the right key for file-authored collections, but monchujo session transcripts
 * all live under one innocuous root: a session that read a private file carries
 * that file's content in its transcript, and nothing about the transcript's own
 * path says so. The ruling: classify each monchujo document at index time on
 * path-reference and crypt-boundary heuristics, move the flagged ones to the
 * nsfw vault, and leave the rest in sfw — "keyed on the existing tripwire". The
 * classifier therefore only ever ESCALATES (resolveVault's `contentVault` hint):
 * it can send a document to nsfw, and it can never pull one out.
 *
 * ---------------------------------------------------------------------------
 * TWO TIERS
 *
 *  - PRIVATE_BOUNDARY_SIGNALS: references to the NSFW, persona and personal
 *    crypt roots. A hit ROUTES the session to the nsfw vault.
 *  - AMBIGUOUS_BOUNDARY_SIGNALS: references to the PII and business crypt roots
 *    (job search, financial, household, customer data, eval transcripts). A hit
 *    is REPORTED ONLY and never routes. Routing PII into the NSFW vault would be
 *    a conflation of its own: those roots are private for a different reason
 *    and belong to a different boundary. The operator rules on them through a
 *    hub item; until then they stay sfw and are counted.
 *
 * ---------------------------------------------------------------------------
 * THE UNIT OF ROUTING IS THE SESSION, NOT THE CHUNK
 *
 * A transcript is cut into chunk files (`<session>__c00.md`, `__c01.md`, ...).
 * If ANY chunk of a session carries a private signal, EVERY chunk of that
 * session routes nsfw. A chunk boundary is arbitrary, and the realistic leak is
 * the assistant paraphrasing a private file in the chunk NEXT to the one that
 * read it, where the path itself never appears.
 *
 * ---------------------------------------------------------------------------
 * CHANGING THE LISTS
 *
 * Widening (or narrowing) either list is a reviewed code change, never config.
 * Both lists are mirrored in master-harness tools/clawmem-vault-isolation-check,
 * which parses them TEXTUALLY from this file and fails on drift. Keep the literal
 * shape: one id/pattern object literal per line, double-quoted strings, no
 * backslashes inside a pattern (use `[.]` for a literal dot).
 *
 * Boundary rule shared by the path signals: the root must start at a path
 * segment boundary (no letter, digit, `_`, `.` or `-` immediately before it) and
 * end at one (no letter, digit, `_` or `-` immediately after it). So
 * `intelligence/personal/bj/x.md` matches while `intelligence/personalize`,
 * `intelligence/personal_notes` and `xintelligence/personal/` do not. The one
 * exception is `openclaw-workspace`, which deliberately has NO trailing
 * boundary: `[.]openclaw/workspace` matches `workspace/`, `workspace-utamaro/`
 * AND `workspaces-config`. Every sibling under `.openclaw/workspace*` is a
 * persona workspace or its config, so over-matching is the conservative choice.
 * ---------------------------------------------------------------------------
 */

import { basename } from "path";
import type { Vault } from "./vaults.ts";

export const CONTENT_CLASSIFIED_COLLECTIONS: readonly string[] = ["monchujo"] as const;

export interface BoundarySignal { readonly id: string; readonly pattern: string }

export const PRIVATE_BOUNDARY_SIGNALS: readonly BoundarySignal[] = [
  { id: "nsfw-ops", pattern: "(?<![A-Za-z0-9_.-])intelligence/operations/nsfw(?![A-Za-z0-9_-])" },
  { id: "personal", pattern: "(?<![A-Za-z0-9_.-])intelligence/personal(?![A-Za-z0-9_-])" },
  { id: "comfyui-nsfw", pattern: "(?<![A-Za-z0-9_.-])agents/skills/comfyui/nsfw(?![A-Za-z0-9_-])" },
  { id: "comfyui-runs-nsfw", pattern: "(?<![A-Za-z0-9_.-])agents/skills/comfyui/evals/imagegen/runs-nsfw(?![A-Za-z0-9_-])" },
  { id: "openclaw-workspace", pattern: "(?<![A-Za-z0-9_-])[.]openclaw/workspace" },
] as const;

export const AMBIGUOUS_BOUNDARY_SIGNALS: readonly BoundarySignal[] = [
  { id: "job-search", pattern: "(?<![A-Za-z0-9_.-])projects/job-search(?![A-Za-z0-9_-])" },
  { id: "private-financial", pattern: "(?<![A-Za-z0-9_.-])private/financial(?![A-Za-z0-9_-])" },
  { id: "private-household", pattern: "(?<![A-Za-z0-9_.-])private/household(?![A-Za-z0-9_-])" },
  { id: "fub", pattern: "(?<![A-Za-z0-9_.-])projects/fub(?![A-Za-z0-9_-])" },
  { id: "eval-hosho", pattern: "(?<![A-Za-z0-9_.-])intelligence/technical/eval-hosho(?![A-Za-z0-9_-])" },
  { id: "household-consumer-private", pattern: "(?<![A-Za-z0-9_.-])tools/household-consumer/private(?![A-Za-z0-9_-])" },
] as const;

interface CompiledSignal { readonly id: string; readonly re: RegExp }

/** Compiled once at module load; `classifyText` runs once per origin file. */
const COMPILED_PRIVATE: readonly CompiledSignal[] =
  PRIVATE_BOUNDARY_SIGNALS.map(s => ({ id: s.id, re: new RegExp(s.pattern) }));
const COMPILED_AMBIGUOUS: readonly CompiledSignal[] =
  AMBIGUOUS_BOUNDARY_SIGNALS.map(s => ({ id: s.id, re: new RegExp(s.pattern) }));

export interface TextClassification {
  /** PRIVATE signal ids that matched. Non-empty routes the session nsfw. */
  private: string[];
  /** AMBIGUOUS signal ids that matched. Reported only; never routes. */
  ambiguous: string[];
}

/**
 * Signal ids that match anywhere in `raw`, frontmatter included (a transcript's
 * `cwd` / `source_path` keys are as much a reference as its body).
 */
export function classifyText(raw: string): TextClassification {
  return {
    private: COMPILED_PRIVATE.filter(s => s.re.test(raw)).map(s => s.id),
    ambiguous: COMPILED_AMBIGUOUS.filter(s => s.re.test(raw)).map(s => s.id),
  };
}

/**
 * The session a chunk file belongs to: the frontmatter `session_id` when present
 * and non-empty, else the file's basename with a trailing `__c<digits>.md` (or a
 * bare `.md`) stripped.
 */
export function sessionKeyFor(relPath: string, frontmatterSessionId?: string | null): string {
  if (typeof frontmatterSessionId === "string" && frontmatterSessionId.trim() !== "") {
    return frontmatterSessionId.trim();
  }
  const base = basename(relPath);
  const chunked = base.replace(/__c\d+\.md$/, "");
  if (chunked !== base) return chunked;
  return base.replace(/\.md$/, "");
}

export interface SessionVerdict {
  /** "nsfw" iff at least one chunk of the session carries a PRIVATE signal. */
  vault: Vault;
  /** Union of PRIVATE signal ids across the session's chunks, sorted. */
  private: string[];
  /** Union of AMBIGUOUS signal ids, sorted. Only meaningful when vault is "sfw". */
  ambiguous: string[];
}

export interface ClassifiableFile {
  rel: string;
  raw: string;
  sessionId?: string | null;
}

/**
 * Streaming form of `classifySessions`: feed files one at a time, read the
 * verdicts at the end. The origin loader's pre-pass uses this so it never holds
 * a whole corpus of transcript text in memory just to decide where it goes.
 */
export class SessionClassifier {
  private readonly acc = new Map<string, { private: Set<string>; ambiguous: Set<string> }>();

  /** Add one chunk; returns its session key. */
  add(file: ClassifiableFile): string {
    const key = sessionKeyFor(file.rel, file.sessionId);
    let a = this.acc.get(key);
    if (!a) {
      a = { private: new Set(), ambiguous: new Set() };
      this.acc.set(key, a);
    }
    const c = classifyText(file.raw);
    for (const id of c.private) a.private.add(id);
    for (const id of c.ambiguous) a.ambiguous.add(id);
    return key;
  }

  verdicts(): Map<string, SessionVerdict> {
    const out = new Map<string, SessionVerdict>();
    for (const [key, a] of this.acc) {
      out.set(key, {
        vault: a.private.size > 0 ? "nsfw" : "sfw",
        private: [...a.private].sort(),
        ambiguous: [...a.ambiguous].sort(),
      });
    }
    return out;
  }
}

/** Per-session verdicts, keyed by `sessionKeyFor`. */
export function classifySessions(files: ClassifiableFile[]): Map<string, SessionVerdict> {
  const c = new SessionClassifier();
  for (const f of files) c.add(f);
  return c.verdicts();
}

export function isContentClassified(collection: string): boolean {
  return CONTENT_CLASSIFIED_COLLECTIONS.includes(collection);
}
