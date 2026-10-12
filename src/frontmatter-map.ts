/**
 * Per-collection frontmatter key map (master-harness-wzwh8.1, wzwh8 phase 2).
 *
 * parseDocument reads the canonical keys `title:` and `content_type:`. Large
 * parts of a real corpus are authored against EXTERNAL specs that use other
 * keys: Claude Code auto-memory writes `name:` + `metadata.type`, the Agent
 * Skills spec writes `name:`. Those files cannot be rewritten, so a collection
 * may DECLARE where its title / content_type live:
 *
 *   collections:
 *     memory-topics:
 *       path: ...
 *       frontmatter_map:
 *         title: name
 *         content_type: metadata.type
 *         content_type_values: { feedback: preference, project: project }
 *
 * Rules (each one is asserted in tests/unit/frontmatter-map.test.ts):
 *   - OPT-IN and per collection. Absent = today's behaviour byte-for-byte.
 *     There is no global alias and no cross-collection default.
 *   - The canonical key ALWAYS wins when it yields a value; the mapped key is
 *     read only when the canonical one is absent.
 *   - A mapped content_type must end up a valid ContentType (after the optional
 *     value map). If it does not, it is NOT stored: the document falls back to
 *     today's chain (collection default -> filename inference) and the
 *     rejection is counted.
 *   - A malformed map fails LOUD at config load (validateFrontmatterMap).
 *
 * SUPERSESSION KEYS (master-harness-vn4rz.84). Two more OPTIONAL source keys
 * let a document declare that it is superseded, so explicit supersession can
 * replace age-based decay:
 *
 *       frontmatter_map:
 *         invalidated_at: invalidated_at   # ISO date or datetime
 *         superseded_by: superseded_by     # collection-relative path, or a title
 *
 *   - `invalidated_at`: the value marks the document invalidated as of that
 *     instant. A value that is not an ISO date/datetime is NOT applied; it is
 *     recorded on the outcome (`invalidatedAtRejected`) and counted, exactly
 *     like a rejected content_type. It never throws at index time.
 *   - `superseded_by`: its PRESENCE alone invalidates the document (as of index
 *     time when no valid invalidated_at value is given). The value names the
 *     replacement: first as a path relative to the collection root, then as a
 *     title that exactly one active document in the same collection carries.
 *     Unresolvable = the superseded_by column stays NULL; the tombstone still
 *     applies.
 *   - An invalidated document is hidden by every search path, which already
 *     filters `invalidated_at IS NULL`. A future-dated invalidated_at hides the
 *     document immediately: the date is recorded provenance, not a schedule.
 *
 * DECISION: the canonical (unmapped) keys `invalidated_at:` / `superseded_by:`
 * are NOT honoured on their own, for any collection. Supersession hides
 * documents from search, so it must be declared per collection, the same
 * opt-in rule as above. Honouring the bare keys everywhere would change what
 * an unmapped collection returns: a real corpus already holds a
 * `superseded_by:` written as a FILE-relative link, which a global reading
 * would silently tombstone. A collection that authors the canonical names
 * opts in with an identity map (`invalidated_at: invalidated_at`). Unlike
 * title / content_type there is therefore no "canonical wins" rule here: the
 * mapped key is the only key read.
 *
 * Provenance: an invalidation set from frontmatter is written with
 * `invalidation_source = 'frontmatter'`. Re-indexing a document without a
 * tombstone clears ONLY such rows; an invalidation written by anything else
 * (the decision extractor, or a legacy row with no recorded source) survives
 * every re-index.
 */

import { CONTENT_TYPE_VALUES, isContentType, type ContentType } from "./memory.ts";
import { normalizeIsoTimestamp } from "./normalize.ts";

export interface FrontmatterMap {
  /** Source key for the title, used only when `title:` is absent. Dotted path allowed. */
  title?: string;
  /** Source key for content_type, used only when `content_type:` is absent. Dotted path allowed. */
  content_type?: string;
  /** Optional source-value -> ContentType translation applied to the mapped content_type. */
  content_type_values?: Record<string, ContentType>;
  /** Source key for the invalidation instant (ISO date/datetime). Dotted path allowed. */
  invalidated_at?: string;
  /** Source key naming the replacement document; its presence alone invalidates. Dotted path allowed. */
  superseded_by?: string;
}

const MAP_KEYS = new Set([
  "title",
  "content_type",
  "content_type_values",
  "invalidated_at",
  "superseded_by",
]);
const MAP_KEYS_LIST = "title, content_type, content_type_values, invalidated_at, superseded_by";

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v) && !(v instanceof Date);
}

function validateKeyPath(v: unknown, field: string, where: string): string {
  if (typeof v !== "string" || v.trim() === "") {
    throw new Error(
      `${where}: frontmatter_map.${field} must be a non-empty string naming a frontmatter key ` +
        `(dotted path allowed, e.g. "name" or "metadata.type"); got ${JSON.stringify(v)}`,
    );
  }
  if (v.split(".").some((seg) => seg === "")) {
    throw new Error(
      `${where}: frontmatter_map.${field} ${JSON.stringify(v)} has an empty path segment ` +
        `(leading, trailing or doubled '.')`,
    );
  }
  return v;
}

/**
 * Validate a raw `frontmatter_map` value from the config file. Throws an
 * actionable Error naming the collection and the offending field; returns a
 * normalized copy otherwise. `where` is e.g. `collection "memory-topics"`.
 */
export function validateFrontmatterMap(raw: unknown, where: string): FrontmatterMap {
  if (!isPlainObject(raw)) {
    throw new Error(
      `${where}: frontmatter_map must be a mapping with any of ${MAP_KEYS_LIST}; ` +
        `got ${JSON.stringify(raw)}. Remove the key to disable mapping.`,
    );
  }
  for (const k of Object.keys(raw)) {
    if (!MAP_KEYS.has(k)) {
      throw new Error(
        `${where}: frontmatter_map has unknown key ${JSON.stringify(k)} ` +
          `(allowed: ${MAP_KEYS_LIST})`,
      );
    }
  }
  const out: FrontmatterMap = {};
  if (raw.title !== undefined) out.title = validateKeyPath(raw.title, "title", where);
  if (raw.content_type !== undefined) {
    out.content_type = validateKeyPath(raw.content_type, "content_type", where);
  }
  if (raw.invalidated_at !== undefined) {
    out.invalidated_at = validateKeyPath(raw.invalidated_at, "invalidated_at", where);
  }
  if (raw.superseded_by !== undefined) {
    out.superseded_by = validateKeyPath(raw.superseded_by, "superseded_by", where);
  }
  if (raw.content_type_values !== undefined) {
    if (out.content_type === undefined) {
      throw new Error(
        `${where}: frontmatter_map.content_type_values has no effect without ` +
          `frontmatter_map.content_type (the source key it translates)`,
      );
    }
    const vals = raw.content_type_values;
    if (!isPlainObject(vals)) {
      throw new Error(
        `${where}: frontmatter_map.content_type_values must be a mapping of ` +
          `source value -> content_type; got ${JSON.stringify(vals)}`,
      );
    }
    const table: Record<string, ContentType> = {};
    for (const [from, to] of Object.entries(vals)) {
      if (!isContentType(to)) {
        throw new Error(
          `${where}: frontmatter_map.content_type_values.${from} = ${JSON.stringify(to)} ` +
            `is not a valid content_type (expected one of: ${CONTENT_TYPE_VALUES.join(", ")})`,
        );
      }
      table[from] = to;
    }
    out.content_type_values = table;
  }
  return out;
}

/** Read a dotted key path out of parsed frontmatter; undefined when any hop is missing. */
export function readKeyPath(data: unknown, path: string): unknown {
  let cur: unknown = data;
  for (const seg of path.split(".")) {
    if (!isPlainObject(cur)) return undefined;
    cur = cur[seg];
  }
  return cur;
}

/** What the map did for one document. Only produced when a map is in effect. */
export interface FrontmatterMapOutcome {
  /** The title came from the mapped key (canonical `title:` was absent). */
  titleMapped: boolean;
  /** content_type came from the mapped key (canonical `content_type:` was absent). */
  contentTypeMapped: boolean;
  /**
   * The mapped key yielded a string that is not a valid ContentType even after
   * content_type_values; it was NOT stored. Holds the raw source value.
   */
  contentTypeRejected?: string;
  /**
   * master-harness-vn4rz.84: the mapped invalidated_at key held a value that is
   * not an ISO date/datetime; it was NOT applied. Holds the raw value as text.
   */
  invalidatedAtRejected?: string;
}

const str = (v: unknown): string | undefined =>
  typeof v === "string" ? v || undefined : undefined;

/**
 * Fill title / content_type from the map where the canonical key is absent.
 * `canonicalTitle` / `canonicalContentType` are what parseDocument already read
 * from `title:` / `content_type:`; they always win.
 */
export function applyFrontmatterMap(
  data: unknown,
  map: FrontmatterMap,
  canonicalTitle: string | undefined,
  canonicalContentType: string | undefined,
): { title: string | undefined; contentType: string | undefined; outcome: FrontmatterMapOutcome } {
  const outcome: FrontmatterMapOutcome = { titleMapped: false, contentTypeMapped: false };

  let title = canonicalTitle;
  if (title === undefined && map.title !== undefined) {
    const v = str(readKeyPath(data, map.title));
    if (v !== undefined) {
      title = v;
      outcome.titleMapped = true;
    }
  }

  let contentType = canonicalContentType;
  if (contentType === undefined && map.content_type !== undefined) {
    const rawValue = str(readKeyPath(data, map.content_type));
    if (rawValue !== undefined) {
      const table = map.content_type_values;
      const translated = table && Object.hasOwn(table, rawValue) ? table[rawValue] : rawValue;
      if (isContentType(translated)) {
        contentType = translated;
        outcome.contentTypeMapped = true;
      } else {
        outcome.contentTypeRejected = rawValue;
      }
    }
  }

  return { title, contentType, outcome };
}

// =============================================================================
// Supersession (master-harness-vn4rz.84)
// =============================================================================

/** Strict date-only form, normalized to UTC midnight. */
const DATE_ONLY_RE = /^(\d{4})-(\d{2})-(\d{2})$/;

/**
 * One frontmatter instant -> UTC ISO, or null when invalid/absent. gray-matter
 * coerces an unquoted `2025-03-01` to a Date, so a Date is validated as a
 * finite Date and a string is parsed as a date-only or RFC 3339 value. The
 * single definition behind both `authored_at` (indexer.ts
 * authoredAtFromFrontmatter) and the mapped `invalidated_at`.
 */
export function frontmatterInstant(value: unknown): string | null {
  if (value instanceof Date) {
    return isNaN(value.getTime()) ? null : value.toISOString();
  }
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (DATE_ONLY_RE.test(trimmed)) {
    return normalizeIsoTimestamp(`${trimmed}T00:00:00Z`);
  }
  return normalizeIsoTimestamp(trimmed);
}

/** A document's own declaration that it is superseded. */
export interface FrontmatterInvalidation {
  /** UTC ISO instant from the mapped invalidated_at key; null = index time. */
  at: string | null;
  /** The replacement reference as written (path or title); null = none named. */
  supersededBy: string | null;
}

/** YAML `key:` with no value parses as null; null, "" and false all mean "not declared". */
function declared(v: unknown): boolean {
  return v !== undefined && v !== null && v !== "" && v !== false;
}

/**
 * Read the supersession keys a map declares. Pure: returns the tombstone (or
 * null when the document declares none) plus a rejected raw invalidated_at
 * value. Without `invalidated_at` / `superseded_by` in the map this always
 * returns `{ invalidation: null }`.
 */
export function readFrontmatterInvalidation(
  data: unknown,
  map: FrontmatterMap | undefined,
): { invalidation: FrontmatterInvalidation | null; invalidatedAtRejected?: string } {
  if (!map || (map.invalidated_at === undefined && map.superseded_by === undefined)) {
    return { invalidation: null };
  }
  let at: string | null = null;
  let atDeclared = false;
  let invalidatedAtRejected: string | undefined;
  if (map.invalidated_at !== undefined) {
    const raw = readKeyPath(data, map.invalidated_at);
    if (declared(raw)) {
      at = frontmatterInstant(raw);
      if (at === null) invalidatedAtRejected = typeof raw === "string" ? raw : JSON.stringify(raw);
      else atDeclared = true;
    }
  }
  let supersededBy: string | null = null;
  let refDeclared = false;
  if (map.superseded_by !== undefined) {
    const raw = readKeyPath(data, map.superseded_by);
    if (declared(raw)) {
      refDeclared = true;
      if (typeof raw === "string" && raw.trim() !== "") supersededBy = raw.trim();
      else if (typeof raw === "number" && Number.isFinite(raw)) supersededBy = String(raw);
    }
  }
  const invalidation = atDeclared || refDeclared ? { at, supersededBy } : null;
  return invalidatedAtRejected === undefined
    ? { invalidation }
    : { invalidation, invalidatedAtRejected };
}

/**
 * Normalize a superseded_by reference for a path lookup: strip a leading
 * `./`. Resolution order (both backends): exact collection-relative path, then
 * a title that exactly ONE other active document in the collection carries.
 */
export function supersededByPathCandidate(ref: string): string {
  return ref.replace(/^(\.\/)+/, "");
}
