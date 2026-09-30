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
 */

import { CONTENT_TYPE_VALUES, isContentType, type ContentType } from "./memory.ts";

export interface FrontmatterMap {
  /** Source key for the title, used only when `title:` is absent. Dotted path allowed. */
  title?: string;
  /** Source key for content_type, used only when `content_type:` is absent. Dotted path allowed. */
  content_type?: string;
  /** Optional source-value -> ContentType translation applied to the mapped content_type. */
  content_type_values?: Record<string, ContentType>;
}

const MAP_KEYS = new Set(["title", "content_type", "content_type_values"]);

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
      `${where}: frontmatter_map must be a mapping with any of title / content_type / ` +
        `content_type_values; got ${JSON.stringify(raw)}. Remove the key to disable mapping.`,
    );
  }
  for (const k of Object.keys(raw)) {
    if (!MAP_KEYS.has(k)) {
      throw new Error(
        `${where}: frontmatter_map has unknown key ${JSON.stringify(k)} ` +
          `(allowed: title, content_type, content_type_values)`,
      );
    }
  }
  const out: FrontmatterMap = {};
  if (raw.title !== undefined) out.title = validateKeyPath(raw.title, "title", where);
  if (raw.content_type !== undefined) {
    out.content_type = validateKeyPath(raw.content_type, "content_type", where);
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
