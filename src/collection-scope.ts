/**
 * Per-collection scope decision for the retrieval commands (search / vsearch /
 * query). Pure and IO-free — config, env and process.exit live in the CLI
 * wrapper (resolveCollectionScope in clawmem.ts) so this precedence logic is
 * unit-testable in isolation.
 */

export interface CollectionScope {
  collections: string[];
  source: "flag" | "focus";
}

/**
 * Decide the effective collection scope. Precedence:
 *   1. explicit -c/--collection flag (comma-separated; every name validated
 *      against `knownNames` — an unknown name returns { error } so a typo
 *      can't masquerade as an empty result set).
 *   2. session focus, but ONLY when the focus topic exactly names a real
 *      collection (case-insensitive). This is the focus -> auto-scope path:
 *      `clawmem focus set stormlight` fences unscoped queries to stormlight.
 *   3. null -> unscoped, byte-identical to the pre-scoping baseline. A
 *      free-text focus that is NOT a collection name falls here, so existing
 *      topic-boost focuses keep behaving exactly as before.
 *
 * Matching is case-insensitive against `knownNames`; the returned names are
 * the canonical (configured) spellings.
 */
export function computeCollectionScope(
  flag: string | undefined,
  focusTopic: string | undefined,
  knownNames: string[],
): CollectionScope | { error: string } | null {
  const byLower = new Map(knownNames.map(n => [n.toLowerCase(), n]));

  if (flag && flag.trim()) {
    const requested = flag.split(",").map(s => s.trim()).filter(Boolean);
    const resolved: string[] = [];
    for (const name of requested) {
      const canonical = byLower.get(name.toLowerCase());
      if (!canonical) {
        return { error: `Unknown collection '${name}'. Available: ${knownNames.join(", ")}` };
      }
      if (!resolved.includes(canonical)) resolved.push(canonical);
    }
    return resolved.length > 0 ? { collections: resolved, source: "flag" } : null;
  }

  if (focusTopic && focusTopic.trim()) {
    const canonical = byLower.get(focusTopic.trim().toLowerCase());
    if (canonical) return { collections: [canonical], source: "focus" };
  }

  return null;
}
