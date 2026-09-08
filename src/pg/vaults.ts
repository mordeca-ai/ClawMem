/**
 * Vault routing for the clawmem PG write path (master-harness-0ynkd, ADR-0162 §1).
 *
 * ---------------------------------------------------------------------------
 * WHY THIS MODULE EXISTS
 *
 * ADR-0162 §1 puts the bj-corpus/NSFW boundary in the DATABASE and the LOGIN
 * ROLE — "never an application `WHERE` clause". The substrate half of that is
 * real and proven (role `clawmem` has no CONNECT on database `clawmem_nsfw`,
 * and vice versa). What was missing was the half that decides WHICH of the two
 * connections a given write goes down: src/pg/* had exactly one global pool
 * built from one CLAWMEM_PG_URL, so every write — private or not — landed in
 * the `clawmem` database by construction. 2006 bj-corpus rows in `clawmem`
 * are the measured consequence.
 *
 * ---------------------------------------------------------------------------
 * THE TWO LAYERS, AND WHY THEY ARE NOT THE SAME LAYER
 *
 *  1. **Routing is DATA.** A collection declares `vault: nsfw` in
 *     ~/.config/clawmem/index.yml. Undeclared means "sfw". That is the normal,
 *     extensible path: a new private collection is a config edit.
 *
 *  2. **The guarantee is CODE.** `PRIVATE_ROOTS` below is a hard-coded list of
 *     collection names and path prefixes that MUST land in the nsfw vault. If
 *     routing ever produces "sfw" for something under a private root, this
 *     module THROWS instead of returning.
 *
 * Layer 2 is not redundant with layer 1: a config-driven tripwire is defeated
 * by editing the config, which is exactly the failure it would need to catch
 * (a collection silently declared — or left undeclared — as sfw). Putting the
 * private set in code means a mis-declaration is impossible rather than merely
 * detectable, and changing the private set is a reviewed code change.
 *
 * PRIVATE_ROOTS is seeded with EXACTLY what ADR-0162 §1 and
 * infra/clawmem/clawmem-recall.toml's `excluded_collections` declare today —
 * the bj-corpus surface, nothing more. Widening the private set (openclaw
 * workspaces, monchujo, …) is a policy decision that belongs to whoever makes
 * it, not to this module.
 * ---------------------------------------------------------------------------
 */

import { isAbsolute, join, normalize } from "path";
import { getCollection } from "../collections.ts";
import { PrivateContentRoutingError } from "./errors.ts";

export type Vault = "sfw" | "nsfw";

export const VAULTS: readonly Vault[] = ["sfw", "nsfw"] as const;

export function isVault(v: unknown): v is Vault {
  return v === "sfw" || v === "nsfw";
}

/**
 * Collection names that MUST route to the nsfw vault. Hard-coded on purpose —
 * see the header. Set-equal to `excluded_collections` in
 * infra/clawmem/clawmem-recall.toml as of master-harness-0ynkd.
 */
export const PRIVATE_COLLECTIONS: readonly string[] = ["bj-corpus"] as const;

/**
 * Source-path prefixes that MUST route to the nsfw vault. Matched
 * SEGMENT-WISE (see `pathIsPrivate`) against both the document's
 * collection-relative path and its resolved absolute path, so a private root
 * is caught whether it is expressed repo-relative (as `documents.path` is) or
 * absolute (as the collection root joined with it is).
 */
export const PRIVATE_PATH_PREFIXES: readonly string[] = [
  "intelligence/personal/bj/",
] as const;

/** Everything the tripwire treats as private, in one place for reporting. */
export const PRIVATE_ROOTS = {
  collections: PRIVATE_COLLECTIONS,
  pathPrefixes: PRIVATE_PATH_PREFIXES,
} as const;

// ---------------------------------------------------------------------------
// Collection lookup (memoized — resolveVault runs once per document on a
// reindex, and getCollection re-reads + re-parses the YAML config every call).
// ---------------------------------------------------------------------------

const collectionMemo = new Map<string, { vault?: string; path?: string } | null>();

/** Drop the memo. Tests that rewrite the config between assertions need this. */
export function resetVaultCache(): void {
  collectionMemo.clear();
}

function lookupCollection(name: string): { vault?: string; path?: string } | null {
  if (collectionMemo.has(name)) return collectionMemo.get(name)!;
  let found: { vault?: string; path?: string } | null = null;
  try {
    const c = getCollection(name);
    if (c) found = { vault: c.vault, path: c.path };
  } catch {
    // A missing/unreadable config must not be able to turn a private write
    // into an sfw write: fall through to the tripwire below, which does not
    // consult the config at all.
    found = null;
  }
  collectionMemo.set(name, found);
  return found;
}

// ---------------------------------------------------------------------------
// The tripwire
// ---------------------------------------------------------------------------

/**
 * Does `candidate` fall under a private path prefix?
 *
 * Compared on NORMALIZED, slash-terminated segment boundaries so that
 * `intelligence/personal/bjorn/x.md` does NOT match the
 * `intelligence/personal/bj/` prefix, while both
 * `intelligence/personal/bj/approved/x.md` and
 * `/home/bj/claude/master-harness/intelligence/personal/bj/approved/x.md` do.
 */
export function pathIsPrivate(candidate: string | null | undefined): boolean {
  if (!candidate) return false;
  const norm = `${normalize(candidate).replace(/\\/g, "/").replace(/\/+$/, "")}/`;
  for (const prefix of PRIVATE_PATH_PREFIXES) {
    const p = `${prefix.replace(/\/+$/, "")}/`;
    if (isAbsolute(p)) {
      if (norm.startsWith(p)) return true;
    } else if (norm.startsWith(p) || norm.includes(`/${p}`)) {
      return true;
    }
  }
  return false;
}

export function collectionIsPrivate(collection: string): boolean {
  return PRIVATE_COLLECTIONS.includes(collection);
}

/**
 * Resolve the absolute source path for a document, when the collection root is
 * knowable. Returns null when it is not — the relative path is still checked.
 */
function resolvedSourcePath(collection: string, relPath?: string): string | null {
  if (!relPath) return null;
  if (isAbsolute(relPath)) return relPath;
  const root = lookupCollection(collection)?.path;
  return root ? join(root, relPath) : null;
}

/**
 * THE routing decision. Data decides; code guarantees.
 *
 * @throws PrivateContentRoutingError when the resolved vault is "sfw" but the
 *   collection name or source path falls under a hard-coded private root. This
 *   is a programming/configuration defect, never a runtime condition to handle:
 *   the correct response is to declare the collection `vault: nsfw`, not to
 *   catch this.
 */
export function resolveVault(collection: string, relPath?: string): Vault {
  const declared = lookupCollection(collection)?.vault;
  let vault: Vault = "sfw";
  if (declared !== undefined) {
    if (!isVault(declared)) {
      throw new Error(
        `Collection ${JSON.stringify(collection)} declares vault ` +
        `${JSON.stringify(declared)} in the clawmem index config, but the only ` +
        `valid values are "sfw" and "nsfw". Refusing to guess which vault a ` +
        `write belongs in.`,
      );
    }
    vault = declared;
  }

  if (vault === "sfw") {
    const abs = resolvedSourcePath(collection, relPath);
    const byCollection = collectionIsPrivate(collection);
    const byPath = pathIsPrivate(relPath) || pathIsPrivate(abs);
    if (byCollection || byPath) {
      throw new PrivateContentRoutingError(
        collection,
        relPath ?? null,
        abs,
        byCollection ? "collection" : "path",
        declared === undefined ? "undeclared (defaulted to sfw)" : `declared "${declared}"`,
      );
    }
  }

  return vault;
}
