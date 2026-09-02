/**
 * Vault routing unit tests (master-harness-0ynkd, ADR-0162 §1).
 *
 * These are the CHEAP tier of the guarantee: pure decisions, no substrate. They
 * cover the four things that, if any one of them silently flipped, would put
 * bj-corpus back in the general database:
 *
 *   1. a declared nsfw collection routes nsfw;
 *   2. an UNDECLARED collection defaults to sfw (routing is opt-in, not guessy);
 *   3. a private path routing sfw THROWS, rather than being logged/repaired;
 *   4. the tripwire fires EVEN WHEN THE CONFIG SAYS "sfw" — which is the whole
 *      reason PRIVATE_ROOTS is code and not config.
 *
 * Every case drives the REAL exported resolveVault against a REAL config file
 * (via CLAWMEM_CONFIG_DIR), not a stubbed collection lookup: a mocked lookup
 * would have proven the tripwire fires when handed a private collection, which
 * was never in doubt — what needed proving is that a config an operator can
 * actually write cannot defeat it.
 */

import { describe, it, expect, beforeEach, afterAll } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import {
  PRIVATE_COLLECTIONS,
  PRIVATE_PATH_PREFIXES,
  collectionIsPrivate,
  pathIsPrivate,
  resetVaultCache,
  resolveVault,
} from "../../src/pg/vaults.ts";
import { PrivateContentRoutingError } from "../../src/pg/errors.ts";

const REPO = "/home/bj/claude/master-harness";
const dirs: string[] = [];
const prevConfigDir = process.env.CLAWMEM_CONFIG_DIR;

/** Write a real index.yml and point clawmem's config loader at it. */
function useConfig(yaml: string): void {
  const dir = mkdtempSync(join(tmpdir(), "clawmem-vault-cfg-"));
  dirs.push(dir);
  writeFileSync(join(dir, "index.yml"), yaml, "utf-8");
  process.env.CLAWMEM_CONFIG_DIR = dir;
  resetVaultCache();
}

beforeEach(() => {
  resetVaultCache();
});

afterAll(() => {
  if (prevConfigDir === undefined) delete process.env.CLAWMEM_CONFIG_DIR;
  else process.env.CLAWMEM_CONFIG_DIR = prevConfigDir;
  resetVaultCache();
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

describe("PRIVATE_ROOTS is exactly the ADR-0162 §1 / clawmem-recall.toml surface", () => {
  it("names bj-corpus and nothing else", () => {
    // Widening this set is a POLICY decision (openclaw-workspace, monchujo and
    // friends were deliberately left out at 0ynkd). This assertion is what makes
    // a quiet widening show up as a failing test rather than as a diff nobody read.
    expect([...PRIVATE_COLLECTIONS]).toEqual(["bj-corpus"]);
    expect([...PRIVATE_PATH_PREFIXES]).toEqual(["intelligence/personal/bj/"]);
  });

  it("does NOT treat openclaw-workspace or monchujo as private", () => {
    expect(collectionIsPrivate("openclaw-workspace")).toBe(false);
    expect(collectionIsPrivate("monchujo")).toBe(false);
  });
});

describe("resolveVault — the data half", () => {
  it("routes a collection DECLARED vault: nsfw to the nsfw vault", () => {
    useConfig(`collections:\n  bj-corpus:\n    path: ${REPO}\n    pattern: "intelligence/personal/bj/approved/**/*.md"\n    vault: nsfw\n`);
    expect(resolveVault("bj-corpus", "intelligence/personal/bj/approved/x.md")).toBe("nsfw");
  });

  it("routes a NON-private collection declared vault: nsfw to nsfw too (the key is data, not the tripwire)", () => {
    useConfig(`collections:\n  someday-private:\n    path: ${REPO}\n    pattern: "**/*.md"\n    vault: nsfw\n`);
    expect(resolveVault("someday-private", "library/x.md")).toBe("nsfw");
  });

  it("defaults an UNDECLARED collection to sfw", () => {
    useConfig(`collections:\n  docs:\n    path: ${REPO}/library\n    pattern: "**/*.md"\n`);
    expect(resolveVault("docs", "reference/x.md")).toBe("sfw");
  });

  it("defaults a collection missing from the config entirely to sfw", () => {
    useConfig("collections: {}\n");
    expect(resolveVault("never-heard-of-it", "some/path.md")).toBe("sfw");
  });

  it("REFUSES an unparseable vault declaration rather than guessing", () => {
    useConfig(`collections:\n  weird:\n    path: ${REPO}\n    pattern: "**/*.md"\n    vault: private\n`);
    expect(() => resolveVault("weird", "a.md")).toThrow(/only valid values are "sfw" and "nsfw"/);
  });
});

describe("resolveVault — the hard-coded tripwire (the guarantee half)", () => {
  it("THROWS when a private COLLECTION is declared sfw", () => {
    // This is the case a config-driven tripwire cannot catch: the config is the
    // thing that is wrong.
    useConfig(`collections:\n  bj-corpus:\n    path: ${REPO}\n    pattern: "intelligence/personal/bj/approved/**/*.md"\n    vault: sfw\n`);
    expect(() => resolveVault("bj-corpus", "intelligence/personal/bj/approved/x.md"))
      .toThrow(PrivateContentRoutingError);
  });

  it("THROWS when a private collection is left UNDECLARED (defaulting to sfw)", () => {
    useConfig(`collections:\n  bj-corpus:\n    path: ${REPO}\n    pattern: "intelligence/personal/bj/approved/**/*.md"\n`);
    expect(() => resolveVault("bj-corpus", "intelligence/personal/bj/approved/x.md"))
      .toThrow(PrivateContentRoutingError);
  });

  it("THROWS on a private PATH even under an innocuous collection name", () => {
    // The collection name is clean; only the path is private. A rename or a
    // widened glob is exactly how this happens for real.
    useConfig(`collections:\n  synthesis:\n    path: ${REPO}\n    pattern: "**/*.md"\n`);
    expect(() => resolveVault("synthesis", "intelligence/personal/bj/approved/x.md"))
      .toThrow(PrivateContentRoutingError);
  });

  it("THROWS on the ABSOLUTE resolved path when only the collection root carries the prefix", () => {
    useConfig(`collections:\n  bjnotes:\n    path: ${REPO}/intelligence/personal/bj\n    pattern: "**/*.md"\n`);
    expect(() => resolveVault("bjnotes", "approved/x.md")).toThrow(PrivateContentRoutingError);
  });

  it("names the collection, the path and the fix in the refusal", () => {
    useConfig(`collections:\n  bj-corpus:\n    path: ${REPO}\n    pattern: "**/*.md"\n    vault: sfw\n`);
    let err: unknown;
    try {
      resolveVault("bj-corpus", "intelligence/personal/bj/approved/x.md");
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(PrivateContentRoutingError);
    const msg = (err as Error).message;
    expect(msg).toContain("bj-corpus");
    expect(msg).toContain("vault: nsfw");
    expect(msg).toContain("Nothing was written");
  });

  it("does NOT fire on a near-miss path (segment-wise, not substring)", () => {
    useConfig(`collections:\n  synthesis:\n    path: ${REPO}\n    pattern: "**/*.md"\n`);
    // A guard that fires on everything proves nothing; prove it can also stay quiet.
    expect(resolveVault("synthesis", "intelligence/personal/bjorn/x.md")).toBe("sfw");
    expect(resolveVault("synthesis", "intelligence/personal/bjarne/notes.md")).toBe("sfw");
  });

  it("does NOT fire when the private content is correctly routed nsfw", () => {
    useConfig(`collections:\n  bj-corpus:\n    path: ${REPO}\n    pattern: "**/*.md"\n    vault: nsfw\n`);
    expect(resolveVault("bj-corpus", "intelligence/personal/bj/approved/x.md")).toBe("nsfw");
  });
});

describe("pathIsPrivate", () => {
  it.each([
    ["intelligence/personal/bj/approved/a.md", true],
    ["intelligence/personal/bj/", true],
    ["/home/bj/claude/master-harness/intelligence/personal/bj/approved/a.md", true],
    ["./intelligence/personal/bj/approved/a.md", true],
    ["intelligence/personal/bjorn/a.md", false],
    ["intelligence/personal/bjx/a.md", false],
    ["library/intelligence/personal/bj-notes/a.md", false],
    ["", false],
  ])("%s -> %s", (p, expected) => {
    expect(pathIsPrivate(p)).toBe(expected);
  });

  it("matches a private prefix nested under any parent directory", () => {
    expect(pathIsPrivate("/some/checkout/intelligence/personal/bj/approved/a.md")).toBe(true);
  });
});
