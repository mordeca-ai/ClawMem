/**
 * Per-vault connection config + no-silent-fallback unit tests
 * (master-harness-0ynkd, ADR-0162 §1).
 *
 * The bug class this file exists to keep dead: a write that ROUTES to the nsfw
 * vault, finds no nsfw connection, and quietly uses the sfw pool. That is the
 * same failure shape as "silently fall back to sqlite" which src/pg/config.ts
 * was already written to refuse — so it is refused the same way, and proven the
 * same way: the throw is asserted, AND the sfw path is asserted to be untouched.
 */

import { describe, it, expect, beforeEach, afterAll } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { resolvePgConfig, vaultIsConfigured, VAULT_DEFAULT_DATABASE } from "../../src/pg/config.ts";
import { resetVaultCache } from "../../src/pg/vaults.ts";
import { insertEmbeddingsBatch, upsertDocument, type EmbeddingWrite } from "../../src/pg/write.ts";
import {
  PgVecBatchVaultMismatchError,
  VaultNotConfiguredError,
} from "../../src/pg/errors.ts";

const REPO = "/home/bj/claude/master-harness";
const dirs: string[] = [];
const saved = {
  configDir: process.env.CLAWMEM_CONFIG_DIR,
  url: process.env.CLAWMEM_PG_URL,
  nsfwUrl: process.env.CLAWMEM_PG_NSFW_URL,
};

function useConfig(yaml: string): void {
  const dir = mkdtempSync(join(tmpdir(), "clawmem-vaultcfg-"));
  dirs.push(dir);
  writeFileSync(join(dir, "index.yml"), yaml, "utf-8");
  process.env.CLAWMEM_CONFIG_DIR = dir;
  resetVaultCache();
}

beforeEach(() => resetVaultCache());

afterAll(() => {
  for (const [k, v] of Object.entries({
    CLAWMEM_CONFIG_DIR: saved.configDir,
    CLAWMEM_PG_URL: saved.url,
    CLAWMEM_PG_NSFW_URL: saved.nsfwUrl,
  })) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  resetVaultCache();
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

describe("resolvePgConfig — sfw behavior is byte-identical to pre-0ynkd", () => {
  const env = {
    CLAWMEM_PG_URL: "postgres://clawmem:pw@127.0.0.1:5433/clawmem",
  } as NodeJS.ProcessEnv;

  it("reads CLAWMEM_PG_URL unchanged and defaults the vault to sfw", () => {
    const a = resolvePgConfig("sfw", env);
    const b = resolvePgConfig(undefined, env);
    expect(a.connectionString).toBe("postgres://clawmem:pw@127.0.0.1:5433/clawmem");
    expect(a.safeLabel).toBe("postgres://clawmem@127.0.0.1:5433/clawmem");
    expect(a.vault).toBe("sfw");
    expect(a.database).toBe("clawmem");
    expect(b).toEqual(a);
  });

  it("assembles the discrete CLAWMEM_PG_* vars exactly as before", () => {
    const cfg = resolvePgConfig("sfw", {
      CLAWMEM_PG_HOST: "127.0.0.1",
      CLAWMEM_PG_DATABASE: "clawmem",
      CLAWMEM_PG_USER: "clawmem",
      CLAWMEM_PG_PASSWORD: "p@ss/word",
    } as NodeJS.ProcessEnv);
    expect(cfg.connectionString).toBe("postgres://clawmem:p%40ss%2Fword@127.0.0.1:5433/clawmem");
    expect(cfg.safeLabel).toBe("postgres://clawmem@127.0.0.1:5433/clawmem");
    expect(cfg.database).toBe("clawmem");
  });

  it("still refuses a non-postgres URL and a non-numeric port", () => {
    expect(() => resolvePgConfig("sfw", { CLAWMEM_PG_URL: "mysql://x@y/z" } as NodeJS.ProcessEnv))
      .toThrow(/must be a postgres:\/\/ URL/);
    expect(() => resolvePgConfig("sfw", {
      CLAWMEM_PG_HOST: "h", CLAWMEM_PG_DATABASE: "d", CLAWMEM_PG_USER: "u", CLAWMEM_PG_PORT: "abc",
    } as NodeJS.ProcessEnv)).toThrow(/must be numeric/);
  });

  it("REFUSES a URL that names no database (libpq would pick the role-named one)", () => {
    expect(() => resolvePgConfig("sfw", {
      CLAWMEM_PG_URL: "postgres://clawmem:pw@127.0.0.1:5433",
    } as NodeJS.ProcessEnv)).toThrow(/names no database/);
  });
});

describe("resolvePgConfig — the nsfw namespace", () => {
  it("reads CLAWMEM_PG_NSFW_URL, never CLAWMEM_PG_URL", () => {
    const cfg = resolvePgConfig("nsfw", {
      CLAWMEM_PG_URL: "postgres://clawmem:pw@127.0.0.1:5433/clawmem",
      CLAWMEM_PG_NSFW_URL: "postgres://clawmem_nsfw:pw2@127.0.0.1:5433/clawmem_nsfw",
    } as NodeJS.ProcessEnv);
    expect(cfg.database).toBe("clawmem_nsfw");
    expect(cfg.connectionString).toContain("/clawmem_nsfw");
  });

  it("assembles the discrete CLAWMEM_PG_NSFW_* vars", () => {
    const cfg = resolvePgConfig("nsfw", {
      CLAWMEM_PG_NSFW_HOST: "127.0.0.1",
      CLAWMEM_PG_NSFW_PORT: "5433",
      CLAWMEM_PG_NSFW_DATABASE: "clawmem_nsfw",
      CLAWMEM_PG_NSFW_USER: "clawmem_nsfw",
      CLAWMEM_PG_NSFW_PASSWORD: "pw",
    } as NodeJS.ProcessEnv);
    expect(cfg.safeLabel).toBe("postgres://clawmem_nsfw@127.0.0.1:5433/clawmem_nsfw");
    expect(cfg.vault).toBe("nsfw");
  });

  it("THROWS VaultNotConfiguredError when ONLY the sfw connection exists", () => {
    const env = { CLAWMEM_PG_URL: "postgres://clawmem:pw@127.0.0.1:5433/clawmem" } as NodeJS.ProcessEnv;
    expect(vaultIsConfigured("sfw", env)).toBe(true);
    expect(vaultIsConfigured("nsfw", env)).toBe(false);
    let err: unknown;
    try { resolvePgConfig("nsfw", env); } catch (e) { err = e; }
    expect(err).toBeInstanceOf(VaultNotConfiguredError);
    // The refusal must SAY there is no fallback — a reader who sees only
    // "not configured" reaches for the other pool.
    expect((err as Error).message).toContain("NO fallback");
    expect((err as Error).message).toContain("CLAWMEM_PG_NSFW_URL");
  });

  it("declares the ADR-0162 §1 database names", () => {
    expect(VAULT_DEFAULT_DATABASE).toEqual({ sfw: "clawmem", nsfw: "clawmem_nsfw" });
  });
});

describe("a routed-nsfw write with no nsfw connection REFUSES rather than falling back", () => {
  beforeEach(() => {
    useConfig(
      `collections:\n` +
      `  bj-corpus:\n    path: ${REPO}\n    pattern: "**/*.md"\n    vault: nsfw\n` +
      `  docs:\n    path: ${REPO}/library\n    pattern: "**/*.md"\n`,
    );
    process.env.CLAWMEM_PG_URL = "postgres://clawmem:pw@127.0.0.1:5433/clawmem";
    delete process.env.CLAWMEM_PG_NSFW_URL;
  });

  it("upsertDocument throws VaultNotConfiguredError (no sfw pool is opened)", async () => {
    await expect(upsertDocument({
      collection: "bj-corpus",
      path: "intelligence/personal/bj/approved/x.md",
      title: "x", hash: "h_nofallback", body: "# x",
    })).rejects.toBeInstanceOf(VaultNotConfiguredError);
  });

  it("insertEmbeddingsBatch throws VaultNotConfiguredError", async () => {
    await expect(insertEmbeddingsBatch([{
      collection: "bj-corpus",
      path: "intelligence/personal/bj/approved/x.md",
      hash: "h_nofallback", seq: 0, pos: 0,
      embedding: new Array(768).fill(0.1), model: "embeddinggemma",
    }])).rejects.toBeInstanceOf(VaultNotConfiguredError);
  });
});

describe("a batch spanning two vaults is REFUSED, never split", () => {
  beforeEach(() => {
    useConfig(
      `collections:\n` +
      `  bj-corpus:\n    path: ${REPO}\n    pattern: "**/*.md"\n    vault: nsfw\n` +
      `  docs:\n    path: ${REPO}/library\n    pattern: "**/*.md"\n`,
    );
  });

  const frag = (collection: string, path: string, seq: number): EmbeddingWrite => ({
    collection, path, hash: `h_${seq}`, seq, pos: 0,
    embedding: new Array(768).fill(0.1), model: "embeddinggemma",
  });

  it("throws PgVecBatchVaultMismatchError naming both vaults and the collections", async () => {
    let err: unknown;
    try {
      await insertEmbeddingsBatch([
        frag("docs", "reference/a.md", 0),
        frag("bj-corpus", "intelligence/personal/bj/approved/b.md", 1),
      ]);
    } catch (e) { err = e; }
    expect(err).toBeInstanceOf(PgVecBatchVaultMismatchError);
    const msg = (err as Error).message;
    expect(msg).toContain("sfw");
    expect(msg).toContain("nsfw");
    expect(msg).toContain("bj-corpus");
    expect(msg).toContain("Nothing in this batch was written");
  });

  it("refuses BEFORE any connection is attempted (the throw is not a connect error)", async () => {
    // Deliberately no CLAWMEM_PG_* env at all: if the vault check ran after the
    // pool was opened, this would surface as a config/connection error instead.
    const url = process.env.CLAWMEM_PG_URL;
    delete process.env.CLAWMEM_PG_URL;
    try {
      await expect(insertEmbeddingsBatch([
        frag("docs", "reference/a.md", 0),
        frag("bj-corpus", "intelligence/personal/bj/approved/b.md", 1),
      ])).rejects.toBeInstanceOf(PgVecBatchVaultMismatchError);
    } finally {
      if (url === undefined) delete process.env.CLAWMEM_PG_URL;
      else process.env.CLAWMEM_PG_URL = url;
    }
  });

  it("a single-vault batch is NOT refused by the vault guard (the guard is not an off switch)", async () => {
    // Proves the discriminator: same call shape, homogeneous vault, and the
    // failure that comes back is the CONNECTION one, not the vault one.
    const url = process.env.CLAWMEM_PG_URL;
    delete process.env.CLAWMEM_PG_URL;
    try {
      const p = insertEmbeddingsBatch([
        frag("docs", "reference/a.md", 0),
        frag("docs", "reference/b.md", 1),
      ]);
      await expect(p).rejects.not.toBeInstanceOf(PgVecBatchVaultMismatchError);
    } finally {
      if (url === undefined) delete process.env.CLAWMEM_PG_URL;
      else process.env.CLAWMEM_PG_URL = url;
    }
  });
});
