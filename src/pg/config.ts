/**
 * PostgreSQL connection + geometry configuration for the clawmem PG write path
 * (master-harness-vn4rz.7, ADR-0162).
 *
 * Two rules this module exists to enforce:
 *
 *  1. **One named constant for the embedding dimension.** pgvector fixes the
 *     dimension at CREATE TABLE; sqlite's vec0 table discovered it at runtime
 *     from the first embed response. That runtime discovery is gone, so the
 *     dimension must have exactly ONE authoritative home. This is it. The
 *     migration runner substitutes it into `:EMBED_DIM`, and the write path
 *     asserts the live column against it at connect time.
 *
 *  2. **Never silently fall back to sqlite.** With no PG connection configured,
 *     every entry point here THROWS. A write path that quietly re-routed to the
 *     old store on a config typo would produce a green run and an empty
 *     database.
 */

import { VaultDatabaseCollisionError, VaultNotConfiguredError } from "./errors.ts";
import type { Vault } from "./vaults.ts";

/**
 * THE embedding dimension. Single source of truth.
 *
 * 768 = `embeddinggemma`, the model pinned in
 * ~/.config/environment.d/50-clawmem.conf (CLAWMEM_EMBED_DIMENSIONS=768) and
 * served by yoshiee's ollama at http://192.168.2.15:11434. Changing this is a
 * new vector space (ADR-0162 §7: "a different model is a different space"), so
 * it is a MIGRATION plus a full re-embed, never an edit.
 */
export const EMBED_DIM = 768 as const;

/** Overridable only for tests that stand up a small-dimension fixture schema. */
export function embedDim(): number {
  const raw = process.env.CLAWMEM_PG_EMBED_DIM;
  if (!raw) return EMBED_DIM;
  const n = Number(raw);
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(`CLAWMEM_PG_EMBED_DIM must be a positive integer, got ${JSON.stringify(raw)}`);
  }
  return n;
}

/**
 * THE SCHEMA KNOB (master-harness-vn4rz.7 pass C, Task 3.1).
 *
 * Every statement in src/pg/write.ts names its tables UNQUALIFIED and resolves
 * them through the connection's `search_path`. Before this knob existed the
 * integration suite could not point the real exported functions at a throwaway
 * schema, so it RE-EXPRESSED the write transaction instead — a test that
 * re-implements the code under test proves nothing about that code, and a
 * future drift between write.ts and the copy would not have been caught. That
 * was pass B's self-reported weakest link.
 *
 * The knob is deliberately NOT a parameter threaded through every write
 * signature: search_path is a *connection* property, so the right place to set
 * it is where the connection is checked out (src/pg/client.ts::withClient),
 * once, for every helper at the same time.
 *
 * Precedence: the programmatic override (setPgSchema) wins over the
 * CLAWMEM_PG_SCHEMA env var; null/empty means "the server's default
 * search_path", which is what production uses.
 */
const SCHEMA_IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/;

let schemaOverride: string | null | undefined;

/** Programmatic override, for tests. Pass null to restore env/default behavior. */
export function setPgSchema(schema: string | null): void {
  if (schema !== null) assertSchemaIdent(schema);
  schemaOverride = schema;
}

function assertSchemaIdent(raw: string): void {
  if (!SCHEMA_IDENT.test(raw)) {
    // A schema name reaches SQL as an IDENTIFIER, which cannot be parameterized.
    // Rejecting anything but a plain identifier is what keeps that safe even
    // though client.ts also quotes it.
    throw new Error(
      `Invalid PostgreSQL schema name ${JSON.stringify(raw)}: must match ${SCHEMA_IDENT}`,
    );
  }
}

/** The schema every pooled connection should resolve unqualified tables in. */
export function pgSchema(env: NodeJS.ProcessEnv = process.env): string | null {
  const raw = schemaOverride !== undefined ? schemaOverride : (env.CLAWMEM_PG_SCHEMA ?? null);
  if (raw === null || raw === "") return null;
  assertSchemaIdent(raw);
  return raw;
}

export interface PgConfig {
  /** libpq connection string. */
  connectionString: string;
  /** Redacted form, safe to log or put in an error message. */
  safeLabel: string;
  /** Which vault this connection serves. */
  vault: Vault;
  /**
   * The database name this configuration NAMES. The write path asserts
   * `current_database()` against it inside the transaction (the belt) —
   * a connection string with no path silently connects to the database named
   * after the role, which routing logic cannot see.
   */
  database: string;
}

/**
 * Env-var prefix per vault. The sfw vault keeps TODAY'S names byte-for-byte so
 * every existing deployment, test and shell one-liner behaves identically; the
 * nsfw vault gets its own namespace, which is what makes "configured for one
 * vault but not the other" a representable — and therefore refusable — state.
 */
const VAULT_ENV_PREFIX: Record<Vault, string> = {
  sfw: "CLAWMEM_PG",
  nsfw: "CLAWMEM_PG_NSFW",
};

/** Database name each vault is EXPECTED to resolve to, per ADR-0162 §1. */
export const VAULT_DEFAULT_DATABASE: Record<Vault, string> = {
  sfw: "clawmem",
  nsfw: "clawmem_nsfw",
};

function databaseFromUrl(parsed: URL, varName: string): string {
  const name = decodeURIComponent(parsed.pathname.replace(/^\//, "")).trim();
  if (!name) {
    throw new Error(
      `${varName} names no database (path is empty). libpq would silently connect ` +
      `to the database named after the role, which is exactly the mis-wire the ` +
      `write path's current_database() belt exists to catch. Spell the database out: ` +
      `postgres://user:pass@host:5433/<database>.`,
    );
  }
  return name;
}

/**
 * Resolve the PG connection for `vault` from the environment.
 *
 * Precedence, per vault: `<PREFIX>_URL` wins outright; otherwise the discrete
 * `<PREFIX>_{HOST,PORT,DATABASE,USER,PASSWORD}` vars are assembled. If neither
 * is usable this THROWS — there is deliberately no default, no sqlite fallback,
 * and (master-harness-0ynkd) no fallback to the OTHER vault's connection.
 *
 * `vault` defaults to "sfw" so that every pre-existing call site keeps its exact
 * previous behavior; the write path never relies on that default — it passes the
 * vault explicitly.
 */
export function resolvePgConfig(
  vault: Vault = "sfw",
  env: NodeJS.ProcessEnv = process.env,
): PgConfig {
  const cfg = resolveOneVault(vault, env);
  if (vault === "nsfw") {
    // The one mis-configuration the per-connection belt is blind to: if both
    // vaults name the SAME database, current_database() agrees with both
    // expectations and the boundary is gone with every check green. Only the
    // comparison BETWEEN the two configs can see it, so it is made here, before
    // a pool is ever opened. An UNCONFIGURED sfw vault is not a collision.
    let sfw: PgConfig | null = null;
    try {
      sfw = resolveOneVault("sfw", env);
    } catch {
      sfw = null;
    }
    if (sfw && sfw.database === cfg.database) {
      throw new VaultDatabaseCollisionError(cfg.database);
    }
  }
  return cfg;
}

function resolveOneVault(vault: Vault, env: NodeJS.ProcessEnv): PgConfig {
  const P = VAULT_ENV_PREFIX[vault];
  const url = env[`${P}_URL`]?.trim();
  if (url) {
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      throw new Error(`${P}_URL is set but is not a valid URL`);
    }
    if (parsed.protocol !== "postgres:" && parsed.protocol !== "postgresql:") {
      throw new Error(`${P}_URL must be a postgres:// URL, got ${parsed.protocol}//`);
    }
    return {
      connectionString: url,
      safeLabel: `${parsed.protocol}//${parsed.username || "?"}@${parsed.host}${parsed.pathname}`,
      vault,
      database: databaseFromUrl(parsed, `${P}_URL`),
    };
  }

  const host = env[`${P}_HOST`]?.trim();
  const database = env[`${P}_DATABASE`]?.trim();
  const user = env[`${P}_USER`]?.trim();
  const password = env[`${P}_PASSWORD`] ?? "";
  const port = env[`${P}_PORT`]?.trim() || "5433";

  const missing = [
    [`${P}_HOST`, host],
    [`${P}_DATABASE`, database],
    [`${P}_USER`, user],
  ].filter(([, v]) => !v).map(([k]) => k as string);

  if (missing.length > 0) {
    throw new VaultNotConfiguredError(vault, [
      `${P}_URL (postgres://user:pass@host:5433/${VAULT_DEFAULT_DATABASE[vault]})`,
      `all of ${P}_HOST / ${P}_DATABASE / ${P}_USER (missing: ${missing.join(", ")})`,
    ]);
  }
  if (!/^\d+$/.test(port)) {
    throw new Error(`${P}_PORT must be numeric, got ${JSON.stringify(port)}`);
  }

  const enc = encodeURIComponent;
  return {
    connectionString: `postgres://${enc(user!)}:${enc(password)}@${host}:${port}/${enc(database!)}`,
    safeLabel: `postgres://${user}@${host}:${port}/${database}`,
    vault,
    database: database!,
  };
}

/** Is a connection configured for `vault` at all? Never throws. */
export function vaultIsConfigured(
  vault: Vault,
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  try {
    resolvePgConfig(vault, env);
    return true;
  } catch {
    return false;
  }
}
