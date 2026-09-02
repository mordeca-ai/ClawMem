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
}

/**
 * Resolve the PG connection from the environment.
 *
 * Precedence: CLAWMEM_PG_URL wins outright; otherwise the discrete
 * CLAWMEM_PG_{HOST,PORT,DATABASE,USER,PASSWORD} vars are assembled. If neither
 * is usable this THROWS — there is deliberately no default and no fallback.
 */
export function resolvePgConfig(env: NodeJS.ProcessEnv = process.env): PgConfig {
  const url = env.CLAWMEM_PG_URL?.trim();
  if (url) {
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      throw new Error("CLAWMEM_PG_URL is set but is not a valid URL");
    }
    if (parsed.protocol !== "postgres:" && parsed.protocol !== "postgresql:") {
      throw new Error(`CLAWMEM_PG_URL must be a postgres:// URL, got ${parsed.protocol}//`);
    }
    return {
      connectionString: url,
      safeLabel: `${parsed.protocol}//${parsed.username || "?"}@${parsed.host}${parsed.pathname}`,
    };
  }

  const host = env.CLAWMEM_PG_HOST?.trim();
  const database = env.CLAWMEM_PG_DATABASE?.trim();
  const user = env.CLAWMEM_PG_USER?.trim();
  const password = env.CLAWMEM_PG_PASSWORD ?? "";
  const port = env.CLAWMEM_PG_PORT?.trim() || "5433";

  const missing = [
    ["CLAWMEM_PG_HOST", host],
    ["CLAWMEM_PG_DATABASE", database],
    ["CLAWMEM_PG_USER", user],
  ].filter(([, v]) => !v).map(([k]) => k as string);

  if (missing.length > 0) {
    throw new Error(
      "No PostgreSQL connection configured for clawmem. Set CLAWMEM_PG_URL " +
      "(postgres://user:pass@host:5433/clawmem), or all of CLAWMEM_PG_HOST / " +
      `CLAWMEM_PG_DATABASE / CLAWMEM_PG_USER (missing: ${missing.join(", ")}). ` +
      "There is no default and no sqlite fallback: a misconfigured PG write path " +
      "must fail loudly, not write somewhere else.",
    );
  }
  if (!/^\d+$/.test(port)) {
    throw new Error(`CLAWMEM_PG_PORT must be numeric, got ${JSON.stringify(port)}`);
  }

  const enc = encodeURIComponent;
  return {
    connectionString: `postgres://${enc(user!)}:${enc(password)}@${host}:${port}/${enc(database!)}`,
    safeLabel: `postgres://${user}@${host}:${port}/${database}`,
  };
}
