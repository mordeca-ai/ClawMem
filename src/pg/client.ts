/**
 * node-postgres pool wiring (master-harness-vn4rz.7).
 *
 * `pg` uses $1-style placeholders, NOT sqlite's `?`. Every query in src/pg/ is
 * parameterized; no value is ever interpolated into SQL text.
 */

import pg from "pg";
import { pgSchema, resolvePgConfig } from "./config.ts";
import type { Vault } from "./vaults.ts";

const { Pool, types } = pg;

// pgvector arrives as a text literal like "[0.1,0.2,...]". Parse it back to a
// Float32Array-compatible number[] so callers never see the wire format.
// 1700 = numeric; we only override the vector OID, looked up lazily below.

/**
 * ONE POOL PER VAULT (master-harness-0ynkd, ADR-0162 §1).
 *
 * Before this, a single module-global pool meant the sfw and nsfw vaults were
 * not merely un-routed — they were unrepresentable. Keying the pools by vault
 * is what makes "which database did this write go to" a question with an
 * answer. Each pool resolves its OWN connection config and never borrows the
 * other's: resolvePgConfig THROWS for an unconfigured vault rather than
 * returning a usable-looking fallback.
 */
const pools = new Map<Vault, pg.Pool>();

export function getPool(vault: Vault): pg.Pool {
  const existing = pools.get(vault);
  if (existing) return existing;
  const cfg = resolvePgConfig(vault);
  const pool = new Pool({
    connectionString: cfg.connectionString,
    // Single-node WSL2 box shared with everything else on yoriserver; a small
    // pool is correct here. ADR-0162 §1's substrate, not a warehouse.
    max: Number(process.env.CLAWMEM_PG_POOL_MAX ?? 8),
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 10_000,
    application_name: `clawmem-${vault}`,
  });
  pool.on("error", err => {
    // An idle-client error must not take the process down.
    console.error(`[clawmem-pg:${vault}] idle client error: ${err.message}`);
  });
  pools.set(vault, pool);
  return pool;
}

/**
 * Check a client out of `vault`'s pool with its `search_path` PINNED (vn4rz.7 pass C).
 *
 * `vault` is a REQUIRED FIRST PARAMETER WITH NO DEFAULT, on purpose
 * (master-harness-0ynkd). A default would make "forgot to name a vault" a
 * silent runtime routing decision; with no default it is a compile error, which
 * is the only version of this guarantee that survives a future call site being
 * added by someone who has not read ADR-0162 §1.
 *
 * Every write helper names its tables unqualified, so the schema they land in
 * is whatever search_path the pooled session happens to carry. Pooled sessions
 * are REUSED, so a previous checkout's `SET search_path` would otherwise leak
 * into the next caller. Setting it unconditionally on every checkout — to the
 * configured schema, or explicitly back to DEFAULT — makes the resolution
 * deterministic instead of order-dependent, and is what lets the integration
 * suite run the REAL exported functions against a throwaway schema.
 *
 * Cost is one extra round trip per checkout against a loopback socket; the
 * alternative is a class of silently-wrote-to-the-wrong-schema bug.
 */
export async function withClient<T>(
  vault: Vault,
  fn: (c: pg.PoolClient) => Promise<T>,
): Promise<T> {
  const c = await getPool(vault).connect();
  try {
    const schema = pgSchema();
    // A schema name is an identifier and cannot be a bind parameter. config.ts
    // has already rejected anything that is not a bare identifier; quoting here
    // is the second, independent layer.
    await c.query(
      schema ? `SET search_path TO ${c.escapeIdentifier(schema)}, public` : "SET search_path TO DEFAULT",
    );
    return await fn(c);
  } finally {
    c.release();
  }
}

/**
 * Run `fn` inside ONE transaction. Every write helper that must be atomic with
 * its preflight takes a client from here — the preflight and the INSERT share a
 * transaction or the fence is worthless (ADR-0162 §7 / master-harness-vn4rz.21).
 */
export async function withTransaction<T>(
  vault: Vault,
  fn: (c: pg.PoolClient) => Promise<T>,
): Promise<T> {
  return withClient(vault, async c => {
    await c.query("BEGIN");
    try {
      const out = await fn(c);
      await c.query("COMMIT");
      return out;
    } catch (e) {
      await c.query("ROLLBACK");
      throw e;
    }
  });
}

/** Close EVERY vault pool. Called from the CLI teardown and test afterAll hooks. */
export async function closePool(): Promise<void> {
  const open = [...pools.values()];
  pools.clear();
  await Promise.all(open.map(p => p.end()));
}

/** Serialize a JS number array into pgvector's text input format. */
export function toVectorLiteral(v: ArrayLike<number>): string {
  const parts: string[] = new Array(v.length);
  for (let i = 0; i < v.length; i++) parts[i] = String(v[i]);
  return `[${parts.join(",")}]`;
}

export { types };
