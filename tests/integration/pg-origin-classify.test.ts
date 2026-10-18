/**
 * Index-time origin classification against real Postgres (master-harness-vn4rz.66,
 * Review Hub a7981735).
 *
 * Same substrate pattern as pg-vault-isolation.test.ts: TWO EPHEMERAL DATABASES
 * (`clawmem_vaulttest_<suffix>` / `clawmem_vaulttest_nsfw_<suffix>`) created and
 * dropped per run on the local clawmem-pg cluster — never the live `clawmem` /
 * `clawmem_nsfw` vaults. Every transcript below is SYNTHETIC fixture text.
 *
 * Proves, on rows:
 *   - a session with ONE private chunk lands wholly in nsfw, a control session
 *     wholly in sfw;
 *   - "flagged later": a session first loaded clean, then given a private chunk,
 *     moves to nsfw on reload and its sfw residue (origin rows, unshared content,
 *     their vectors) is removed — while a hash another sfw row still references
 *     is kept;
 *   - the embed coverage check reads the vault the document routes to;
 *   - mixed-vault embed batches are split, and nsfw vectors land in nsfw;
 *   - an unclassified collection and an explicit nsfw load behave as before;
 *   - the vn4rz.109 read-back guard covers the nsfw-routed write: a stale
 *     session_id on an nsfw origin row throws OriginWriteReadBackError and the
 *     transaction commits nothing in either vault.
 *
 * SKIPS WITH A REASON when the substrate is unreachable — never silently passes.
 */

import { describe, it, expect, beforeAll, afterAll } from "bun:test";
import pg from "pg";
import { assertNotProductionDatabase, PG_TEST_SETUP_TIMEOUT_MS, PG_TEST_TEARDOWN_TIMEOUT_MS } from "./pg-test-schema.ts";
import { readFileSync, readdirSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { randomBytes } from "crypto";
import { MIGRATIONS_DIR, substituteMigrationParams } from "../../src/pg/migrate.ts";
import { closePool } from "../../src/pg/client.ts";
import { setPgSchema } from "../../src/pg/config.ts";
import { resetVaultCache } from "../../src/pg/vaults.ts";
import { insertEmbeddingsBatch, upsertDocument } from "../../src/pg/write.ts";
import {
  loadOriginCollection, OriginWriteReadBackError, type OriginLoadStats,
} from "../../src/pg/origin.ts";
import { hashContent } from "../../src/indexer.ts";
import { setDefaultLlamaCpp } from "../../src/llm.ts";

const REPO = "/home/bj/claude/master-harness";
const DIM = 768;
const MODEL = "embeddinggemma";

function superuserUrl(): string | null {
  const explicit = process.env.CLAWMEM_PG_SUPERUSER_URL?.trim();
  if (explicit) return explicit;
  try {
    const pw = readFileSync(`${REPO}/.secrets/local/clawmem-pg/superuser.pw`, "utf-8").trim();
    if (!pw) return null;
    return `postgres://postgres:${encodeURIComponent(pw)}@127.0.0.1:5433/postgres`;
  } catch {
    return null;
  }
}

const SU = superuserUrl();
if (!SU) {
  console.log(
    "[SKIP] pg-origin-classify: no superuser connection " +
    "(CLAWMEM_PG_SUPERUSER_URL unset and .secrets/local/clawmem-pg/superuser.pw unreadable) — " +
    "ephemeral vault databases cannot be created.",
  );
}
const d = SU ? describe : describe.skip;

const suffix = `${process.pid}_${randomBytes(4).toString("hex")}`;
const SFW_DB = `clawmem_vaulttest_${suffix}`;
const NSFW_DB = `clawmem_vaulttest_nsfw_${suffix}`;

const saved: Record<string, string | undefined> = {};
const ENV_KEYS = [
  "CLAWMEM_CONFIG_DIR", "CLAWMEM_PG_URL", "CLAWMEM_PG_NSFW_URL",
  "CLAWMEM_PG_HOST", "CLAWMEM_PG_DATABASE", "CLAWMEM_PG_USER", "CLAWMEM_PG_PASSWORD",
];

function setEnv(k: string, v: string | undefined): void {
  if (v === undefined) delete process.env[k];
  else process.env[k] = v;
}

/** A synthetic chunk file. `sessionId` null = no frontmatter key (basename keys the session). */
function chunk(sessionId: string | null, day: number, body: string): string {
  return (
    "---\n" +
    (sessionId ? `session_id: ${sessionId}\n` : "") +
    `ts: 2026-09-${String(day).padStart(2, "0")}T10:00:00Z\n` +
    "content_type: session-transcript\n" +
    "---\n\n" +
    `# synthetic transcript\n\n${body}\n`
  );
}

/** Identical raw text in two sessions -> one content hash referenced from both. */
const SHARED_RAW = chunk(null, 3, "user: run the tests\nassistant: 12 passed");

d("PG origin classification (vn4rz.66)", () => {
  let admin: pg.Pool;
  let configDir: string;
  let root: string;
  let otherRoot: string;
  let explicitRoot: string;
  let guardRoot: string;
  const files: Record<string, string> = {};

  function dbUrl(db: string): string {
    const u = new URL(SU!);
    u.pathname = `/${db}`;
    return u.toString();
  }

  async function migrateInto(db: string): Promise<void> {
    const p = new pg.Pool({ connectionString: dbUrl(db) });
    try {
      const c = await p.connect();
      try {
        await assertNotProductionDatabase(c);
        await c.query("CREATE EXTENSION IF NOT EXISTS vector");
        await c.query("CREATE EXTENSION IF NOT EXISTS pg_trgm");
        for (const f of readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith(".sql")).sort()) {
          await c.query(substituteMigrationParams(readFileSync(join(MIGRATIONS_DIR, f), "utf-8"), undefined, DIM));
        }
      } finally {
        c.release();
      }
    } finally {
      await p.end();
    }
  }

  async function q<T = { n: string }>(db: string, sql: string, params: unknown[] = []): Promise<T[]> {
    const p = new pg.Pool({ connectionString: dbUrl(db) });
    try {
      return (await p.query(sql, params as never[])).rows as T[];
    } finally {
      await p.end();
    }
  }
  async function count(db: string, sql: string, params: unknown[] = []): Promise<number> {
    const rows = await q(db, sql, params);
    return Number(rows[0]?.n ?? -1);
  }
  const originRows = (db: string, like: string) =>
    count(db, "SELECT count(*)::text AS n FROM origin_documents WHERE collection = 'monchujo' AND path LIKE $1", [like]);
  const contentRows = (db: string, hashes: string[]) =>
    count(db, "SELECT count(*)::text AS n FROM content WHERE hash = ANY($1::text[])", [hashes]);
  const vectorRows = (db: string, hashes: string[]) =>
    count(db, "SELECT count(*)::text AS n FROM content_vectors WHERE hash = ANY($1::text[])", [hashes]);

  function put(rel: string, raw: string): void {
    files[rel] = raw;
    writeFileSync(join(root, rel), raw);
  }
  const h = (rel: string) => hashContent(files[rel]!);
  const hashesOf = (prefix: string) =>
    Object.keys(files).filter(f => f.startsWith(prefix)).sort().map(h);

  async function load(extra: Partial<Parameters<typeof loadOriginCollection>[0]> = {}): Promise<OriginLoadStats> {
    const s = await loadOriginCollection({
      collection: "monchujo", root, pattern: "**/*.md", skipEmbed: true, ...extra,
    });
    await closePool();
    return s;
  }

  beforeAll(async () => {
    for (const k of ENV_KEYS) saved[k] = process.env[k];
    setPgSchema(null);

    admin = new pg.Pool({ connectionString: SU! });
    await admin.query(`CREATE DATABASE ${SFW_DB}`);
    await admin.query(`CREATE DATABASE ${NSFW_DB}`);
    await migrateInto(SFW_DB);
    await migrateInto(NSFW_DB);

    root = mkdtempSync(join(tmpdir(), "clawmem-originclass-root-"));
    otherRoot = mkdtempSync(join(tmpdir(), "clawmem-originclass-other-"));
    explicitRoot = mkdtempSync(join(tmpdir(), "clawmem-originclass-explicit-"));
    guardRoot = mkdtempSync(join(tmpdir(), "clawmem-originclass-guard-"));
    mkdirSync(join(root, "2026", "09"), { recursive: true });

    // Session A: six chunks, ONE (c03) reads a private path.
    for (let i = 0; i < 6; i++) {
      put(`2026/09/sess-a__c0${i}.md`, chunk("sess-a", 1,
        i === 3
          ? "assistant: $ cat ~/claude/master-harness/intelligence/operations/nsfw/x.md"
          : `assistant: ordinary step ${i}`));
    }
    // Session B: ordinary control, with an AMBIGUOUS-only reference (reported, not routed).
    put("2026/09/sess-b__c00.md", chunk("sess-b", 2, "assistant: bd ready"));
    put("2026/09/sess-b__c01.md", chunk("sess-b", 2, "assistant: ls projects/job-search/"));
    // Session C: clean at first; flagged later. c00 shares its raw text with D.
    put("2026/09/sess-c__c00.md", SHARED_RAW);
    put("2026/09/sess-c__c01.md", chunk(null, 3, "assistant: c01 unique text"));
    put("2026/09/sess-c__c02.md", chunk(null, 3, "assistant: c02 unique text"));
    // Session D: one chunk, byte-identical to C's c00.
    put("2026/09/sess-d__c00.md", SHARED_RAW);

    writeFileSync(join(otherRoot, "telemetry__c00.md"),
      chunk("tel-1", 4, "assistant: cat intelligence/operations/nsfw/x.md"));
    writeFileSync(join(explicitRoot, "explicit__c00.md"), chunk("exp-1", 5, "assistant: bd ready"));
    // One session, every chunk nsfw-routed: the guard test must see ONLY nsfw writes.
    writeFileSync(join(guardRoot, "sess-g__c00.md"),
      chunk("sid-guard-7", 6, "assistant: $ cat ~/claude/master-harness/intelligence/operations/nsfw/g.md"));
    writeFileSync(join(guardRoot, "sess-g__c01.md"), chunk("sid-guard-7", 6, "assistant: ordinary step g1"));

    configDir = mkdtempSync(join(tmpdir(), "clawmem-originclass-cfg-"));
    writeFileSync(
      join(configDir, "index.yml"),
      `collections:\n  monchujo:\n    path: ${root}\n    pattern: "**/*.md"\n` +
      `  docs:\n    path: ${otherRoot}\n    pattern: "**/*.md"\n`,
      "utf-8",
    );
    setEnv("CLAWMEM_CONFIG_DIR", configDir);
    for (const k of ["CLAWMEM_PG_HOST", "CLAWMEM_PG_DATABASE", "CLAWMEM_PG_USER", "CLAWMEM_PG_PASSWORD"]) {
      setEnv(k, undefined);
    }
    setEnv("CLAWMEM_PG_URL", dbUrl(SFW_DB));
    setEnv("CLAWMEM_PG_NSFW_URL", dbUrl(NSFW_DB));
    resetVaultCache();
  }, PG_TEST_SETUP_TIMEOUT_MS);

  afterAll(async () => {
    setDefaultLlamaCpp(null);
    await closePool();
    for (const k of ENV_KEYS) setEnv(k, saved[k]);
    resetVaultCache();
    for (const dir of [configDir, root, otherRoot, explicitRoot, guardRoot]) {
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
    if (admin) {
      for (const db of [SFW_DB, NSFW_DB]) {
        await admin.query(`SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1`, [db]);
        await admin.query(`DROP DATABASE IF EXISTS ${db}`);
      }
      await admin.end();
    }
  }, PG_TEST_TEARDOWN_TIMEOUT_MS);

  it("routes the session with one private chunk WHOLLY to nsfw and the control WHOLLY to sfw", async () => {
    const s = await load();
    expect(s.recordsWritten).toBe(12);
    expect(s.classifiedNsfwDocs).toBe(6);
    expect(s.classifiedNsfwSessions).toBe(1);
    expect(s.ambiguousSessions).toBe(1);
    expect(s.sfwResidueRemoved).toEqual({ originRows: 0, contentRows: 0, vectorRows: 0 });

    // Positive session: every chunk, not just c03, and ONLY in nsfw.
    expect(await originRows(NSFW_DB, "2026/09/sess-a__%")).toBe(6);
    expect(await originRows(SFW_DB, "2026/09/sess-a__%")).toBe(0);
    expect(await contentRows(NSFW_DB, hashesOf("2026/09/sess-a__"))).toBe(6);
    expect(await contentRows(SFW_DB, hashesOf("2026/09/sess-a__"))).toBe(0);

    // Controls: ONLY in sfw (the ambiguous-only session included).
    for (const [like, n] of [["2026/09/sess-b__%", 2], ["2026/09/sess-c__%", 3], ["2026/09/sess-d__%", 1]] as const) {
      expect(await originRows(SFW_DB, like)).toBe(n);
      expect(await originRows(NSFW_DB, like)).toBe(0);
    }
    expect(await contentRows(NSFW_DB, hashesOf("2026/09/sess-b__"))).toBe(0);

    // Provenance records the verdict as signal ids only.
    const [prov] = await q<{ p: { content_classification: { vault: string; private: string[] } } }>(
      NSFW_DB, "SELECT provenance AS p FROM origin_documents WHERE path = $1", ["2026/09/sess-a__c00.md"]);
    expect(prov!.p.content_classification.vault).toBe("nsfw");
    expect(prov!.p.content_classification.private).toEqual(["nsfw-ops"]);
  });

  it("moves a LATER-flagged session to nsfw, removes its sfw residue, keeps a shared hash", async () => {
    const [c0, c1, c2] = [h("2026/09/sess-c__c00.md"), h("2026/09/sess-c__c01.md"), h("2026/09/sess-c__c02.md")];
    // sfw state before the flag: one vector per C hash, and c01's hash ALSO
    // referenced by an unrelated sfw documents row (c00's is referenced by D).
    await insertEmbeddingsBatch([c0, c1, c2].map((hash, i) => ({
      collection: "monchujo", path: `2026/09/sess-c__c0${i}.md`,
      hash, seq: 0, pos: 0, embedding: new Array(DIM).fill(0.25), model: MODEL,
    })));
    await upsertDocument({
      collection: "docs", path: "shared/c01-copy.md", title: "c01 copy",
      hash: c1, body: files["2026/09/sess-c__c01.md"]!,
    });
    await closePool();
    expect(await vectorRows(SFW_DB, [c0, c1, c2])).toBe(3);

    // The flag: a new chunk of session C references a persona workspace.
    put("2026/09/sess-c__c03.md", chunk(null, 3, "assistant: Read /home/bj/.openclaw/workspace/SOUL.md"));
    const c3 = h("2026/09/sess-c__c03.md");

    const s = await load();
    expect(s.classifiedNsfwDocs).toBe(10);
    expect(s.classifiedNsfwSessions).toBe(2);
    // 3 sfw origin rows for C; only c02's content is unshared, with its 1 vector.
    expect(s.sfwResidueRemoved).toEqual({ originRows: 3, contentRows: 1, vectorRows: 1 });
    // Coverage read the ROUTED vault: C's sfw vectors did not count for nsfw
    // docs. The single "already embedded" hit is D (sfw), whose hash is c0.
    expect(s.hashesAlreadyEmbedded).toBe(1);

    expect(await originRows(SFW_DB, "2026/09/sess-c__%")).toBe(0);
    expect(await originRows(NSFW_DB, "2026/09/sess-c__%")).toBe(4);
    expect(await contentRows(NSFW_DB, [c0, c1, c2, c3])).toBe(4);
    // Unshared content + vectors: gone from sfw.
    expect(await contentRows(SFW_DB, [c2, c3])).toBe(0);
    expect(await vectorRows(SFW_DB, [c2])).toBe(0);
    // Shared with a remaining sfw row: KEPT, vectors too.
    expect(await contentRows(SFW_DB, [c0])).toBe(1); // still D's
    expect(await contentRows(SFW_DB, [c1])).toBe(1); // still the docs row's
    expect(await vectorRows(SFW_DB, [c0, c1])).toBe(2);
    expect(await originRows(SFW_DB, "2026/09/sess-d__%")).toBe(1);

    // Idempotent: a second reload finds no residue.
    const again = await load();
    expect(again.sfwResidueRemoved).toEqual({ originRows: 0, contentRows: 0, vectorRows: 0 });
    expect(await originRows(NSFW_DB, "2026/09/sess-c__%")).toBe(4);
  });

  it("splits mixed-vault embed batches and lands nsfw-classified vectors in nsfw only", async () => {
    setDefaultLlamaCpp({
      embedBatch: async (texts: string[]) =>
        texts.map(() => ({ embedding: new Array(DIM).fill(0.1), model: MODEL, endpoint: "stub-embed" })),
    } as never);
    try {
      // Batch size 3 guarantees batches that straddle sess-a (nsfw) / sess-b (sfw).
      const s = await load({ skipEmbed: false, embedBatchSize: 3 });
      expect(s.embedFailures).toBe(0);
      expect(s.fragmentsEmbedded).toBeGreaterThan(0);
    } finally {
      setDefaultLlamaCpp(null);
    }
    const nsfwHashes = [...hashesOf("2026/09/sess-a__"), ...hashesOf("2026/09/sess-c__")];
    const covered = await count(NSFW_DB,
      "SELECT count(DISTINCT hash)::text AS n FROM content_vectors WHERE hash = ANY($1::text[])", [nsfwHashes]);
    expect(covered).toBe(nsfwHashes.length);
    expect(await vectorRows(SFW_DB, hashesOf("2026/09/sess-a__"))).toBe(0);
    expect(await vectorRows(SFW_DB, [h("2026/09/sess-c__c02.md"), h("2026/09/sess-c__c03.md")])).toBe(0);
    // The sfw control got its vectors in sfw, none in nsfw.
    expect(await vectorRows(SFW_DB, hashesOf("2026/09/sess-b__"))).toBeGreaterThan(0);
    expect(await vectorRows(NSFW_DB, hashesOf("2026/09/sess-b__"))).toBe(0);
  });

  it("leaves an unclassified collection and an explicit --vault nsfw load unchanged", async () => {
    const other = await loadOriginCollection({
      collection: "telemetry-origin", root: otherRoot, pattern: "**/*.md", skipEmbed: true,
    });
    await closePool();
    expect(other.classifiedNsfwDocs).toBe(0);
    expect(other.sfwResidueRemoved).toEqual({ originRows: 0, contentRows: 0, vectorRows: 0 });
    // Not content-classified: a private reference does NOT move it (behaviour as before).
    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM origin_documents WHERE collection = 'telemetry-origin'")).toBe(1);
    expect(await count(NSFW_DB, "SELECT count(*)::text AS n FROM origin_documents WHERE collection = 'telemetry-origin'")).toBe(0);

    const explicit = await loadOriginCollection({
      collection: "monchujo", root: explicitRoot, pattern: "**/*.md", skipEmbed: true, vault: "nsfw",
    });
    await closePool();
    expect(explicit.classifiedNsfwDocs).toBe(0);
    expect(explicit.classifiedNsfwSessions).toBe(0);
    expect(explicit.sfwResidueRemoved).toEqual({ originRows: 0, contentRows: 0, vectorRows: 0 });
    expect(await count(NSFW_DB, "SELECT count(*)::text AS n FROM origin_documents WHERE path = 'explicit__c00.md'")).toBe(1);
    expect(await count(SFW_DB, "SELECT count(*)::text AS n FROM origin_documents WHERE path = 'explicit__c00.md'")).toBe(0);
    const [prov] = await q<{ p: Record<string, unknown> }>(
      NSFW_DB, "SELECT provenance AS p FROM origin_documents WHERE path = 'explicit__c00.md'");
    expect("content_classification" in prov!.p).toBe(false);
  });

  it("guards the CLASSIFIED nsfw write: a stale session_id throws OriginWriteReadBackError, nothing commits", async () => {
    const guardHashes = ["sess-g__c00.md", "sess-g__c01.md"]
      .map(f => hashContent(readFileSync(join(guardRoot, f), "utf-8")));
    // Stale-parameter stand-in: the nsfw database rewrites session_id on its way
    // in, so the RETURNING row disagrees with what the loader sent. Attached in
    // NSFW_DB only: the sfw vault is never written by this all-nsfw session.
    const nsfw = new pg.Pool({ connectionString: dbUrl(NSFW_DB) });
    const partitions = async () => (await nsfw.query<{ relname: string }>(
      `SELECT c.relname FROM pg_inherits i
         JOIN pg_class c ON i.inhrelid = c.oid
         JOIN pg_class p ON i.inhparent = p.oid
        WHERE p.relname = 'origin_documents'`)).rows.map(r => r.relname);
    try {
      await nsfw.query(`
        CREATE OR REPLACE FUNCTION stale_session_trigger() RETURNS trigger AS $$
        BEGIN NEW.session_id := 'stale-session'; RETURN NEW; END;
        $$ LANGUAGE plpgsql`);
      try {
        await nsfw.query(`CREATE TRIGGER trg_stale_session BEFORE INSERT OR UPDATE ON origin_documents
                          FOR EACH ROW EXECUTE FUNCTION stale_session_trigger()`);
      } catch {
        for (const t of await partitions()) {
          await nsfw.query(`CREATE TRIGGER trg_stale_session BEFORE INSERT OR UPDATE ON ${t}
                            FOR EACH ROW EXECUTE FUNCTION stale_session_trigger()`);
        }
      }

      let thrown: unknown = null;
      try {
        await loadOriginCollection({
          collection: "monchujo", root: guardRoot, pattern: "**/*.md", skipEmbed: true,
        });
      } catch (err) {
        thrown = err;
      } finally {
        await closePool();
      }

      expect(thrown).toBeInstanceOf(OriginWriteReadBackError);
      const e = thrown as OriginWriteReadBackError;
      expect(e.collection).toBe("monchujo");
      expect(e.mismatches).toEqual(["session_id"]);
      // Privacy: the message names columns and the path, never the session id.
      expect(e.message).not.toContain("sid-guard-7");
      expect(e.message).not.toContain("stale-session");

      // Rolled back: no origin row and no content row in nsfw, nothing leaked to sfw.
      for (const db of [NSFW_DB, SFW_DB]) {
        expect(await originRows(db, "sess-g__%")).toBe(0);
        expect(await contentRows(db, guardHashes)).toBe(0);
      }
    } finally {
      await nsfw.query("DROP TRIGGER IF EXISTS trg_stale_session ON origin_documents").catch(() => {});
      for (const t of await partitions()) {
        await nsfw.query(`DROP TRIGGER IF EXISTS trg_stale_session ON ${t}`);
      }
      await nsfw.query("DROP FUNCTION IF EXISTS stale_session_trigger()");
      await nsfw.end();
    }
  });
});
