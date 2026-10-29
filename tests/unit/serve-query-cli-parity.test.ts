import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { createStore } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";

/**
 * master-harness-vn4rz.123 — the REAL `clawmem serve` POST /query path, end to end.
 *
 * WHY a subprocess and not startServer() with a stub handler (server-query.test.ts covers the
 * route's input validation that way): the defects under test live in cmdServe's queryHandler,
 * which only exists inside the CLI process.
 *   1. An unknown collection reached cmdQuery's die() -> process.exit(1) and killed the whole
 *      persistent endpoint (one probe took down the live 7438 instance). It must answer 4xx
 *      and keep serving.
 *   2. `collection` is optional: absent => unscoped, exactly `clawmem query` with no -c.
 *   3. `session_id` applies that session's focus for the one request (CLI precedence).
 *   4. The response is the same JSON `clawmem query --json` prints (the recall hook parses
 *      either interchangeably).
 * Hermetic: per-run temp vault + config dir + focus root, embed/rerank/LLM pointed nowhere.
 */

const REPO_ROOT = new URL("../..", import.meta.url).pathname;
const ROOT = mkdtempSync(join(tmpdir(), "clawmem-serve-query-"));
const DB = join(ROOT, "index.sqlite");
const CONFIG_DIR = join(ROOT, "config");
const FOCUS_ROOT = join(ROOT, "sessions");
const SESSION = "0f2c4d6e-serve-parity";

function cliEnv(): Record<string, string> {
  const env: Record<string, string> = { ...(process.env as Record<string, string>) };
  for (const k of [
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_SESSION_ID", "CLAWMEM_SESSION_ID", "CLAWMEM_SESSION_FOCUS",
    "CLAWMEM_LLM_URL", "CLAWMEM_LLM_MODEL", "CLAWMEM_RERANK_URL", "CLAWMEM_API_TOKEN",
  ]) delete env[k];
  return {
    ...env,
    INDEX_PATH: DB,
    CLAWMEM_CONFIG_DIR: CONFIG_DIR,
    CLAWMEM_FOCUS_ROOT: FOCUS_ROOT,
    CLAWMEM_NO_LOCAL_MODELS: "true",
    // Nothing listens here: the vector leg degrades and BM25 carries the result set.
    CLAWMEM_EMBED_URL: "http://127.0.0.1:9",
  };
}

function seed(): void {
  mkdirSync(CONFIG_DIR, { recursive: true });
  mkdirSync(FOCUS_ROOT, { recursive: true });
  writeFileSync(join(CONFIG_DIR, "config.yaml"), [
    "collections:",
    "  alpha:",
    `    path: ${join(ROOT, "alpha")}`,
    '    pattern: "**/*.md"',
    "  beta:",
    `    path: ${join(ROOT, "beta")}`,
    '    pattern: "**/*.md"',
    "",
  ].join("\n"));
  const store = createStore(DB);
  const now = new Date().toISOString();
  const doc = (col: string, path: string, body: string) => {
    const hash = hashContent(body + col + path);
    store.insertContent(hash, body, now);
    store.insertDocument(col, path, path, hash, now, now);
  };
  doc("alpha", "zanzibar-alpha.md", "zanzibar protocol handshake ordering for the alpha relay");
  doc("beta", "zanzibar-beta.md", "zanzibar protocol rotation notes for the beta relay");
  for (let i = 0; i < 20; i++) doc(i % 2 ? "alpha" : "beta", `filler-${i}.md`, `assorted planning notes ${i}`);
  store.close();
}

async function freePort(): Promise<number> {
  const probe = Bun.serve({ port: 0, hostname: "127.0.0.1", fetch: () => new Response("") });
  const port = probe.port as number;
  probe.stop(true);
  return port;
}

let serve: ReturnType<typeof Bun.spawn> | null = null;
let base = "";

async function waitHealthy(): Promise<void> {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(`${base}/health`)).ok) return;
    } catch { /* not listening yet */ }
    await Bun.sleep(100);
  }
  throw new Error("clawmem serve never became healthy");
}

async function cliQuery(args: string[]): Promise<unknown> {
  const proc = Bun.spawn([process.execPath, "src/clawmem.ts", "query", ...args, "--json"], {
    cwd: REPO_ROOT, env: cliEnv(), stdout: "pipe", stderr: "pipe",
  });
  const [code, out, err] = await Promise.all([
    proc.exited, new Response(proc.stdout).text(), new Response(proc.stderr).text(),
  ]);
  expect(code, `clawmem query failed: ${err}`).toBe(0);
  return JSON.parse(out);
}

const post = (body: unknown) => fetch(`${base}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/** recencyScore/score move with the wall clock between two runs; compare everything else. */
function stable(hits: unknown): unknown {
  return (hits as Array<Record<string, unknown>>).map(h => ({
    keys: Object.keys(h),
    file: h.file,
    title: h.title,
    contentType: h.contentType,
    searchScore: h.searchScore,
    types: Object.values(h).map(v => typeof v),
  }));
}

beforeAll(async () => {
  seed();
  const port = await freePort();
  base = `http://127.0.0.1:${port}`;
  serve = Bun.spawn([process.execPath, "src/clawmem.ts", "serve", "--host", "127.0.0.1", "--port", String(port)], {
    cwd: REPO_ROOT, env: cliEnv(), stdout: "pipe", stderr: "pipe",
  });
  await waitHealthy();
}, 30_000);

afterAll(async () => {
  if (serve) {
    serve.kill();
    await serve.exited;
  }
  rmSync(ROOT, { recursive: true, force: true });
});

describe("clawmem serve POST /query (vn4rz.123)", () => {
  test("an unknown collection answers 4xx JSON and the server keeps serving", async () => {
    const res = await post({ query: "zanzibar protocol", collection: "no-such-collection", limit: 3 });
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
    const body = await res.json() as { error?: string };
    expect(body.error).toContain("no-such-collection");
    // The process is still alive and still answers both routes.
    expect(serve!.exitCode).toBeNull();
    expect((await fetch(`${base}/health`)).status).toBe(200);
    const again = await post({ query: "zanzibar protocol", collection: "alpha", limit: 3 });
    expect(again.status).toBe(200);
  }, 30_000);

  test("collection is optional: absent => unscoped, same hits as `clawmem query` with no -c", async () => {
    const res = await post({ query: "zanzibar protocol", limit: 5 });
    expect(res.status).toBe(200);
    const served = await res.json() as Array<{ file: string }>;
    expect(served.map(h => h.file).sort()).toEqual(["alpha/zanzibar-alpha.md", "beta/zanzibar-beta.md"]);
    expect(stable(served)).toEqual(stable(await cliQuery(["zanzibar protocol", "-n", "5"])));
  }, 30_000);

  test("a scoped request matches `clawmem query -c <collection> --json` shape and hits", async () => {
    const res = await post({ query: "zanzibar protocol", collection: "beta", limit: 5 });
    expect(res.status).toBe(200);
    const served = await res.json() as Array<{ file: string }>;
    expect(served.map(h => h.file)).toEqual(["beta/zanzibar-beta.md"]);
    const cli = await cliQuery(["zanzibar protocol", "-n", "5", "-c", "beta"]);
    expect(stable(served)).toEqual(stable(cli));
    expect(Object.keys((served as unknown[])[0] as object)).toEqual(
      ["file", "title", "score", "searchScore", "recencyScore", "contentType"],
    );
  }, 30_000);

  test("session_id applies that session's focus for the one request, then is restored", async () => {
    writeFileSync(join(FOCUS_ROOT, `${SESSION}.focus`), "alpha\n");
    try {
      const focused = await post({ query: "zanzibar protocol", limit: 5, session_id: SESSION });
      expect(focused.status).toBe(200);
      expect((await focused.json() as Array<{ file: string }>).map(h => h.file)).toEqual(["alpha/zanzibar-alpha.md"]);
      // Next request carries no session: unscoped again (the env override did not leak).
      const plain = await post({ query: "zanzibar protocol", limit: 5 });
      expect((await plain.json() as Array<{ file: string }>).map(h => h.file).sort())
        .toEqual(["alpha/zanzibar-alpha.md", "beta/zanzibar-beta.md"]);
      // An explicit collection still beats the session focus (CLI precedence).
      const scoped = await post({ query: "zanzibar protocol", limit: 5, session_id: SESSION, collection: "beta" });
      expect((await scoped.json() as Array<{ file: string }>).map(h => h.file)).toEqual(["beta/zanzibar-beta.md"]);
    } finally {
      rmSync(join(FOCUS_ROOT, `${SESSION}.focus`), { force: true });
    }
  }, 30_000);

  test("a prompt that starts with a dash is a query, not a CLI option", async () => {
    // Parsed as options, "-n zanzibar" would set limit=NaN and answer an EMPTY list.
    const res = await post({ query: "-n zanzibar protocol", limit: 5 });
    expect(res.status).toBe(200);
    expect((await res.json() as unknown[]).length).toBeGreaterThan(0);
    expect(serve!.exitCode).toBeNull();
  }, 30_000);
});
