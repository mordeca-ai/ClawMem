import { describe, it, expect, beforeAll } from "bun:test";

/**
 * master-harness-zkjyh — CLI EXIT-CODE CONTRACT.
 *
 * The defect class here is "a gate that cannot fail": several abort paths print a red
 * diagnostic to stderr and then `return` without touching `process.exitCode`, so a
 * caller (`clawmem update --embed`, a systemd timer, a shell `set -e` pipeline) sees
 * exit 0 and treats a refused/partial run as success.
 *
 * Every row below was watched go RED against the unmodified code before the fix landed
 * (see the bead's worker report for the captured pre-fix stderr + exit codes). Each row
 * asserts BOTH the exit code AND a stderr/stdout marker, so a row can never pass by
 * reaching a *different* abort than the one it names — the geometry canary runs before
 * all of these and fails closed with its own exit 1, which is exactly the false
 * attribution the message assertion rules out.
 *
 * Everything runs the REAL CLI as a subprocess (`bun src/clawmem.ts …`) because the
 * contract under test is a process exit code, not a return value.
 */

import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore } from "../../src/store.ts";
import { hashContent } from "../../src/indexer.ts";
import { canaryProbeInputs, canaryMargins, CANARY_ABSOLUTE_MARGIN_FLOOR } from "../../src/canary.ts";
import {
  startFakeEmbedServer,
  fakeEmbedVector,
  CANARY_SAFE_DIMS,
  type FakeEmbedServer,
} from "../helpers/fake-embed-server.ts";

const CLI = new URL("../../src/clawmem.ts", import.meta.url).pathname;
const DIM_A = 512;
const DIM_B = 1024;
const MODEL_A = "fake-embed-a";
const MODEL_B = "fake-embed-b";

type Vault = { dir: string; db: string; configDir: string; docsDir: string };

function newVault(slug: string): Vault {
  const dir = mkdtempSync(join(tmpdir(), `zkjyh-${slug}-`));
  const docsDir = join(dir, "docs");
  mkdirSync(docsDir, { recursive: true });
  // A configured, existing collection keeps doctor's "No collections configured" /
  // "directory not found" checks green so a doctor row isolates the site it names.
  writeFileSync(
    join(dir, "config.yaml"),
    `collections:\n  testcol:\n    path: ${docsDir}\n    pattern: "**/*.md"\n`,
  );
  return { dir, db: join(dir, "index.sqlite"), configDir: dir, docsDir };
}

/** Insert a document + its body. Returns the content hash. */
function seedDoc(vault: Vault, path: string, body: string): string {
  const s = createStore(vault.db);
  const hash = hashContent(body + path);
  const now = new Date().toISOString();
  s.insertContent(hash, body, now);
  s.insertDocument("testcol", path, path, hash, now, now);
  s.db.close();
  return hash;
}

/** Open the vault for a direct mutation (constructing states the CLI refuses to create). */
function withStore<T>(vault: Vault, fn: (s: ReturnType<typeof createStore>) => T): T {
  const s = createStore(vault.db);
  try {
    return fn(s);
  } finally {
    s.db.close();
  }
}

type CliResult = { code: number; stdout: string; stderr: string };

async function runCli(vault: Vault, args: string[], embedUrl?: string): Promise<CliResult> {
  // Scrub every CLAWMEM_* / INDEX_PATH inherited from the developer's shell. Without
  // this the operator's real CLAWMEM_RERANK_URL / CLAWMEM_LLM_URL leak in and `doctor`
  // probes LIVE services — which added an unrelated second issue and destroyed the
  // attribution of the doctor rows (observed on the first baseline run).
  const cleanEnv: Record<string, string> = {};
  for (const [k, val] of Object.entries(process.env)) {
    if (val === undefined) continue;
    if (k.startsWith("CLAWMEM_") || k === "INDEX_PATH") continue;
    cleanEnv[k] = val;
  }
  const proc = Bun.spawn(["bun", CLI, ...args], {
    env: {
      ...cleanEnv,
      INDEX_PATH: vault.db,
      CLAWMEM_CONFIG_DIR: vault.configDir,
      // Never let a missing endpoint fall back to downloading a local GGUF mid-test.
      CLAWMEM_NO_LOCAL_MODELS: "true",
      // The fake server also serves /v1/rerank (golden-set lookup) so `doctor` can
      // reach a genuine zero-issue state for the negative control.
      ...(embedUrl ? { CLAWMEM_EMBED_URL: embedUrl, CLAWMEM_RERANK_URL: embedUrl } : { CLAWMEM_EMBED_URL: "" }),
    },
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  return { code, stdout, stderr };
}

/** Run a body with a fake embed server, guaranteeing teardown. */
async function withServer<T>(
  opts: Parameters<typeof startFakeEmbedServer>[0],
  fn: (srv: FakeEmbedServer) => Promise<T>,
): Promise<T> {
  const srv = startFakeEmbedServer(opts);
  try {
    return await fn(srv);
  } finally {
    srv.stop();
  }
}

// ---------------------------------------------------------------------------
// Rig self-check — the canary runs BEFORE every branch below and fails closed.
// If the fake embedder stopped clearing the canary floors, every row would abort
// at the canary instead of the branch it names. Assert the premise explicitly.
// ---------------------------------------------------------------------------
describe("rig premise: the fake embedder clears the geometry canary", () => {
  for (const dim of [DIM_A, DIM_B]) {
    it(`dim ${dim} clears every canary pair-separation floor`, () => {
      expect(CANARY_SAFE_DIMS).toContain(dim as never);
      const vecs = new Map<string, Float32Array>();
      for (const [id, text] of canaryProbeInputs()) {
        vecs.set(id, new Float32Array(fakeEmbedVector(text, dim)));
      }
      for (const [name, margin] of Object.entries(canaryMargins(vecs))) {
        expect(`${name}=${margin.toFixed(3)}`).toBe(`${name}=${margin.toFixed(3)}`);
        expect(margin).toBeGreaterThan(CANARY_ABSOLUTE_MARGIN_FLOOR);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// (1) The four `embed` abort branches — printed red, returned bare, exited 0.
// ---------------------------------------------------------------------------
describe("(1) embed abort branches exit non-zero", () => {
  it("force + unreachable endpoint → exit 1 (\"Force re-embed aborted\")", async () => {
    const v = newVault("force-abort");
    seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file.\n");
    // Canary probes succeed (so the run reaches the guard); ONLY the dimension probe
    // input fails — precisely the "endpoint unreachable at probe time" branch.
    const r = await withServer(
      { dim: DIM_A, model: MODEL_A, failInputs: t => t === "clawmem dimension probe" },
      srv => runCli(v, ["embed", "--force"], srv.url),
    );
    expect(r.stderr).toContain("Force re-embed aborted: could not reach the embedding endpoint");
    expect(r.code).toBe(1);
  }, 120_000);

  it("dimension changed → exit 1 (\"Embedding dimension changed\")", async () => {
    const v = newVault("dim-changed");
    seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file.\n");
    const first = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    expect(first.code).toBe(0);
    const r = await withServer({ dim: DIM_B, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    expect(r.stderr).toContain(`Embedding dimension changed (${DIM_A} → ${DIM_B})`);
    expect(r.code).toBe(1);
  }, 120_000);

  it("mixed embedding models in the vault → exit 1 (\"mixed embedding models\")", async () => {
    const v = newVault("mixed-models");
    const h1 = seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file.\n");
    const h2 = seedDoc(v, "b.md", "# B\n\nthe sandbox policy uses a syscall filter.\n");
    // The CLI refuses to CREATE this state, so construct it directly — that is the
    // point of the guard: an already-heterogeneous vault must refuse to embed.
    withStore(v, s => {
      const now = new Date().toISOString();
      s.ensureVecTable(DIM_A);
      s.insertEmbedding(h1, 0, 1, new Float32Array(fakeEmbedVector("a", DIM_A)), "m1", now, "full", undefined, "testcol/a.md");
      s.insertEmbedding(h2, 0, 1, new Float32Array(fakeEmbedVector("b", DIM_A)), "m2", now, "full", undefined, "testcol/b.md");
    });
    const r = await withServer({ dim: DIM_A, model: "m1" }, srv => runCli(v, ["embed"], srv.url));
    expect(r.stderr).toContain("Vault already contains mixed embedding models");
    expect(r.code).toBe(1);
  }, 120_000);

  it("model changed at the same dimension → exit 1 (\"Embedding model changed\")", async () => {
    const v = newVault("model-changed");
    seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file.\n");
    const first = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    expect(first.code).toBe(0);
    const r = await withServer({ dim: DIM_A, model: MODEL_B }, srv => runCli(v, ["embed"], srv.url));
    expect(r.stderr).toContain(`Embedding model changed (${MODEL_A} → ${MODEL_B})`);
    expect(r.code).toBe(1);
  }, 120_000);
});

// ---------------------------------------------------------------------------
// (2) doctor must not fail open on an issue class that was never paired with an
//     exitCode assignment. `getVecModels().length > 1` (clawmem.ts §3) is one of
//     the eleven unpaired sites.
// ---------------------------------------------------------------------------
describe("(2) doctor exits non-zero for an UNPAIRED issue class", () => {
  it("mixed embedding models reported by doctor → exit 1", async () => {
    const v = newVault("doctor-mixed");
    seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file and validates its signature.\n");
    seedDoc(v, "b.md", "# B\n\nthe sandbox policy uses a syscall filter so the manifest is unreadable.\n");
    const embed = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    expect(embed.code).toBe(0);
    // Retag ONE persisted vector's model. Everything else in the vault stays healthy,
    // so the ONLY issue doctor can find is the unpaired mixed-models site.
    withStore(v, s => {
      s.db.exec(`UPDATE content_vectors SET model = '${MODEL_B}' WHERE rowid = (SELECT MIN(rowid) FROM content_vectors)`);
    });
    const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["doctor"], srv.url));
    expect(r.stdout).toContain("vault has MIXED models");
    expect(r.stdout).toMatch(/1 issue\(s\) found/);
    expect(r.code).toBe(1);
  }, 180_000);
});

// ---------------------------------------------------------------------------
// (3) partial embed failure must be non-zero by default, exit 0 under --lenient.
// ---------------------------------------------------------------------------
describe("(3) embed with failed fragments", () => {
  const POISON = "POISONFRAGMENT";
  const body = (marker: string) => `# Doc\n\nthe deploy pipeline reads the manifest file ${marker}.\n`;

  it("strict default → exit 1 and the failure count stays in the summary", async () => {
    const v = newVault("partial-strict");
    seedDoc(v, "ok.md", body("cleanly"));
    seedDoc(v, "bad.md", body(POISON));
    const r = await withServer(
      { dim: DIM_A, model: MODEL_A, failInputs: t => t.includes(POISON) },
      srv => runCli(v, ["embed"], srv.url),
    );
    expect(r.stdout).toMatch(/Embedded \d+ documents \(\d+ fragments, [1-9]\d* failed\)/);
    expect(r.stderr).toContain("failed fragment(s)");
    expect(r.code).toBe(1);
  }, 120_000);

  it("--lenient → exit 0 on the same partial failure (back-compat)", async () => {
    const v = newVault("partial-lenient");
    seedDoc(v, "ok.md", body("cleanly"));
    seedDoc(v, "bad.md", body(POISON));
    const r = await withServer(
      { dim: DIM_A, model: MODEL_A, failInputs: t => t.includes(POISON) },
      srv => runCli(v, ["embed", "--lenient"], srv.url),
    );
    expect(r.stdout).toMatch(/Embedded \d+ documents \(\d+ fragments, [1-9]\d* failed\)/);
    expect(r.code).toBe(0);
  }, 120_000);
});

// ---------------------------------------------------------------------------
// (4) vsearch / query zero-result signalling — OPT-IN only. The default must not
//     move: ~20 live consumers run these under `set -e`.
// ---------------------------------------------------------------------------
describe("(4) --fail-on-empty distinguishes 'no match' from 'ran fine'", () => {
  let empty: Vault;
  let populated: Vault;

  beforeAll(async () => {
    empty = newVault("search-empty");
    populated = newVault("search-full");
    seedDoc(populated, "a.md", "# Deploy\n\nthe deploy pipeline reads the manifest file and validates its signature.\n");
    const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(populated, ["embed"], srv.url));
    expect(r.code).toBe(0);
  }, 120_000);

  for (const cmd of ["vsearch", "query"] as const) {
    it(`${cmd}: zero results WITHOUT the flag → exit 0 (default unchanged)`, async () => {
      const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv =>
        runCli(empty, [cmd, "nothing here at all", "--min-score", "0"], srv.url));
      expect(r.stdout).toContain("No results found");
      expect(r.code).toBe(0);
    }, 120_000);

    it(`${cmd}: zero results WITH --fail-on-empty → exit 2`, async () => {
      const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv =>
        runCli(empty, [cmd, "nothing here at all", "--min-score", "0", "--fail-on-empty"], srv.url));
      expect(r.stdout).toContain("No results found");
      expect(r.code).toBe(2);
    }, 120_000);

    it(`${cmd}: results WITH --fail-on-empty → exit 0 (flag is not a blanket failure)`, async () => {
      const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv =>
        runCli(populated, [cmd, "deploy pipeline manifest signature", "--min-score", "0", "--fail-on-empty"], srv.url));
      expect(r.stdout).not.toContain("No results found");
      expect(r.code).toBe(0);
    }, 120_000);
  }
});

// ---------------------------------------------------------------------------
// NEGATIVE CONTROLS — an over-broad fix that makes everything non-zero dies here.
// ---------------------------------------------------------------------------
describe("negative controls: a healthy run still exits 0", () => {
  it("healthy embed → exit 0", async () => {
    const v = newVault("healthy-embed");
    seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file and validates its signature.\n");
    const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    expect(r.stdout).toMatch(/Embedded \d+ documents \(\d+ fragments, 0 failed\)/);
    expect(r.code).toBe(0);
  }, 120_000);

  it("healthy re-embed (no work) → exit 0", async () => {
    const v = newVault("healthy-nowork");
    seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file and validates its signature.\n");
    await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    expect(r.stdout).toContain("All documents already embedded");
    expect(r.code).toBe(0);
  }, 180_000);

  it("healthy doctor (zero issues) → exit 0", async () => {
    const v = newVault("healthy-doctor");
    seedDoc(v, "a.md", "# A\n\nthe deploy pipeline reads the manifest file and validates its signature.\n");
    const embed = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["embed"], srv.url));
    expect(embed.code).toBe(0);
    const r = await withServer({ dim: DIM_A, model: MODEL_A }, srv => runCli(v, ["doctor"], srv.url));
    expect(r.stdout).toContain("All checks passed.");
    expect(r.code).toBe(0);
  }, 180_000);
});
