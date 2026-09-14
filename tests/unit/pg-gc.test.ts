/**
 * Content GC vault fence — unit tier (master-harness-vn4rz.49).
 *
 * No database. Proves the GC refuses the nsfw vault BEFORE any configuration
 * is resolved or any pool is opened: the refusal is ContentGcVaultRefusedError
 * even with the nsfw connection env deliberately UNSET, where any attempt to
 * resolve config would instead have thrown VaultNotConfiguredError.
 */

import { describe, it, expect } from "bun:test";
import { join } from "path";
import { gcOrphanedContent } from "../../src/pg/write.ts";
import { ContentGcVaultRefusedError } from "../../src/pg/errors.ts";

const NSFW_ENV_KEYS = [
  "CLAWMEM_PG_NSFW_URL", "CLAWMEM_PG_NSFW_HOST", "CLAWMEM_PG_NSFW_DATABASE",
  "CLAWMEM_PG_NSFW_USER", "CLAWMEM_PG_NSFW_PASSWORD",
];

describe("content GC vault fence", () => {
  it("REFUSES the nsfw vault before resolving config or opening a pool", async () => {
    const saved = Object.fromEntries(NSFW_ENV_KEYS.map(k => [k, process.env[k]]));
    for (const k of NSFW_ENV_KEYS) delete process.env[k];
    try {
      let err: unknown;
      try {
        await gcOrphanedContent({ vault: "nsfw" });
      } catch (e) {
        err = e;
      }
      expect(err).toBeInstanceOf(ContentGcVaultRefusedError);
      expect((err as Error).message).toContain("SFW-only");
    } finally {
      for (const [k, v] of Object.entries(saved)) if (v !== undefined) process.env[k] = v;
    }
  });

  it("REFUSES the nsfw vault in dry-run mode too (a count is still a connection)", async () => {
    await expect(gcOrphanedContent({ vault: "nsfw", dryRun: true }))
      .rejects.toBeInstanceOf(ContentGcVaultRefusedError);
  });

  it("the CLI `gc --vault nsfw` exits non-zero with the refusal", () => {
    const cli = join(import.meta.dir, "../../src/pg/cli.ts");
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined && !k.startsWith("CLAWMEM_PG")) env[k] = v;
    }
    const p = Bun.spawnSync(["bun", cli, "gc", "--vault", "nsfw", "--dry-run"], { env });
    expect(p.exitCode).toBe(1);
    expect(p.stderr.toString()).toContain("Refusing content GC");
  });
});
