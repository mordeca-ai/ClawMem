#!/usr/bin/env bun
/**
 * `bun src/pg/cli.ts <migrate|reindex|status>` — the PG-path entry point
 * (master-harness-vn4rz.7).
 *
 * Deliberately a SEPARATE entry point from src/clawmem.ts: the PG path is
 * additive and must not be reachable by accident from the sqlite CLI while both
 * writers coexist (ADR-0162 §6).
 */

import { applyMigrations } from "./migrate.ts";
import { closePool, withClient } from "./client.ts";
import { resolvePgConfig } from "./config.ts";
import { isVault, type Vault } from "./vaults.ts";
import { reindex } from "./reindex.ts";
import { assertSchemaGeometry, getVecModels } from "./write.ts";

function flagValue(argv: string[], name: string): string | undefined {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : undefined;
}

/**
 * Which vault the command targets (master-harness-0ynkd). Defaults to "sfw" so
 * every existing invocation is byte-identical; `--vault nsfw` reads the
 * CLAWMEM_PG_NSFW_* namespace instead. Deliberately NOT inferred from anything:
 * pointing a migration at the wrong database is exactly the class of mistake
 * this bead is closing, so the operator says it out loud.
 */
function vaultFlag(argv: string[]): Vault {
  const raw = flagValue(argv, "--vault");
  if (raw === undefined) return "sfw";
  if (!isVault(raw)) {
    throw new Error(`--vault must be "sfw" or "nsfw", got ${JSON.stringify(raw)}`);
  }
  return raw;
}

async function main() {
  const [cmd, ...argv] = process.argv.slice(2);
  const vault = vaultFlag(argv);
  const cfg = resolvePgConfig(vault);

  switch (cmd) {
    case "migrate": {
      const r = await applyMigrations(undefined, vault);
      console.log(`vault:   ${vault}`);
      console.log(`target:  ${cfg.safeLabel}`);
      console.log(`applied: ${r.applied.length ? r.applied.join(", ") : "(none)"}`);
      console.log(`already: ${r.skipped.length ? r.skipped.join(", ") : "(none)"}`);
      break;
    }
    case "status": {
      await withClient(vault, async c => {
        const dim = await assertSchemaGeometry(c);
        const models = await getVecModels(c);
        const { rows } = await c.query<{ collection: string; n: string }>(
          "SELECT collection, count(*)::text AS n FROM documents WHERE active GROUP BY 1 ORDER BY 1",
        );
        console.log(`vault:      ${vault}`);
        console.log(`target:     ${cfg.safeLabel}`);
        console.log(`embed dim:  ${dim}`);
        console.log(`vec models: ${models.length ? models.join(", ") : "(none embedded yet)"}`);
        for (const r of rows) console.log(`  ${r.collection.padEnd(28)} ${r.n.padStart(7)}`);
      });
      break;
    }
    case "reindex": {
      const cols = flagValue(argv, "--collection");
      const limitRaw = flagValue(argv, "--limit");
      const stats = await reindex({
        collections: cols ? cols.split(",").map(s => s.trim()) : undefined,
        limit: limitRaw ? Number(limitRaw) : undefined,
        skipEmbed: argv.includes("--no-embed"),
        embedBatchSize: argv.includes("--batch-size") ? Number(flagValue(argv, "--batch-size")) : undefined,
        onProgress: m => console.log(m),
      });
      for (const s of stats) {
        console.log(
          `${s.collection}: ${s.documentsWritten} docs written, ` +
          `${s.fragmentsEmbedded} embedded, ${s.embedFailures} embed failures, ` +
          `${(s.wallClockMs / 1000).toFixed(1)}s`,
        );
        // Reported HERE, in the reindex summary, and deliberately NOT in
        // `clawmem status` / `pg status` (master-harness-vn4rz.34). Both status
        // verbs are pure reads of stored row state; "this document's frontmatter
        // did not parse" is a SCAN-TIME fact with no column behind it. Surfacing
        // it there would mean either a schema migration to persist it, or turning
        // a cheap status read into a full filesystem walk of every collection.
        // Neither is warranted for a signal whose natural moment is the scan that
        // produced it — which is also why contentTypeRetagBacklog below reports
        // here rather than inventing a second channel.
        const fmFailures = Object.entries(s.frontmatterParseFailures);
        if (fmFailures.length > 0) {
          console.log(
            `  UNPARSEABLE FRONTMATTER: ${fmFailures.length} document(s) — metadata ` +
            `(title/description/tags/domain/workstream) was DROPPED, content_type was ` +
            `INFERRED from the filename, and the raw YAML was embedded as body prose:`,
          );
          for (const [path, msg] of fmFailures) console.log(`    ${path}: ${msg}`);
        }
        const backlog = Object.entries(s.contentTypeRetagBacklog);
        if (backlog.length > 0) {
          console.log(
            `  retag backlog (content_type outside the closed ADR-0058 enum, ` +
            `stored as 'unknown' with the raw value in content_type_raw): ` +
            backlog.map(([k, v]) => `${k}=${v}`).join(", "),
          );
        }
      }
      break;
    }
    default:
      console.error(
        "usage: bun src/pg/cli.ts <migrate|status|reindex> [--vault sfw|nsfw] " +
        "[--collection a,b] [--limit N] [--no-embed]",
      );
      process.exit(2);
  }
  await closePool();
}

main().catch(async e => {
  console.error(e instanceof Error ? e.message : String(e));
  await closePool();
  process.exit(1);
});
