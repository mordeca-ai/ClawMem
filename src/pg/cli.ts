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
import {
  dropLegacyDocumentRows, dropPartitionsBefore, listPartitions, loadOriginCollection,
} from "./origin.ts";
import { listCollections } from "../collections.ts";

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
        // Two-directional convergence is the DEFAULT (master-harness-vn4rz.41);
        // --no-sweep is the opt-out, never the other way around.
        sweep: !argv.includes("--no-sweep"),
        embedBatchSize: argv.includes("--batch-size") ? Number(flagValue(argv, "--batch-size")) : undefined,
        onProgress: m => console.log(m),
      });
      for (const s of stats) {
        console.log(
          `${s.collection}: ${s.documentsWritten} docs written, ` +
          // The deactivation count is APPENDED AFTER "embed failures" and
          // before the wall-clock on purpose: master-harness's
          // tools/clawmem-pg-reindex matches a PREFIX ending at "embed
          // failures", so an older wrapper keeps parsing a newer summary.
          `${s.fragmentsEmbedded} embedded, ${s.embedFailures} embed failures, ` +
          `${s.documentsDeactivated} deactivated, ` +
          `${(s.wallClockMs / 1000).toFixed(1)}s`,
        );
        // NEVER SILENTLY SKIP. A one-directional run that says nothing about
        // being one-directional is the vn4rz.41 defect wearing a green run as
        // camouflage, so the reason is always printed when there is one.
        if (s.sweepSkippedReason !== null) {
          console.log(`  sweep SKIPPED: ${s.sweepSkippedReason}`);
        }
        // Same channel and same reason as the three reports below: a row the
        // sweep retired must be answerable from the summary, by path.
        if (s.deactivatedPaths.length > 0) {
          console.log(
            `  deactivated: ${s.deactivatedPaths.length} document(s) whose source file ` +
            `was ABSENT from the walk were set active=false (soft — the rows and their ` +
            `vectors are retained, and a file that returns reactivates on the next run):`,
          );
          for (const path of s.deactivatedPaths) console.log(`    ${path}`);
        }
        // Reported HERE, in the reindex summary, and deliberately NOT in
        // `clawmem status` / `pg status` (master-harness-vn4rz.34). Both status
        // verbs are pure reads of stored row state; "this document's frontmatter
        // did not parse" is a SCAN-TIME fact with no column behind it. Surfacing
        // it there would mean either a schema migration to persist it, or turning
        // a cheap status read into a full filesystem walk of every collection.
        // Neither is warranted for a signal whose natural moment is the scan that
        // produced it — which is also why contentTypeRetagBacklog below reports
        // here rather than inventing a second channel.
        // Same channel, same reason as the two reports below: a file the scan
        // silently dropped is as much a defect as one whose parse it silently
        // swallowed (master-harness-vn4rz.7 pass D). Reported even though it is
        // the EXPECTED steady state — the number going to zero on a collection
        // that used to report one is itself the signal.
        const skipped = Object.entries(s.skippedOutOfScope);
        if (skipped.length > 0) {
          console.log(
            `  out of default scope: ${skipped.length} file(s) matched the glob but ` +
            `were NOT indexed (master-harness ADR-0071 — leading '_' segments such as ` +
            `_superseded/ / _reviews/, dotted segments, and EXCLUDED_DIRS):`,
          );
          for (const [path, why] of skipped) console.log(`    ${path}: ${why}`);
        }
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
    // ---------------------------------------------------------------------
    // ORIGIN TIER (master-harness-vn4rz.8) — the ONE sanctioned copy path.
    // Kept as its own verb rather than folded into `reindex` precisely so the
    // copy/reindex boundary is visible at the command line: `reindex` may never
    // copy, `origin-load` may never reindex.
    // ---------------------------------------------------------------------
    case "origin-load": {
      const name = flagValue(argv, "--collection");
      if (!name) throw new Error("origin-load requires --collection <name>");
      const c = listCollections().find(x => x.name === name);
      if (!c) throw new Error(`no collection ${name} in the clawmem index config`);
      const limitRaw = flagValue(argv, "--limit");
      const s = await loadOriginCollection({
        collection: c.name, root: c.path, pattern: c.pattern,
        month: flagValue(argv, "--month"),
        limit: limitRaw ? Number(limitRaw) : undefined,
        skipEmbed: argv.includes("--no-embed"),
        embedBatchSize: argv.includes("--batch-size")
          ? Number(flagValue(argv, "--batch-size")) : undefined,
        vault,
        onProgress: m => console.log(m),
      });
      console.log(
        `${s.collection}: ${s.recordsWritten} records written of ${s.filesSeen} seen, ` +
        `${s.hashesAlreadyEmbedded} hashes already embedded (not re-embedded), ` +
        `${s.fragmentsEmbedded} fragments embedded, ${s.embedFailures} embed failures, ` +
        `${(s.wallClockMs / 1000).toFixed(1)}s`);
      console.log("  per-partition:");
      for (const [k, v] of Object.entries(s.perMonth).sort())
        console.log(`    ${k.padEnd(32)} ${String(v).padStart(7)}`);
      // ALWAYS printed, including when it is zero. The DEFAULT partition is a
      // monitored number, not a catch-all nobody revisits (migration 004,
      // DESIGN DECISION (b)); a reader of a green run has to see the count.
      console.log(
        `  DEFAULT partition (origin_documents_unassigned): ${s.unassigned} record(s) ` +
        `with no resolvable authored_at`);
      for (const p of s.unassignedPaths) console.log(`    ${p}`);
      const backlog = Object.entries(s.recordTypeBacklog);
      console.log(
        `  record_type outside the closed origin vocabulary: ` +
        (backlog.length ? backlog.map(([k, v]) => `${k}=${v}`).join(", ") : "(none)") +
        ` -- stored as 'unknown' with the raw value in record_type_raw`);
      break;
    }
    case "origin-partitions": {
      for (const p of await listPartitions(vault)) {
        console.log(
          `${p.partition_name.padEnd(32)} ${p.is_default ? "DEFAULT" : p.bound_expr}  ` +
          `${(Number(p.total_bytes) / 1048576).toFixed(1)} MiB`);
      }
      break;
    }
    case "origin-drop-legacy": {
      const name = flagValue(argv, "--collection");
      if (!name) throw new Error("origin-drop-legacy requires --collection <name>");
      const apply = argv.includes("--apply");
      const r = await dropLegacyDocumentRows(name, vault, { apply });
      console.log(
        `${name}: ${r.inDocuments} rows in documents, ${r.inOrigin} in origin_documents, ` +
        `${r.missing} unmatched. ${apply ? `DELETED ${r.deleted}` : "DRY RUN (pass --apply)"}`);
      break;
    }
    case "origin-retention": {
      const before = flagValue(argv, "--before");
      if (!before) throw new Error("origin-retention requires --before YYYY-MM-DD");
      const apply = argv.includes("--apply");
      const r = await dropPartitionsBefore(before, vault, { apply });
      console.log(`cutoff ${before} -- DETACH + DROP (never a DELETE sweep)`);
      console.log(`  detach mode: ${r.detachMode} -- ${r.detachModeReason}`);
      console.log(`  ${apply ? "dropped" : "would drop"}: ${r.dropped.join(", ") || "(none)"}`);
      console.log(`  kept:    ${r.kept.join(", ") || "(none)"}`);
      console.log(
        `  refused (DEFAULT partition has no upper bound; dropping it is unbounded ` +
        `data loss, never retention): ${r.refusedDefault.join(", ") || "(none)"}`);
      if (!apply) console.log("  DRY RUN -- pass --apply to execute");
      break;
    }
    default:
      console.error(
        "usage: bun src/pg/cli.ts <migrate|status|reindex|origin-load|origin-partitions|" +
        "origin-drop-legacy|origin-retention> [--vault sfw|nsfw] " +
        "[--collection a,b] [--limit N] [--no-embed] [--no-sweep] [--month YYYY-MM] " +
        "[--before DATE] [--apply]",
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
