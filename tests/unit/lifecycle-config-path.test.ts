/**
 * The lifecycle policy reads the file the collection list reads, and a policy alone never makes
 * the SessionStart hook archive (master-harness-xbgy1.22).
 *
 * Before this, loadVaultConfig opened config.yaml only while collections.ts fell back to
 * index.yml, so a host on index.yml could not configure a policy at all.
 */

import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfig, saveConfig } from "../../src/collections.ts";
import { clearConfigCache, loadVaultConfig } from "../../src/config.ts";
import { stalenessCheck } from "../../src/hooks/staleness-check.ts";
import { createStore, insertContent, insertDocument, type Store } from "../../src/store.ts";

const policyLines = (days: number, extra: string[] = []): string[] => [
  "collections: {}",
  "lifecycle:",
  `  archive_after_days: ${days}`,
  "  type_overrides: {}",
  "  dry_run: false",
  ...extra,
];

let cfgDir: string;
let saved: string | undefined;

beforeEach(() => {
  cfgDir = mkdtempSync(join(tmpdir(), "clawmem-lifecycle-path-"));
  saved = process.env.CLAWMEM_CONFIG_DIR;
  process.env.CLAWMEM_CONFIG_DIR = cfgDir;
  clearConfigCache();
});

afterEach(() => {
  if (saved === undefined) delete process.env.CLAWMEM_CONFIG_DIR;
  else process.env.CLAWMEM_CONFIG_DIR = saved;
  clearConfigCache();
  rmSync(cfgDir, { recursive: true, force: true });
});

const write = (name: string, lines: string[]): void =>
  writeFileSync(join(cfgDir, name), lines.join("\n") + "\n");

describe("loadVaultConfig picks the same file as the collection list", () => {
  it("reads the policy from index.yml when there is no config.yaml", () => {
    write("index.yml", policyLines(7));
    expect(loadVaultConfig().lifecycle?.archive_after_days).toBe(7);
  });

  it("prefers config.yaml, and an index.yml policy does not leak through it", () => {
    write("index.yml", policyLines(7));
    write("config.yaml", ["collections: {}"]);
    expect(loadVaultConfig().lifecycle).toBeUndefined();
    clearConfigCache();
    write("config.yaml", policyLines(9));
    expect(loadVaultConfig().lifecycle?.archive_after_days).toBe(9);
  });

  it("reports no policy when neither file has one", () => {
    expect(loadVaultConfig().lifecycle).toBeUndefined();
    write("index.yml", ["collections: {}"]);
    clearConfigCache();
    expect(loadVaultConfig().lifecycle).toBeUndefined();
  });

  it("keeps the opt-in through a collection rewrite", () => {
    write("index.yml", policyLines(7, ["  session_start_archive: true"]));
    saveConfig(loadConfig());
    clearConfigCache();
    expect(loadVaultConfig().lifecycle?.session_start_archive).toBe(true);
  });
});

describe("the SessionStart hook archives only when the policy opts in", () => {
  let store: Store;

  const inactive = (): number =>
    (store.db.prepare("SELECT COUNT(*) AS n FROM documents WHERE active = 0").get() as { n: number })
      .n;

  beforeEach(() => {
    store = createStore(":memory:");
    const iso = new Date(Date.now() - 400 * 86_400_000).toISOString();
    insertContent(store.db, "h-stale", "stale body", iso);
    insertDocument(store.db, "test", "stale.md", "T", "h-stale", iso, iso);
  });

  it("a live policy alone archives nothing at session start, and a sweep still sees the doc", async () => {
    write("index.yml", policyLines(1));
    await stalenessCheck(store, { sessionId: "s", hookEventName: "SessionStart" });
    expect(inactive()).toBe(0);
    expect(store.getArchiveCandidates(loadVaultConfig().lifecycle!).length).toBe(1);
  });

  it("the opt-in restores the archive, so the check above can fail", async () => {
    write("index.yml", policyLines(1, ["  session_start_archive: true"]));
    await stalenessCheck(store, { sessionId: "s", hookEventName: "SessionStart" });
    expect(inactive()).toBe(1);
  });

  it("the opt-in still honours dry_run", async () => {
    write("index.yml", [
      "collections: {}",
      "lifecycle:",
      "  archive_after_days: 1",
      "  dry_run: true",
      "  session_start_archive: true",
    ]);
    await stalenessCheck(store, { sessionId: "s", hookEventName: "SessionStart" });
    expect(inactive()).toBe(0);
  });
});
