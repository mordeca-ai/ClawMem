/**
 * Index-time origin classifier (master-harness-vn4rz.66; Review Hub a7981735).
 *
 * Pure decisions plus the no-database `origin-classify` verb. All text below is
 * SYNTHETIC fixture text written for this test — never a real transcript.
 *
 * Covers: every PRIVATE signal fires on a representative reference; ordinary
 * session text and segment near-misses stay quiet; AMBIGUOUS signals report but
 * never route; the session (not the chunk) is the unit of routing; session keys
 * come from frontmatter when present; and the CLI verb's output is content-free.
 */

import { describe, it, expect, afterAll } from "bun:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import {
  AMBIGUOUS_BOUNDARY_SIGNALS,
  CONTENT_CLASSIFIED_COLLECTIONS,
  PRIVATE_BOUNDARY_SIGNALS,
  classifySessions,
  classifyText,
  isContentClassified,
  sessionKeyFor,
} from "../../src/pg/classify.ts";
import { hashContent } from "../../src/indexer.ts";

const dirs: string[] = [];
afterAll(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
});

describe("signal lists", () => {
  it("only monchujo is content-classified", () => {
    expect([...CONTENT_CLASSIFIED_COLLECTIONS]).toEqual(["monchujo"]);
    expect(isContentClassified("monchujo")).toBe(true);
    expect(isContentClassified("docs")).toBe(false);
    expect(isContentClassified("bj-corpus")).toBe(false);
  });

  it("keeps the textual shape the master-harness isolation check parses (no backslashes)", () => {
    for (const s of [...PRIVATE_BOUNDARY_SIGNALS, ...AMBIGUOUS_BOUNDARY_SIGNALS]) {
      expect(s.pattern.includes("\\")).toBe(false);
      expect(() => new RegExp(s.pattern)).not.toThrow();
    }
    expect(PRIVATE_BOUNDARY_SIGNALS.map(s => s.id)).toEqual([
      "nsfw-ops", "personal", "comfyui-nsfw", "comfyui-runs-nsfw", "openclaw-workspace",
    ]);
    expect(AMBIGUOUS_BOUNDARY_SIGNALS.map(s => s.id)).toEqual([
      "job-search", "private-financial", "private-household", "fub", "eval-hosho",
      "household-consumer-private",
    ]);
  });
});

describe("classifyText — PRIVATE positives", () => {
  it.each([
    ["nsfw-ops", "$ cat ~/claude/master-harness/intelligence/operations/nsfw/x.md"],
    ["nsfw-ops", "Read intelligence/operations/nsfw"],
    ["personal", "opened intelligence/personal/bj/notes.md for the summary"],
    ["personal", "ls intelligence/personal"],
    ["comfyui-nsfw", "workflow at agents/skills/comfyui/nsfw/"],
    ["comfyui-runs-nsfw", "see agents/skills/comfyui/evals/imagegen/runs-nsfw/run-01"],
    ["openclaw-workspace", "Read /home/bj/.openclaw/workspace/SOUL.md"],
    ["openclaw-workspace", "cd ~/.openclaw/workspace-utamaro/"],
  ])("%s fires on: %s", (id, text) => {
    expect(classifyText(text).private).toContain(id);
  });

  it("matches inside frontmatter too (cwd / source_path keys count)", () => {
    const raw = "---\ncwd: /home/bj/.openclaw/workspace\nsession_id: s1\n---\n\nhello\n";
    expect(classifyText(raw).private).toEqual(["openclaw-workspace"]);
  });

  it("[.]openclaw/workspace is DELIBERATELY suffix-open: workspaces-config matches (conservative)", () => {
    // Documented in classify.ts: every sibling under .openclaw/workspace* is a
    // persona workspace or its config, so over-matching is the safe direction.
    expect(classifyText("edit ~/.openclaw/workspaces-config.json").private).toEqual(["openclaw-workspace"]);
  });
});

describe("classifyText — NEGATIVES", () => {
  it("ordinary session text matches nothing", () => {
    const ordinary = [
      "bd ready",
      "bd update master-harness-abc1 --status=in_progress",
      "============================= test session starts ==============================",
      "tests/test_tools.py::test_parse PASSED                                    [ 50%]",
      "========================= 12 passed, 1 skipped in 0.42s =========================",
      "Read intelligence/operations/handoff-x.md and intelligence/technical/notes.md",
      "git diff --stat agents/skills/comfyui/README.md",
      "cat ~/.openclaw/config.yaml",
    ].join("\n");
    expect(classifyText(ordinary)).toEqual({ private: [], ambiguous: [] });
  });

  it.each([
    "intelligence/personalize",
    "intelligence/personal_notes",
    "intelligence/personal-old/x.md",
    "xintelligence/personal/",
    "myintelligence/operations/nsfw",
    "intelligence/operations/nsfw2/x.md",
    "intelligence/operations/nsfw_draft",
    "agents/skills/comfyui/nsfwish/",
    "agents/skills/comfyui/evals/imagegen/runs-nsfw-old",
    "notopenclaw/workspace",
    "projects/job-searching",
  ])("near-miss stays quiet: %s", text => {
    expect(classifyText(text)).toEqual({ private: [], ambiguous: [] });
  });
});

describe("classifyText — AMBIGUOUS tier reports, never routes", () => {
  it.each([
    ["job-search", "ls projects/job-search/"],
    ["private-financial", "cat private/financial/2026/statement.md"],
    ["private-household", "private/household/plan.md"],
    ["fub", "projects/fub/export.csv"],
    ["eval-hosho", "intelligence/technical/eval-hosho/run.md"],
    ["household-consumer-private", "tools/household-consumer/private/x.json"],
  ])("%s fires on: %s", (id, text) => {
    const c = classifyText(text);
    expect(c.ambiguous).toContain(id);
    expect(c.private).toEqual([]);
  });

  it("an ambiguous-only session stays sfw with its ambiguous ids recorded", () => {
    const v = classifySessions([
      { rel: "sA__c00.md", raw: "ran pytest" },
      { rel: "sA__c01.md", raw: "cat private/financial/x.md and projects/fub/y.csv" },
    ]);
    expect(v.get("sA")).toEqual({ vault: "sfw", private: [], ambiguous: ["fub", "private-financial"] });
  });
});

describe("classifySessions — the session is the unit of routing", () => {
  it("one positive chunk (c03) routes every chunk c00..c05; another session stays sfw", () => {
    const files = [];
    for (let i = 0; i < 6; i++) {
      files.push({
        rel: `2026/09/sess-pos__c0${i}.md`,
        raw: i === 3 ? "Read intelligence/personal/bj/notes.md" : `ordinary chunk ${i}`,
      });
    }
    files.push({ rel: "2026/09/sess-ctl__c00.md", raw: "bd ready" });
    files.push({ rel: "2026/09/sess-ctl__c01.md", raw: "pytest -q" });
    const v = classifySessions(files);
    expect(v.size).toBe(2);
    expect(v.get("sess-pos")).toEqual({ vault: "nsfw", private: ["personal"], ambiguous: [] });
    expect(v.get("sess-ctl")).toEqual({ vault: "sfw", private: [], ambiguous: [] });
  });

  it("groups by frontmatter session_id across differently named files", () => {
    const v = classifySessions([
      { rel: "a.md", raw: "x", sessionId: "S-1" },
      { rel: "b.md", raw: "cat intelligence/operations/nsfw/q.md", sessionId: "S-1" },
      { rel: "c.md", raw: "y", sessionId: "S-2" },
    ]);
    expect(v.get("S-1")!.vault).toBe("nsfw");
    expect(v.get("S-2")!.vault).toBe("sfw");
  });
});

describe("sessionKeyFor", () => {
  it("prefers a non-empty frontmatter session_id", () => {
    expect(sessionKeyFor("2026/09/abc__c03.md", "11111111-2222-3333-4444-555555555555"))
      .toBe("11111111-2222-3333-4444-555555555555");
  });
  it("falls back to the basename with __c<digits>.md stripped", () => {
    expect(sessionKeyFor("2026/09/abc__c03.md")).toBe("abc");
    expect(sessionKeyFor("2026/09/abc__c117.md", null)).toBe("abc");
    expect(sessionKeyFor("abc__c00.md", "   ")).toBe("abc");
  });
  it("strips a bare .md when there is no chunk suffix", () => {
    expect(sessionKeyFor("2026/09/single-session.md")).toBe("single-session");
    expect(sessionKeyFor("x/abc__cX.md")).toBe("abc__cX");
  });
});

describe("origin-classify CLI verb — no database, content-free output", () => {
  const BODY_MARKER = "ZZQ-SYNTHETIC-BODY-MARKER";
  const TITLE_MARKER = "ZZQ Synthetic Title Marker";

  function fixture(): { cfg: string; root: string; files: Record<string, string> } {
    const root = mkdtempSync(join(tmpdir(), "clawmem-origin-classify-root-"));
    const cfg = mkdtempSync(join(tmpdir(), "clawmem-origin-classify-cfg-"));
    dirs.push(root, cfg);
    mkdirSync(join(root, "2026", "09"), { recursive: true });
    const files: Record<string, string> = {};
    const mk = (sid: string, i: number, extra: string) =>
      `---\nsession_id: ${sid}\nts: 2026-09-0${i + 1}T00:00:00Z\ncontent_type: session-transcript\n---\n\n` +
      `# ${TITLE_MARKER}\n\n${BODY_MARKER} ${extra}\n`;
    for (let i = 0; i < 3; i++) {
      files[`2026/09/pos__c0${i}.md`] = mk("sess-pos", i, i === 1 ? "cat intelligence/operations/nsfw/x.md" : "");
    }
    files["2026/09/ctl__c00.md"] = mk("sess-ctl", 0, "bd ready");
    files["2026/09/amb__c00.md"] = mk("sess-amb", 0, "cat private/household/plan.md");
    for (const [rel, raw] of Object.entries(files)) writeFileSync(join(root, rel), raw);
    writeFileSync(
      join(cfg, "index.yml"),
      `collections:\n  monchujo:\n    path: ${root}\n    pattern: "**/*.md"\n`,
    );
    return { cfg, root, files };
  }

  function run(cfg: string, args: string[]) {
    return Bun.spawnSync([process.execPath, "src/pg/cli.ts", "origin-classify", ...args], {
      cwd: join(import.meta.dir, "..", ".."),
      env: {
        ...process.env,
        CLAWMEM_CONFIG_DIR: cfg,
        // An unreachable database: if the verb touched PG it would fail here.
        CLAWMEM_PG_URL: "postgres://nobody@127.0.0.1:1/clawmem_none",
        CLAWMEM_PG_NSFW_URL: "postgres://nobody@127.0.0.1:1/clawmem_none_nsfw",
      },
      stdout: "pipe",
      stderr: "pipe",
    });
  }

  it("--json reports per-session verdicts, chunk hashes and totals without any text", () => {
    const { cfg, root, files } = fixture();
    const p = run(cfg, ["--collection", "monchujo", "--json"]);
    const out = p.stdout.toString();
    expect(p.exitCode).toBe(0);
    const r = JSON.parse(out);
    expect(r.collection).toBe("monchujo");
    const byKey = Object.fromEntries(r.sessions.map((s: { session: string }) => [s.session, s]));
    expect(byKey["sess-pos"].vault).toBe("nsfw");
    expect(byKey["sess-pos"].docs).toBe(3);
    expect(byKey["sess-pos"].private).toEqual(["nsfw-ops"]);
    expect(byKey["sess-pos"].hashes).toEqual(
      ["2026/09/pos__c00.md", "2026/09/pos__c01.md", "2026/09/pos__c02.md"].map(f => hashContent(files[f]!)),
    );
    expect(byKey["sess-pos"].chunks).toEqual(["pos__c00.md", "pos__c01.md", "pos__c02.md"]);
    expect(byKey["sess-ctl"]).toMatchObject({ vault: "sfw", docs: 1, private: [], ambiguous: [] });
    expect(byKey["sess-amb"]).toMatchObject({ vault: "sfw", private: [], ambiguous: ["private-household"] });
    expect(r.totals).toEqual({
      files: 5, sessions: 3, nsfwSessions: 1, sfwSessions: 2, ambiguousSessions: 1, nsfwDocs: 3, sfwDocs: 2,
    });
    // CONTENT-FREE: no body text, title, matched path or directory.
    for (const forbidden of [BODY_MARKER, TITLE_MARKER, "intelligence/operations/nsfw", "private/household", root, "2026/09"]) {
      expect(out.includes(forbidden)).toBe(false);
    }
  });

  it("text mode is content-free too", () => {
    const { cfg, root } = fixture();
    const p = run(cfg, ["--collection", "monchujo"]);
    const out = p.stdout.toString();
    expect(p.exitCode).toBe(0);
    expect(out).toContain("sess-pos  vault=nsfw docs=3 private=nsfw-ops ambiguous=-");
    expect(out).toContain("monchujo: 5 file(s), 3 session(s) -- nsfw 1 session(s) / 3 doc(s)");
    for (const forbidden of [BODY_MARKER, TITLE_MARKER, "intelligence/operations/nsfw", root]) {
      expect(out.includes(forbidden)).toBe(false);
    }
  });
});
