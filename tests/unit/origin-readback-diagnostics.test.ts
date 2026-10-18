/**
 * Content-free read-back mismatch diagnostics (master-harness-vn4rz.114).
 *
 * The vn4rz.109 message printed "hash expected 0458e203 got 0458e203" for a
 * 2026-10-18 recurrence, which could not tell stale values from missing keys or
 * mistyped values. These tests pin the replacement: every guarded column gets
 * presence, type, constructor, length and equality, hex columns get both
 * prefixes and the first differing index, and no title, session id or document
 * text ever reaches the message or the structured log line.
 */

import { describe, expect, it } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { hashContent } from "../../src/indexer.ts";
import {
  checkOriginWriteReadBack,
  rereadSource,
  describeOriginWriteReadBack,
  formatOriginReadBackDiagnostics,
  originReadBackLogLine,
  OriginWriteReadBackError,
  summarizeReadBackResult,
  type ExpectedOriginWrite,
  type ReadBackFieldDiagnostic,
} from "../../src/pg/origin.ts";

const SECRET_TITLE = "Zebra Quorum Title 7781";
const SECRET_SESSION = "sess-QUOKKA-55102";
const SECRET_DOC = "the transcript body nobody may print 4471";
const HASH = "0458e203" + "ab".repeat(28);

const expected: ExpectedOriginWrite = {
  path: "records/0000aaaa__c06.md",
  title: SECRET_TITLE,
  hash: HASH,
  authoredAt: new Date("2026-03-15T14:30:00.000Z"),
  sessionId: SECRET_SESSION,
};

function field(fields: ReadBackFieldDiagnostic[], name: string): ReadBackFieldDiagnostic {
  const f = fields.find((x) => x.field === name);
  if (!f) throw new Error(`no diagnostic for ${name}`);
  return f;
}

function assertNoSecrets(text: string): void {
  expect(text).not.toContain(SECRET_TITLE);
  expect(text).not.toContain(SECRET_SESSION);
  expect(text).not.toContain(SECRET_DOC);
  // Never more than the 8-char hash prefix.
  expect(text).not.toContain(HASH.slice(0, 12));
}

describe("describeOriginWriteReadBack", () => {
  it("the 2026-10-18 shape: origin row has only `path`, doc_sha is a String object", () => {
    const contentRow = { hash: HASH, doc_sha: new String(HASH) as unknown as string };
    const originRow = { path: expected.path };
    expect(checkOriginWriteReadBack(expected, contentRow, originRow)).toEqual([
      "title",
      "hash",
      "authored_at",
      "session_id",
      "doc_sha",
    ]);
    const d = describeOriginWriteReadBack(expected, contentRow, originRow);

    expect(field(d.fields, "origin.path")).toMatchObject({
      present: true,
      type: "string",
      equal: true,
      firstDiff: -1,
      len: expected.path.length,
    });
    for (const name of ["origin.title", "origin.hash", "origin.authored_at", "origin.session_id"]) {
      expect(field(d.fields, name)).toMatchObject({
        present: false,
        type: "undefined",
        equal: false,
      });
    }
    expect(field(d.fields, "origin.hash")).toMatchObject({ exp8: "0458e203", got8: "missing" });
    expect(field(d.fields, "content.hash")).toMatchObject({
      present: true,
      type: "string",
      equal: true,
      exp8: "0458e203",
      got8: "0458e203",
      firstDiff: -1,
    });
    // Same text, different type: the case the old message could not show.
    expect(field(d.fields, "content.doc_sha")).toMatchObject({
      present: true,
      type: "object",
      ctor: "String",
      len: 64,
      expLen: 64,
      equal: false,
      exp8: "0458e203",
      got8: "0458e203",
      firstDiff: -1,
    });

    const line = formatOriginReadBackDiagnostics(d);
    expect(line).toContain(
      "origin.hash[has=0 type=undefined len=-/64 eq=0 exp8=0458e203 got8=missing diff=n/a]",
    );
    expect(line).toContain("content.doc_sha[has=1 type=object ctor=String len=64/64 eq=0");
    expect(line).toContain("diff=same-text");
    assertNoSecrets(line);
  });

  it("stale values: a different hash prints both prefixes and the first differing index", () => {
    const stale = "bfbd720a" + "cd".repeat(28);
    const originRow = {
      path: expected.path,
      title: "Some Other Title",
      hash: stale,
      authored_at: new Date("2026-03-15T14:30:05.000Z"),
      session_id: "sess-other",
    };
    const contentRow = { hash: HASH, doc_sha: HASH };
    const d = describeOriginWriteReadBack(expected, contentRow, originRow);
    expect(field(d.fields, "origin.hash")).toMatchObject({
      present: true,
      equal: false,
      exp8: "0458e203",
      got8: "bfbd720a",
      firstDiff: 0,
    });
    expect(field(d.fields, "origin.authored_at")).toMatchObject({
      present: true,
      ctor: "Date",
      equal: false,
      deltaMs: 5000,
    });
    expect(field(d.fields, "origin.title")).toMatchObject({
      present: true,
      type: "string",
      len: 16,
      expLen: SECRET_TITLE.length,
      equal: false,
    });
    const line = formatOriginReadBackDiagnostics(d);
    expect(line).toContain("got8=bfbd720a diff=0");
    expect(line).toContain("dms=5000");
    expect(line).not.toContain("Some Other Title");
    expect(line).not.toContain("sess-other");
    assertNoSecrets(line);
  });

  it("a late difference is located even when the prefixes agree", () => {
    const late = HASH.slice(0, 40) + "f" + HASH.slice(41);
    const d = describeOriginWriteReadBack(
      expected,
      { hash: HASH, doc_sha: late },
      {
        path: expected.path,
        title: SECRET_TITLE,
        hash: HASH,
        authored_at: expected.authoredAt,
        session_id: SECRET_SESSION,
      },
    );
    expect(field(d.fields, "content.doc_sha")).toMatchObject({
      equal: false,
      got8: "0458e203",
      firstDiff: 40,
    });
  });

  it("document text in a hex column prints as non-hex, never as text", () => {
    const d = describeOriginWriteReadBack(
      expected,
      { hash: SECRET_DOC, doc_sha: SECRET_DOC },
      null,
    );
    expect(field(d.fields, "content.hash").got8).toBe("non-hex");
    expect(field(d.fields, "content.doc_sha").got8).toBe("non-hex");
    const line = formatOriginReadBackDiagnostics(d);
    expect(line).not.toContain(SECRET_DOC.slice(0, 8));
    assertNoSecrets(line);
  });

  it("Buffer and null values are typed, not stringified", () => {
    const d = describeOriginWriteReadBack(
      expected,
      { hash: null, doc_sha: Buffer.from(HASH) as unknown as string },
      { path: expected.path, title: null, hash: HASH, authored_at: null, session_id: null },
    );
    expect(field(d.fields, "content.hash")).toMatchObject({
      present: true,
      type: "null",
      got8: "null",
    });
    expect(field(d.fields, "content.doc_sha")).toMatchObject({
      type: "object",
      ctor: "Buffer",
      len: 64,
      equal: false,
      got8: "0458e203",
      firstDiff: -1,
    });
    expect(field(d.fields, "origin.authored_at")).toMatchObject({ type: "null", deltaMs: null });
  });

  it("carries runtime versions", () => {
    const d = describeOriginWriteReadBack(expected, null, null);
    expect(d.runtime.bun).toBe(process.versions.bun ?? null);
    expect(d.runtime.pg).toMatch(/^\d+\.\d+\.\d+/);
    expect(formatOriginReadBackDiagnostics(d)).toContain(`pg=${d.runtime.pg}`);
  });
});

describe("summarizeReadBackResult", () => {
  it("reports rows, command, key names and RowDescription without values", () => {
    const s = summarizeReadBackResult({
      command: "INSERT",
      rowCount: 1,
      rows: [{ path: "p", title: SECRET_TITLE }],
      fields: [
        { name: "path", dataTypeID: 25 },
        { name: "title", dataTypeID: 25 },
      ],
    });
    expect(s).toEqual({
      rows: 1,
      rowCount: 1,
      command: "INSERT",
      keys: ["path", "title"],
      fields: ["path:25", "title:25"],
    });
    expect(JSON.stringify(s)).not.toContain(SECRET_TITLE);
  });

  it("a key that is not a plain identifier prints as its length only", () => {
    const s = summarizeReadBackResult({
      rows: [{ [SECRET_DOC]: 1 }],
      fields: [{ name: SECRET_DOC, dataTypeID: 25 }],
    });
    expect(s!.keys).toEqual([`<non-ident len ${SECRET_DOC.length}>`]);
    expect(s!.fields).toEqual([`<non-ident len ${SECRET_DOC.length}>:25`]);
    expect(summarizeReadBackResult({ rows: [] })).toMatchObject({ rows: 0, keys: [], fields: [] });
  });
});

describe("OriginWriteReadBackError message", () => {
  it("prints exp8 and got8 for every hex field, `missing` when absent (legacy details)", () => {
    const err = new OriginWriteReadBackError("monchujo", expected.path, ["hash", "doc_sha"], {
      expectedHash: HASH,
      contentHash: HASH,
      originHash: undefined,
      docSha: new String(HASH),
    });
    expect(err.message).toContain(
      "content.hash[has=1 type=string len=64/64 eq=1 exp8=0458e203 got8=0458e203",
    );
    expect(err.message).toContain(
      "origin.hash[has=0 type=undefined len=-/64 eq=0 exp8=0458e203 got8=missing",
    );
    expect(err.message).toContain("content.doc_sha[has=1 type=object ctor=String len=64/64 eq=0");
    // The lossy combined fallback is gone.
    expect(err.message).not.toContain("hash expected");
    assertNoSecrets(err.message);
  });

  it("full diagnostics: message and log line carry client, shapes, re-read and versions, no secrets", () => {
    const contentRow = { hash: HASH, doc_sha: HASH };
    const originRow = { path: expected.path };
    const d = describeOriginWriteReadBack(expected, contentRow, originRow, {
      content: summarizeReadBackResult({
        command: "INSERT",
        rowCount: 1,
        rows: [contentRow],
        fields: [
          { name: "hash", dataTypeID: 25 },
          { name: "doc_sha", dataTypeID: 25 },
        ],
      }),
      origin: summarizeReadBackResult({
        command: "INSERT",
        rowCount: 1,
        rows: [originRow],
        fields: [{ name: "path", dataTypeID: 25 }],
      }),
      client: {
        vault: "sfw",
        processID: 4242,
        database: "clawmem_x",
        applicationName: "clawmem-sfw",
        inOwnPool: true,
        inOtherPools: [],
        poolTotal: 2,
        poolIdle: 1,
        poolWaiting: 0,
        activeQuery: false,
        queueDepth: 0,
        readyForQuery: true,
      },
      stored: {
        pathRows: 1,
        exactRows: 1,
        hashRows: 1,
        titleRows: 1,
        authoredRows: 1,
        sessionRows: 1,
        contentOkRows: 1,
        backendPid: 4242,
        database: "clawmem_x",
      },
    });
    const mism = checkOriginWriteReadBack(expected, contentRow, originRow);
    const err = new OriginWriteReadBackError("monchujo", expected.path, mism, { diagnostics: d });
    expect(err.diagnostics).toBe(d);
    expect(err.message).toContain(
      "origin{rows=1 rowCount=1 cmd=INSERT keys=[path] desc=[path:25]}",
    );
    expect(err.message).toContain(
      "client{vault=sfw pid=4242 db=clawmem_x app=clawmem-sfw ownPool=true otherPools=[]",
    );
    expect(err.message).toContain("active=false queue=0 ready=true}");
    expect(err.message).toContain(
      "stored{pathRows=1 exact=1 hash=1 title=1 authored=1 session=1 contentOk=1 serverPid=4242",
    );
    expect(err.message).toContain("bun=");
    assertNoSecrets(err.message);

    const line = originReadBackLogLine("monchujo", expected.path, mism, d);
    const parsed = JSON.parse(line);
    expect(parsed.event).toBe("origin_write_readback_mismatch");
    expect(parsed.mismatches).toEqual(mism);
    expect(parsed.client.processID).toBe(4242);
    expect(parsed.fields).toHaveLength(7);
    assertNoSecrets(line);
  });
});

describe("independent source re-read (the 2026-10-18T08:53Z guard hole)", () => {
  const good = {
    path: expected.path,
    title: SECRET_TITLE,
    hash: HASH,
    authored_at: expected.authoredAt,
    session_id: SECRET_SESSION,
  };
  const content = { hash: HASH, doc_sha: HASH };

  it("every RETURNING column agrees but the source re-read does not -> ['source']", () => {
    // `raw` carried another file's text: every expected value, and the server's
    // sha256(doc), derive from it, so only the second read disagrees.
    const realSource = "f00dfeed" + "11".repeat(28);
    const exp = { ...expected, sourceHash: realSource };
    expect(checkOriginWriteReadBack(exp, content, good)).toEqual(["source"]);
    const d = describeOriginWriteReadBack(exp, content, good, {
      source: { index: 79, fileBytes: 936, rawBytes: 2806, readError: null, matchesRunFirst: true },
    });
    expect(field(d.fields, "source.sha")).toMatchObject({
      equal: false,
      exp8: "0458e203",
      got8: "f00dfeed",
      firstDiff: 0,
    });
    const line = formatOriginReadBackDiagnostics(d);
    expect(line).toContain("source{index=79 fileBytes=936 rawBytes=2806 matchesRunFirst=true}");
    assertNoSecrets(line);
  });

  it("a failed re-read (null) is a mismatch; an equal one and an absent one are not", () => {
    expect(checkOriginWriteReadBack({ ...expected, sourceHash: null }, content, good)).toEqual([
      "source",
    ]);
    expect(checkOriginWriteReadBack({ ...expected, sourceHash: HASH }, content, good)).toEqual([]);
    expect(checkOriginWriteReadBack(expected, content, good)).toEqual([]);
    const d = describeOriginWriteReadBack(expected, content, good);
    expect(d.fields.find((f) => f.field === "source.sha")).toBeUndefined();
  });

  it("rereadSource agrees with hashContent(readFileSync utf-8), invalid UTF-8 included", () => {
    const dir = mkdtempSync(join(tmpdir(), "clawmem-reread-"));
    try {
      const a = join(dir, "a.md");
      writeFileSync(a, "---\ntitle: caf\u00e9 \u2014 \ud83d\ude80\n---\nbody\n", "utf-8");
      const b = join(dir, "b.md");
      writeFileSync(b, Buffer.from([0x2d, 0x2d, 0x2d, 0x0a, 0xff, 0xfe, 0xc3, 0x0a, 0x41]));
      for (const f of [a, b]) {
        const r = rereadSource(f);
        expect(r.error).toBeNull();
        expect(r.hash).toBe(hashContent(readFileSync(f, "utf-8")));
        expect(r.bytes).toBe(readFileSync(f).byteLength);
      }
      const missing = rereadSource(join(dir, "nope.md"));
      expect(missing).toMatchObject({ hash: null, bytes: null });
      expect(missing.error).toBeString();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
