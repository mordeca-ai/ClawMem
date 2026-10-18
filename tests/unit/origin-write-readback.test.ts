/**
 * Unit tests for origin-load write read-back verification (master-harness-vn4rz.109).
 */

import { describe, expect, it } from "bun:test";
import {
  checkOriginWriteReadBack,
  OriginWriteReadBackError,
  type ExpectedOriginWrite,
  type ContentWriteReadBack,
  type OriginWriteReadBack,
} from "../../src/pg/origin.ts";

describe("checkOriginWriteReadBack", () => {
  const baseExpected: ExpectedOriginWrite = {
    path: "2026/03/session-123.md",
    title: "Session 123: Architecture Discussion",
    hash: "a1b2c3d4e5f67890123456789012345678901234567890123456789012345678",
    authoredAt: new Date("2026-03-15T14:30:00.000Z"),
    sessionId: "sess-abc-123",
  };

  const baseContentRow: ContentWriteReadBack = {
    hash: baseExpected.hash,
    doc_sha: baseExpected.hash,
  };

  const baseOriginRow: OriginWriteReadBack = {
    path: baseExpected.path,
    title: baseExpected.title,
    hash: baseExpected.hash,
    authored_at: new Date("2026-03-15T14:30:00.000Z"),
    session_id: baseExpected.sessionId,
  };

  it("all-match -> []", () => {
    const mismatches = checkOriginWriteReadBack(baseExpected, baseContentRow, baseOriginRow);
    expect(mismatches).toEqual([]);
  });

  describe("each single column mismatch is reported", () => {
    it("reports path mismatch", () => {
      const originRow: OriginWriteReadBack = { ...baseOriginRow, path: "2026/03/wrong-path.md" };
      expect(checkOriginWriteReadBack(baseExpected, baseContentRow, originRow)).toEqual(["path"]);
    });

    it("reports title mismatch", () => {
      const originRow: OriginWriteReadBack = { ...baseOriginRow, title: "Tampered or Stale Title" };
      expect(checkOriginWriteReadBack(baseExpected, baseContentRow, originRow)).toEqual(["title"]);
    });

    it("reports hash mismatch when contentRow.hash differs", () => {
      const contentRow: ContentWriteReadBack = {
        ...baseContentRow,
        hash: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
      };
      expect(checkOriginWriteReadBack(baseExpected, contentRow, baseOriginRow)).toEqual(["hash"]);
    });

    it("reports hash mismatch when originRow.hash differs", () => {
      const originRow: OriginWriteReadBack = {
        ...baseOriginRow,
        hash: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
      };
      expect(checkOriginWriteReadBack(baseExpected, baseContentRow, originRow)).toEqual(["hash"]);
    });

    it("reports hash mismatch once when both contentRow and originRow hashes differ", () => {
      const contentRow: ContentWriteReadBack = {
        ...baseContentRow,
        hash: "1111111111111111111111111111111111111111111111111111111111111111",
      };
      const originRow: OriginWriteReadBack = {
        ...baseOriginRow,
        hash: "2222222222222222222222222222222222222222222222222222222222222222",
      };
      expect(checkOriginWriteReadBack(baseExpected, contentRow, originRow)).toEqual(["hash"]);
    });

    it("reports authored_at mismatch when date differs", () => {
      const originRow: OriginWriteReadBack = {
        ...baseOriginRow,
        authored_at: new Date("2026-03-16T00:00:00.000Z"),
      };
      expect(checkOriginWriteReadBack(baseExpected, baseContentRow, originRow)).toEqual([
        "authored_at",
      ]);
    });

    it("reports authored_at mismatch when expected is Date but got null", () => {
      const originRow: OriginWriteReadBack = { ...baseOriginRow, authored_at: null };
      expect(checkOriginWriteReadBack(baseExpected, baseContentRow, originRow)).toEqual([
        "authored_at",
      ]);
    });

    it("reports authored_at mismatch when expected is null but got Date", () => {
      const exp: ExpectedOriginWrite = { ...baseExpected, authoredAt: null };
      expect(checkOriginWriteReadBack(exp, baseContentRow, baseOriginRow)).toEqual(["authored_at"]);
    });

    it("reports session_id mismatch when values differ", () => {
      const originRow: OriginWriteReadBack = { ...baseOriginRow, session_id: "other-session" };
      expect(checkOriginWriteReadBack(baseExpected, baseContentRow, originRow)).toEqual([
        "session_id",
      ]);
    });

    it("reports session_id mismatch when expected is string but got null", () => {
      const originRow: OriginWriteReadBack = { ...baseOriginRow, session_id: null };
      expect(checkOriginWriteReadBack(baseExpected, baseContentRow, originRow)).toEqual([
        "session_id",
      ]);
    });

    it("reports session_id mismatch when expected is null but got string", () => {
      const exp: ExpectedOriginWrite = { ...baseExpected, sessionId: null };
      expect(checkOriginWriteReadBack(exp, baseContentRow, baseOriginRow)).toEqual(["session_id"]);
    });

    it("reports doc_sha mismatch when contentRow.doc_sha differs from expected.hash", () => {
      const contentRow: ContentWriteReadBack = {
        ...baseContentRow,
        doc_sha: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      };
      expect(checkOriginWriteReadBack(baseExpected, contentRow, baseOriginRow)).toEqual([
        "doc_sha",
      ]);
    });
  });

  it("null authored_at/session_id both sides -> []", () => {
    const exp: ExpectedOriginWrite = {
      ...baseExpected,
      authoredAt: null,
      sessionId: null,
    };
    const originRow: OriginWriteReadBack = {
      ...baseOriginRow,
      authored_at: null,
      session_id: null,
    };
    expect(checkOriginWriteReadBack(exp, baseContentRow, originRow)).toEqual([]);
  });

  it("Date vs equal-instant Date -> []", () => {
    const d1 = new Date("2026-06-01T12:34:56.789Z");
    const d2 = new Date(d1.getTime());
    const exp: ExpectedOriginWrite = { ...baseExpected, authoredAt: d1 };
    const originRow: OriginWriteReadBack = { ...baseOriginRow, authored_at: d2 };
    expect(checkOriginWriteReadBack(exp, baseContentRow, originRow)).toEqual([]);

    // Also verify ISO string with equal millisecond instant
    const originRowString: OriginWriteReadBack = {
      ...baseOriginRow,
      authored_at: d1.toISOString(),
    };
    expect(checkOriginWriteReadBack(exp, baseContentRow, originRowString)).toEqual([]);
  });
});

describe("OriginWriteReadBackError", () => {
  it("formats privacy-safe message without title, session, or doc content", () => {
    const SECRET_TITLE = "TOP_SECRET_TITULAR_DATA_CLASSIFIED";
    const SECRET_SESSION = "SESSION_IDENTIFIER_HIGH_CONFIDENTIALITY_789";

    const err = new OriginWriteReadBackError(
      "test-collection",
      "path/to/transcript.md",
      ["title", "session_id", "hash", "doc_sha"],
      {
        expectedHash: "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
        contentHash: "99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa",
        originHash: "99887766554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa",
        docSha: "554433221100ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766",
      },
    );

    // Names collection, relative path, and mismatching column names
    expect(err.message).toContain("test-collection");
    expect(err.message).toContain("path/to/transcript.md");
    expect(err.message).toContain("title");
    expect(err.message).toContain("session_id");
    expect(err.message).toContain("hash");
    expect(err.message).toContain("doc_sha");

    // Only first 8 hex chars for hash/doc_sha
    expect(err.message).toContain("11223344");
    expect(err.message).toContain("99887766");
    expect(err.message).toContain("55443322");
    // Full hashes must NOT be present
    expect(err.message).not.toContain("11223344556677889900");
    expect(err.message).not.toContain("99887766554433221100");

    // Privacy assertions: title text, session id must NOT appear in error message
    expect(err.message).not.toContain(SECRET_TITLE);
    expect(err.message).not.toContain(SECRET_SESSION);
  });
});
