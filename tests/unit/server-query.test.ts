import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { QUERY_MAX_CHARS, QueryRequestError, startServer, type QueryJsonHit } from "../../src/server.ts";
import type { Store } from "../../src/store.ts";

const hit: QueryJsonHit = {
  file: "openclaw-workspace/AGENTS.md",
  title: "AGENTS",
  score: 0.72,
  searchScore: 0.61,
  recencyScore: 0.4,
  contentType: "note",
};

describe("POST /query", () => {
  const calls: Array<[string, string | undefined, number, string | undefined]> = [];
  let server: ReturnType<typeof startServer>;
  let base: string;

  beforeAll(() => {
    server = startServer({} as Store, 0, "127.0.0.1", async (query, collection, limit, sessionId) => {
      calls.push([query, collection, limit, sessionId]);
      if (collection === "unknown-collection") throw new QueryRequestError("Unknown collection 'unknown-collection'.");
      if (query === "explode") throw new Error("pipeline fault");
      return [hit];
    });
    base = `http://127.0.0.1:${server.port}`;
  });
  afterAll(() => server.stop());

  const post = (body: unknown) => fetch(`${base}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  test("returns the CLI query JSON shape and passes a bounded collection scope", async () => {
    const response = await post({ query: "relay policy", collection: "openclaw-workspace", limit: 3 });
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual([hit]);
    expect(calls.at(-1)).toEqual(["relay policy", "openclaw-workspace", 3, undefined]);
  });

  // master-harness-vn4rz.123: collection is optional (absent => unscoped) and session_id is
  // forwarded to the handler for the one request.
  test("an absent collection passes through as unscoped, with the session id", async () => {
    const response = await post({ query: "relay policy", session_id: "0f2c4d6e-a1b2" });
    expect(response.status).toBe(200);
    expect(calls.at(-1)).toEqual(["relay policy", undefined, 3, "0f2c4d6e-a1b2"]);
  });

  test("a whole long prompt (past the old 2000-char cap) is accepted up to QUERY_MAX_CHARS", async () => {
    const long = "relay policy ".repeat(1200).slice(0, 15_000);
    expect((await post({ query: long })).status).toBe(200);
    expect(calls.at(-1)?.[0]).toBe(long);
    expect((await post({ query: "x".repeat(QUERY_MAX_CHARS) })).status).toBe(200);
    expect((await post({ query: "x".repeat(QUERY_MAX_CHARS + 1) })).status).toBe(400);
  });

  test("a handler QueryRequestError answers its 4xx JSON, not a 500", async () => {
    const response = await post({ query: "relay policy", collection: "unknown-collection" });
    expect(response.status).toBe(400);
    expect(((await response.json()) as { error: string }).error).toContain("unknown-collection");
  });

  test("any other handler failure is still a 500", async () => {
    const response = await post({ query: "explode" });
    expect(response.status).toBe(500);
  });

  test.each([
    {},
    { query: "   " },
    { query: "relay policy", collection: "../other-vault" },
    { query: "relay policy", collection: "" },
    { query: "relay policy", collection: 7 },
    { query: "relay policy", session_id: "../escape" },
    { query: "relay policy", session_id: "a".repeat(129) },
    { query: "relay policy", session_id: 42 },
    { query: "relay policy", collection: "openclaw-workspace", limit: 0 },
    { query: "relay policy", collection: "openclaw-workspace", limit: 21 },
  ])("refuses invalid request before calling the query pipeline", async (body) => {
    const before = calls.length;
    const response = await post(body);
    expect(response.status).toBe(400);
    expect(calls).toHaveLength(before);
  });
});
