import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { startServer, type QueryJsonHit } from "../../src/server.ts";
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
  const calls: Array<[string, string, number]> = [];
  let server: ReturnType<typeof startServer>;
  let base: string;

  beforeAll(() => {
    server = startServer({} as Store, 0, "127.0.0.1", async (query, collection, limit) => {
      calls.push([query, collection, limit]);
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
    expect(calls.at(-1)).toEqual(["relay policy", "openclaw-workspace", 3]);
  });

  test.each([
    {},
    { query: "relay policy" },
    { query: "relay policy", collection: "../other-vault" },
    { query: "relay policy", collection: "openclaw-workspace", limit: 0 },
    { query: "relay policy", collection: "openclaw-workspace", limit: 21 },
  ])("refuses invalid request before calling the query pipeline", async (body) => {
    const before = calls.length;
    const response = await post(body);
    expect(response.status).toBe(400);
    expect(calls).toHaveLength(before);
  });
});
