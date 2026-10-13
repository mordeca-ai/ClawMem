/**
 * Subprocess driver for mcp-query-vec-legs.test.ts (master-harness-vn4rz.96): builds the real
 * MCP server over an in-memory transport, calls the `query` tool twice on the SAME server (the
 * second call reuses the warm leg pool) and prints the ordered result paths as JSON.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { buildMcpServer } from "../../src/mcp.ts";

const query = process.env.DRIVER_QUERY!;
const built = buildMcpServer();
const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
await built.server.connect(serverTransport);
const client = new Client({ name: "vec-legs-driver", version: "0.0.0" });
await client.connect(clientTransport);

const pathsOf = (r: any): string[] =>
  (r.structuredContent?.results ?? []).map((x: any) => x.path ?? x.file ?? x.displayPath ?? "");

const out: Record<string, string[]> = {};
for (const [label, extra] of [
  ["default", {}],
  ["scoped", { collection: "colA" }],
  ["includeInternal", { includeInternal: true }],
  ["default-again", {}],
] as const) {
  const res = await client.callTool({
    name: "query",
    arguments: { query, compact: true, minScore: 0, limit: 10, ...extra },
  });
  out[label] = pathsOf(res);
}
console.log("DRIVER_RESULT " + JSON.stringify(out));
built.closeAllStores();
process.exit(0);
