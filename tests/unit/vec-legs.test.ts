import { describe, expect, it } from "bun:test";
import { vecLegParallelism, runVecLegs } from "../../src/vec-legs.ts";

/**
 * master-harness-vn4rz.93 — Parallel vector legs in hybrid query
 * (a) vecLegParallelism env parsing (unset->4, "1"->1, "0"->1, "3"->3, "99"->8, "abc"->4, " 2 "->2)
 * (b) runVecLegs preserves input order when later legs finish first
 * (c) runVecLegs never exceeds concurrency (track max in-flight; 6 legs, concurrency 2 -> max 2)
 * (d) concurrency 1 -> max in-flight 1
 * (e) a rejecting runLeg makes runVecLegs reject
 */

describe("vecLegParallelism — env parsing", () => {
  it("parses CLAWMEM_QUERY_VEC_PARALLEL correctly", () => {
    expect(vecLegParallelism({})).toBe(4);
    expect(vecLegParallelism({ CLAWMEM_QUERY_VEC_PARALLEL: "" })).toBe(4);
    expect(vecLegParallelism({ CLAWMEM_QUERY_VEC_PARALLEL: "1" })).toBe(1);
    expect(vecLegParallelism({ CLAWMEM_QUERY_VEC_PARALLEL: "0" })).toBe(1);
    expect(vecLegParallelism({ CLAWMEM_QUERY_VEC_PARALLEL: "3" })).toBe(3);
    expect(vecLegParallelism({ CLAWMEM_QUERY_VEC_PARALLEL: "99" })).toBe(8);
    expect(vecLegParallelism({ CLAWMEM_QUERY_VEC_PARALLEL: "abc" })).toBe(4);
    expect(vecLegParallelism({ CLAWMEM_QUERY_VEC_PARALLEL: " 2 " })).toBe(2);
  });
});

describe("runVecLegs — concurrency and ordering", () => {
  it("preserves input order when later legs finish first", async () => {
    const queries = ["q0", "q1", "q2", "q3"];
    const delays = [30, 10, 20, 0];
    const runLeg = async (q: string) => {
      const idx = queries.indexOf(q);
      const delay = delays[idx] ?? 0;
      await new Promise((resolve) => setTimeout(resolve, delay));
      return `result-${q}`;
    };

    const results = await runVecLegs(queries, runLeg, 4);
    expect(results).toEqual([
      "result-q0",
      "result-q1",
      "result-q2",
      "result-q3",
    ]);
  });

  it("never exceeds concurrency (6 legs, concurrency 2 -> max 2)", async () => {
    const queries = ["a", "b", "c", "d", "e", "f"];
    let currentInFlight = 0;
    let maxInFlight = 0;

    const runLeg = async (q: string) => {
      currentInFlight++;
      if (currentInFlight > maxInFlight) {
        maxInFlight = currentInFlight;
      }
      await new Promise((resolve) => setTimeout(resolve, 20));
      currentInFlight--;
      return q;
    };

    const results = await runVecLegs(queries, runLeg, 2);
    expect(results).toEqual(queries);
    expect(maxInFlight).toBe(2);
  });

  it("concurrency 1 -> max in-flight 1", async () => {
    const queries = ["a", "b", "c", "d"];
    let currentInFlight = 0;
    let maxInFlight = 0;

    const runLeg = async (q: string) => {
      currentInFlight++;
      if (currentInFlight > maxInFlight) {
        maxInFlight = currentInFlight;
      }
      await new Promise((resolve) => setTimeout(resolve, 10));
      currentInFlight--;
      return q;
    };

    const results = await runVecLegs(queries, runLeg, 1);
    expect(results).toEqual(queries);
    expect(maxInFlight).toBe(1);
  });

  it("a rejecting runLeg makes runVecLegs reject", async () => {
    const queries = ["ok1", "fail", "ok2"];
    const runLeg = async (q: string) => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      if (q === "fail") {
        throw new Error("leg exploded");
      }
      return q;
    };

    expect(runVecLegs(queries, runLeg, 2)).rejects.toThrow("leg exploded");
  });
});
