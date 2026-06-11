/**
 * buildFTS5Query — uppercase OR is an operator, not a literal term.
 *
 * Regression: `cocoa OR frosting` used to compile to
 * `"cocoa"* AND "or"* AND "frosting"*`, requiring an or-prefixed token in
 * the doc ("oregano" masked the bug for some corpora; recipes without one
 * returned nothing).
 */
import { describe, test, expect } from "bun:test";
import { buildFTS5Query } from "../../src/store.ts";

describe("buildFTS5Query", () => {
  test("single term → quoted prefix", () => {
    expect(buildFTS5Query("cocoa")).toBe(`"cocoa"*`);
  });

  test("multi term → AND-joined (default semantics unchanged)", () => {
    expect(buildFTS5Query("chocolate cake")).toBe(`"chocolate"* AND "cake"*`);
  });

  test("uppercase OR between terms → FTS5 OR operator", () => {
    expect(buildFTS5Query("cocoa OR frosting")).toBe(`"cocoa"* OR "frosting"*`);
  });

  test("OR groups with multi-term sides get parenthesized", () => {
    expect(buildFTS5Query("dark cocoa OR chocolate frosting"))
      .toBe(`("dark"* AND "cocoa"*) OR ("chocolate"* AND "frosting"*)`);
  });

  test("many OR terms (search-pp shape)", () => {
    expect(buildFTS5Query("cocoa OR hershey OR cake"))
      .toBe(`"cocoa"* OR "hershey"* OR "cake"*`);
  });

  test("lowercase 'or' stays a plain term (natural language)", () => {
    expect(buildFTS5Query("this or that")).toBe(`"this"* AND "or"* AND "that"*`);
  });

  test("empty OR sides are dropped", () => {
    expect(buildFTS5Query("cocoa OR !!!")).toBe(`"cocoa"*`);
  });

  test("all-empty query → null", () => {
    expect(buildFTS5Query("!!! ???")).toBeNull();
  });

  test("punctuation inside groups still tokenizes (v0.10.6 separator rule)", () => {
    expect(buildFTS5Query("before_compaction OR after-restart"))
      .toBe(`("before"* AND "compaction"*) OR ("after"* AND "restart"*)`);
  });
});
