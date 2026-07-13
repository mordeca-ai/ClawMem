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

describe("buildFTS5Query — stopword relaxation (master-harness-cxj0u)", () => {
  test("natural-language question drops stopwords, keeps content tokens only", () => {
    expect(buildFTS5Query("How does ClawMem's hybrid retrieval actually work end-to-end?"))
      .toBe(`"clawmem"* AND "hybrid"* AND "retrieval"* AND "work"* AND "end"* AND "end"*`);
  });

  test("all-stopword query falls back to AND-all (pre-fix behavior, unchanged)", () => {
    expect(buildFTS5Query("this or that")).toBe(`"this"* AND "or"* AND "that"*`);
  });

  test("all-stopword query (question shape) falls back to AND-all", () => {
    expect(buildFTS5Query("to be or not to be"))
      .toBe(`"to"* AND "be"* AND "or"* AND "not"* AND "to"* AND "be"*`);
  });

  test("mixed OR groups drop stopwords independently per branch", () => {
    expect(buildFTS5Query("what is a cocoa OR how does frosting work"))
      .toBe(`"cocoa"* OR ("frosting"* AND "work"*)`);
  });

  test("single stopword query still yields a query (fallback, single-token shape unchanged)", () => {
    expect(buildFTS5Query("the")).toBe(`"the"*`);
  });

  test("single content word unchanged", () => {
    expect(buildFTS5Query("cocoa")).toBe(`"cocoa"*`);
  });

  test("multi term with no stopwords unaffected", () => {
    expect(buildFTS5Query("chocolate cake")).toBe(`"chocolate"* AND "cake"*`);
  });
});
