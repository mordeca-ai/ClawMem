/** A parameterized PostgreSQL tsquery expression and its bind values. */
export interface PgFtsQueryFragment {
  text: string;
  values: string[];
}

/**
 * Mirror of src/store.ts's private FTS5_STOPWORDS set.
 *
 * Keep this list in lockstep with that reference. The unit suite compares the
 * two builders for every word here because src/store.ts is read-only in this
 * slice and therefore cannot export its set.
 */
export const PG_FTS_STOPWORDS: ReadonlySet<string> = new Set([
  "a", "an", "the",
  "how", "what", "when", "where", "why", "which", "who", "whom", "whose",
  "am", "is", "are", "was", "were", "be", "been", "being",
  "do", "does", "did", "has", "have", "had",
  "can", "could", "will", "would", "shall", "should", "may", "might", "must",
  "to", "of", "in", "on", "at", "for", "with", "from", "by", "about",
  "into", "over", "under", "through", "between", "during", "against",
  "above", "below", "up", "down", "out", "off", "again", "further",
  "and", "or", "not", "but", "if", "because", "as", "until", "while",
  "so", "than", "then",
  "it", "its", "this", "that", "these", "those",
  "i", "me", "my", "we", "us", "our", "you", "your",
  "he", "him", "his", "she", "her", "they", "them", "their",
  "itself", "himself", "herself", "themselves",
  "actually", "really", "just", "also", "very", "there", "here",
  "all", "any", "both", "each", "few", "more", "most", "other", "some",
  "such", "no", "nor", "only", "own", "same", "too", "s", "t",
]);

/** Tokenize exactly as sqlite's unicode61-facing tokenizeForFTS5 helper. */
export function tokenizeForPgFts(query: string): string[] {
  return query.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter(token => token.length > 0);
}

function effectiveTerms(group: string): string[] {
  const terms = tokenizeForPgFts(group);
  if (terms.length <= 1) return terms;
  const content = terms.filter(term => !PG_FTS_STOPWORDS.has(term));
  return content.length > 0 ? content : terms;
}

/**
 * Build the PG equivalent of buildFTS5Query(). Every token is passed to the
 * server's english stemmer independently and then marked as a prefix node.
 * Uppercase ` OR ` separates groups; every other separator is tokenization.
 */
export function buildPgFtsQuery(
  query: string,
  firstParameter = 1,
): PgFtsQueryFragment | null {
  const values: string[] = [];
  const groups: string[] = [];

  for (const group of query.split(/\s+OR\s+/)) {
    const terms = effectiveTerms(group);
    if (terms.length === 0) continue;
    const nodes = terms.map(term => {
      values.push(term);
      return `to_tsquery('english', $${firstParameter + values.length - 1}::text || ':*')`;
    });
    const expression = nodes.join(" && ");
    groups.push(nodes.length > 1 ? `(${expression})` : expression);
  }

  if (groups.length === 0) return null;
  return { text: groups.join(" || "), values };
}
