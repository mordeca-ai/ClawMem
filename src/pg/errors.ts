/**
 * Refusals for the PG write path (master-harness-vn4rz.7).
 *
 * Every error here is a LOUD refusal of a write that would otherwise corrupt the
 * vector space or the schema contract silently. None of them is recoverable by
 * retry; each names the operator action that clears it.
 */

/** The vault already holds vectors under a different model (ADR-0162 §7). */
export class PgVecWriteModelMismatchError extends Error {
  readonly storedModels: string[];
  readonly writeModel: string;
  constructor(storedModels: string[], writeModel: string, endpoint: string) {
    super(
      `Refusing vector write: this database already holds embeddings from ` +
      `[${storedModels.join(", ")}], but ${endpoint} produced "${writeModel}". ` +
      `A different model is a different vector space — mixing them makes cosine ` +
      `distance meaningless (ADR-0162 §7, master-harness-p2ib3). Either point the ` +
      `embed endpoint back at "${storedModels[0]}", or clear the vectors and ` +
      `re-embed the whole database under the new model.`,
    );
    this.name = "PgVecWriteModelMismatchError";
    this.storedModels = storedModels;
    this.writeModel = writeModel;
  }
}

/** A single batch carries rows from more than one model — drift by definition. */
export class PgVecBatchModelMismatchError extends Error {
  constructor(a: string, b: string) {
    super(
      `Refusing vector write: one batch carries embeddings from two models ` +
      `("${a}" and "${b}"). The embed endpoint changed model mid-flight. ` +
      `Nothing in this batch was written.`,
    );
    this.name = "PgVecBatchModelMismatchError";
  }
}

/** The embedding array length does not match the schema's fixed dimension. */
export class PgVecDimensionMismatchError extends Error {
  readonly expected: number;
  readonly actual: number;
  constructor(expected: number, actual: number, where: string) {
    super(
      `Refusing vector write: ${where} has ${actual} dimensions but the schema ` +
      `column is vector(${expected}). pgvector fixes the dimension at CREATE TABLE; ` +
      `clawmem will not truncate or pad an embedding to make it fit. Check ` +
      `CLAWMEM_EMBED_MODEL / CLAWMEM_EMBED_DIMENSIONS against EMBED_DIM in ` +
      `src/pg/config.ts.`,
    );
    this.name = "PgVecDimensionMismatchError";
    this.expected = expected;
    this.actual = actual;
  }
}

/** The live schema's vector column disagrees with the compiled-in constant. */
export class PgSchemaGeometryError extends Error {
  constructor(expected: number, actual: number) {
    super(
      `Schema geometry mismatch: content_vectors.embedding is vector(${actual}) but ` +
      `EMBED_DIM in src/pg/config.ts is ${expected}. One of the two is stale. ` +
      `Do not write until they agree — every row written under the wrong constant ` +
      `is a silently wrong vector.`,
    );
    this.name = "PgSchemaGeometryError";
  }
}
