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

// ===========================================================================
// Vault routing refusals (master-harness-0ynkd, ADR-0162 §1)
// ===========================================================================

/**
 * A write routed to the SFW vault but its collection/path is under a
 * hard-coded private root (src/pg/vaults.ts PRIVATE_ROOTS). The by-construction
 * tripwire: config can mis-declare, this cannot be mis-declared.
 */
export class PrivateContentRoutingError extends Error {
  readonly collection: string;
  readonly relPath: string | null;
  readonly absPath: string | null;
  readonly matchedOn: "collection" | "path";
  constructor(
    collection: string,
    relPath: string | null,
    absPath: string | null,
    matchedOn: "collection" | "path",
    declaration: string,
  ) {
    super(
      `Refusing write: ${matchedOn === "collection" ? "collection" : "source path"} ` +
      `${JSON.stringify(matchedOn === "collection" ? collection : (relPath ?? absPath ?? collection))} ` +
      `is under a PRIVATE root (src/pg/vaults.ts PRIVATE_ROOTS), but it routed to the ` +
      `"sfw" vault — the collection is ${declaration} in ~/.config/clawmem/index.yml. ` +
      `ADR-0162 §1 puts this boundary in the database and the login role, so a private ` +
      `document landing in database "clawmem" is a privacy defect, not a mis-tag. ` +
      `Fix: add \`vault: nsfw\` to the ${JSON.stringify(collection)} entry of ` +
      `~/.config/clawmem/index.yml. Nothing was written.`,
    );
    this.name = "PrivateContentRoutingError";
    this.collection = collection;
    this.relPath = relPath;
    this.absPath = absPath;
    this.matchedOn = matchedOn;
  }
}

/**
 * A write routed to a vault that has no connection configured.
 *
 * NEVER falls back to another vault's pool. That silent fallback is precisely
 * the bug master-harness-0ynkd exists to kill, and it is the same rule as
 * config.ts's "never silently fall back to sqlite": a misconfigured write path
 * must fail loudly, not write somewhere else.
 */
export class VaultNotConfiguredError extends Error {
  readonly vault: string;
  constructor(vault: string, missing: string[]) {
    super(
      `No PostgreSQL connection configured for the "${vault}" vault. Set ` +
      `${missing.join(" or ")}. There is deliberately NO fallback to the other ` +
      `vault's pool: a private write with no private connection must fail loudly, ` +
      `never land in the general database (ADR-0162 §1, master-harness-0ynkd). ` +
      `Nothing was written.`,
    );
    this.name = "VaultNotConfiguredError";
    this.vault = vault;
  }
}

/** One batch carries rows bound for two different vaults. */
export class PgVecBatchVaultMismatchError extends Error {
  constructor(a: string, b: string, collections: string[]) {
    super(
      `Refusing vector write: one batch carries fragments bound for two different ` +
      `vaults ("${a}" and "${b}"; collections: ${collections.join(", ")}). A batch is ` +
      `written in ONE transaction against ONE database, so a cross-vault batch cannot ` +
      `be honoured — and splitting it silently would put the caller's private and ` +
      `general content on two different code paths without anyone deciding to. This is ` +
      `a programming error at the call site. Nothing in this batch was written.`,
    );
    this.name = "PgVecBatchVaultMismatchError";
  }
}

/**
 * The connection we are about to INSERT through is attached to a different
 * database than the vault's configuration names. The belt inside the
 * transaction: routing logic cannot see a mis-wired connection string (a
 * `postgres://user@host:5433` with no path silently connects to the database
 * named after the user), so the session is asked directly.
 */
export class PgWrongDatabaseError extends Error {
  readonly vault: string;
  readonly expected: string;
  readonly actual: string;
  constructor(vault: string, expected: string, actual: string) {
    super(
      `Refusing write: the "${vault}" vault's connection is attached to database ` +
      `"${actual}", but its configuration names database "${expected}". A connection ` +
      `string that resolves to the wrong database is invisible to the routing layer, ` +
      `so the session is asked directly before every write (ADR-0162 §1, ` +
      `master-harness-0ynkd). Check CLAWMEM_PG${vault === "nsfw" ? "_NSFW" : ""}_URL / ` +
      `CLAWMEM_PG${vault === "nsfw" ? "_NSFW" : ""}_DATABASE. Nothing was written.`,
    );
    this.name = "PgWrongDatabaseError";
    this.vault = vault;
    this.expected = expected;
    this.actual = actual;
  }
}

/**
 * Both vaults resolve to the SAME database — the boundary collapsed.
 *
 * The current_database() belt cannot see this one: if CLAWMEM_PG_NSFW_URL names
 * database "clawmem", then expected and actual agree and every private write
 * lands in the general database with every check green. The only place it is
 * visible is the comparison BETWEEN the two configurations, so that is where it
 * is made.
 */
export class VaultDatabaseCollisionError extends Error {
  constructor(database: string) {
    super(
      `Refusing to open the "nsfw" vault: it resolves to database ${JSON.stringify(database)}, ` +
      `which is the SAME database the "sfw" vault resolves to. ADR-0162 §1 puts the ` +
      `bj-corpus boundary IN the database and the login role — two vaults sharing one ` +
      `database is that boundary silently removed, and no per-connection check can see ` +
      `it. Point CLAWMEM_PG_NSFW_URL / CLAWMEM_PG_NSFW_DATABASE at a different database ` +
      `than CLAWMEM_PG_URL / CLAWMEM_PG_DATABASE. Nothing was written.`,
    );
    this.name = "VaultDatabaseCollisionError";
  }
}
