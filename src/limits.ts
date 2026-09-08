/**
 * Centralized limits for input validation and resource bounding.
 */

// Search & query
export const MAX_QUERY_LENGTH = 10_000;
export const MAX_SEARCH_LIMIT = 100;

// LLM
export const MAX_LLM_INPUT_CHARS = 100_000;
export const MAX_LLM_GENERATE_TIMEOUT_MS = 120_000; // 2 minutes

// Transcripts & hooks
export const MAX_TRANSCRIPT_BYTES = 50 * 1024 * 1024; // 50 MB
export const MAX_FILES_EXTRACTED = 200;

// Document processing
export const MAX_FRAGMENTS_PER_DOC = 500;
export const MAX_SPLITTER_INPUT_CHARS = 500_000;
export const MAX_FILE_LINES_READ = 100_000;

// Profile
export const MAX_LEVENSHTEIN_LENGTH = 1_000;

// Paths
export const MAX_PATH_LENGTH = 1_000;

// ---------------------------------------------------------------------------
// Embed input bounding (master-harness-vn4rz.42)
//
// Why this exists: the splitter emits one `full` fragment per document that is
// pushed RAW (bounded only by MAX_SPLITTER_INPUT_CHARS above = 500_000), while
// every other fragment class is routed through chunkContent at
// MAX_FRAGMENT_CHARS = 2000. So `full` is the only class that can hand the
// embed path a six-figure-character string. An ollama endpoint serving a
// 2048-token model does not reject such a body — it stalls on it; the 60s
// remote-fetch deadline then fires, isTransportError classifies the abort as
// transport, markRemoteEmbedDown trips the breaker, and EVERY remaining
// fragment fails fast (measured: 4,200 embedded vs 4,332 embed failures on a
// single reindex). The bound therefore lives at the embed chokepoint, not only
// at the splitter, so it protects every caller regardless of which code path
// produced the text.
//
// The budget is DERIVED from the model's advertised context_length (ollama
// `POST /api/show` → model_info's `*.context_length`), never guessed, and
// converted to characters with a deliberately conservative chars-per-token
// ratio — under-estimating chars can only ever truncate MORE, which is the
// safe direction. Truncation (not chunking) is the right shape: the model
// physically cannot attend past its context, and ollama's own `truncate:true`
// default would drop the tail server-side anyway.
// ---------------------------------------------------------------------------

/**
 * Conservative characters-per-token used to convert an advertised token
 * context into a character budget. Real mixed prose/code tokenizes at ~3-4
 * chars/token; 2 is deliberately pessimistic so the derived char budget stays
 * under the true token ceiling even for dense code. Override with
 * CLAWMEM_EMBED_CHARS_PER_TOKEN.
 */
export const EMBED_CHARS_PER_TOKEN = 2;

/**
 * Token context assumed when the endpoint cannot tell us (no /api/show, a
 * non-ollama OpenAI-compatible server, a response with no `*.context_length`).
 * 2048 is EmbeddingGemma's context and the smallest of the commonly-configured
 * models here, so assuming it under-shoots rather than over-shoots.
 */
export const FALLBACK_EMBED_CONTEXT_TOKENS = 2048;

/**
 * Deadline for the one-shot `/api/show` context probe. Short by design: it sits
 * in front of the embed hot path, is memoized per (endpoint, model), and a
 * failure is not an error — it just falls back to
 * FALLBACK_EMBED_CONTEXT_TOKENS. Override with
 * CLAWMEM_EMBED_CONTEXT_PROBE_TIMEOUT_MS.
 */
export const EMBED_CONTEXT_PROBE_TIMEOUT_MS = 5_000;
