/**
 * Collections configuration management
 *
 * This module manages the YAML-based collection configuration at ~/.config/clawmem/config.yaml.
 * Collections define which directories to index and their associated contexts.
 */

import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import YAML from "yaml";

// ============================================================================
// Types
// ============================================================================

/**
 * Context definitions for a collection
 * Key is path prefix (e.g., "/", "/2024", "/Board of Directors")
 * Value is the context description
 */
export type ContextMap = Record<string, string>;

/**
 * A single collection configuration
 */
export interface Collection {
  path: string;           // Absolute path to index
  pattern: string;        // Glob pattern (e.g., "**/*.md")
  context?: ContextMap;   // Optional context definitions
  update?: string;        // Optional bash command to run during qmd update
  content_type?: string;  // Default content_type for docs WITHOUT explicit frontmatter
                          // (kills filename inference for this collection — rvzn8.2:
                          // frontmatter-less ADRs were inferring `note` and decaying at 60d)
}

/**
 * The complete configuration file structure
 */
export interface LifecyclePolicy {
  archive_after_days: number;
  type_overrides: Record<string, number | null>;
  purge_after_days: number | null;
  exempt_collections: string[];
  dry_run: boolean;
}

export interface CollectionConfig {
  global_context?: string;                    // Context applied to all collections
  collections: Record<string, Collection>;    // Collection name -> config
  directoryContext?: boolean;                 // Opt-in: auto-generate CLAUDE.md in directories
  lifecycle?: LifecyclePolicy;                // Lifecycle management policy
}

/**
 * Collection with its name (for return values)
 */
export interface NamedCollection extends Collection {
  name: string;
}

// ============================================================================
// Configuration paths
// ============================================================================

function getConfigDir(): string {
  // Allow override via CLAWMEM_CONFIG_DIR for testing
  if (process.env.CLAWMEM_CONFIG_DIR) {
    return process.env.CLAWMEM_CONFIG_DIR;
  }
  return join(homedir(), ".config", "clawmem");
}

function getConfigFilePath(): string {
  const dir = getConfigDir();
  const preferred = join(dir, "config.yaml");
  if (existsSync(preferred)) return preferred;
  return join(dir, "index.yml");
}

/**
 * Ensure config directory exists
 */
function ensureConfigDir(): void {
  const configDir = getConfigDir();
  if (!existsSync(configDir)) {
    mkdirSync(configDir, { recursive: true });
  }
}

// ============================================================================
// Core functions
// ============================================================================

// ---------------------------------------------------------------------------
// loadConfig memoization (master-harness-hxa17)
// ---------------------------------------------------------------------------
//
// loadConfig() used to readFileSync + YAML.parse the config on EVERY call (twice,
// in fact — `config` and `raw` were two independent parses of the same string).
// It is called once per result row by getContextForFile(), and again per row via
// listCollections(), so a single 60-row vector hydrate paid ~120 config loads and
// a 6-leg query paid several hundred parses of one small file. Measured: 60x
// loadConfig() = 235 ms, of which essentially all is redundant YAML parsing.
//
// The memo is keyed on the config file's IDENTITY — (resolved path, mtimeMs, size) —
// so an edit on disk invalidates it; a missing file is itself a cacheable state
// (sentinel mtime/size of -1). On any mismatch we re-read and re-parse.
//
// CORRECTNESS: callers MUTATE the object they get back (addCollection and friends
// read-modify-write it, then saveConfig it). Handing out the cached reference would
// let one caller's mutation leak into every later reader, silently corrupting the
// vault config. So the memo caches the parsed value and EVERY call returns a fresh
// independent deep copy.

interface ConfigCacheEntry {
  path: string;
  mtimeMs: number;
  size: number;
  value: CollectionConfig;
}

let configCache: ConfigCacheEntry | null = null;

/**
 * Ops bypass: `CLAWMEM_DISABLE_CONFIG_CACHE=true` forces the uncached path.
 * Read at call time (same convention as ftsBypassEnabled() in src/search-utils.ts)
 * so a harness can toggle it per invocation.
 */
export function configCacheDisabled(): boolean {
  return process.env.CLAWMEM_DISABLE_CONFIG_CACHE === "true";
}

/** Drop the memo. Used by saveConfig() and by tests. */
export function clearConfigCache(): void {
  configCache = null;
}

/** Identity of the config file right now; (-1, -1) means "does not exist". */
function configFileIdentity(configPath: string): { mtimeMs: number; size: number } {
  try {
    const st = statSync(configPath);
    return { mtimeMs: st.mtimeMs, size: st.size };
  } catch {
    return { mtimeMs: -1, size: -1 };
  }
}

/**
 * Load configuration from ~/.config/clawmem/config.yaml
 * Returns empty config if file doesn't exist
 *
 * Memoized on (path, mtimeMs, size); always returns a fresh deep copy.
 */
export function loadConfig(): CollectionConfig {
  const configPath = getConfigFilePath();

  if (configCacheDisabled()) {
    return parseConfigFile(configPath);
  }

  const { mtimeMs, size } = configFileIdentity(configPath);

  if (
    configCache !== null &&
    configCache.path === configPath &&
    configCache.mtimeMs === mtimeMs &&
    configCache.size === size
  ) {
    return structuredClone(configCache.value);
  }

  const value = parseConfigFile(configPath);
  configCache = { path: configPath, mtimeMs, size, value };
  return structuredClone(value);
}

/** The uncached read + parse. Sole source of truth for config semantics. */
function parseConfigFile(configPath: string): CollectionConfig {
  if (!existsSync(configPath)) {
    return { collections: {} };
  }

  try {
    const content = readFileSync(configPath, "utf-8");
    const config = YAML.parse(content) as CollectionConfig;

    // Ensure collections object exists
    if (!config.collections) {
      config.collections = {};
    }

    // `clawmem collection add` enforces isValidCollectionName, but a hand-edited config.yaml
    // bypasses it entirely. A name containing '/' makes every virtual path for that collection
    // ambiguous — `clawmem://a/b/doc.md` parses as collection 'a', path 'b/doc.md' — so lookups
    // that round-trip through parseVirtualPath silently resolve to the wrong pair or to nothing.
    // Warn rather than throw: rejecting outright would lock an existing vault out of its own
    // config, and the damage is confined to virtual-path round-trips.
    for (const name of Object.keys(config.collections)) {
      if (!isValidCollectionName(name)) {
        console.warn(
          `[clawmem] collection name ${JSON.stringify(name)} in ${configPath} is not valid ` +
          `(expected only letters, digits, '_' and '-'). Virtual-path lookups for this ` +
          `collection may resolve incorrectly — rename it with 'clawmem collection add'.`,
        );
      }
    }

    // Parse lifecycle policy if present. (Historically this re-parsed `content` a second
    // time into `raw`; one parse is equivalent — the lifecycle fields are read off the
    // object before `config.lifecycle` is reassigned.)
    const lc = (config as { lifecycle?: Record<string, unknown> }).lifecycle;
    if (lc && typeof lc === "object") {
      config.lifecycle = {
        archive_after_days: typeof lc.archive_after_days === "number" ? lc.archive_after_days : 90,
        type_overrides: typeof lc.type_overrides === "object" && lc.type_overrides !== null ? lc.type_overrides as Record<string, number | null> : {},
        // INERT since v0.30.0 (see src/config.ts for the same guard) — only a positive
        // finite value is accepted; a negative or infinite one previously yielded a future
        // cutoff that deleted every archived row.
        purge_after_days:
          typeof lc.purge_after_days === "number" &&
          Number.isFinite(lc.purge_after_days) &&
          lc.purge_after_days > 0
            ? lc.purge_after_days
            : null,
        exempt_collections: Array.isArray(lc.exempt_collections) ? lc.exempt_collections as string[] : [],
        dry_run: lc.dry_run !== false,
      };
    }

    return config;
  } catch (error) {
    throw new Error(`Failed to parse ${configPath}: ${error}`);
  }
}

/**
 * Save configuration to ~/.config/clawmem/index.yml
 */
export function saveConfig(config: CollectionConfig): void {
  ensureConfigDir();
  const configPath = getConfigFilePath();

  try {
    const yaml = YAML.stringify(config, {
      indent: 2,
      lineWidth: 0,  // Don't wrap lines
    });
    writeFileSync(configPath, yaml, "utf-8");
    clearConfigCache();
  } catch (error) {
    throw new Error(`Failed to write ${configPath}: ${error}`);
  }
}

/**
 * Get a specific collection by name
 * Returns null if not found
 */
export function getCollection(name: string): NamedCollection | null {
  const config = loadConfig();
  const collection = config.collections[name];

  if (!collection) {
    return null;
  }

  return { name, ...collection };
}

/**
 * List all collections
 */
export function listCollections(): NamedCollection[] {
  const config = loadConfig();
  return Object.entries(config.collections).map(([name, collection]) => ({
    name,
    ...collection,
  }));
}

/**
 * Add or update a collection
 */
export function addCollection(
  name: string,
  path: string,
  pattern: string = "**/*.md",
  contentType?: string
): void {
  const config = loadConfig();

  config.collections[name] = {
    path,
    pattern,
    context: config.collections[name]?.context, // Preserve existing context
    // Explicit arg wins; otherwise preserve an existing default (same posture as context).
    content_type: contentType ?? config.collections[name]?.content_type,
  };

  saveConfig(config);
}

/**
 * Remove a collection
 */
export function removeCollection(name: string): boolean {
  const config = loadConfig();

  if (!config.collections[name]) {
    return false;
  }

  delete config.collections[name];
  saveConfig(config);
  return true;
}

/**
 * Rename a collection
 */
export function renameCollection(oldName: string, newName: string): boolean {
  const config = loadConfig();

  if (!config.collections[oldName]) {
    return false;
  }

  if (config.collections[newName]) {
    throw new Error(`Collection '${newName}' already exists`);
  }

  config.collections[newName] = config.collections[oldName];
  delete config.collections[oldName];
  saveConfig(config);
  return true;
}

// ============================================================================
// Context management
// ============================================================================

/**
 * Get global context
 */
export function getGlobalContext(): string | undefined {
  const config = loadConfig();
  return config.global_context;
}

/**
 * Set global context
 */
export function setGlobalContext(context: string | undefined): void {
  const config = loadConfig();
  config.global_context = context;
  saveConfig(config);
}

/**
 * Get all contexts for a collection
 */
export function getContexts(collectionName: string): ContextMap | undefined {
  const collection = getCollection(collectionName);
  return collection?.context;
}

/**
 * Add or update a context for a specific path in a collection
 */
export function addContext(
  collectionName: string,
  pathPrefix: string,
  contextText: string
): boolean {
  const config = loadConfig();
  const collection = config.collections[collectionName];

  if (!collection) {
    return false;
  }

  if (!collection.context) {
    collection.context = {};
  }

  collection.context[pathPrefix] = contextText;
  saveConfig(config);
  return true;
}

/**
 * Remove a context from a collection
 */
export function removeContext(
  collectionName: string,
  pathPrefix: string
): boolean {
  const config = loadConfig();
  const collection = config.collections[collectionName];

  if (!collection?.context?.[pathPrefix]) {
    return false;
  }

  delete collection.context[pathPrefix];

  // Remove empty context object
  if (Object.keys(collection.context).length === 0) {
    delete collection.context;
  }

  saveConfig(config);
  return true;
}

/**
 * List all contexts across all collections
 */
export function listAllContexts(): Array<{
  collection: string;
  path: string;
  context: string;
}> {
  const config = loadConfig();
  const results: Array<{ collection: string; path: string; context: string }> = [];

  // Add global context if present
  if (config.global_context) {
    results.push({
      collection: "*",
      path: "/",
      context: config.global_context,
    });
  }

  // Add collection contexts
  for (const [name, collection] of Object.entries(config.collections)) {
    if (collection.context) {
      for (const [path, context] of Object.entries(collection.context)) {
        results.push({
          collection: name,
          path,
          context,
        });
      }
    }
  }

  return results;
}

/**
 * Find best matching context for a given collection and path
 * Returns the most specific matching context (longest path prefix match)
 */
export function findContextForPath(
  collectionName: string,
  filePath: string
): string | undefined {
  const config = loadConfig();
  const collection = config.collections[collectionName];

  if (!collection?.context) {
    return config.global_context;
  }

  // Find all matching prefixes
  const matches: Array<{ prefix: string; context: string }> = [];

  for (const [prefix, context] of Object.entries(collection.context)) {
    // Normalize paths for comparison
    const normalizedPath = filePath.startsWith("/") ? filePath : `/${filePath}`;
    const normalizedPrefix = prefix.startsWith("/") ? prefix : `/${prefix}`;

    if (normalizedPath.startsWith(normalizedPrefix)) {
      matches.push({ prefix: normalizedPrefix, context });
    }
  }

  // Return most specific match (longest prefix)
  if (matches.length > 0) {
    matches.sort((a, b) => b.prefix.length - a.prefix.length);
    return matches[0]!.context;
  }

  // Fallback to global context
  return config.global_context;
}

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Get the config file path (useful for error messages)
 */
export function getConfigPath(): string {
  return getConfigFilePath();
}

/**
 * Check if config file exists
 */
export function configExists(): boolean {
  return existsSync(getConfigFilePath());
}

/**
 * Validate a collection name
 * Collection names must be valid and not contain special characters
 */
export function isValidCollectionName(name: string): boolean {
  // Allow alphanumeric, hyphens, underscores
  return /^[a-zA-Z0-9_-]+$/.test(name);
}
