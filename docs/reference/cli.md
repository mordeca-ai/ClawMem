# ClawMem CLI reference

Complete command reference for the ClawMem memory engine. Always use the `bin/clawmem` wrapper, which sets GPU endpoint defaults.

## Core commands

```bash
clawmem init                    # Initialize vault (creates SQLite DB)
clawmem status                  # Quick index status
clawmem doctor                  # Full health check (GPU connectivity, index integrity)
```

## Collection management

```bash
clawmem collection add <path> --name <name>   # Add a collection
clawmem collection list                        # List all collections
clawmem collection remove <name>               # Forget the collection in config.yaml
                                                # (does NOT delete indexed rows)
clawmem collection purge <name>                # Preview a hard-delete of this collection's rows
clawmem collection purge <name> --yes          # Hard-delete: documents (active+inactive) + orphaned content/vectors
```

`collection remove` and `collection purge` are different operations. `remove` only
edits `config.yaml` so the collection stops being re-scanned — existing rows in the
SQLite index are untouched. `purge` (v0.10.8+, master-harness-t5i0) hard-deletes that
collection's rows from the index — active and inactive alike — plus any content/vector
rows that become orphaned as a result, without touching any other collection. It
refuses an empty or wildcard-bearing name, and without `--yes` only prints a preview
(active/inactive counts) — nothing is deleted. This is the operation to use for tearing
down a disposable/scratch/smoke-test collection; **do not** reach for the DB-wide
maintenance functions (`deleteInactiveDocuments`/`cleanupOrphaned*` in `src/store.ts`)
for a single-collection teardown — see [Backup](#backup) below for why that's
dangerous.

## Indexing

```bash
clawmem update                  # Index all collections (BM25 only)
clawmem update --embed          # Index + embed in one pass
clawmem mine <dir>                             # Import conversation exports (Claude, ChatGPT, Slack)
clawmem mine <dir> -c convos                   # Import with custom collection name
clawmem mine <dir> --embed                     # Import + embed in one pass
clawmem mine <dir> --dry-run                   # Preview without importing
clawmem mine <dir> --synthesize                # v0.7.2: import + post-import LLM fact extraction
clawmem mine <dir> --synthesize --synthesis-max-docs 50   # Cap synthesis to first 50 conversations (default 20)
clawmem reindex                                # Force re-scan all collections
clawmem embed                   # Embed all un-embedded fragments
clawmem embed --force           # Re-embed everything (clears existing vectors)
```

### Conversation Import

`clawmem mine` normalizes and imports conversation exports from multiple AI chat formats:

- **Claude Code JSONL** — session transcripts (`.jsonl`)
- **Claude.ai JSON** — flat messages or privacy export with `chat_messages`
- **ChatGPT JSON** — `conversations.json` with mapping tree
- **Slack JSON** — 2-party DM exports
- **Plain text** — files with `User:`/`Assistant:` markers

Each user+assistant exchange pair becomes one indexed document with `content_type: conversation`. Files are chunked, written to a temporary staging directory, indexed through the standard pipeline (including A-MEM enrichment), then staging is cleaned up.

#### `--synthesize` (v0.7.2)

Adds a post-import LLM fact extraction pass. After `indexCollection` commits the raw conversations, the synthesis pipeline walks the freshly imported docs and extracts structured facts (`decision`, `preference`, `milestone`, `problem`) plus cross-fact relations via a two-pass LLM pipeline. Each extracted fact is saved as a first-class searchable document alongside the raw conversation exchanges. Cross-fact links bind across conversations in the same batch — not just within a single conversation.

- **Off by default.** Raw mine import semantics are byte-identical when `--synthesize` is omitted.
- **Opt-in user consent required** — each pass drives one additional LLM call per conversation doc.
- **`--synthesis-max-docs N`** caps the number of conversations scanned per run (default 20).
- **Idempotent reruns** — synthesized fact paths are hash-stable, so rerunning over the same collection updates facts in place rather than creating parallel rows. Relation weights are monotone (`MAX(weight, excluded.weight)`).
- **Non-fatal failures** — any LLM failure, JSON parse error, or relation insert error is counted and logged. Synthesis failure never rolls back the mine import.
- See [post-import conversation synthesis](../concepts/architecture.md#post-import-conversation-synthesis-v072) for the full architectural walkthrough.

## Search (CLI)

```bash
clawmem search <query>          # BM25 search
clawmem search <query> --vec    # Vector search
clawmem search <query> --hybrid # BM25 + vector (default)
```

## Bootstrap

```bash
clawmem bootstrap <path> --name <name>   # One-command setup: init + collection + index + embed + hooks + mcp
```

## Watch

```bash
clawmem watch                   # Start file watcher (indexes on .md changes)
```

## Setup

```bash
clawmem setup hooks             # Install Claude Code hooks
clawmem setup hooks --remove    # Remove installed hooks
clawmem setup mcp               # Register MCP server
clawmem setup openclaw                   # Install OpenClaw memory plugin. v0.10.4+ delegates to `openclaw plugins install --force` when the CLI is on PATH (profile-aware via OPENCLAW_STATE_DIR; auto-enabled). Falls back to recursive copy honoring OPENCLAW_STATE_DIR when the CLI is absent.
clawmem setup openclaw --link            # Load-path mode: delegates `openclaw plugins install -l` when CLI is on PATH (records source in plugins.load.paths — NOT a filesystem symlink). In CLI-absent fallback, creates a real symlink (note: OpenClaw v2026.4.11+ discovery skips fallback symlinks).
clawmem setup openclaw --remove          # Uninstall. Tries `openclaw plugins uninstall clawmem --force` first; falls back to manual cleanup at the resolved extensions path for legacy unmanaged installs.
clawmem setup openclaw --help            # Print full flag + env-var reference (v0.10.4+).
clawmem setup curator                    # Install curator agent
```

### `setup openclaw` env vars (v0.10.4+)

Both the delegated and fallback paths honor:

| Env var | Effect |
|---------|--------|
| `OPENCLAW_STATE_DIR` | Override the OpenClaw config root. Plugin installs into `<OPENCLAW_STATE_DIR>/extensions/clawmem`. |
| `OPENCLAW_CONFIG_PATH` | Override the OpenClaw config file path; config root becomes `dirname(OPENCLAW_CONFIG_PATH)`. |
| `OPENCLAW_HOME` | Override the home directory used to resolve the default `~/.openclaw` root. |
| `HOME` / `USERPROFILE` | Standard home-dir env vars; consulted in that order when `OPENCLAW_HOME` is unset. |

```bash
# Install ClawMem into the `dev` profile (~/.openclaw-dev/extensions/clawmem)
OPENCLAW_STATE_DIR=~/.openclaw-dev clawmem setup openclaw
```

## Server

```bash
clawmem serve                            # Start REST API (localhost:7438)
clawmem serve --port 8080                # Custom port
clawmem serve --host 0.0.0.0             # Listen on all interfaces
```

## Hook execution (internal)

```bash
clawmem hook context-surfacing    # Execute a hook (reads JSON from stdin)
clawmem hook decision-extractor
clawmem hook handoff-generator
clawmem hook feedback-loop
clawmem hook precompact-extract
clawmem hook session-bootstrap
clawmem hook staleness-check
clawmem hook curator-nudge
```

## IO6 surface commands (daemon integration)

For non-hook integrations where a host process needs to inject context programmatically (e.g., daemon mode, custom orchestrators):

```bash
echo "user query" | clawmem surface --context --stdin     # Per-prompt context injection
echo "session-id" | clawmem surface --bootstrap --stdin    # Per-session bootstrap
```

## Analysis

```bash
clawmem reflect [N]             # Cross-session reflection (last N days, default 14)
clawmem consolidate [--dry-run] # Find and archive duplicate low-confidence documents
```

## Backup

```bash
clawmem backup                                 # Snapshot to ~/.cache/clawmem/backups/, keep 7
clawmem backup --dest /path/to/backups         # Custom destination directory
clawmem backup --keep 14                       # Custom retention count
```

**v0.10.8+ (master-harness-t5i0).** Prior to this there was no backup mechanism for
the ClawMem SQLite index at all — a scope-guard incident during unrelated testing hard
deleted 3566 pre-existing document rows with no recovery path available, because
nothing had ever snapshotted the DB. `clawmem backup` produces a consistent, compacted
copy of the live index at `<dest>/index-<timestamp>.sqlite` via SQLite's `VACUUM INTO`
(the `bun:sqlite`-native equivalent of `sqlite3 <db> ".backup <dest>"` — a single
transactional statement, no external `sqlite3` binary required, safe to run against a
live, in-use database). After each backup, files beyond the retention count (default
7, oldest first) are pruned automatically.

| Env var | Default | Description |
|---|---|---|
| `CLAWMEM_BACKUP_DIR` | `~/.cache/clawmem/backups` (or `$XDG_CACHE_HOME/clawmem/backups`) | Backup destination directory |

### Scheduling

`clawmem backup` does not install its own timer — run it periodically via systemd or
cron. Example systemd user timer (daily at 03:15):

```ini
# ~/.config/systemd/user/clawmem-backup.service
[Unit]
Description=ClawMem index backup

[Service]
Type=oneshot
ExecStart=%h/path/to/clawmem/bin/clawmem backup
```

```ini
# ~/.config/systemd/user/clawmem-backup.timer
[Unit]
Description=Daily ClawMem index backup

[Timer]
OnCalendar=*-*-* 03:15:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now clawmem-backup.timer
```

Or via cron:

```cron
15 3 * * * /path/to/clawmem/bin/clawmem backup >> ~/.cache/clawmem/backup.log 2>&1
```

## Session focus topic (v0.9.0)

Per-session topic biasing for the context-surfacing hook. Writes a focus file at `~/.cache/clawmem/sessions/<session_id>.focus` that steers query expansion, reranking, snippet extraction, and applies a post-composite-score topic boost (1.4× match, 0.75× demote, NO-OP on zero matches). Session-scoped — never writes to SQLite or mutates any lifecycle column.

```bash
clawmem focus set "<topic>"                        # uses CLAUDE_SESSION_ID / CLAWMEM_SESSION_ID env
clawmem focus set "<topic>" --session-id <id>      # explicit session id
clawmem focus show                                 # reads session id from env
clawmem focus show --session-id <id>
clawmem focus clear                                # uses env-resolved session id
clawmem focus clear --session-id <id>
```

The session ID is resolved from `--session-id <id>`, then `CLAUDE_SESSION_ID`, then `CLAWMEM_SESSION_ID`. `CLAWMEM_SESSION_FOCUS` env var is a debug-only override that does NOT provide per-session scoping on multi-session hosts. `CLAWMEM_FOCUS_ROOT` overrides the focus file root directory for hermetic testing.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAWMEM_EMBED_URL` | `http://localhost:8088` | Embedding server |
| `CLAWMEM_EMBED_API_KEY` | — | API key for cloud embedding |
| `CLAWMEM_EMBED_MODEL` | `embedding` | Model name for embedding requests |
| `CLAWMEM_EMBED_MAX_CHARS` | `6000` | Max chars per embedding input |
| `CLAWMEM_EMBED_TPM_LIMIT` | `100000` | Tokens-per-minute limit for cloud embedding pacing |
| `CLAWMEM_EMBED_DIMENSIONS` | — | Output dimensions for OpenAI `text-embedding-3-*` models |
| `CLAWMEM_LLM_URL` | `http://localhost:8089` | LLM server |
| `CLAWMEM_LLM_MODEL` | `qwen3` | Model name sent to the configured LLM endpoint |
| `CLAWMEM_LLM_REASONING_EFFORT` | — | Optional top-level `reasoning_effort` field for Chat Completions endpoints that support it (for example OpenAI reasoning models). Leave unset for llama-server/vLLM unless explicitly supported. |
| `CLAWMEM_LLM_NO_THINK` | `true` | Append `/no_think` to remote prompts; set `false` for standard OpenAI models and other endpoints that reject or treat it as literal prompt text |
| `CLAWMEM_RERANK_URL` | `http://localhost:8090` | Reranker server |
| `CLAWMEM_NO_LOCAL_MODELS` | `false` | Block node-llama-cpp auto-downloads |
| `CLAWMEM_PROFILE` | `balanced` | Performance profile: `speed` (BM25 only), `balanced` (BM25+vector), `deep` (BM25+vector+expansion+reranking) |
| `CLAWMEM_VAULTS` | — | JSON map of vault name to SQLite path |
| `CLAWMEM_API_TOKEN` | — | Bearer token for REST API auth |
| `CLAWMEM_ENABLE_AMEM` | enabled | A-MEM note construction during indexing |
| `CLAWMEM_ENABLE_CONSOLIDATION` | disabled | Background consolidation worker (light lane, 5-min interval). **v0.8.2:** every tick wraps in a `worker_leases` row (`light-consolidation` key) so dual-host (`clawmem watch` + `clawmem mcp`) is safe. Hosted by either `cmdWatch` (canonical, long-lived) or `cmdMcp` (per-session fallback). |
| `CLAWMEM_CONSOLIDATION_INTERVAL` | `300000` | Light-lane worker interval in ms |
| `CLAWMEM_HEAVY_LANE` | disabled | **v0.8.0.** Enable the quiet-window heavy maintenance lane (second consolidation worker with DB-backed lease + stale-first batching + `maintenance_runs` journaling). See [heavy maintenance lane](../concepts/architecture.md#heavy-maintenance-lane-v080). **v0.8.2:** canonical host is `clawmem watch`; `clawmem mcp` retains the same gate as a fallback host but emits a stderr warning advising operators to move heavy-lane hosting to the watcher because per-session stdio MCPs may never be alive during the configured quiet window. |
| `CLAWMEM_HEAVY_LANE_INTERVAL` | `1800000` | **v0.8.0.** Heavy-lane tick interval in ms (default 30 min, min 30 s). |
| `CLAWMEM_HEAVY_LANE_WINDOW_START` | — | **v0.8.0.** Start hour (0-23) of the quiet window. Unset → no window. |
| `CLAWMEM_HEAVY_LANE_WINDOW_END` | — | **v0.8.0.** End hour (0-23, exclusive). Supports midnight wrap (22→6). |
| `CLAWMEM_HEAVY_LANE_MAX_USAGES` | `30` | **v0.8.0.** Max `context_usage` rows in the last 10 min before the heavy lane skips with `reason='query_rate_high'`. |
| `CLAWMEM_HEAVY_LANE_OBS_LIMIT` | `100` | **v0.8.0.** Phase 2 stale-first observation batch size for the heavy lane. |
| `CLAWMEM_HEAVY_LANE_DED_LIMIT` | `40` | **v0.8.0.** Phase 3 stale-first deductive candidate batch size for the heavy lane. |
| `CLAWMEM_HEAVY_LANE_SURPRISAL` | `false` | **v0.8.0.** When `true`, seed Phase 2 with k-NN anomaly-ranked doc ids from `computeSurprisalScores` instead of stale-first ordering. Degrades to stale-first on vaults without embeddings. |
| `CLAWMEM_SESSION_FOCUS` | — | **v0.9.0 §11.4.** Debug-only override for the session focus topic. NOT session-scoped — do not use in multi-session deployments. Use `clawmem focus set <topic> --session-id <id>` instead. |
| `CLAWMEM_FOCUS_ROOT` | `~/.cache/clawmem/sessions` | **v0.9.0 §11.4.** Override directory for per-session focus files. Primarily for hermetic testing. |
| `CLAWMEM_BACKUP_DIR` | `~/.cache/clawmem/backups` | **v0.10.8+.** Destination directory for `clawmem backup`. See [Backup](#backup). |
| `INDEX_PATH` | `~/.cache/clawmem/index.sqlite` | Override default vault path |

The `bin/clawmem` wrapper sets endpoint defaults. Always use it instead of `bun run src/clawmem.ts` directly.
