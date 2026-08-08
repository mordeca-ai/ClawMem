# Multi-vault configuration

By default, ClawMem stores all AI agent memory in a single vault at `~/.cache/clawmem/index.sqlite`. Multi-vault is opt-in for users who want separate memory domains.

## When to use multiple vaults

- **Project isolation** — keep work and personal memory separate
- **Shared vs private** — a shared team vault alongside a personal vault
- **Runtime separation** — separate vaults for Claude Code and OpenClaw (though shared single vault is the recommended default)

## Configuration

### Via config.yaml

```yaml
# ~/.config/clawmem/config.yaml
vaults:
  work: ~/.cache/clawmem/work.sqlite
  personal: ~/.cache/clawmem/personal.sqlite
```

### Via environment variable

```bash
export CLAWMEM_VAULTS='{"work":"~/.cache/clawmem/work.sqlite","personal":"~/.cache/clawmem/personal.sqlite"}'
```

Environment variables override config file values. Vault paths support `~` expansion.

## Using vaults with tools

All MCP tools accept an optional `vault` parameter. Omit it for the default vault.

```
# Search the work vault
query("authentication setup", vault="work", compact=true)

# Pin a memory in the personal vault
memory_pin("remember preference X", vault="personal")

# Get lifecycle stats for a specific vault
lifecycle_status(vault="work")
```

Tools that support vault: `query`, `search`, `vsearch`, `intent_search`, `memory_retrieve`, `query_plan`, `get`, `multi_get`, `find_similar`, `find_causal_links`, `timeline`, `memory_evolution_status`, `session_log`, `memory_pin`, `memory_snooze`, `memory_forget`, `lifecycle_status`, `lifecycle_sweep`, `lifecycle_restore`, `status`, `reindex`, `index_stats`, `build_graphs`.

## Automatic surfacing and vault isolation

Vaults are isolated by default in the automatic feed too (v0.35.0): the `context-surfacing`
hook reads only the general vault, so a secondary vault's content reaches a session only
through a deliberate `vault`-parameter call. To have the hook also merge secondary-vault
results into every prompt's injected context, opt in:

```yaml
retrieval:
  surface_secondary_vaults: true
```

or `CLAWMEM_SURFACE_SECONDARY_VAULTS=true` (env wins; only the literal `true` enables).
The automatic lane is the configured named `skill` vault — the knob does not fan out across
every named vault. Config is process-cached: restart long-lived processes (watcher, MCP
server) after changing it. Note the gate governs the hook's automatic feed only — content
that is *also* indexed into the general vault as an ordinary collection surfaces regardless;
remove that collection from the general config if full isolation is the goal.

## Populating a vault

Use `vault_sync` to index content into a named vault:

```
vault_sync(vault="work", content_root="~/projects/work-notes", pattern="**/*.md")
```

`vault_sync` includes restricted-path validation — it rejects sensitive directories like `/etc/`, `/root/`, `.ssh`, `.env`, `credentials`, `.aws`, `.kube`.

## Listing vaults

```
list_vaults()
```

Returns configured vault names and paths. Empty in single-vault mode.

## Implementation details

- Named vault stores are cached in memory (`Map<string, Store>`) and reused across tool calls
- All cached stores are closed on MCP server shutdown (SIGINT/SIGTERM)
- Each vault is an independent SQLite database with its own documents, embeddings, graphs, and sessions
- WAL mode + `busy_timeout=5000ms` ensures safe concurrent access
- Hooks always operate on the default vault — vault selection is only available through MCP tools and REST API
