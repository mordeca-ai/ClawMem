# clawmem migrations — corrections live here, not in the `.sql`

## Applied migrations are immutable

`src/pg/migrate.ts` `loadMigrations()` checksums the **whole substituted file text**
(`sha256(sql).slice(0, 16)`), and `migrate` refuses to run when an already-applied
migration's on-disk checksum no longer matches the one recorded in `schema_migrations`
("An applied migration was edited in place"). **Any byte change — a comment included —
breaks `migrate` on every environment that already applied it.**

Measured 2026-09-02 (`master-harness-e7d4z`):

```
$ psql -h 127.0.0.1 -p 5433 -U postgres -d clawmem -Atc \
    "select version, checksum from schema_migrations where version like '005%'"
005_fts_functions_pin_search_path|d268d7b0d7ff11e4

$ node -e 'const {createHash}=require("crypto");const fs=require("fs");
  console.log(createHash("sha256").update(
    fs.readFileSync("migrations/005_fts_functions_pin_search_path.sql","utf-8"),"utf-8"
  ).digest("hex").slice(0,16))'
d268d7b0d7ff11e4
```

So a stale or wrong comment in an applied migration is corrected **here**. This file is a
`.md`; `loadMigrations()` filters `f.endsWith(".sql")`, so the migrator ignores it.

## Correction — 005, `WHY IT SHOWS ON THE ORIGIN TIER FIRST`

`005_fts_functions_pin_search_path.sql` (lines 27–31) currently says:

> WHY IT SHOWS ON THE ORIGIN TIER FIRST: a partitioned table's data restores
> through its PARTITIONS, as separate TOC entries handled by parallel workers
> that each apply the empty search_path. `documents` happened to survive the
> same dump. That is luck, not a difference in correctness […]

**That explanation is wrong** — it is equally true of `documents`, which restored fine, so it
explains nothing. The measured mechanism:

Under `pg_restore -j`, a post-data item (`CREATE TRIGGER`) is held back only while a
`TABLE DATA` item **for the same relation** is in flight. A partitioned parent has **no
`TABLE DATA` item of its own** (its rows are dumped per child), so nothing defers its
trigger: it is created early, PostgreSQL clones it onto every partition, and every child
`COPY` dispatched afterwards fires the broken function. `documents_fts_trg` sits on a plain
table that *does* have its own data item, so it waited until after that table's `COPY` —
it survived by winning a race, and the race is decided by archive shape and size.

Instrumented on the real 621 MiB SFW archive: trigger visible at t+0.5 s, first
origin-partition `COPY` at t+2.3 s, `documents_fts_trg` not until t+7.4 s (after its own
`COPY` finished at t+4.2 s).

The migration's *conclusion* is unaffected and correct: `documents` survived by luck, not by
correctness, so all three functions are pinned. Only the stated reason changes.

Full write-up, TOC evidence and both traces:
`/home/bj/claude/master-harness/intelligence/technical/research/clawmem-restore-mode-mechanism-2026-09.md`

## Never `--disable-triggers` on a restore drill

A schema-primed `--data-only --disable-triggers -j4` restores a **definitely broken** database
at rc=0 with zero `COPY` failures and exact row-count parity (mode E in the doc above) — it
suppresses the cloned partition triggers, i.e. the only thing that can detect this defect
class. Adding it for speed turns the drill into a check that cannot fail.
