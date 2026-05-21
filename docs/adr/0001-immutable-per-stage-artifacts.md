# Immutable per-stage artifacts for the arc table

## Status

Accepted — 2026-05-21.

## Context and decision

The per-arc table began as a single `{station}_{year}_arc_table.parquet` created
by the arc-table builder and then **mutated in place** by three later stages:
`extract` merged feature columns in, `af-correct` dropped and replaced the `AF`
column, and `normalize` added `*_norm` columns. Caching was keyed
inconsistently across those stages, and the builder's skip check ("does
`arc_table.parquet` exist") could not tell a fresh build from a
fully-processed table — so re-running the builder after re-running gnssrefl
`subdaily` silently kept the stale table.

We decided that **each stage writes its own immutable, named output file** and
reads the previous stage's file by name. The pipeline chains them, and a
stage's skip check becomes simply "does *my own* output already exist".

## Considered options

- **One arc-table file with consistent caching** — keep the single in-place
  mutated file but give every stage uniform skip/`force` logic keyed on a
  marker column. Rejected: the file's processing state still cannot be
  identified from disk, and in-place mutation keeps every stage coupled to the
  others' side effects.
- **Immutable per-stage artifacts** (chosen) — every artifact is traceable to
  the stage that produced it, each stage is independently cacheable, and
  `--force` semantics become obvious.

## Consequences

- A station-year produces a short chain of `arc_*` files instead of one
  (negligible disk; per-arc tables are small).
- The documented output-file list grows; the README and worked example are
  updated to describe the chain.
- Each stage's contract is "read file X, write file Y" — no hidden mutation of
  a shared artifact, which also makes the per-stage CLI subcommands genuinely
  standalone.
