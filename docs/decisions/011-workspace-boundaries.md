# DEC-011: Workspace Crate Boundaries

**Status:** Active
**Date:** 2025

## Decision

The workspace has two crates with strict responsibility boundaries:

### `ringgrid` (library crate)

- All detection algorithms, math primitives, result types.
- No file I/O, no CLI argument parsing, no image file loading.
- Image input: `&image::GrayImage` (in-memory).
- Board input: `BoardLayout` struct (constructed by caller or loaded from JSON
  via `BoardLayout::from_json_file`).
- Output: `DetectionResult` struct (serializable to JSON via serde).

### `ringgrid-cli` (binary crate)

- CLI binary (`ringgrid`) with `clap`-based argument parsing.
- File I/O: image loading, JSON output writing.
- Depends on `ringgrid` — never the reverse.

### `tools/` (Python)

- Synthetic data generation, evaluation, scoring, visualization.
- Not part of the Rust workspace.
- Generates `codebook.rs` and board JSON — Rust code never generates these.

### Anti-patterns

- Adding `std::fs` or `clap` to the `ringgrid` library crate.
- Adding detection algorithm logic to `ringgrid-cli`.
- Making `ringgrid` depend on `ringgrid-cli`.
