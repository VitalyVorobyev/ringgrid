# DEC-012: Error Handling Patterns

**Status:** Active
**Date:** 2025

## Decision

The crate uses different error strategies at different layers:

### Algorithm layer (internal)

- **`Option<T>`** for operations that may fail to produce a result
  (e.g., `fit_ellipse_direct` → `Option<Ellipse>`, degenerate inputs).
- **`Result<T, DomainError>`** for operations with categorized failure modes
  (e.g., `ConicError`, `HomographyError`, `ProjectiveCenterError`).
- **String reasons** for pipeline-level rejection (`Result<T, String>` with
  short diagnostic tags like `"fit_outer:too_few_points"`).

### Pipeline layer

- Individual marker failures are **silently skipped** — a failed fit/decode
  simply means that candidate is not included in the output.
- Aggregate diagnostics are logged via `tracing::debug!` / `tracing::warn!`
  (e.g., projective center application summary).
- No panics in the detection pipeline. All `unwrap()` calls are on
  mathematically guaranteed paths (e.g., `min()` on non-empty iterators).

### Public API layer

- `Detector::detect` and variants return `DetectionResult` (never `Result`).
  An empty `detected_markers` vec is the "failure" case.
- `BoardLayout::from_json_file` returns `Result<Self, Box<dyn Error>>`.

### Logging

- `tracing` crate, controlled by `RUST_LOG=debug|info|trace`.
- No `println!` or `eprintln!` in library code.
