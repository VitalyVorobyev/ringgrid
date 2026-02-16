---
name: api-shaping
description: Apply this skill when changing ringgrid public API, output schemas, or configuration types. It guides small-surface API design, explicit contracts, serde-safe evolution, and migration-aware integration.
metadata:
  short-description: Public API and config contract design
---

# API Shaping

Use this skill for any public-surface change in `crates/ringgrid`: `Detector`, result types, config structs, and re-exports.

## Design Rules

1. Keep surface area small.
- Extend existing types before adding new entry points.
2. Make contracts explicit.
- Avoid implicit semantics (especially frames and optional behavior).
3. Keep `lib.rs` as re-exports only.
- Define types at construction sites, then re-export.
4. Keep defaults safe.
- New config fields must preserve existing behavior via `Default`.
5. Respect crate boundaries.
- Algorithms in `ringgrid`; CLI and file I/O in `ringgrid-cli`.

## API Change Workflow

1. Describe the API diff up front.
- Added, removed, renamed fields/types/methods.
2. Check compatibility scope.
- Is this non-breaking, soft-breaking, or intentional hard break?
3. Verify serialization contract.
- `DetectionResult` and related structs serialize cleanly.
- If format meaning changes, call out schema/version implications.
4. Align pipeline and tooling.
- Ensure pipeline producers and Python tools agree on field semantics.
5. Update docs and examples.
- README/API usage snippets must match the new contract.

## Acceptance Checklist

- Public diff is intentional and documented.
- No redundant config mirrors or adapter layers.
- Changed semantics are represented by explicit fields/types.
- Workflow handoff includes caller impact and migration notes.

## Anti-Patterns

- Adding new methods when config can express the behavior.
- Hiding frame or mode semantics behind "best effort" inference.
- Mixing public API definitions into `lib.rs`.
