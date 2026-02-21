---
name: ab-algorithm-implementer
description: Implement the new algorithm alongside the old one, add synthetic comparison tests, keep a config toggle, and produce validation results.
---

## When to use
You have an approved design + integration constraints and need PR-ready code with evidence (accuracy/robustness/edge cases).

## Prerequisites
- Approved backlog item in `state/backlog.md`
- ADR exists (and covers any format/API/policy implications)
- Integration constraints from Pipeline Architect

## Outputs (deliverables)
- New algorithm implementation (isolated module/function)
- Config toggle wiring (old + new coexisting during validation)
- Synthetic fixture tests + A/B comparison results
- Validation gate results (fmt/clippy/test + targeted eval)
- Handoff note → Performance Engineer with A/B data

## Implementation rules
- **Do not replace the old code path initially**; land new code in parallel.
- Keep diffs minimal and localized to the owning seam.
- Do not introduce new public API/types without explicit approval.

## Required A/B test suite (synthetic)
### Accuracy
Report center error in px:
- mean, p50, p95 on standard eval set

### Robustness
Success rate under noise sweep (example sweep):
- blur: 0.5, 1.0, 2.0, 3.0 px

### Edge cases
Include at least:
- near-circular vs high-eccentricity ellipses
- small marker count
- markers near image borders

## Validation gates (mandatory)
- `cargo fmt --all`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-targets`
- plus any repo-specific eval command(s)

## Handoff note template (to Performance Engineer)
- What was benchmarked (functions + input classes)
- A/B results summary table (accuracy + robustness)
- How to reproduce tests and evaluations
- Any known performance risks (allocations, iteration counts, solver iterations)
