# ADR-001: Adopt 3-Role Agent Development Workflow

- **Status:** accepted
- **Date:** 2026-02-15
- **Author role:** pipeline-architect
- **Supersedes:** none

## Context

ringgrid has grown to ~10K lines of math-heavy Rust with Python tooling. Development spans four distinct concerns that require different expertise:

1. **Algorithm correctness** — subpixel geometry, RANSAC, ellipse fitting, projective center
2. **API stability** — v1 public surface, config ergonomics, module boundaries
3. **Performance** — hot inner loops, allocation patterns, benchmark tracking

Without structured handoffs, changes in one area can silently regress another (e.g., a performance optimization that shifts center error by 0.02 px, or an algorithm change that breaks API stability). Validation gates (tests, clippy, synthetic eval) are run by each implementor role before handoff rather than by a separate validation role.

The project already has `.claude/skills/` defining coding patterns for specific concerns. What was missing was the coordination layer: who does what, in what order, and how work transfers between roles.

## Decision

Adopt a 3-role agent workflow:

- **Algorithm Engineer** — math primitives, uses `metrology-invariants` + `tests-synthetic-fixtures` skills
- **Pipeline Architect** — orchestration + API, uses `api-shaping` skill
- **Performance Engineer** — hot paths + benchmarks, uses `hotpath-rust` + `criterion-bench` skills

Each implementor role runs its own validation gates (tests, clippy, synthetic eval with accuracy thresholds) before handoff, eliminating the need for a separate validation role.

Four workflows cover the common task types: feature development, bug fix, performance optimization, and algorithm improvement. Each workflow defines a specific role sequence with handoff points.

## Consequences

**Positive:**
- Structured handoffs prevent silent accuracy/performance regressions
- ADRs capture algorithmic decisions with evidence
- Backlog provides visibility into project state
- Skills are leveraged systematically rather than ad-hoc

**Negative:**
- Process overhead for trivial changes (mitigated: use judgment, skip workflow for single-line fixes)
- Markdown files to maintain

**Neutral:**
- No Rust or Python code changes
- No build or CI impact

## Evidence

N/A — process-only change.

## Affected Modules

None. This ADR creates `.ai/` directory only.
