# ADR-001: Adopt 4-Role Agent Development Workflow

- **Status:** accepted
- **Date:** 2026-02-15
- **Author role:** pipeline-architect
- **Supersedes:** none

## Context

ringgrid has grown to ~10K lines of math-heavy Rust with Python tooling. Development spans four distinct concerns that require different expertise:

1. **Algorithm correctness** — subpixel geometry, RANSAC, ellipse fitting, projective center
2. **API stability** — v1 public surface, config ergonomics, module boundaries
3. **Performance** — hot inner loops, allocation patterns, benchmark tracking
4. **Validation** — end-to-end scoring, synthetic data, cross-language tooling

Without structured handoffs, changes in one area can silently regress another (e.g., a performance optimization that shifts center error by 0.02 px, or an algorithm change that breaks API stability).

The project already has `.claude/skills/` defining coding patterns for specific concerns. What was missing was the coordination layer: who does what, in what order, and how work transfers between roles.

## Decision

Adopt a 4-role agent workflow:

- **Algorithm Engineer** — math primitives, uses `metrology-invariants` + `tests-synthetic-fixtures` skills
- **Pipeline Architect** — orchestration + API, uses `api-shaping` skill
- **Performance Engineer** — hot paths + benchmarks, uses `hotpath-rust` + `criterion-bench` skills
- **Validation Engineer** — testing + scoring + Python tools, uses `tests-synthetic-fixtures` + `metrology-invariants` skills

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
