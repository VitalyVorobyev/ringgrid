# AI Agent Development Workflow

Structured workflow for AI-assisted development of ringgrid.

## Agent Team

| Role | Skills | Responsibility |
|------|--------|---------------|
| **Project Lead** | — | Backlog, task specs, prioritization, milestone tracking. Owns `state/`. First point of contact for planning |
| **Algorithm Engineer** | `metrology-invariants`, `tests-synthetic-fixtures` | Math primitives: ellipse fitting, RANSAC, homography, projective center, decode. Owns `conic/`, `homography/`, `ring/`, `marker/` |
| **Pipeline Architect** | `api-shaping` | Pipeline orchestration, public API, config design, module boundaries. Owns `pipeline/`, `api.rs`, `lib.rs`, `detector/config.rs` |
| **Performance Engineer** | `hotpath-rust`, `criterion-bench` | Hot loop optimization, benchmarking, allocation profiling. Reviews any changed inner loop |
| **Validation Engineer** | `tests-synthetic-fixtures`, `metrology-invariants` | End-to-end testing, synthetic eval, scoring, Python tooling. Owns `tools/`, CI verification |

## Quick Start

1. **To discuss priorities or plan work** — start with [planning](workflows/planning.md) workflow (Project Lead)
2. **To execute a task** — pick a workflow based on task type (see below)
3. **Read the role prompt** for your assigned role in `roles/`
4. **Follow the workflow steps** in `workflows/`
5. **Hand off** using `templates/handoff-note.md` when your phase completes
6. **Human** reviews and merges to main

## Workflow Selection

| Task Type | Workflow | Starting Role |
|-----------|----------|--------------|
| Discuss priorities, plan milestones, triage new work | [planning](workflows/planning.md) | Project Lead |
| New pipeline stage, detection mode, or API entry point | [feature-development](workflows/feature-development.md) | Pipeline Architect |
| Correctness regression or defect | [bug-fix](workflows/bug-fix.md) | Validation Engineer |
| Latency or throughput improvement | [performance-optimization](workflows/performance-optimization.md) | Performance Engineer |
| Replace or enhance a math primitive | [algorithm-improvement](workflows/algorithm-improvement.md) | Algorithm Engineer |

## Directory Layout

```
.ai/
  roles/           System prompts for each agent role
  workflows/       Step-by-step process definitions
  state/
    backlog.md     Living task board
    decisions/     Architecture Decision Records
    sessions/      Handoff notes between agents
  templates/       Reusable templates for specs, handoffs, and reports
```

## Key Files

- Handoff rules: [workflows/handoff.md](workflows/handoff.md)
- Current tasks: [state/backlog.md](state/backlog.md)
- Task spec template: [templates/task-spec.md](templates/task-spec.md)

## Conventions

### Branch Naming
- `feat/` — new features
- `fix/` — bug fixes
- `perf/` — performance optimizations
- `algo/` — algorithm improvements

### Handoff Protocol
Every role transition uses `templates/handoff-note.md` saved to `state/sessions/YYYY-MM-DD-TASKID-from-to.md`. See [workflows/handoff.md](workflows/handoff.md) for details.

### CI Gates (must pass before handoff)
```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

### Dual-Language Context
- **Rust** (`crates/`): algorithms and library code
- **Python** (`tools/`): synthetic data generation, scoring, visualization, benchmarking
- Algorithm and performance changes are Rust-only; validation may involve both
