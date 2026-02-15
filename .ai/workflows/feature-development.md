# Feature Development Workflow

Use this workflow for: new pipeline stages, detection modes, API entry points, config additions, or new module capabilities.

## Prerequisites
- Task spec created from `templates/task-spec.md`
- Task added to `state/backlog.md`

## Phases

### 1. Specification (Pipeline Architect)

**Goal:** Define scope, API impact, and pipeline integration plan.

**Steps:**
1. Identify which of the 13 pipeline stages are affected
2. Assess public API impact:
   - New types needed? → define at construction site, re-export from `lib.rs`
   - New config fields? → extend `DetectConfig` sub-structs with `Default` values
   - New entry point? → justify why `Detector` methods don't suffice
3. Check backward compatibility — existing callers must not break
4. Create ADR if significant architectural decision involved
5. Write handoff note → Algorithm Engineer

**Deliverables:** Task spec, ADR (if needed), handoff note

### 2. Algorithm Implementation (Algorithm Engineer)

**Goal:** Implement the math primitives with proven correctness.

**Steps:**
1. Implement in the appropriate module:
   - Fitting/geometry → `conic/` or `ring/`
   - Detection logic → `detector/`
   - Decode/codebook → `marker/`
   - Camera/distortion → `pixelmap/`
2. Write unit tests with synthetic fixtures:
   - Deterministic inputs (seeded or hand-crafted)
   - Documented tolerances (0.1 px quick, 0.05 px precision)
   - Edge cases: image borders, extreme aspect ratios, small marker count
3. Verify pixel-center coordinate convention throughout
4. Write handoff note → Pipeline Architect

**Deliverables:** Implementation, unit tests, handoff note with test results

### 3. Pipeline Integration (Pipeline Architect)

**Goal:** Wire the new capability into the detection pipeline.

**Steps:**
1. Add to `pipeline/fit_decode.rs` (stages 1-6) or `pipeline/finalize.rs` (stages 7-13)
2. Update `pipeline/run.rs` if orchestration logic changes
3. Add config fields to `DetectConfig` hierarchy in `detector/config.rs`
4. Update `lib.rs` re-exports for any new public types
5. Update CLI in `ringgrid-cli` if new flags needed
6. Update `CLAUDE.md` pipeline documentation if stage flow changed
7. Write handoff note → Validation Engineer

**Deliverables:** Integrated pipeline, updated config, handoff note

### 4. Validation (Validation Engineer)

**Goal:** Verify correctness end-to-end.

**Steps:**
1. Run CI checks:
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --workspace --all-features
   ```
2. Run synthetic eval:
   ```bash
   python3 tools/run_synth_eval.py --n 3 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_feature
   ```
3. Compare scoring metrics against baseline
4. Update Python tooling if needed (new viz, new scoring metrics)
5. Write handoff note → Performance Engineer (if perf-sensitive) or Pipeline Architect (if not)

**Deliverables:** CI results, scoring comparison, handoff note

### 5. Performance Check (Performance Engineer) — conditional

**Goal:** Ensure no performance regression; add benchmarks for new hot paths.

**Steps:**
1. Add Criterion benchmark if new hot paths introduced
2. Profile for allocation regressions (no per-candidate `Vec` creation in loops)
3. Report benchmark results
4. Write handoff note → Pipeline Architect

**Deliverables:** Benchmark results, handoff note

### 6. Finalize (Pipeline Architect)

**Goal:** Close the loop.

**Steps:**
1. Review all handoff notes for the task
2. Verify CI passes on feature branch
3. Update `state/backlog.md` — mark task done
4. Human reviews and merges
