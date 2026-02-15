# Bug Fix Workflow

Use this workflow for: correctness regressions, detection failures, decode errors, accuracy degradation, or coordinate convention violations.

## Prerequisites
- Bug reported or observed (scoring regression, test failure, visual inspection)
- Task added to `state/backlog.md`

## Phases

### 1. Triage (Validation Engineer)

**Goal:** Reproduce, quantify, and localize the defect.

**Steps:**
1. Reproduce with minimal input (synthetic preferred over real images)
2. Quantify the defect:
   - Which markers are affected? (all, specific IDs, specific image regions)
   - Center error magnitude? Decode failure mode?
   - Detection precision/recall change?
3. Identify pipeline stage where defect originates:
   ```bash
   RUST_LOG=debug cargo run -- detect --image <path> --out /tmp/debug.json --marker-diameter 32.0
   ```
4. Use visualization for diagnosis:
   ```bash
   python3 tools/viz_detect_debug.py --debug <debug.json> --image <path> --out /tmp/debug_overlay.png
   ```
5. Create synthetic fixture that triggers the bug (deterministic, seeded)
6. Write handoff note → Algorithm Engineer with:
   - Reproduction case (command + input)
   - Pipeline stage identification
   - Quantified defect metrics

**Deliverables:** Reproduction fixture, defect quantification, handoff note

### 2. Root Cause Analysis & Fix (Algorithm Engineer)

**Goal:** Find and fix the root cause with a regression test.

**Steps:**
1. Trace through the identified pipeline stage code
2. Check common root causes:
   - Coordinate convention violation (pixel-center vs pixel-edge)
   - RANSAC threshold too tight/loose for the input
   - Edge cases: markers near image border, very small/large markers, high eccentricity
   - Numerical instability in eigenvalue solver or matrix operations
3. Implement fix in the appropriate primitive module
4. Write regression test using the provided synthetic fixture
5. Verify the fixture passes with documented tolerance
6. Write handoff note → Validation Engineer

**Deliverables:** Fix, regression test, handoff note with root cause explanation

### 3. Validation (Validation Engineer)

**Goal:** Verify fix and check for side effects.

**Steps:**
1. Run full test suite: `cargo test --workspace --all-features`
2. Run synthetic eval and compare to pre-bug baseline:
   ```bash
   python3 tools/run_synth_eval.py --n 5 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_bugfix
   ```
3. Verify the regression fixture passes
4. Verify no other metrics regressed (center error, decode rate, precision, recall)
5. Write handoff note → Performance Engineer (if fix touched hot path) or Pipeline Architect

**Deliverables:** CI results, scoring comparison, regression verification, handoff note

### 4. Performance Sanity (Performance Engineer) — conditional

**Goal:** Ensure the fix didn't introduce a performance regression.

**Steps:**
1. Run relevant Criterion benchmarks (if they exist for the changed code)
2. Confirm no significant latency regression
3. Write handoff note → Pipeline Architect

**Deliverables:** Benchmark comparison, handoff note

### 5. Close (Pipeline Architect)

**Goal:** Verify API integrity and close the task.

**Steps:**
1. Verify public API surface is unchanged
2. Verify CI passes
3. Update `state/backlog.md` — mark task done
4. Write session note summarizing root cause and fix
5. Human reviews and merges
