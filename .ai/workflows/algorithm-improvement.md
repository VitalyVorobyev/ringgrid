# Algorithm Improvement Workflow

Use this workflow for: replacing or enhancing a mathematical primitive (e.g., better ellipse fitting, improved decode strategy, new center correction method, alternative RANSAC variant).

## Prerequisites
- Limitation of current algorithm identified (accuracy ceiling, robustness issue, theoretical improvement available)
- Task added to `state/backlog.md`

## Phases

### 1. Design (Algorithm Engineer)

**Goal:** Document the current limitation and propose an improvement with mathematical justification.

**Steps:**
1. Document the current algorithm:
   - What it does, where it's implemented, what its known limitations are
   - Accuracy on current synthetic eval baselines
2. Reference relevant literature or known improvements
3. Write mathematical specification of the proposed change
4. Identify affected modules and downstream pipeline stages
5. Create ADR from `templates/adr.md` documenting the algorithmic choice
6. Write handoff note → Pipeline Architect with design document

**Deliverables:** Algorithm design document, ADR, handoff note

### 2. API Impact Assessment (Pipeline Architect)

**Goal:** Determine how the improvement integrates with the existing system.

**Steps:**
1. Does the improvement need new config knobs?
   - New field on existing config struct (preferred)
   - New enum variant (e.g., extending `CircleRefinementMethod`)
   - New config struct (last resort)
2. Does it require new public types?
3. Can old and new algorithms coexist behind a config toggle during validation?
4. Design migration path: toggle → validate → remove old (or keep as fallback)
5. Write handoff note → Algorithm Engineer with integration constraints

**Deliverables:** Integration design, handoff note

### 3. Implementation (Algorithm Engineer)

**Goal:** Implement new algorithm with A/B comparison capability.

**Steps:**
1. Implement new algorithm in isolated module/function first (don't replace old code yet)
2. Write synthetic fixture tests comparing old vs. new:
   - **Accuracy:** center error px (mean, p50, p95) on standard eval
   - **Robustness:** success rate under noise sweep (blur 0.5, 1.0, 2.0, 3.0 px)
   - **Edge cases:** near-circular vs high-eccentricity ellipses, small marker count, image borders
3. Keep old implementation accessible behind config toggle
4. Write handoff note → Validation Engineer with A/B comparison data

**Deliverables:** New algorithm, comparison tests, A/B results, handoff note

### 4. End-to-End Validation (Validation Engineer)

**Goal:** Full pipeline comparison: old algorithm vs new.

**Steps:**
1. Run full synthetic eval with old algorithm (baseline):
   ```bash
   python3 tools/run_synth_eval.py --n 10 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_algo_baseline
   ```
2. Run full synthetic eval with new algorithm:
   ```bash
   python3 tools/run_synth_eval.py --n 10 --blur_px 1.0 --marker_diameter 32.0 --out_dir tools/out/eval_algo_new
   ```
3. Produce comparison report using `templates/accuracy-report.md`:
   - Precision, recall, F1
   - Center error (mean, p50, p95, max)
   - Decode success rate
   - Homography reprojection error
4. Test under stress conditions (blur 3.0, noise):
   ```bash
   python3 tools/run_synth_eval.py --n 10 --blur_px 3.0 --marker_diameter 32.0 --out_dir tools/out/eval_algo_stress
   ```
5. Write handoff note → Performance Engineer

**Deliverables:** Accuracy report (baseline vs new), stress test results, handoff note

### 5. Performance Comparison (Performance Engineer)

**Goal:** Quantify the latency impact.

**Steps:**
1. Benchmark old vs new algorithm on representative inputs
2. Report per-function and per-detect-call latency change
3. If new algorithm is slower, quantify the accuracy-per-microsecond tradeoff
4. Fill in `templates/benchmark-report.md`
5. Write handoff note → Pipeline Architect + Algorithm Engineer

**Deliverables:** Benchmark report, handoff note

### 6. Decision & Integration (Pipeline Architect)

**Goal:** Make the adopt/reject decision and integrate.

**Steps:**
1. Review all evidence:
   - Accuracy: improved / same / regressed (from accuracy report)
   - Performance: faster / same / slower (from benchmark report)
   - Robustness: better / same / worse (from stress test)
2. Decide:
   - **Adopt:** Remove old code path, new algorithm becomes default
   - **Adopt as option:** Keep both behind config toggle (only if both have clear use cases)
   - **Reject:** Document reasoning in ADR, remove new code
3. If adopted:
   - Update `CLAUDE.md` pipeline documentation if applicable
   - Update ADR with decision outcome and evidence summary
4. Update `state/backlog.md` — mark task done
5. Human reviews and merges
