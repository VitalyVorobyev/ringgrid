# Reviewer Report - TASK-014-optional-extended-codebook-mode

- Task ID: `TASK-014-optional-extended-codebook-mode`
- Backlog ID: `ALGO-014`
- Role: `reviewer`
- Date: `2026-03-11`
- Status: `complete`

## Inputs Consulted
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/03-reviewer.md` (previous revision)
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/01-architect.md`
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/02-implementer.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/reviewer/SKILL.md`
- `tools/gen_codebook.py`
- `tools/tests/test_gen_codebook.py`
- `tools/codebook.json`
- `crates/ringgrid/src/marker/codebook.rs`
- `crates/ringgrid/src/marker/codec.rs`
- `crates/ringgrid/src/marker/decode.rs`
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/detector/id_correction/bootstrap.rs`
- `crates/ringgrid/src/detector/id_correction/consistency.rs`
- `crates/ringgrid/src/detector/id_correction/diagnostics.rs`
- `crates/ringgrid/src/detector/id_correction/engine.rs`
- `crates/ringgrid/src/detector/id_correction/homography.rs`
- `crates/ringgrid/src/detector/id_correction/local.rs`
- `crates/ringgrid/src/detector/id_correction/workspace.rs`
- `crates/ringgrid/src/pipeline/finalize.rs`
- `crates/ringgrid-cli/src/main.rs`
- `README.md`
- `crates/ringgrid/README.md`
- `book/src/cli-guide.md`
- `book/src/configuration/detect-config.md`
- `book/src/detection-modes/simple.md`
- `book/src/detection-pipeline/decode.md`
- `book/src/introduction.md`
- `book/src/marker-anatomy/coding-scheme.md`
- `book/src/output-types/detected-marker.md`
- `book/src/output-types/fit-metrics.md`
- `book/src/why-rings.md`
- `docs/decisions/006-codebook-invariants.md`
- reviewer-reproduced command outputs listed below

## Summary
- Re-reviewed the implementation after the requested changes and confirmed both previous blockers are resolved in the current tree.
- Reproduced the full required local CI baseline (`fmt`, `clippy`, workspace tests, rustdoc, doctests, `mdbook`, typing-artifact check, `maturin develop`, Python tests); all passed.
- Reproduced the risky reviewer-only checks directly:
  - the seed-mismatched `--base_json` path now emits source-seed metadata in both JSON and Rust artifacts
  - the previous `extended` polarity repro no longer exact-matches a complement entry
- Found no new blocking issues. The implementation is ready for approval with the documented deviation that `extended` is now the largest additive profile that does not widen shipped polarity ambiguity, yielding `2180` total entries instead of the earlier `4080` draft.

## Decisions Made
- Accepted the implementer’s deviation from the initial “full primitive-pool” extension attempt.
  - Reason: the architect plan prioritized preserving shipped baseline IDs and default behavior; the narrowed `extended` profile is the smallest change that satisfies that contract after the previous polarity regression was reproduced.
- Treated the seed-provenance fix as complete.
  - Reason: `tools/gen_codebook.py` now threads `effective_seed` through both emitters, and a reviewer-run mismatched-seed repro confirmed JSON and Rust outputs both record the loaded source seed.
- Treated the `extended` polarity regression as complete.
  - Reason: the previous exact-match repro (`0xFCA2`) no longer resolves to an appended exact match, and new Rust regressions cover both baseline ID `0` and the first appended ID under inverted polarity in `extended`.

## Files/Modules Reviewed
- `tools/gen_codebook.py` - loaded-seed handling, additive extension construction, and emitter inputs.
- `tools/tests/test_gen_codebook.py` - new generator regressions.
- `tools/codebook.json` - baseline compatibility, seed metadata, and updated `extended` counts.
- `crates/ringgrid/src/marker/codebook.rs` - regenerated constants and emitted seed metadata.
- `crates/ringgrid/src/marker/codec.rs` - profile invariants and complement-collision regression.
- `crates/ringgrid/src/marker/decode.rs` - profile-aware decode behavior and inverted-polarity regressions.
- `crates/ringgrid/src/detector/completion.rs` - active-profile exact-decode gate.
- `crates/ringgrid/src/detector/id_correction/*.rs` - active-profile min-distance propagation.
- `crates/ringgrid/src/pipeline/finalize.rs` - active-profile handoff into `id_correction`.
- `crates/ringgrid-cli/src/main.rs` - `codebook-info` / `decode-test` profile surfacing.
- `README.md`, `crates/ringgrid/README.md`, `book/src/*.md`, `docs/decisions/006-codebook-invariants.md` - documentation updates for the revised `extended` contract and counts.

## Validation / Tests
- Commands run:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `cargo doc --workspace --all-features --no-deps`
  - `cargo test --doc --workspace`
  - `mdbook build book`
  - `.venv/bin/python crates/ringgrid-py/tools/generate_typing_artifacts.py --check`
  - `.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release`
  - `.venv/bin/python -m pytest crates/ringgrid-py/tests -q`
  - `.venv/bin/python -m pytest tools/tests/test_gen_codebook.py -q`
  - `cargo test inverted_polarity_stays_stable_in_extended_profile -- --nocapture`
  - `cargo test appendix_excludes_new_complement_collisions -- --nocapture`
  - `cargo run --quiet -- codebook-info`
  - `cargo run --quiet -- decode-test --word 0x035D --profile extended`
  - `cargo run --quiet -- decode-test --word 0xFCA2 --profile extended`
  - reviewer-only mismatched-seed repro of `tools/gen_codebook.py` with a temporary baseline JSON whose top-level/base/extended seed was set to `99`
- Results:
  - All required local CI baseline commands passed.
  - `tools/tests/test_gen_codebook.py` passed (`2 passed`).
  - Targeted Rust regressions passed:
    - `test_decode_inverted_polarity_stays_stable_in_extended_profile`
    - `test_decode_appended_word_inverted_polarity_stays_stable_in_extended_profile`
    - `test_extended_profile_appendix_excludes_new_complement_collisions`
  - `codebook-info` reports:
    - `base`: `893 / dist 2 / seed 1 / first 0x035D / last 0x0E63`
    - `extended`: `2180 / dist 1 / seed 1 / first 0x035D / last 0x2CD3`
  - `decode-test --word 0x035D --profile extended` returns baseline ID `0`, distance `0`, margin `1`, confidence `1.000`.
  - `decode-test --word 0xFCA2 --profile extended` returns ID `496`, distance `1`, margin `0`, confidence `0.000`; it no longer exact-matches an appended extended entry.
  - Reviewer-only mismatched-seed repro produced:
    - warning on stderr: `using source JSON`
    - JSON output seeds: `99 / 99 / 99`
    - Rust output contained:
      - `Generated by \`tools/gen_codebook.py --n 893 --seed 99\`.`
      - `pub const CODEBOOK_SEED: u64 = 99;`
      - `pub const CODEBOOK_EXTENDED_SEED: u64 = 99;`

## Risks / Open Questions
- The shipped baseline still contains historical complement-equivalent classes. This rework prevents `extended` from introducing new ones, but it does not retroactively redesign the shipped baseline.
- If the project still wants a “full primitive-pool” 16-bit extension in the future, that goal now clearly conflicts with the current baseline/polarity contract and should be handled as a separate architect decision rather than folded into this task.

## Next Handoff
- To: `Human`
- Requested action: accept the task as complete and merge when convenient.

---

## Reviewer Required Sections

### Review Scope
- Reviewed the rework against:
  - architect requirements for baseline preservation, explicit opt-in extension, backward-compatible config/JSON behavior, and minimal docs/CLI surfacing
  - the two prior blocking findings:
    - `extended` inverted-polarity ambiguity
    - mismatched `--base_json` seed provenance
- Focused on correctness of the new generator constraint, truthfulness of emitted metadata, adequacy of the added regressions, and consistency of the updated docs/CLI outputs.

### Inputs Reviewed
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/01-architect.md`
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/02-implementer.md`
- prior `03-reviewer.md`
- actual changed code, generated artifacts, docs, and reproduced command outputs

### What Was Checked
- Seed handling through the `--base_json` load path into both JSON and Rust emitters.
- Additive extension construction and the complement-collision exclusion rule for appended words.
- `DecodeConfig` default/serde behavior and `extended` decode behavior under inverted polarity.
- CLI helper output for the updated profile counts.
- Documentation updates for the narrowed `extended` contract.
- Full local CI baseline coherence and reproducibility.

### Findings
- No blocking findings.
- Previous reviewer finding 1 is resolved.
  - Evidence: `tools/gen_codebook.py` now computes `effective_seed`, passes it to both `write_json(...)` and `write_rust(...)`, and the reviewer-only mismatched-seed repro emitted `99` in JSON and Rust metadata while still warning that source JSON was used.
- Previous reviewer finding 2 is resolved.
  - Evidence: `decode-test --word 0x035D --profile extended` still returns ID `0`, while `decode-test --word 0xFCA2 --profile extended` is no longer an exact match. New Rust regressions cover inverted polarity for both a baseline ID and the first appended extended ID.

### Test Assessment
- Adequate for approval.
- The new coverage directly exercises the two previous blind spots:
  - loaded-baseline seed provenance in generator output
  - `extended` inverted-polarity decode stability
- The full repo-local validation baseline also reproduced cleanly on the reviewer side.

### Risks
- Residual risk is limited to documented behavior, not an untested bug:
  - `extended` now trades capacity for a stricter ambiguity envelope than the earlier `4080` draft
  - baseline historical complement ambiguity remains by design

### Required Changes Or Approval Notes
- Approval note:
  - the documented deviation from the earlier `4080` extension attempt is accepted for this task because it preserves the shipped baseline contract and resolves the previously reproduced correctness regression.

### Final Verdict
- `approved`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: merge the approved implementation. No further implementer changes are required for `ALGO-014`.
