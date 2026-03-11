# Implementer Report - TASK-014-optional-extended-codebook-mode

- Task ID: `TASK-014-optional-extended-codebook-mode`
- Backlog ID: `ALGO-014`
- Role: `implementer`
- Date: `2026-03-11`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/03-reviewer.md`
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/01-architect.md`
- `docs/handoffs/TASK-014-optional-extended-codebook-mode/02-implementer.md` (previous revision)
- `.agents/skills/implementer/SKILL.md`
- `tools/gen_codebook.py`
- `tools/tests/test_gen_codebook.py`
- `tools/codebook.json`
- `crates/ringgrid/src/marker/codebook.rs`
- `crates/ringgrid/src/marker/codec.rs`
- `crates/ringgrid/src/marker/decode.rs`
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/detector/id_correction/*.rs`
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

## Summary
- Addressed the reviewer’s seed-provenance finding by carrying the loaded baseline JSON seed through both JSON and Rust emitters whenever `--base_json` is used.
- Reworked extended-profile generation so it appends the largest additive set that does not introduce new polarity-complement collisions beyond the fixed shipped baseline.
- Regenerated the embedded/runtime artifacts and updated docs/CLI examples to match the revised profile shape: `2180` total extended entries, `1287` appended, `min_cyclic_dist = 1`, last extended codeword `0x2CD3`.
- Added Rust and Python regressions covering the loaded-seed path, appended-word complement-collision exclusion, baseline ID `0` inverted-polarity stability in `extended`, and appended-word inverted-polarity stability in `extended`.

## Decisions Made
- Preserved the committed 893-word baseline as the source-of-truth prefix. Reviewer reproduction showed the shipped baseline already contains some historical complement-equivalent classes, so changing the baseline or redefining shipped IDs would have been out of scope.
- Fixed reviewer finding 1 by constraining the generated extension rather than by special-casing decode tie-breaks. The full primitive-pool extension (`4080`) was incompatible with the current fixed-baseline polarity behavior; the reworked extension is the maximal additive profile that does not widen that ambiguity envelope.
- Fixed reviewer finding 2 by making loaded-baseline provenance truthful: when the generator loads a baseline JSON, the emitted `seed` metadata now reflects the source JSON seed, not the CLI `--seed`.
- Kept the public API and runtime plumbing introduced in the first implementation pass (`CodebookProfile`, `DecodeConfig.codebook_profile`, active-profile min-distance propagation) unchanged apart from the new regression coverage and doc wording.

## Files/Modules Affected
- `tools/gen_codebook.py` - fixed loaded-seed provenance and reworked extended-profile construction to exclude new complement collisions beyond the shipped baseline.
- `tools/tests/test_gen_codebook.py` - added generator regressions for loaded-seed metadata and appended-word complement-collision exclusion.
- `tools/codebook.json` - regenerated with truthful seed metadata and the revised `extended` profile (`2180` total / `1287` appended).
- `crates/ringgrid/src/marker/codebook.rs` - regenerated embedded constants for the revised `extended` profile.
- `crates/ringgrid/src/marker/codec.rs` - updated profile docs and added the appended-word complement-collision invariant test.
- `crates/ringgrid/src/marker/decode.rs` - added `extended` inverted-polarity regressions for baseline ID `0` and the first appended ID.
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/detector/config.rs`
- `crates/ringgrid/src/detector/id_correction/bootstrap.rs`
- `crates/ringgrid/src/detector/id_correction/consistency.rs`
- `crates/ringgrid/src/detector/id_correction/diagnostics.rs`
- `crates/ringgrid/src/detector/id_correction/engine.rs`
- `crates/ringgrid/src/detector/id_correction/homography.rs`
- `crates/ringgrid/src/detector/id_correction/local.rs`
- `crates/ringgrid/src/detector/id_correction/workspace.rs`
- `crates/ringgrid/src/detector/marker_build.rs`
- `crates/ringgrid/src/lib.rs`
- `crates/ringgrid/src/marker/mod.rs`
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

## Validation / Tests
- Required baseline rerun completed on the final post-regeneration, post-`cargo fmt` tree.
- Reviewer-specific proof points also passed:
  - `codebook-info` now reports `extended: 2180 / dist 1 / seed 1 / last 0x2CD3`
  - `decode-test --word 0x035D --profile extended` resolves to baseline ID `0`
  - `decode-test --word 0xFCA2 --profile extended` no longer exact-matches an appended extended entry; it falls back to distance `1`, margin `0`
  - `tools/tests/test_gen_codebook.py` passes both the seed-provenance and complement-collision regressions

## Risks / Open Questions
- The shipped baseline still contains some historical complement-equivalent classes. This rework prevents `extended` from introducing new ones, but it does not retroactively change baseline ambiguity that already exists in shipped artifacts.
- The revised extension size (`2180`) is materially smaller than the original `4080` implementation because the stronger “no new complement collisions beyond baseline” constraint is now enforced.

## Next Handoff
- To: `Reviewer`
- Requested action: verify that the reviewer’s two findings are resolved, confirm the new `extended` profile count/docs (`2180` total / `1287` appended), and validate that the narrowed extension invariant is acceptable relative to the architect’s original capacity target.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1: preserved the committed baseline prefix and regenerated the additive profile around explicit `base` / `extended` artifacts.
- Reviewer rework: corrected the loaded-baseline seed contract and rebuilt the extension so `extended` no longer adds new complement-equivalent ambiguity beyond the shipped baseline.
- Architect step 2 and step 3: kept the profile-aware runtime plumbing intact, added targeted Rust/Python regressions, refreshed the docs/CLI surfaces, and reran the full validation baseline.

### Changes Made
- Generator and artifact changes:
  - added `complement_canonical(...)` in `tools/gen_codebook.py`
  - changed `build_extended_profile(...)` to block complement classes from the shipped baseline and from already-appended entries
  - introduced `effective_seed` so emitters use `base_seed` when the baseline comes from `--base_json`
  - regenerated `tools/codebook.json` and `crates/ringgrid/src/marker/codebook.rs` to the new `extended` size
- Rust regression coverage:
  - `crates/ringgrid/src/marker/codec.rs` now asserts that appended `extended` entries do not introduce new complement collisions against the final profile
  - `crates/ringgrid/src/marker/decode.rs` now covers inverted-polarity decoding in `extended` for baseline ID `0` (`0x035D`) and for the first appended ID
- Python regression coverage:
  - `tools/tests/test_gen_codebook.py` verifies mismatched `--base_json` seed handling preserves the source seed in emitted metadata
  - `tools/tests/test_gen_codebook.py` verifies appended `extended` entries do not collide with complement classes already present in the final profile
- Documentation refresh:
  - updated docs and CLI examples from `4080` to `2180`
  - updated wording from “full primitive-pool extension” to the narrower “no new complement collisions beyond the shipped baseline” contract

### Files Changed
- `tools/gen_codebook.py`
- `tools/tests/test_gen_codebook.py`
- `tools/codebook.json`
- `crates/ringgrid/src/marker/codebook.rs`
- `crates/ringgrid/src/marker/codec.rs`
- `crates/ringgrid/src/marker/decode.rs`
- `crates/ringgrid/src/detector/completion.rs`
- `crates/ringgrid/src/detector/config.rs`
- `crates/ringgrid/src/detector/id_correction/bootstrap.rs`
- `crates/ringgrid/src/detector/id_correction/consistency.rs`
- `crates/ringgrid/src/detector/id_correction/diagnostics.rs`
- `crates/ringgrid/src/detector/id_correction/engine.rs`
- `crates/ringgrid/src/detector/id_correction/homography.rs`
- `crates/ringgrid/src/detector/id_correction/local.rs`
- `crates/ringgrid/src/detector/id_correction/workspace.rs`
- `crates/ringgrid/src/detector/marker_build.rs`
- `crates/ringgrid/src/lib.rs`
- `crates/ringgrid/src/marker/mod.rs`
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

### Deviations From Plan
- The original implementation expanded `extended` to the full primitive-pool size (`4080`). Reviewer finding 1 demonstrated that this violates the fixed-baseline polarity behavior once complement classes are embedded as exact matches.
- Final implementation therefore narrows the extension to `2180` entries (`1287` appended). This is the maximal additive profile under the constraints:
  - preserve the shipped 893-word baseline prefix exactly
  - do not introduce new complement collisions beyond that baseline
- Risk impact:
  - positive: reviewer repro is resolved without changing shipped baseline IDs or decode defaults
  - tradeoff: `extended` capacity is lower than the original first-pass implementation
- Deviation approval note:
  - reviewer should confirm this narrower invariant is acceptable, because the shipped task semantics are now “largest safe additive extension” rather than “full primitive-pool extension”

### Tests Added/Updated
- `crates/ringgrid/src/marker/codec.rs`
  - `test_extended_profile_appendix_excludes_new_complement_collisions`
- `crates/ringgrid/src/marker/decode.rs`
  - `test_decode_inverted_polarity_stays_stable_in_extended_profile`
  - `test_decode_appended_word_inverted_polarity_stays_stable_in_extended_profile`
- `tools/tests/test_gen_codebook.py`
  - `test_gen_codebook_preserves_loaded_seed_metadata`
  - `test_gen_codebook_excludes_complement_collisions_in_extended_profile`

### Commands Run
- `python3 tools/gen_codebook.py --n 893 --seed 1 --out_json tools/codebook.json --out_rs crates/ringgrid/src/marker/codebook.rs`
- `cargo fmt --all`
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
- `cargo run --quiet -- codebook-info`
- `cargo run --quiet -- decode-test --word 0x035D --profile extended`
- `cargo run --quiet -- decode-test --word 0xFCA2 --profile extended`

### Results
- All required validation commands passed on the final tree.
- `cargo test --workspace --all-features` passed:
  - `153` crate tests
  - `4` target-generation integration tests
  - `6` CLI tests
  - `5` doctests
- `cargo doc --workspace --all-features --no-deps` and `cargo test --doc --workspace` both passed.
- `mdbook build book` passed and refreshed the rendered book away from stale `4080` references.
- Python binding checks passed:
  - typing artifacts up to date
  - `maturin develop` rebuilt and reinstalled the editable package
  - `crates/ringgrid-py/tests` passed (`28 passed`)
- Generator regression tests passed (`2 passed`).
- `cargo run --quiet -- codebook-info` reports:
  - `base`: `893`, min cyclic Hamming `2`, seed `1`, first `0x035D`, last `0x0E63`
  - `extended`: `2180`, min cyclic Hamming `1`, seed `1`, first `0x035D`, last `0x2CD3`
- `cargo run --quiet -- decode-test --word 0x035D --profile extended` reports baseline ID `0`, distance `0`, margin `1`, confidence `1.000`.
- `cargo run --quiet -- decode-test --word 0xFCA2 --profile extended` reports ID `496`, distance `1`, margin `0`, confidence `0.000`, confirming the reviewer’s former exact-match repro is no longer possible.

### Remaining Concerns
- The baseline profile remains historically fixed, including its pre-existing complement ambiguities. This change intentionally avoids widening that set rather than attempting a baseline redesign.
- If the architect intended “full primitive-pool extension” as a hard requirement independent of decode semantics, that requirement now conflicts with the current shipped baseline/polarity contract and would need an explicit follow-up design decision.

### Handoff To Reviewer
- Re-run the reviewer’s polarity repro:
  - `cargo run --quiet -- decode-test --word 0x035D --profile extended`
  - `cargo run --quiet -- decode-test --word 0xFCA2 --profile extended`
  - Expected: `0x035D` still maps to baseline ID `0`; `0xFCA2` is no longer an exact extended match.
- Re-run the mismatched-seed provenance path with a temporary baseline JSON whose top-level/base/extended `seed` is not `1`.
  - Expected: stderr still warns about using the source JSON, and emitted JSON/Rust metadata keep the source seed.
- Verify docs and CLI output now match the final artifact shape:
  - `extended`: `2180` total entries
  - `extension_n`: `1287`
  - last extended codeword: `0x2CD3`
