# Implementer Report - TASK-010-rust-target-generation-api

- Task ID: `TASK-010-rust-target-generation-api`
- Role: `implementer`
- Date: `2026-03-08`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-010-rust-target-generation-api/03-reviewer.md`
- `docs/handoffs/TASK-010-rust-target-generation-api/01-architect.md`
- previous `docs/handoffs/TASK-010-rust-target-generation-api/02-implementer.md`
- `docs/templates/task-handoff-report.md`
- `.agents/skills/implementer/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/api-shaping/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/metrology-invariants/SKILL.md`
- `/Users/vitalyvorobyev/.codex/skills/tests-synthetic-fixtures/SKILL.md`
- `Cargo.toml`
- `crates/ringgrid/Cargo.toml`
- `crates/ringgrid/src/board_layout.rs`
- `crates/ringgrid/src/target_generation.rs`
- `crates/ringgrid/src/lib.rs`
- `crates/ringgrid/tests/target_generation.rs`
- `crates/ringgrid/tests/fixtures/target_generation/fixture_compact_hex.png`
- `crates/ringgrid/README.md`
- `tools/gen_synth.py`
- local dependency references:
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/image-0.25.9/src/images/buffer.rs`
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/png-0.18.1/src/encoder.rs`
  - `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/png-0.18.1/src/common.rs`

## Summary
- Kept the original additive `BoardLayout` and SVG/PNG generation API, then reworked the PNG file writer per reviewer feedback.
- `write_target_png` now uses an explicit PNG encoder instead of extension-driven `image.save(...)`, so it always emits PNG bytes.
- The PNG writer now embeds the requested DPI as PNG physical pixel dimensions (`pHYs`), matching the existing Python print-target contract.
- Updated the integration test to verify file-level PNG signature, `pHYs` metadata, and decoded pixel parity from the written artifact.

## Decisions Made
- Kept `BoardLayout` as the single public geometry contract; the rework is localized to the PNG file-writing path and test coverage.
- Chose the reviewer-approved “always write PNG regardless of path suffix” contract rather than rejecting non-`.png` paths, because it avoids extension-driven behavior entirely and keeps `write_target_png` semantically self-contained.
- Added a direct `png` crate dependency to encode `pHYs` metadata explicitly; the `image` crate path used before cannot set pixel dimensions metadata.
- Preserved the existing parity stance: SVG exactness is deterministic enough to compare as text, while PNG correctness is enforced at file contract plus decoded pixel level rather than compressed-byte identity.

## Files/Modules Affected (Or Expected)
- `Cargo.toml` - added workspace `png` dependency for explicit PNG encoding/metadata support.
- `crates/ringgrid/Cargo.toml` - pulled the workspace `png` dependency into the crate.
- `crates/ringgrid/src/target_generation.rs` - replaced generic image save with explicit PNG encoding and `pHYs` metadata emission.
- `crates/ringgrid/tests/target_generation.rs` - upgraded the file-writer test to inspect PNG signature and pixel-dimension metadata in addition to decoded pixels.
- `crates/ringgrid/README.md` - documented that the PNG writer embeds DPI print metadata.

## Validation / Tests
- Commands run:
  - `cargo fmt --all`
  - `cargo test -p ringgrid --test target_generation`
  - `cargo test -p ringgrid`
- Results:
  - `cargo fmt --all` passed.
  - `cargo test -p ringgrid --test target_generation` passed (`4 passed`).
  - `cargo test -p ringgrid` passed (`143` unit tests, `4` integration tests, `5` doc tests).

## Risks / Open Questions
- PNG byte-for-byte output is still intentionally not treated as stable because compression/filter details are encoder-specific; the enforced contract is now PNG format, DPI metadata, dimensions, and decoded pixels.

## Next Handoff
- To: `Reviewer`
- Requested action: confirm the explicit PNG writer now satisfies the file-level print contract (`pHYs` + PNG-only output) and that the strengthened integration test closes the gap from the prior review.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1 remained unchanged: `BoardLayout` still owns direct geometry construction and canonical JSON emission.
- Architect step 2 was refined to satisfy the reviewer: the file-oriented PNG wrapper now uses explicit PNG encoding with physical pixel metadata instead of delegating to the generic image save path.
- Architect step 3 was extended: the integration test now verifies the written file contract, not only rendered raster parity.

### Changes Made
- Added the `png` crate as a direct dependency so the library can control PNG encoding metadata.
- Added `TargetGenerationError::PngEncoding(...)` and a private `encode_png(...)` helper in `crates/ringgrid/src/target_generation.rs`.
- Changed `BoardLayout::write_target_png(...)` to:
  - render the raster as before,
  - create the output file directly,
  - encode PNG bytes explicitly,
  - embed `PngTargetOptions::dpi` as `pHYs` pixels-per-meter metadata.
- Kept the write method format-stable by always writing PNG bytes regardless of filename suffix.
- Strengthened the integration test to verify:
  - the written file starts with the PNG signature,
  - the file contains `pHYs` metadata with the expected pixels-per-meter value,
  - decoded pixels still match the committed reference fixture.
- Updated the README line covering `write_target_png` so the DPI metadata behavior is discoverable.

### Files Changed
- `Cargo.toml` - added workspace `png = "0.18"`.
- `crates/ringgrid/Cargo.toml` - added `png = { workspace = true }`.
- `crates/ringgrid/src/target_generation.rs` - explicit PNG encoding and metadata handling.
- `crates/ringgrid/tests/target_generation.rs` - file-level PNG contract regression coverage.
- `crates/ringgrid/README.md` - clarified PNG metadata behavior.

### Deviations From Plan
- None.
- The reviewer-approved contract choice is “always write PNG bytes regardless of suffix” rather than “reject non-`.png` paths”; this stays within the requested bounded follow-up options.

### Tests Added/Updated
- `crates/ringgrid/tests/target_generation.rs`
  - file writer now uses a non-`.png` suffix to prove the method is no longer extension-driven,
  - inspects PNG signature,
  - inspects `pixel_dims` / `pHYs`,
  - still checks decoded pixel parity against the committed fixture.

### Commands Run
- `cargo fmt --all`
- `cargo test -p ringgrid --test target_generation`
- `cargo test -p ringgrid`

### Results
- The PNG write path now emits real PNG files independent of filename suffix.
- The written file now carries DPI-derived physical pixel metadata, matching the Python print-target behavior the reviewer cited.
- Target-generation integration tests passed after the rework.
- The full `ringgrid` crate test suite still passes after the dependency and writer change.

### Remaining Concerns
- Exact compressed PNG bytes remain intentionally non-contractual; if future consumers need deterministic chunk ordering/compression choices, that should be a separate scope decision.

### Handoff To Reviewer
- Focus on `crates/ringgrid/src/target_generation.rs` to verify the new `encode_png(...)` path and the “always write PNG bytes” contract are appropriate for the public API.
- Verify the integration test now closes the reviewer finding by checking metadata and format at the file level instead of only decoding pixels.
- Confirm the added `png` dependency is a justified, localized implementation detail rather than an unwanted surface expansion.
