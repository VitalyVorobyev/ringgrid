# Implementer Report - TASK-011-reconcile-codebook-docs-invariants

- Task ID: `TASK-011-reconcile-codebook-docs-invariants`
- Role: `implementer`
- Date: `2026-03-08`
- Status: `ready_for_review`

## Inputs Consulted
- `docs/handoffs/TASK-011-reconcile-codebook-docs-invariants/01-architect.md`
- `README.md`
- `book/src/cli-guide.md`
- `book/src/introduction.md`
- `book/src/why-rings.md`
- `book/src/detection-pipeline/decode.md`
- `book/src/marker-anatomy/coding-scheme.md`
- `docs/decisions/006-codebook-invariants.md`
- `docs/decisions/011-workspace-boundaries.md`
- `docs/decisions/013-deterministic-reproducibility.md`
- `crates/ringgrid/src/marker/codebook.rs`
- `crates/ringgrid-cli/src/main.rs`
- `tools/codebook.json`
- `tools/gen_codebook.py`

## Summary
- Updated the targeted docs so the shipped codebook facts now consistently match the committed artifacts: `n = 893`, `bits = 16`, `min_cyclic_dist = 2`, `seed = 1`.
- Corrected stale user-facing examples and formulas, including the `ringgrid codebook-info` sample output and the decode-confidence normalization wording.
- Added concise provenance/reproducibility notes tying `tools/gen_codebook.py --n 893 --seed 1` to the committed JSON/Rust codebook artifacts and clarifying that the achieved generated value is authoritative.
- Kept scope doc-only; no runtime Rust or Python behavior was changed.

## Decisions Made
- Used `crates/ringgrid/src/marker/codebook.rs`, `tools/codebook.json`, and the `codebook-info` CLI path in `crates/ringgrid-cli/src/main.rs` as the source of truth for all numeric/codebook example updates.
- Corrected `README.md`'s regeneration command to the actual generated Rust artifact path `crates/ringgrid/src/marker/codebook.rs`.
- Tightened codebook-guarantee wording to avoid overclaiming beyond minimum cyclic Hamming distance `2`, especially around single-bit correction.
- Narrowed `docs/decisions/011-workspace-boundaries.md` to codebook-artifact generation ownership instead of trying to re-document broader target-generation responsibilities that belong to other docs work.

## Files/Modules Affected (Or Expected)
- `README.md` - corrected top-level codebook facts, added provenance note, and fixed the regeneration path.
- `book/src/cli-guide.md` - updated the `ringgrid codebook-info` example output to match the current CLI.
- `book/src/introduction.md` - replaced overstated codebook-guarantee wording with the shipped baseline invariant.
- `book/src/why-rings.md` - aligned the codebook capacity/error-tolerance description with minimum cyclic Hamming distance `2`.
- `book/src/detection-pipeline/decode.md` - updated decode-confidence wording to use `CODEBOOK_MIN_CYCLIC_DIST`.
- `book/src/marker-anatomy/coding-scheme.md` - corrected decode defaults/formula wording and added explicit generated-artifact provenance.
- `docs/decisions/006-codebook-invariants.md` - refreshed the formal invariant record and provenance section to the shipped artifacts.
- `docs/decisions/011-workspace-boundaries.md` - replaced the stale board/codebook generation sentence with a tight statement about codebook artifacts.
- `docs/decisions/013-deterministic-reproducibility.md` - added a short generated-artifact reproducibility section for the committed codebook outputs.

## Validation / Tests
- Commands run:
  - `rg -n "minimum cyclic Hamming distance of 5|min cyclic Hamming:\\s+3|CODEBOOK_MIN_CYCLIC_DIST = 5|margin / 3|margin/3|guaranteed minimum Hamming distance" README.md book/src docs/decisions`
  - `sed -n '1,12p' crates/ringgrid/src/marker/codebook.rs && printf '\n---\n' && sed -n '1,12p' tools/codebook.json`
  - `cargo run --quiet -- codebook-info`
  - `mdbook build book`
- Results:
  - `rg`: no matches (exit code `1`), confirming the targeted stale claims/formulas were removed from the intended docs.
  - source-of-truth check: `crates/ringgrid/src/marker/codebook.rs` and `tools/codebook.json` both show `CODEBOOK_N = 893`, `CODEBOOK_BITS = 16`, `CODEBOOK_MIN_CYCLIC_DIST = 2`, `CODEBOOK_SEED = 1`.
  - `cargo run --quiet -- codebook-info`: printed `16`, `893`, `2`, `1`, first codeword `0x035D`, last codeword `0x0E63`, matching the updated CLI example.
  - `mdbook build book`: passed with no reported errors.

## Risks / Open Questions
- `book/src/marker-anatomy/coding-scheme.md` contained additional stale decode-default wording (`min_decode_confidence`, acceptance bullets). I corrected that while updating the codebook formula because leaving the surrounding defaults stale would have kept the chapter internally inconsistent.
- Broader README structure cleanup and wider docs organization are still out of scope here and remain better fits for `DOCS-002`.
- Full Python/Rust `DetectConfig` surface documentation is still out of scope and belongs to `DOCS-001`.

## Next Handoff
- To: `Reviewer`
- Requested action: verify the doc set now matches the committed codebook artifacts and current CLI output, and confirm the small scope expansion inside `book/src/marker-anatomy/coding-scheme.md` stayed aligned with shipped decoder defaults rather than drifting into broader config documentation.

---

## Implementer Required Sections

### Plan Followed
- Architect step 1: completed a targeted consistency sweep across the requested user-facing docs and examples.
- Architect step 2: refreshed the canonical decision docs with current invariants and reproducibility/provenance notes tied to generated artifacts.
- Architect step 3: ran the bounded validation set (`rg`, artifact inspection, `codebook-info`, `mdbook build`) and recorded the outcomes for review.

### Changes Made
- Replaced stale codebook claims (`5`, `3`, `margin/3`, generic “guaranteed minimum Hamming distance”) with wording that matches the shipped baseline profile and current decoder semantics.
- Updated the CLI example output to the current `codebook-info` values, including first/last codewords.
- Added concise provenance language explaining that `tools/gen_codebook.py --n 893 --seed 1` produces the committed codebook artifacts and that the achieved minimum cyclic Hamming distance recorded there is authoritative.
- Fixed the stale README regeneration output path from `crates/ringgrid/src/codebook.rs` to `crates/ringgrid/src/marker/codebook.rs`.

### Files Changed
- `README.md` - corrected codebook summary and regeneration/provenance text.
- `book/src/cli-guide.md` - corrected CLI example output.
- `book/src/introduction.md` - corrected high-level codebook capability wording.
- `book/src/why-rings.md` - corrected codebook guarantees/error-tolerance wording.
- `book/src/detection-pipeline/decode.md` - corrected confidence formula wording.
- `book/src/marker-anatomy/coding-scheme.md` - corrected decode defaults/formula/acceptance bullets and added provenance note.
- `docs/decisions/006-codebook-invariants.md` - corrected invariants and generation contract.
- `docs/decisions/011-workspace-boundaries.md` - corrected codebook artifact ownership wording.
- `docs/decisions/013-deterministic-reproducibility.md` - added generated-artifact reproducibility note.

### Deviations From Plan
- Minor deviation: while updating `book/src/marker-anatomy/coding-scheme.md` for the codebook formula/provenance work, I also corrected the stale documented `min_decode_confidence` default and acceptance bullets.
  - Reason + impact: the surrounding text would otherwise remain internally inconsistent with the shipped decoder even after the codebook formula update. This stayed within doc-only scope and reduced reviewer churn.

### Tests Added/Updated
- None.
- This task is documentation-only; validation was done through consistency checks, CLI output reproduction, and a docs build.

### Commands Run
- `rg -n "minimum cyclic Hamming distance of 5|min cyclic Hamming:\\s+3|CODEBOOK_MIN_CYCLIC_DIST = 5|margin / 3|margin/3|guaranteed minimum Hamming distance" README.md book/src docs/decisions`
- `sed -n '1,12p' crates/ringgrid/src/marker/codebook.rs && printf '\n---\n' && sed -n '1,12p' tools/codebook.json`
- `cargo run --quiet -- codebook-info`
- `mdbook build book`

### Results
- The stale numeric/formula claims targeted by the architect handoff are absent from the updated docs.
- The authoritative artifact files still show the expected shipped values: `893`, `16`, `2`, `1`.
- The CLI output now matches the updated book example exactly.
- The book builds successfully after the doc edits.

### Remaining Concerns
- I did not do a full prose sweep across all non-targeted docs outside `README`, `book/src`, and `docs/decisions`; review should stay focused on the architect-listed surfaces.
- `docs/decisions/011-workspace-boundaries.md` was corrected only enough to remove the stale codebook-generation claim. Any broader post-`INFRA-010` boundary rewrite should be handled separately if desired.

### Handoff To Reviewer
- Check that every explicit numeric codebook invariant in the edited docs matches `crates/ringgrid/src/marker/codebook.rs` and `tools/codebook.json`.
- Confirm the `ringgrid codebook-info` example in `book/src/cli-guide.md` matches the actual command output, including first/last codewords.
- Review the `book/src/marker-anatomy/coding-scheme.md` default/acceptance-bullet corrections as the only deliberate scope expansion beyond pure codebook-distance/provenance text.
