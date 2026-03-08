# Reviewer Report - TASK-011-reconcile-codebook-docs-invariants

- Task ID: `TASK-011-reconcile-codebook-docs-invariants`
- Role: `reviewer`
- Date: `2026-03-08`
- Status: `complete`

## Inputs Consulted
- `docs/handoffs/TASK-011-reconcile-codebook-docs-invariants/01-architect.md`
- `docs/handoffs/TASK-011-reconcile-codebook-docs-invariants/02-implementer.md`
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
- `git diff -- README.md book/src/cli-guide.md book/src/introduction.md book/src/why-rings.md book/src/detection-pipeline/decode.md book/src/marker-anatomy/coding-scheme.md docs/decisions/006-codebook-invariants.md docs/decisions/011-workspace-boundaries.md docs/decisions/013-deterministic-reproducibility.md`
- Reviewer validation commands:
  - `rg -n "minimum cyclic Hamming distance of 5|min cyclic Hamming:\\s+3|CODEBOOK_MIN_CYCLIC_DIST = 5|margin / 3|margin/3|guaranteed minimum Hamming distance|crates/ringgrid/src/codebook.rs" README.md book/src docs/decisions`
  - `cargo run --quiet -- codebook-info`
  - `mdbook build book`

## Summary
- Re-reviewed the documentation-only change set against the architect acceptance criteria and the actual committed codebook artifacts.
- Reproduced the key validation evidence: stale-claim grep is clean, `codebook-info` prints the expected `16 / 893 / 2 / 1 / 0x035D / 0x0E63` values, and the book builds successfully.
- The updated docs now consistently point at the correct generated artifact path `crates/ringgrid/src/marker/codebook.rs`, remove the stale `5`/`3`/`margin/3` claims, and document seed/provenance expectations.
- I found no blocking correctness or completeness issues for `DOCS-003`; one broader docs-boundary cleanup remains a non-blocking follow-up outside this task's narrow scope.

## Decisions Made
- Assessed this review against the architect handoff's codebook-fact and provenance acceptance criteria, not against broader README/decision-doc restructuring work reserved for other backlog items.
- Treated `crates/ringgrid/src/marker/codebook.rs`, `tools/codebook.json`, and the `ringgrid codebook-info` command as the authoritative evidence for numeric/codeword example validation.
- Accepted the implementer's small scope expansion inside `book/src/marker-anatomy/coding-scheme.md` because it removed adjacent stale defaults and left the chapter internally consistent with the shipped decoder.

## Files/Modules Affected (Or Expected)
- `README.md` - reviewed corrected top-level codebook facts, provenance wording, and regeneration path.
- `book/src/cli-guide.md` - reviewed the updated CLI example against reproduced command output.
- `book/src/introduction.md` - reviewed updated high-level codebook-capacity wording.
- `book/src/why-rings.md` - reviewed updated error-tolerance claims for correctness at minimum cyclic Hamming distance `2`.
- `book/src/detection-pipeline/decode.md` - reviewed updated decode-confidence wording.
- `book/src/marker-anatomy/coding-scheme.md` - reviewed updated defaults/formula/acceptance bullets and provenance wording.
- `docs/decisions/006-codebook-invariants.md` - reviewed updated invariant and generation contract wording.
- `docs/decisions/011-workspace-boundaries.md` - reviewed the narrowed codebook-artifact ownership update.
- `docs/decisions/013-deterministic-reproducibility.md` - reviewed generated-artifact reproducibility additions.

## Validation / Tests
- Commands run:
  - `rg -n "minimum cyclic Hamming distance of 5|min cyclic Hamming:\\s+3|CODEBOOK_MIN_CYCLIC_DIST = 5|margin / 3|margin/3|guaranteed minimum Hamming distance|crates/ringgrid/src/codebook.rs" README.md book/src docs/decisions`
  - `cargo run --quiet -- codebook-info`
  - `mdbook build book`
- Results:
  - `rg`: no matches (exit code `1`), confirming the targeted stale claims and the stale Rust artifact path are absent from the reviewed docs.
  - `cargo run --quiet -- codebook-info`: reproduced `bits=16`, `codewords=893`, `min cyclic Hamming=2`, `seed=1`, `first=0x035D`, `last=0x0E63`, matching the updated example exactly.
  - `mdbook build book`: passed locally with no reported errors.

## Risks / Open Questions
- `docs/decisions/011-workspace-boundaries.md` still contains broader pre-`INFRA-010` boundary language about library/CLI file-I/O responsibilities. The codebook-artifact sentence fixed by this task is now correct, but a fuller boundary refresh remains a separate docs follow-up rather than a blocker for `DOCS-003`.

## Next Handoff
- To: `Human`
- Requested action: mark `TASK-011-reconcile-codebook-docs-invariants` complete. Optional follow-up: track broader `docs/decisions/011-workspace-boundaries.md` boundary cleanup under `DOCS-002` or another docs-focused task if desired.

---

## Reviewer Required Sections

### Review Scope
- Reviewed the documentation changes for `TASK-011-reconcile-codebook-docs-invariants` against the architect plan, the implementer report, the committed codebook artifacts, and the reproduced CLI/docs-build evidence.

### Inputs Reviewed
- `docs/handoffs/TASK-011-reconcile-codebook-docs-invariants/01-architect.md`
- `docs/handoffs/TASK-011-reconcile-codebook-docs-invariants/02-implementer.md`
- Actual modified docs and their `git diff`
- Reproduced validation command outputs

### What Was Checked
- Correctness of all explicit numeric codebook facts in the edited docs
- Consistency of decode-confidence wording with current shipped decoder semantics
- Accuracy of the `ringgrid codebook-info` example output, including first/last codewords
- Provenance/reproducibility wording for `tools/gen_codebook.py`, seed `1`, and committed artifact locations
- Scope discipline: whether the doc-only edits stayed within the architect handoff and whether the one expanded chapter fix remained justified

### Findings
- None. I found no blocking or non-blocking implementation defects within the scope of `DOCS-003`.

### Test Assessment
- Adequate for a documentation-only task. The reproduced `rg` sweep, `codebook-info` output, and `mdbook build book` collectively cover the architect's required validation paths and confirm the updated examples and claims match the shipped artifacts.

### Risks
- Broader boundary-language cleanup in `docs/decisions/011-workspace-boundaries.md` remains open. Impact is limited because the codebook/provenance sentence touched by this task is now accurate, and the remaining drift is outside this task's narrow codebook-doc scope.

### Required Changes Or Approval Notes
- Non-blocking follow-up: if the maintainers want `docs/decisions/011-workspace-boundaries.md` to fully reflect the post-`INFRA-010` library API surface, do that in a broader docs-structure pass rather than reopening this task.

### Final Verdict
- Allowed values only:
  - `approved`
  - `approved_with_minor_followups`
  - `changes_requested`
- `approved_with_minor_followups`

### Handoff To Implementer Or Human
- To: `Human`
- Requested action: close `TASK-011-reconcile-codebook-docs-invariants`; optional minor follow-up is broader DEC-011 boundary wording cleanup in a separate docs task.
