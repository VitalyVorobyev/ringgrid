# Architecture decision records

## The rule

An ADR records one decision in about a page: the context, the decision, its
consequences, and the alternatives rejected, each with a one-line reason.
Measurements, tables and study narrative go in `docs/studies/`, and the ADR
links to them. When the decision changes, amend the ADR in place so that it
states the decision as it stands. There is no rule to keep an ADR forever:
delete it when another ADR replaces it, when it no longer matches the code,
or when it only restates a convention `.claude/CLAUDE.md` already holds. Git
keeps the history. Start from [`000-template.md`](000-template.md).

## Index

Status as checked against the code and the later ADRs on 2026-11-22.
ADR-001…017, 019 and 020 were removed that day as superseded, obsolete or out
of step with the code.

| ADR | Title | Status | Current decision |
|---|---|---|---|
| [018](018-c-cpp-vcpkg-api.md) | C / C++ / vcpkg API | accepted | Flat C ABI in `ringgrid-c` (JSON at the boundary, status codes), cbindgen header, header-only C++ wrapper, CMake package, vcpkg port. |
| [021](021-derived-origin-fiducials.md) | Derived origin fiducials | accepted | Rect lattice coordinates are centred; origin-dot positions are derived from the lattice (an L of three gaps at cell `(0, 0)`); only the dot radius is stored. |
| [022](022-photometric-center-refinement.md) | Photometric centre refinement | accepted | Opt-in `Photometric` refinement fits a model of the blurred printed pattern to every marker with a board position, with a covariance that holds under correlated noise. |
| [023](023-grid-first-finalize.md) | Grid-first finalize | accepted; amended by ADR-024, ADR-026, ADR-027 | One back half for every target: label every fitted ring, anchor, identify by cell, join and complete, final H, verify; ID correction only in the fallback. |
| [024](024-fail-closed-finalize.md) | Fail-closed finalize | accepted; amended by ADR-026, ADR-027, ADR-028 | A wrong output is worse than a missing one: no homography means no IDs, an anchor must beat chance, only perspective-corrected centres are reported, `min_confidence` 0.03, one board per image. |
| [025](025-plain-anchor-by-anchor-cell.md) | Plain anchor by anchor cell | accepted | Origin placements are enumerated by the relative cell that carries the triad (no cap), scored with one homography, and re-anchored after completion. |
| [026](026-large-plain-ring-recall.md) | Large plain ring recall | partially superseded by ADR-029 | Labels checked against the ring shapes (fail closed), ellipse-exact outer fit, adaptive scoring by board position, no code-band probe on plain targets, completion of uncorrected rings, local support for coded refits. Decision 3 is replaced by ADR-029. |
| [027](027-phase-aligned-exact-decode.md) | Phase-aligned exact decode | accepted | Phase-aligned sampling, dark-on-light only, exact match against the target's own codewords, anchor votes checked against the predicted code rotation. |
| [028](028-plain-recall-tail.md) | Verification on refined centres | partially superseded by ADR-029 | Local verification on the deviation field, unverifiable markers removed, photometric configurations verified on refined centres, adaptive candidates joined. Decision 4 is replaced by ADR-029. |
| [029](029-photometric-confidence-error-model.md) | Photometric confidence | accepted | A photometric centre's confidence uses its precision `1/(1 + σ_eff / 0.03 px)`, `σ_eff` from the centre covariance and the blur-gradient bias; no sharpness factor. |

### Length

Still longer than the rule, to be converted when next touched: 023, 024, 026,
027 and 028 carry measurement tables and experiment logs inline; 025 runs
past a page. ADR-022 and ADR-029 follow the rule; their evidence is in
[`docs/studies/photometric-center.md`](../studies/photometric-center.md).
