# ADR-027: Phase-Aligned Sampling, Exact Matching and Rotation-Checked Anchor Votes

- **Status:** accepted
- **Date:** 2026-11-18
- **Author role:** algorithm-engineer
- **Amends:** ADR-006 (codebook invariants: decoding matches the target's own
  codewords), ADR-023/024 (anchor vote)

## Context

ADR-023 measured that 27–30 % of the raw decodes on the rtv3d captures
contradict the lattice cell they sit on, and the grid-first back half has been
overriding them since. A measurement study (branch `study/decode-errors`,
`tools/decode_study/`) asked whether that is the blurred top and bottom of
those Scheimpflug images or the decoder itself. Truth: every pre-finalize ring
of the 86 rtv3d tiles whose anchor is solid, assigned to the cell its final
homography projects nearest; and a synthetic sweep of the real board
(`tests/support/blur_scene.rs`: blur σ 0.5–3 px, outer radius 8–20 px, tilt
0/35/55°, noise 2/6 grey levels, random roll; 180 images, ground-truth
homography).

| | rtv3d | synthetic |
|---|---|---|
| accepted decodes contradicting their cell | 26.9 % (981 / 3653) | 42.6 % (3754 / 8810) |
| by image row (top / middle / bottom) | 45 % (n = 66) / 25.8 % / 27.2 % | — |
| by outer edge width < 2.5 / 2.5–4 / ≥ 4 px | 16.4 % / 31.4 % / 46.2 % | — |

It is the decoder. Ranked by share of the wrong decodes:

1. **Sub-sector phase.** The decoder sampled sectors on a fixed parametric grid
   starting at θ = 0, and cyclic matching absorbs only whole sectors — the
   module doc claimed it absorbed the whole offset. At an offset of 0.4–0.5
   sector between that grid and the printed sectors every sampled sector
   straddles two printed ones: 89 % of the accepted decodes there were wrong
   (rtv3d; recall 6 %), 94 % on the synthetic sweep, against 6.3 % / 24 % at
   offsets below 0.1. Sharp rings: 2.3 % wrong below 0.25 sector, 35.7 %
   above.
2. **Polarity fallback.** All 508 decodes that took the complement word were
   wrong (52 % of the wrong decodes; 1818 of 1820 synthetic). 4240 of the
   14 288 rotated base codewords have a complement that is itself a rotated
   codeword; on the default board some board codeword's complement is another
   board codeword.
3. **Gates that cannot reject.** With minimum cyclic distance 2, 99.9 % of all
   16-bit words lie within one bit of a base codeword (normal polarity alone),
   37 % match one exactly in some polarity. Accepted decodes were only ever
   distance 0 / margin 2 (21.9 % wrong) or distance 1 / margin 1 (89.9 %
   wrong): `max_decode_dist`, `min_decode_margin` and `min_decode_confidence`
   never rejected anything.
4. Blur and noise flip isolated sectors (19.9 % per-bit error on isolated
   white sectors against 5.1 % on runs of four or more).
5. Outer fits on the wrong edge: 30 rings, all wrong, 3 % of the wrong
   decodes.

Rejected: sampling the band off its radius (sampling only the band centre:
27.5 % wrong; three rings inside the band: 27.4 %).

## Decision

1. **Phase-aligned sampling** (`marker/decode.rs`). Sample `16 ×
   samples_per_sector` points (default 8 per sector), each averaged over the
   radial rings; for each of the `samples_per_sector` sub-sector phases average
   the central half of every sector and keep the phase whose 16 means vary the
   most, smoothed over neighbouring phases (the middle of the plateau). The
   decode reports a continuous rotation, `DecodeMetrics::rotation` (whole
   sectors + phase).
2. **Normal polarity only.** Targets are printed dark on light. No capture mode
   in the library produces inverted rings, so polarity is not a per-image or
   per-target decision either — it is fixed.
3. **Exact match against the target's own codewords.** The word is matched
   against the codewords of the target's cells (`TargetCodes`) and accepted
   only at distance 0. The fraction of 16-bit words that match exactly falls
   from 21.8 % (whole base profile) to 5.0 % (203-cell default board).
   `max_decode_dist`, `min_decode_margin`, `min_decode_confidence` and
   `DecodeConfig::codebook_profile` (a copy of the target's) are removed;
   `DecodeMetrics` keeps `observed_word`, `best_id` and the new `rotation`.
4. **Rotation-checked anchor votes** (`pipeline/anchor.rs`). For each decoded,
   labeled ring and each lattice symmetry `g`, the code rotation the board
   predicts at that ring follows from the labeled patch's homography (its
   local linear map at the ring, composed with `g` in millimetres) and the
   ring's fitted ellipse. A decode votes for `g`'s translation only when its
   rotation agrees within `ROTATION_TOL_SECTORS = 0.5`. The chance model
   becomes `C(n, v) · (p_t^v + (R − 1) · p_o^v) · n_cells^(1 − v)` with
   `p_t = P_ROTATION_AGREES_TRUE = 0.6` and `p_o = P_ROTATION_AGREES_OTHER =
   0.1`, unchanged `MAX_CHANCE_WINNER = 1e-3`.

Measured for the new decoder against the true homography: correct decodes
deviate from the predicted rotation by at most 0.31 sector on rtv3d (4141
decodes; 99 % within 0.15) and 0.58 on the synthetic sweep (99.96 % of 7444
within 0.5). Misreads agree with the true symmetry's prediction 39 % of the
time (synthetic, 274 misreads; rtv3d 7 / 13) — they are mostly a few flipped
bits read at the true rotation — and with any one other symmetry at most
5.5 % (rtv3d 7.7 %, 1 / 13). The constants round both up.

Minimum agreeing votes (`min_votes`) for `n` decoded labeled rings, hex boards:

| cells | n = 2 | n = 3 | n = 4–9 | before (n = 2 / 3 / 5+) |
|---|---|---|---|---|
| 152, 203, 363 | none | 3 | 3 | none / 3 / 4 |
| 611, 893 | **2** | 3 | 3 | none / 3 / 3–4 |
| 1239 | **2** | **2** | 3 | none / 3 / 3 |

Two agreeing decodes now anchor only boards of 611 cells or more (rect
boards: the same cut). The rtv3d board (203 cells) still needs three.

## Evidence

Raw decodes, against the same truth (rtv3d: the old run's anchored final
homography on its 86 trusted tiles; synthetic: ground truth):

| | rtv3d before | rtv3d after | synthetic before | synthetic after |
|---|---|---|---|---|
| contradicting / accepted | 26.9 % (981 / 3653) | **0.31 %** (13 / 4154) | 42.6 % (3754 / 8810) | **3.55 %** (274 / 7718) |
| correct decodes / rings at a cell | 0.59 | **0.89** | 0.39 | **0.54** |

Synthetic wrong decodes by blur σ 0.5 / 1 / 1.5 / 2 px: 28 / 46 / 61 / 64 %
before, 0.9 / 4.0 / 8.3 / 5.8 % after; by outer radius 8 / 12 / 20 px: 68 / 50
/ 25 % before, 12.0 / 4.5 / 0.4 % after.

Final output (IDs scored against the same truth):

| | images | final IDs | wrong IDs | coded views anchored on the grid |
|---|---|---|---|---|
| rtv3d trusted, before → after | 86 | 4021 → **4062** | 0 → 0 | — |
| rtv3d all tiles, before → after | 102 | 4088 → **4162** | — | 91 → 93 |
| synthetic sweep, before → after | 180 | 9045 → **9148** | 0 → 0 | 73 → 78 |

Benchmarks (all 71 regression checks pass before and after; precision 1.0
everywhere):

| benchmark | metric | before | after |
|---|---|---|---|
| rtv3d A / B / D | `total_decoded` | 3475 / 2813 / 2059 | **3509 / 2954 / 2163** |
| target matrix, rect coded | `decoded_fraction` | 0.900 | **1.000** |
| target matrix, hex Extended | `decoded_fraction` / `ext_decoded_rate` | 0.696 / 0.554 | **0.994 / 0.979** |
| distortion, no correction | recall | 0.9672 | 0.9655 (floor 0.965) |
| blur2 | recall / centre error | 0.9246 / 0.2630 | 0.9207 / 0.2667 |
| reference | centre / H error | 0.0598 / 0.0190 | 0.0600 / 0.0195 |

Unit tests pin each part: a marker decodes at every sub-sector offset 0…0.5
with the right continuous rotation (a fixed grid rejects ID 0 at 0.35), an
inverted band never decodes as its own ID, a profile codeword no cell carries
is rejected, the predicted rotation of a similarity is exact, every relative
lattice rotation anchors, two correct decodes anchor a 650-cell board but not
the 203-cell one, and two misreads agreeing on one wrong placement do not
anchor when their rotation disagrees (and would without the check).

## Consequences

- **Where recall moved down.** A misread used to take its ring out of the
  labeling, and completion then refitted that cell — with an inner ellipse and
  a projective centre. Read correctly (or left undecoded), the same ring keeps
  its original fit; when that fit has no inner ellipse it has no corrected
  centre and the output contract drops it. In the golden
  `real_single_projective` case (53 → 48 markers, one gained) three of the six
  lost markers are exactly that; two more were completion fills of
  neighbouring cells and one a fit at confidence 0.037. The distortion and
  blur2 recall dips were not traced. Labeled rings without a corrected centre
  should be offered to completion's refit directly instead of relying on a
  misread for it (follow-up; completion is being reworked separately).
- Some synthetic views at the limit (outer radius 8 px, noise 6) lose their
  anchor: the old decoder's 109 decodes held 6 correct ones, 5 of which
  happened to agree; the new decoder's 22 hold 2.
- ID correction (fallback path): every decode is exact, so decoded markers are
  always strong seeds. The confirm-by-consistency pass, which promoted
  untrusted non-exact decodes, could no longer find one and is removed with
  its `confirm_by_consistency` switch; `seed_min_decode_confidence`, which now
  only gates markers carrying an ID without their own decode by their marker
  confidence, is renamed `seed_min_confidence`.
- Breaking API and config changes: see
  [`docs/migrations/0.16-to-0.17.md`](../migrations/0.16-to-0.17.md).

## Alternatives considered

- **Soft-decision matched filter** (correlate the 16 sector means with each
  codeword's blurred ±1 pattern, accept on a correlation margin): on rtv3d,
  phase-aligned, margin ≥ 0.08 gave 3974 correct / 6 wrong. The margin
  threshold was tuned on the same data and is noise-dependent (synthetic
  2.0 % wrong, 5.1 % at noise 6); exact matching needs no threshold.
- **Distance-0 gate for anchor votes only**: 11.7 % of the normal-polarity
  distance-0 decodes were still wrong on rtv3d with the fixed grid.
- **A sparser codebook**: 16 rotation-invariant sectors leave 4080 usable
  necklaces; a greedy cyclic code with minimum distance 3 holds about 108
  words (55 complement-safe), distance 4 about 47 — too few for a 203-cell
  board.
