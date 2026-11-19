# rtv3d: detection status (November 2026, blur-aware inner edge)

Status of ringgrid on the rtv3d captures, the project's most important real
dataset. The strategy table, *Missing rings between detected neighbours* and
*Gate* were measured with the blur-aware inner-edge search (branch
`fix/blur-aware-inner-fit`, on top of `1e5ba23d`); *Contrast-relative edge
gate* on `fix/relative-edge-depth-gate` (on top of `aafce4f1`); every other
number below was measured on main at `e167a14b` (PRs #95–#99 merged), as labelled,
with `tools/run_rtv3d_eval.py` and the one-off analyses listed under
*Reproduce*. The data is private and lives outside the repository, in the
gitignored `data/rtv3d_0`. This document names tiles by file name only and
reports aggregates. Numbers labelled **measured** come from a run. Numbers
labelled **hypothesis** have not been tested yet.

## Setup

- **Scene.** The coded hex target (`default_hex`, 203 cells, pitch 8 mm, outer
  and inner radius 4.8 / 3.2 mm, base codebook) is shown on an e-reader
  display. A rig of six cameras images it, and each camera has a tilted
  (Scheimpflug) sensor. The plane of focus therefore crosses each image in a
  band. The middle rows of a tile are sharp. The top and bottom rows are
  heavily defocused.
- **Images.** There are 19 strips of 4320×540 pixels, and each strip holds six
  camera tiles of 720×540 (`tools/strip_tiles.py`). The eval skips
  `target_0` and `target_7` (`SKIP_STRIPS`). That leaves 17 strips, or
  **102 tiles**.
- **Board spec.** `tools/board/board_spec.json` is the built-in `default_hex`
  board in v6 form, and the eval passes it by default. `ringgrid-dev` gives
  identical results (timings aside) with and without it, checked on three tiles. The old
  `data/rtv3d_ringgrid/board_ringgrid.json` is a `ringgrid.target.v3` file
  with the same geometry and no code-band width. Only the local, untracked
  `py/` scripts read it, so nothing in the repository needs it.
- **Bands.** "Top / middle / bottom" means the decoded marker's centre lies in
  the top, middle or bottom third of the tile height (180 px each). The mean
  p98−p2 grey range of these bands is 79 / 111 / 131 grey levels (top /
  middle / bottom; **measured**). So the top band is both blurred and the
  lowest in contrast. The bottom band is blurred too, but its contrast is the
  highest of the three.

## Strategies and metrics (measured, 102 tiles)

| strategy | decoded IDs | IDs / tile (mean, median) | tiles with 0 IDs | tiles < 10 IDs | top / middle / bottom | runtime / tile (median, max) | total |
|---|---|---|---|---|---|---|---|
| **adaptive** (primary) | **8376** | 82.1, 99.5 | 0 | 0 | 1589 / 4089 / 2698 | 406 ms, 681 ms | 41.7 s |
| adaptive + `--refine photometric` | **9033** | 88.6, 107 | 0 | 0 | 1978 / 4156 / 2899 | 2018 ms, 2527 ms | 205.8 s |
| A, single pass [20, 56] px | 5904 | 57.9, 85.5 | 42 | 42 | 660 / 3012 / 2232 | 40 ms | 3.7 s |
| B, single pass [14, 66] px (default prior) | 4992 | 48.9, 73 | 40 | 42 | 373 / 2491 / 2128 | 37 ms | 3.3 s |
| D, single pass, per-strip prior | 3561 | 34.9, 0 | 58 | 60 | 345 / 1756 / 1460 | 26 ms | 3.1 s |

On `1e5ba23d` (before the blur-aware inner edge) the same columns read:
adaptive 7408 (bands 1298 / 3725 / 2385), adaptive + photometric 8981
(1927 / 4144 / 2910), A 5508, B 4645, D 3341; no tile with 0 or < 10 IDs
in either adaptive strategy.

On `e167a14b` / `aafce4f1` (before the contrast-relative edge gate, identical
on this eval) the same columns read: adaptive 4985 (1 tile with 0 IDs, 2
under 10, bands 212 / 2619 / 2154, 371 ms median), adaptive + photometric
5874 (1, 2, 296 / 2944 / 2634, 1296 ms), A 3987, B 3745, D 2651.

- **adaptive** runs the published CLI, `ringgrid detect --target …`, without
  `--marker-diameter`. That is `Detector::detect_adaptive` with the default
  config. A, B and D run `ringgrid-dev detect` with a fixed scale prior.
- **Runtime** is wall clock per CLI process. It includes start-up, PNG load,
  detection and the JSON write. Strategies run one after another on one macOS
  machine.
- **Photometric runtime is acceptable for offline calibration.** It is
  5× the projective adaptive run and 2.0 s per tile (it refines 52 % more
  markers than before the gate change), or 3.5 min for the whole dataset. It
  is not fast enough for interactive use.
- **Single-pass strategies lose seven whole strips.** A, B and D find (almost)
  no IDs on strips 1, 6, 8, 9, 10, 11 and 12, because the marker size there is
  outside their scale prior. Adaptive finds IDs on every tile.
- **Gate.** All five strategies are gated in
  `tools/ci/regression_baseline.json`. The adaptive strategy is the primary
  target (see *Gate* below).

## Missing rings between detected neighbours (measured)

The owner's complaint: a ring missing in a lattice line whose left and right
neighbours are well identified. A *gap* below is a board cell with no
reported ID whose two opposite lattice neighbours (along any of the three hex
axes) are both reported. Each gap was classified twice, with one-off
instrumentation (not committed): **(a)** by re-running completion's refit
alone at the neighbours' midpoint (outer fit → completion gates → inner
estimate → projective centre → confidence) and **(b)** by the reason the final
output contract dropped the ring the pipeline did hold at that cell.

| 102 tiles, adaptive | `1e5ba23d` | branch |
|---|---|---|
| gaps | 738 | **226** |
| (a) refit alone fails: inner estimate `PeakAtSearchWindowEdge` | 180 (24 %) | 3 |
| (a) inner estimate `ThetaInconsistent` | 73 (10 %) | 50 |
| (a) outer fit fails / confidence < 0.03 | 30 / 23 | 25 / 14 |
| (a) refit alone succeeds | 432 (59 %) | 134 |

What the pipeline held at a gap cell on `1e5ba23d` (all 114 tiles, 762 gaps,
classification (b)): a completion refit without an inner ellipse 473, a
fit-decode ring below `min_confidence` 148, a fit-decode ring without an inner
ellipse 75, a completion refit below `min_confidence` 45, nothing 21.

- **Inner-edge window (fixed).** Blur merges each ring stroke's two edges and
  pulls the inner edge's derivative peak towards the centre (the outer edge's
  outwards). On the reported rings the measured inner/outer ratio falls with
  the outer edge's relative width `w / R` (correlation −0.63; median 0.474 at
  `w/R` < 0.08, 0.427 at 0.2–0.25) down to the nominal window's inner bound
  0.408, which is where every `PeakAtSearchWindowEdge` failure sat (183 of
  183). A search over a wider window found those edges at a median 0.377.
  When the nominal search ends on that bound, the search now runs once more
  with the bound moved in by `(1 + r_inner) σ / R` (σ from the measured ramp
  width). Widening the window for *every* ring was tried first: it gained
  more on the synthetic sweep but on rtv3d the hole of a defocused ring has a
  dome-shaped profile whose slope then won the peak (17 rings in one tile
  removed by the axis-ratio filter at ratios 0.24–0.34), net +65 IDs only.
- **Fragile inner fits under adaptive (fixed).** The same cell's refit gets an
  inner ellipse in one adaptive candidate and not in another (sub-pixel
  differences in the outer fit), but the candidate join let whichever marker
  held a cell first keep it. The join now replaces a held marker the output
  contract would drop with another candidate's reportable one.
- **Low-confidence holders (fixed).** Completion only refit cells held by a
  ring without a corrected centre; a ring below `min_confidence` kept its cell
  and was then dropped. Any ring the output contract would drop now has its
  cell offered to the refit (and is put back when the refit fails).
- **Remaining 226 gaps** (classification (b), branch): completion refit
  without an inner ellipse 113, nothing held 43, refit below `min_confidence`
  40, fit-decode ring below `min_confidence` 25, fit-decode ring without an
  inner ellipse 14. The inner failures left are mostly `ThetaInconsistent`
  (rays disagree on the defocused inner edge).
- **`target_0*` (skipped strip, 6 tiles):** 174 → 195 IDs. Of the 37 in-frame
  cells next to a detection still missing, 20 fail the outer estimate
  (`OuterEstimateNoPolarityCandidates`), nearly all within one ring radius
  (35–40 px) of the tile border; 12 refit alone. The completion seed is the
  lattice prediction, not a proposal, so proposal re-centring cannot help
  them (not implemented). Loosening the outer estimate's per-ray agreement
  (0.75 px) to 0.2 R recovered 7 of the 37 and created 3 refits failing the
  3 px reprojection gate — not pursued.

All 102 tiles, `1e5ba23d` → branch (LOO: the 8-neighbour homography residual
proxy; the display's pixel snapping floors it, so only gross values count):

| | adaptive before | adaptive after | photometric before | photometric after |
|---|---|---|---|---|
| decoded IDs | 7408 | 8376 | 8981 | 9033 |
| duplicate IDs in a tile / ring within 3 px with another ID | 0 / — | 0 / 0 | 0 / — | 0 / 0 |
| same ID moved > 3 px | — | 3 | — | 0 |
| LOO > 2 px, all / conf ≥ 0.1 / conf ≥ 0.2 | 124 / 16 / 2 | 128 / 15 / 1 | 12 / 11 / 10 | 8 / 8 / 5 |
| IDs at conf ≥ 0.1 / ≥ 0.2 | 4175 / 2164 | 4282 / 2189 | 7981 / 5645 | 8033 / 5650 |

- 1227 adaptive IDs are new and 259 lost (all 114 tiles: 7714 → 8706). The new
  ones are low-confidence: 986 in [0.03, 0.1) (LOO > 2 px: 38), 215 in
  [0.1, 0.2) (3), 25 in [0.2, 0.4) (0), 1 above.
- **The two IDs that moved after #104** are back: `target_2_01` ID 93 (LOO
  1.49 → 0.45 px) and `target_7_03` ID 88 (its outer fit had taken a ring
  0.69 × its neighbours' radius; now 0.94 ×, confidence 0.035 → 0.277).
  Rings whose outer radius stays below the inner-as-outer `ratio_threshold`
  (0.75) of their neighbours' after recovery are no longer reported: on main
  both such rings were > 2 px off (LOO); on the branch there are none, so the
  rule changes no output there.

## Contrast-relative edge gate (measured)

Two changes, one PR (`fix/relative-edge-depth-gate`):

- **Edge-depth gate.** An outer edge point needed an absolute step of 0.08 of
  full scale (20 grey levels) between 2 px inside and 2 px outside it. The
  median edge step on dark rtv3d tiles is about 12 grey levels, and doubling
  every tile's brightness alone had lifted adaptive from 5110 to 6541 IDs on
  all 114 tiles. Now the point must sit on the edge's full monotone rise
  along its ray (ink plateau to paper plateau, however wide blur makes it;
  the point's own level in the middle 60 % of the rise), and the rise must
  be at least 0.4 of the ray's swing from the marker centre to half a
  radius past the edge, above an absolute noise floor of 0.03. Two earlier
  drafts failed: a ±2 px step against a ±4 px swing admitted weak code-band /
  inner-ring edges (6 IDs moved 3–6.6 px onto a smaller ring, LOO > 2 px
  51 → 171), and a ±2 px step against the whole ray's swing rejected
  defocused edges (a large plain plate blurred to σ = 6 px lost half its
  rings, `plain_large_ring_recall::focus_gradient` 79 → 39).
- **Completion reach.** The refit's edge search was capped at
  `completion.roi_radius_px` (0.75 × the default prior's nominal diameter =
  30 px), so `detect_adaptive`, which completes with the caller's config,
  could not refit rings of radius > 30 px (the `target_0` rings are 66–76 px
  across). The reach is now 1.4 × the cell's expected radius (neighbour
  median), the per-ray expected-radius gate's own bound.

All 114 tiles (the skipped strips included), `aafce4f1` → branch:

| | adaptive before | adaptive after | photometric before | photometric after |
|---|---|---|---|---|
| decoded IDs | 5110 | 7714 | 6022 | 9327 |
| duplicate IDs in a tile | 0 | 0 | 0 | 0 |
| ring within 3 px reported with another ID | — | 0 | — | 0 |
| same ID moved > 3 px | — | 2 | — | 0 |
| tiles with 0 / < 10 IDs | 3 / 8 | 0 / 0 | 3 / 7 | 0 / 0 |
| `target_0` tiles `_00`…`_05` | 17, 20, 12, 16, 0, 6 | 30, 34, 26, 26, 30, 28 | 20, 26, 16, 21, 0, 7 | 33, 40, 32, 32, 35, 34 |
| `target_7` tiles `_00`…`_05` | 14, 18, 8, 8, 0, 6 | 24, 28, 18, 11, 24, 27 | 14, 20, 10, 8, 0, 6 | 24, 35, 32, 22, 33, 30 |
| darkest third of tiles (median grey 25–51) | 1432 | 2429 | 1597 | 2981 |
| middle / brightest third | 1830 / 1848 | 2731 / 2554 | 2129 / 2296 | 3264 / 3082 |
| LOO centre residual: median, p95 (px) | 0.345, 1.112 | 0.391, 1.381 | 0.193, 0.646 | 0.213, 0.690 |
| LOO > 2 px | 51 | 151 | 19 | 13 |

- **The new adaptive IDs arrive at low confidence.** 2908 are new and 304
  lost (net +2604). Of the new ones 2164 are in [0.03, 0.1), 633 in
  [0.1, 0.2), 109 in [0.2, 0.4), 2 above; 117 of the 128 new IDs with LOO
  > 2 px are in [0.03, 0.1). The lost ones were weak too: 215 of 304 in
  [0.03, 0.1). Cut by confidence, the branch is better than main at equal or
  larger counts:

  | adaptive, LOO | conf ≥ 0.03 | conf ≥ 0.1 | conf ≥ 0.2 | conf ≥ 0.4 |
  |---|---|---|---|---|
  | before: n, > 1 px, > 2 px | 5077, 338, 51 | 3620, 125, 28 | 2295, 48, 16 | 298, 2, 1 |
  | after: n, > 1 px, > 2 px | 7714, 855, 151 | 4328, 175, 22 | 2260, 45, 4 | 251, 0, 0 |

- The two moved IDs (`target_2_01` ID 93, LOO 0.34 → 1.49 px, confidence
  0.048; `target_7_03` ID 88, LOO 3.5 px, confidence 0.035) are now worse;
  both sit just above the default `min_confidence` 0.03.
- The precision audit (synthetic, ground truth; `precision_full_sweep`) has
  0 wrong IDs, board positions or homographies before and after outside the
  unsupported two-board scenes; centres > 1 px off 336 → 290; in the quality
  dump the newly admitted markers are > 1 px off for 1.4 % in [0.03, 0.1),
  0.1 % in [0.1, 0.2), 0 % above (all markers: 312 → 178 > 1 px).
- **Remaining absolute floor (measured on a synthetic view).** The decoder's
  `min_decode_contrast` (0.03 of full scale across the 16 sector means) is
  still absolute: at σ = 2 px blur and 22 grey levels of contrast a view
  decodes nothing (182 → 0 IDs); with the floor at 0 it decodes 181. rtv3d
  tiles do decode, so it did not bind there.

### What changed recently

| | before #95 | after #95 (decoder) | after #98 | after #99 = e167a14b |
|---|---|---|---|---|
| A / B / D decoded | 3475 / 2813 / 2059 | 3509 / 2954 / 2163 | 3976 / 3710 / 2625 | **3987 / 3745 / 2651** (re-measured, identical) |

- **#95: phase-aligned decode with exact target-codeword matching**
  (ADR-027). Accepted raw decodes that contradict their lattice cell fell from
  26.9 % (981 / 3653) to **0.31 %** (13 / 4154). Final IDs on the 86 trusted
  tiles went from 4021 to 4062, with 0 wrong IDs before and after.
- **#98 / #99: recall on large oblique plain plates.** These PRs were aimed at
  IPGS, but they also lifted rtv3d single-pass totals by 13–26 %.
- **Adaptive over the same period.** The figures come from the review runs,
  which were made through the library (`detect_adaptive`, default config).
  The pre-#95 run `fail-closed-adaptive` (5492e3a8) had 4736 IDs, 6 tiles with
  0 IDs, 12 tiles under 10, and band counts of 134 / 2445 / 2157. On
  e167a14b, adaptive has 4985 IDs, 1 tile with 0, 2 tiles under 10, and band
  counts of 212 / 2619 / 2154 (**measured**). The run `quality-adaptive`
  (63ab7680) had 4087 IDs.

### Precision (measured; there is no ground truth, so these are proxies)

- **No duplicate IDs** within a tile, in any strategy.
- **No ID disagreements between strategies.** Every ring that adaptive and
  another strategy both report within 3 px carries the same ID. The pairs
  checked were 4880 (photometric), 3763 (A), 3573 (B) and 2539 (D), with 0
  disagreements.
- **Centre consistency (LOO).** For each decoded marker, a homography is fitted
  from its 8 nearest decoded neighbours (board mm → image px), leaving the
  marker itself out, and the residual of its centre is measured:

  | | median | p95 | over 2 px |
  |---|---|---|---|
  | adaptive | 0.34 px | 1.09 px | 50 / 4980 (1.0 %): 27 bottom, 19 middle, 4 top |
  | adaptive + photometric | **0.19 px** | **0.63 px** | **16 / 5868** (0.27 %) |
  | IDs only photometric reports (988) | 0.27 px | — | 9 |

- **No wrong ID was found.** These proxies cannot rule one out. ADR-027's
  truth set (86 tiles) had 0 wrong IDs.

## Remaining failure modes (with evidence)

1. **Top band (measured).**
   - The top band holds 4.3 % of the adaptive IDs (212 of 4985). The bottom
     band holds 43 %, although it is defocused as well.
   - The top band also has the lowest contrast: 79 grey levels, against 131 in
     the bottom band. Blur alone does not explain the gap; contrast and blur
     together are the likely cause (**hypothesis**).
   - Photometric refinement adds 85 top-band IDs (+40 %).
2. **Markers dropped for lack of a corrected centre (measured).** With the
   default (projective) refinement, every reported marker has an inner
   ellipse. Finalize does not report markers without a corrected centre
   (fail-closed, ADR-024). The photometric run reports 988 IDs that the
   adaptive run does not:
   - 320 of them have no inner ellipse: 222 in the bottom band, 108 in the
     middle, 12 at the top. In total, 342 photometric markers have no inner
     ellipse.
   - 668 have an inner ellipse but are still missing from the projective
     result. The mechanism is not measured yet: a rejected projective centre
     or a different adaptive candidate are both possible.
   - Their centres are consistent: LOO median 0.27 px, and 9 over 2 px.
   - This gap is the largest recall lever found so far, and it is
     concentrated in the blurred bottom band.
3. **The grid labeller falls back (measured, from tracing logs).**
   - Single pass B falls back on 43 of 102 tiles. 41 of them end with 0 IDs:
     the seven strips whose marker size is outside the [14, 66] prior.
   - Adaptive runs about 11 finalize passes per tile (scale tiers ×
     candidates). In 3 tiles no pass labels a grid: `target_12_04` (0 IDs),
     `target_6_04` (5) and `target_9_02` (10). Every other tile has at least
     one grid-labelled candidate.
   - Logs do not show which candidate was selected, so per-tile fallback
     counts for the final adaptive result are not available.
4. **Low-contrast tiles (measured: not the main tile-level factor).**
   - Tile contrast (p98−p2) does not predict the adaptive ID count:
     r = 0.01.
   - Mean IDs per tile by contrast tercile are 48.0, 49.4 and 49.2.
   - The two weak tiles sit at opposite ends of the contrast range.
     `target_6_04` has the lowest contrast in the set (49 grey levels).
     `target_12_04` has the highest (243) and still yields 0 IDs, which points
     to its geometry or labelling rather than to contrast.
5. **Skip list is stale (measured).** `target_0` and `target_7` are skipped as
   "zero-detection". Adaptive now decodes 71 IDs on `target_0` (90 with
   photometric) and 54 on `target_7` (58). Per tile, `target_0` gets
   17 / 20 / 12 / 16 / 0 / 6 IDs and `target_7` gets 14 / 18 / 8 / 8 / 0 / 6.
   The user's remark asks for at least 10 well-reconstructed markers on every
   `target_0` and `target_1` tile. All `target_1` tiles meet it (17–34 IDs).
   On `target_0`, tiles `_04` (0) and `_05` (6) do not.

## User feedback (`tools/out/review/rtv3d_0/feedback.json`)

### What the file holds

- 24 tiles have feedback, but no verdict is set on any of them.
- There are **35 `wrong_center` flags**: 23 on `fail-closed-adaptive` and 12 on
  `v0.15.0`. There are no other flag kinds and no missed-marker pins. Eight
  tiles carry a remark.
- **No flags on `quality-adaptive` does not mean that run is clean.** It was
  never opened: `runs_seen` lists only `fail-closed-adaptive` (21 tiles) and
  `v0.15.0` (7).

### The flagged markers on e167a14b

Flags were matched to current markers by ID in the same tile (**measured**).
32 flags fall on eval tiles, and 3 fall on the skipped strips.

| | still reported | not reported | still within 0.5 px of the flagged centre |
|---|---|---|---|
| adaptive (projective) | 14 | 18 | 2 (`target_20_05` ID 72, 0.08 px; `target_2_00` ID 23, 0.34 px) |
| adaptive + photometric | 30 | 2 | 0. All moved 0.7–5.2 px (median about 2.4 px), and all have LOO ≤ 1.3 px |

- On the skipped tile `target_0_00`, ID 23 is reported only by the photometric
  run, 3.3 px from the flagged centre. On `target_0_01`, ID 106 moved 1.8 px.
- **Whether the moved centres are now right is a hypothesis.** Only the user
  can confirm it in the review runs below.
- **The LOO proxy misses most of the user's flags.** Only 12 of the 34 flagged
  markers that have an ID had an LOO residual above 1.5 px in their own run.
  Centre quality still needs the review loop, or a better automatic
  reference.

### The remarks, checked against e167a14b (measured)

- "146, 142, 105 should be recoverable" (`target_19_00`): all 3 are now
  decoded.
- `target_4_02`: ID 103 is decoded. `target_4_03`: IDs 100 and 102 are
  decoded.
- `target_8_00`: 8 of the 9 listed IDs are decoded. ID 63 is still missing.
- `target_6_02`, "why no ids?": it had 0 IDs and now has 17 (21 with
  photometric).
- `target_1_04`, "about 30 % more": it had 16 IDs and now has 17 (21 with
  photometric, +31 %).
- `target_0_05`, "codes are visible": it had 0 IDs and now has 6 (7 with
  photometric). This is still short of the 10 the remark asks for.
- `target_0_00`, "outer ellipses on inner edges": not checked automatically.
  It needs a visual review.

## Candidate next steps (ranked)

1. **Decide whether photometric refinement becomes the recommended rtv3d
   configuration.**
   - Measured: +889 IDs (+17.8 %), LOO median 0.34 → 0.19 px, p95
     1.09 → 0.63 px, centres over 2 px 50 → 16. The cost is 3.5× runtime
     (1.3 s per tile).
   - Hypothesis: it also fixes the centres the user flagged. 30 of the 32
     flagged markers moved by 0.7–5 px.
   - How to measure: the user reviews `adaptive-photometric-e167a14b` against
     `adaptive-e167a14b` on the 24 flagged tiles (fewer `wrong_center` flags),
     and the eval gate keeps the totals.
2. **Find out why 668 rings with an inner ellipse are not reported under
   projective refinement**, and what blocks the 342 without one.
   - Expected effect (hypothesis): up to +988 IDs (+20 %; the 668 alone +13 %) for the default
     configuration, mostly in the bottom band, without paying the photometric
     runtime.
   - How to measure: count drop reasons per band with
     `detect_adaptive_with_diagnostics` (`inner_fit_status`, projective-centre
     rejection), then gate `total_decoded` and the LOO p95 in the eval.
3. **Bring `target_0` and `target_7` into the eval** (drop them from
   `SKIP_STRIPS`, re-bless the floors).
   - Measured: they yield 125 adaptive IDs, and 4 of their 12 tiles have
     10 or more.
   - Expected effect: the gate then covers the user's "at least 10 per snap"
     remark, which is currently unguarded.
   - Cost: a 114-tile baseline replaces the 102-tile history.
4. **Top-band recall.**
   - Hypothesis: low contrast together with defocus limits proposals and the
     outer fit there.
   - How to measure: per-band proposal and outer-fit counts on the top third,
     then per-band IDs in the eval (top is 212 now).
5. **The three tiles where no adaptive candidate labels a grid**
   (`target_12_04`, `target_6_04`, `target_9_02`).
   - The cause is unknown (hypothesis-level). `target_12_04` is high-contrast,
     so its failure is not photometric.
   - How to measure: the per-candidate labeller output on these tiles.
     Success means each tile gets 10 or more IDs and the eval's
     `tiles_lt10_ids` drops to 0.
6. **An automatic centre-quality reference for the eval.** The LOO proxy
   catches only about a third of the user's flags. Candidates are
   user-confirmed centres from `ok_confirmed` flags, or a cross-camera or pose
   check. The measure would be agreement with the user's flags on the 24
   reviewed tiles.

## Gate

`tools/ci/regression_baseline.json` (`rtv3d_benchmark`) sets each floor at
`round(0.983 × measured)`, the same ~1.7 % slack as before, from the
blur-aware inner-edge run:

| strategy | min_total_decoded | other thresholds |
|---|---|---|
| adaptive (primary) | 8234 | `max_tiles_zero_ids` 0, `max_tiles_lt10_ids` 1 |
| adaptive_photometric | 8879 | `max_tiles_zero_ids` 0, `max_tiles_lt10_ids` 1 |
| A_old_default_20_56 | 5804 | |
| B_new_default_14_66 | 4907 | |
| D_per_strip_best | 3500 | |

## Review runs

Two runs on e167a14b are in `tools/out/review/rtv3d_0/runs/`, next to the
earlier runs: `adaptive-e167a14b` and `adaptive-photometric-e167a14b`. They
cover all 114 tiles, including the skipped strips. The page opens the
`--primary` run as A:

```bash
.venv/bin/python tools/review/server.py --data data/rtv3d_0 --split 6 --open \
    --primary adaptive-e167a14b
```

Without `--primary`, the first run by name is A. That is also
`adaptive-e167a14b`.

## Reproduce

```bash
cargo build --release
.venv/bin/python tools/run_rtv3d_eval.py          # report: tools/out/rtv3d_eval/report.json
.venv/bin/python tools/review/summarize.py tools/out/review/rtv3d_0
```

The precision proxies (LOO residuals, cross-strategy ID agreement), the
contrast and band statistics, the feedback mapping and the fallback counts
came from one-off scripts over `tools/out/rtv3d_eval/detections/`. They were
not committed. Their method is described inline above.
