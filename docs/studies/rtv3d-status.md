# rtv3d: detection status (November 2026, main at e167a14b)

Status of ringgrid on the rtv3d captures, the project's most important real
dataset. Every number below was measured on main at `e167a14b` (PRs #95–#99
merged) with `tools/run_rtv3d_eval.py` and the one-off analyses listed under
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
| **adaptive** (primary) | **4985** | 48.9, 56 | 1 | 2 | 212 / 2619 / 2154 | 371 ms, 557 ms | 37.8 s |
| adaptive + `--refine photometric` | **5874** | 57.6, 63 | 1 | 2 | 296 / 2944 / 2634 | 1296 ms, 2136 ms | 135.6 s |
| A, single pass [20, 56] px | 3987 | 39.1, 54 | 42 | 42 | 22 / 1927 / 2038 | 31 ms | 2.8 s |
| B, single pass [14, 66] px (default prior) | 3745 | 36.7, 51 | 41 | 42 | 27 / 1786 / 1932 | 30 ms | 2.8 s |
| D, single pass, per-strip prior | 2651 | 26.0, 0 | 59 | 60 | 27 / 1301 / 1323 | 24 ms | 2.7 s |

- **adaptive** runs the published CLI, `ringgrid detect --target …`, without
  `--marker-diameter`. That is `Detector::detect_adaptive` with the default
  config. A, B and D run `ringgrid-dev detect` with a fixed scale prior.
- **Runtime** is wall clock per CLI process. It includes start-up, PNG load,
  detection and the JSON write. Strategies run one after another on one macOS
  machine.
- **Photometric runtime is acceptable for offline calibration.** It is
  3.5× the projective adaptive run and 1.3 s per tile, or 2.3 min for the whole
  dataset. It is not fast enough for interactive use.
- **Single-pass strategies lose seven whole strips.** A, B and D find no IDs on
  strips 1, 6, 8, 9, 10, 11 and 12, because the marker size there is outside
  their scale prior. Adaptive finds 98–168 IDs on each of those strips.
- **Gate.** All five strategies are gated in
  `tools/ci/regression_baseline.json`. The adaptive strategy is the primary
  target (see *Gate* below).
- **Determinism.** The review run `adaptive-e167a14b`, made separately through
  the review server, reproduces the eval exactly: 4985 IDs, with the same
  per-band counts.

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
`round(0.983 × measured)`, the same ~1.7 % slack as before:

| strategy | min_total_decoded | other thresholds |
|---|---|---|
| adaptive (primary) | 4900 | `max_tiles_zero_ids` 1, `max_tiles_lt10_ids` 3 |
| adaptive_photometric | 5774 | `max_tiles_zero_ids` 1, `max_tiles_lt10_ids` 3 |
| A_old_default_20_56 | 3919 | |
| B_new_default_14_66 | 3681 | |
| D_per_strip_best | 2606 | |

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
