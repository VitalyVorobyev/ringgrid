# ringgrid (Python)

Python bindings for the `ringgrid` detector (PyO3 + maturin).

## Install

From PyPI:

```bash
pip install ringgrid
```

With plotting helpers:

```bash
pip install "ringgrid[viz]"
```

From source (repository checkout):

```bash
pip install maturin
maturin develop -m crates/ringgrid-py/Cargo.toml --release
```

## Fast Start: Generate `board_spec.json` + Printable SVG/PNG

Installed-package target generation is available directly from `import ringgrid`:

```python
from pathlib import Path
import ringgrid

board = ringgrid.BoardLayout.from_geometry(
    8.0,
    15,
    14,
    4.8,
    3.2,
    name="ringgrid_200mm_hex",
)

board.to_spec_json(Path("board_spec.json"))
board.write_svg(Path("target_print.svg"), margin_mm=5.0)
board.write_png(Path("target_print.png"), dpi=600.0, margin_mm=5.0)
```

Key knobs:

| API | What it controls | Typical value |
|---|---|---|
| `BoardLayout.from_geometry(...)` | Board geometry (`pitch_mm`, `rows`, `long_row_cols`, radii) | `8.0`, `15`, `14`, `4.8`, `3.2` |
| `name=` | Optional explicit board name; omitted uses deterministic geometry-derived name | `"ringgrid_200mm_hex"` |
| `write_svg(..., margin_mm=...)` | Extra white border around the printable page | `3-10` |
| `write_png(..., dpi=...)` | PNG raster resolution and embedded print metadata | `300` or `600` |
| `write_png(..., include_scale_bar=...)` | Include or omit the default scale bar | `True` |

Outputs:

- `board_spec.json`
- `target_print.svg`
- `target_print.png`

Equivalent paths for the same geometry and output set:

- Rust CLI: `ringgrid gen-target --out_dir ... --pitch_mm 8 --rows 15 --long_row_cols 14 --marker_outer_radius_mm 4.8 --marker_inner_radius_mm 3.2 --name ringgrid_200mm_hex --dpi 600 --margin_mm 5`
- Python script from the repo: `tools/gen_target.py` with the same arguments
- Rust API: `BoardLayout::new` / `BoardLayout::with_name` plus `write_json_file`, `write_target_svg`, and `write_target_png`

Load this board in Python:

```python
from pathlib import Path
import ringgrid

board = ringgrid.BoardLayout.from_json_file(Path("tools/out/target_faststart/board_spec.json"))
cfg = ringgrid.DetectConfig(board)
detector = ringgrid.Detector(cfg)
# Convenience defaults: detector = ringgrid.Detector.from_board(board)
```

If you are working from a repository checkout and also need synthetic images or
ground truth, the repo tools under `tools/` still provide the combined
generation/evaluation workflow. The installed package target-generation API is
for board JSON + printable SVG/PNG only. The repo-level `tools/gen_target.py`
is a thin wrapper over this same installed-package surface.

Complete target-generation tutorial and full flag reference:
- https://vitalyvorobyev.github.io/ringgrid/book/target-generation.html

## Features

- Native `BoardLayout` target generation for canonical spec JSON + printable SVG/PNG
- Native `Detector` API with NumPy input support
- Full `DetectionResult` model objects with JSON round-trips
- Optional plotting helpers in `ringgrid.viz` (`pip install ringgrid[viz]`)

## Input Rules

- `Detector.detect(...)` accepts:
  - `np.ndarray` with `dtype=uint8` and shape `(H, W)` (grayscale)
  - `np.ndarray` with `dtype=uint8` and shape `(H, W, 3|4)` (RGB/RGBA, auto-converted to grayscale)
  - image file path (`str` or `pathlib.Path`)
- Other dtypes/shapes raise `TypeError`.

## DetectConfig Field Guide

`DetectConfig` is the full Python tuning surface for `Detector.detect(...)`,
`detect_adaptive(...)`, and `detect_multiscale(...)`.

```python
import ringgrid

board = ringgrid.BoardLayout.default()
cfg = ringgrid.DetectConfig(board)

# Section properties return copies: mutate, then reassign.
decode = cfg.decode
decode.codebook_profile = "extended"
decode.min_decode_margin = 2
cfg.decode = decode

# Or use convenience aliases for common one-field tweaks.
cfg.completion_enable = False
cfg.decode_min_confidence = 0.4

snapshot = cfg.to_dict()
print(snapshot["decode"]["codebook_profile"])  # "extended"
```

How Python `DetectConfig` behaves:

- `cfg.board` is the constructor input and stays read-only. It is not included
  in `cfg.to_dict()`.
- `cfg.to_dict()` returns the resolved native wire view. That is the easiest way
  to inspect the exact config the Rust detector will use.
- Section getters such as `cfg.decode`, `cfg.inner_fit`, and `cfg.self_undistort`
  return copies. Reassign the section after editing it, or use a convenience
  alias such as `cfg.decode_min_margin = 2`.
- `cfg.marker_scale` defaults to `14-66` px outer diameter and re-derives the
  scale-coupled search windows when you replace it.
- Board geometry derives `cfg.marker_spec.r_inner_expected` and
  `cfg.decode.code_band_ratio`. For `BoardLayout.default()`, those resolve to
  `0.48809522` and `0.74404764`.
- `cfg.circle_refinement` uses the Python enum
  `ringgrid.CircleRefinementMethod`, while `cfg.to_dict()["circle_refinement"]`
  stores the native wire strings `"ProjectiveCenter"` or `"None"`.

Default `marker_scale` derivations for `DetectConfig(BoardLayout.default())`:

- `proposal.r_min = max(0.4 * radius_min_px, 2.0)` -> `2.8`
- `proposal.r_max = 1.7 * radius_max_px` -> `56.100002`
- `proposal.nms_radius = max(0.8 * radius_min_px, 2.0)` -> `5.6`
- `edge_sample.r_max = 2.0 * radius_max_px` -> `66.0`
- `outer_estimation.search_halfwidth_px = max(max((radius_max_px - radius_min_px) * 0.5, 2.0), 13.0)` -> `13.0`
- `completion.roi_radius_px = clamp(0.75 * nominal_diameter_px, 24.0, 80.0)` -> `30.0`
- `projective_center.max_center_shift_px = 2.0 * nominal_outer_radius_px` -> `40.0`

Deeper theory and Rust-side derivation details:
- [Book: DetectConfig](https://vitalyvorobyev.github.io/ringgrid/book/configuration/detect-config.html)
- [Book: MarkerScalePrior](https://vitalyvorobyev.github.io/ringgrid/book/configuration/marker-scale-prior.html)

### Surface Map

| Surface | Type | What it controls |
|---|---|---|
| `cfg.board` | `BoardLayout` | Board geometry used to derive geometry-coupled defaults |
| `cfg.marker_scale` | `MarkerScalePrior` | Expected marker diameter range in working pixels |
| `cfg.proposal` | `ProposalConfig` | Scharr-vote proposal generation |
| `cfg.edge_sample` | `EdgeSampleConfig` | Radial edge sampling limits and density |
| `cfg.outer_estimation` | `OuterEstimationConfig` | Outer-radius hypothesis generation from radial peaks |
| `cfg.marker_spec` | `MarkerSpec` | Board-driven ring geometry assumptions |
| `cfg.outer_fit` | `OuterFitConfig` | Outer ellipse fit acceptance and scoring |
| `cfg.inner_fit` | `InnerFitConfig` | Inner ellipse fit acceptance and penalties |
| `cfg.decode` | `DecodeConfig` | Code-band sampling and decode strictness |
| `cfg.seed_proposals` | `SeedProposalParams` | Seed-injected proposals for multi-pass flows |
| `cfg.projective_center` | `ProjectiveCenterParams` | Projective-center recovery gates |
| `cfg.completion` | `CompletionParams` | Homography-guided recovery of missing IDs |
| `cfg.ransac_homography` | `RansacHomographyConfig` | Global homography fitting thresholds |
| `cfg.self_undistort` | `SelfUndistortConfig` | Division-model self-undistort estimation |
| `cfg.id_correction` | `IdCorrectionConfig` | Hex-lattice ID verification and recovery |
| `cfg.inner_as_outer_recovery` | `InnerAsOuterRecoveryConfig` | Recovery when outer fit locked onto the inner ring |

### Top-Level Controls

| Property | Default | Practical notes |
|---|---|---|
| `cfg.board` | constructor input | Read-only board layout. Replace the whole config if you need a different board. |
| `cfg.circle_refinement` | `ringgrid.CircleRefinementMethod.PROJECTIVE_CENTER` | Use `NONE` for raw ellipse centers, or keep `PROJECTIVE_CENTER` for the accuracy-oriented default. |
| `cfg.dedup_radius` | `6.0` | Final marker merge radius in pixels. Raise only if duplicate fits survive; lower if nearby valid markers merge incorrectly. |
| `cfg.max_aspect_ratio` | `3.0` | Rejects very elongated ellipses. Tighten when false positives are obviously non-circular; loosen only for extreme perspective. |
| `cfg.use_global_filter` | `True` | Enables homography-based outlier rejection. Turn it off when debugging local fits or when you want to inspect raw pre-homography detections. |

### Convenience Aliases

| Alias | Expands to | When to use it |
|---|---|---|
| `cfg.completion_enable` | `cfg.completion.enable` | Quick toggle for homography-guided completion |
| `cfg.self_undistort_enable` | `cfg.self_undistort.enable` | Quick toggle for division-model self-undistort inside `detect()` |
| `cfg.inner_fit_required` | `cfg.inner_fit.require_inner_fit` | Promote missing inner fits from soft penalty to hard reject |
| `cfg.homography_inlier_threshold_px` | `cfg.ransac_homography.inlier_threshold` | Tighten or loosen global homography inlier gating |
| `cfg.decode_min_margin` | `cfg.decode.min_decode_margin` | Reject ambiguous decodes more aggressively |
| `cfg.decode_max_dist` | `cfg.decode.max_decode_dist` | Limit how many bit errors a decode may contain |
| `cfg.decode_min_confidence` | `cfg.decode.min_decode_confidence` | Raise or lower overall decode strictness without rebuilding the section |

### `marker_scale`

This is the highest-leverage tuning section. Replacing `cfg.marker_scale`
recomputes the scale-coupled defaults in `proposal`, `edge_sample`,
`outer_estimation.search_halfwidth_px`, `completion.roi_radius_px`, and
`projective_center.max_center_shift_px`.

| Field | Default | Practical notes |
|---|---|---|
| `diameter_min_px` | `14.0` | Minimum expected outer diameter in working pixels. Raise it if markers are never tiny. |
| `diameter_max_px` | `66.0` | Maximum expected outer diameter. Narrow it to cut false positives; widen it only if markers truly get larger. |

If markers span a very wide range, prefer `detect_adaptive(...)` or
`detect_multiscale(...)` over one very wide `marker_scale`.

### `proposal`

Controls the first Scharr-gradient voting stage that proposes candidate marker
centers before local fitting.

| Field | Default | Practical notes |
|---|---|---|
| `r_min` | derived -> `2.8` | Minimum vote radius. Lower only for genuinely tiny markers. Re-derived from `cfg.marker_scale`. |
| `r_max` | derived -> `56.100002` | Maximum vote radius. Raise only if markers exceed your current size prior. Re-derived from `cfg.marker_scale`. |
| `grad_threshold` | `0.05` | Fraction of max gradient magnitude used to keep votes. Raise it in noisy scenes; lower it for low-contrast imagery. |
| `nms_radius` | derived -> `5.6` | Proposal dedup radius. Re-derived from `cfg.marker_scale`. |
| `min_vote_frac` | `0.1` | Minimum accumulator peak fraction relative to the best proposal. Raise to be stricter, lower to keep weaker peaks. |
| `accum_sigma` | `2.0` | Gaussian blur on the accumulator before NMS. Higher values smooth noisy peaks but can merge close candidates. |
| `max_candidates` | `None` | Optional hard cap on proposals. Use only when you must bound runtime in cluttered scenes. |

### `edge_sample`

Controls the radial rays used to collect inner and outer edge evidence around a
proposal.

| Field | Default | Practical notes |
|---|---|---|
| `n_rays` | `48` | Angular sampling density. More rays improve stability on oblique markers at extra cost. |
| `r_max` | derived -> `66.0` | Maximum sampling radius. Re-derived from `cfg.marker_scale`. |
| `r_min` | `1.5` | Minimum sampling radius. Rarely changed directly. |
| `r_step` | `0.5` | Radial step in pixels. Lower values sample more densely but cost more. |
| `min_ring_depth` | `0.08` | Minimum signed edge depth kept during sampling. Raise for noisy false edges; lower for low-contrast targets. |
| `min_rays_with_ring` | `16` | Minimum rays that must see a valid ring-like response. Raise for stricter geometry, lower for partial occlusion. |

### `outer_estimation`

Controls the radial profile stage that predicts an outer radius before the
ellipse fit. This stage is cheaper than a full fit, so it is a good place to
reject weak hypotheses early.

| Field | Default | Practical notes |
|---|---|---|
| `search_halfwidth_px` | derived -> `13.0` | Radius search window around the prior. Re-derived from `cfg.marker_scale` but never below the base default. |
| `radial_samples` | `64` | Samples per ray used to estimate radial peaks. Higher values help blurry targets at extra runtime. |
| `aggregator` | `"median"` | Cross-ray aggregation policy. The shipped value is the only one used in first-party docs/tests. |
| `grad_polarity` | `"dark_to_light"` | Expected outer-edge gradient direction. Match this only if you intentionally invert target contrast assumptions. |
| `min_theta_coverage` | `0.6` | Minimum fraction of rays with valid evidence. Raise it to reject partial arcs sooner. |
| `min_theta_consistency` | `0.35` | Minimum agreement fraction around the selected radius. Raise for stricter peak consensus. |
| `allow_two_hypotheses` | `True` | Lets the stage carry a strong secondary radius hypothesis into later scoring. Helpful for ambiguous profiles. |
| `second_peak_min_rel` | `0.85` | Relative strength required for that second hypothesis. Raise it to keep only nearly-tied alternatives. |
| `refine_halfwidth_px` | `1.0` | Per-ray refinement window around the selected peak. Raise slightly for blurrier edges. |

### `marker_spec`

Describes the expected ring geometry. The board layout drives
`r_inner_expected`, while the remaining fields control radial/theta sampling
and coverage checks.

| Field | Default | Practical notes |
|---|---|---|
| `r_inner_expected` | derived -> `0.48809522` | Expected inner/outer radius ratio after board-geometry padding. Usually change the board geometry, not this field. |
| `inner_search_halfwidth` | `0.08` | Half-width around `r_inner_expected` used to search for the inner ring. Widen only if the target design itself differs. |
| `inner_grad_polarity` | `"light_to_dark"` | Expected inner-edge polarity. Match this only if you intentionally invert the target rendering. |
| `radial_samples` | `64` | Samples per theta spoke. Raise for softer gradients; lower only if you are aggressively trading accuracy for runtime. |
| `theta_samples` | `96` | Angular samples around the ring. More samples help oblique or partially occluded markers. |
| `aggregator` | `"median"` | Cross-theta aggregation policy. The shipped value is the canonical path. |
| `min_theta_coverage` | `0.6` | Minimum angular coverage for a valid marker profile. Lower it only for partial visibility. |
| `min_theta_consistency` | `0.25` | Minimum angular consistency around the chosen profile. Raise to reject uneven or contaminated profiles earlier. |

### `outer_fit`

Controls the final outer-ellipse fit. Nested `ransac` is a
`ringgrid.RansacFitConfig`.

| Field | Default | Practical notes |
|---|---|---|
| `min_direct_fit_points` | `6` | Minimum points required for a direct algebraic fit. Rarely tuned. |
| `min_ransac_points` | `8` | Minimum points before RANSAC is attempted. Lower only for severe occlusion experiments. |
| `ransac.max_iters` | `200` | More iterations help with heavy outliers but cost runtime. |
| `ransac.inlier_threshold` | `1.5` | Sampson-distance threshold in pixels. Tighten for cleaner data; loosen for blur/distortion. |
| `ransac.min_inliers` | `6` | Minimum inlier count accepted by the outer fit. |
| `ransac.seed` | `42` | Deterministic RNG seed for the fit. |
| `size_score_weight` | `0.15` | Weight of size agreement in the outer-hypothesis score. Raise if size priors are highly reliable. |
| `max_angular_gap_rad` | `1.5707963267948966` | Largest allowed missing arc gap (pi/2 by default). Lower for stricter completeness, raise for partial arcs. |

### `inner_fit`

Controls the inner-ellipse fit that refines geometry and influences final
confidence. Nested `ransac` is a `ringgrid.RansacFitConfig`.

| Field | Default | Practical notes |
|---|---|---|
| `min_points` | `20` | Minimum points before an inner fit is attempted. Lower only when many markers are partially cropped. |
| `min_inlier_ratio` | `0.5` | Required RANSAC inlier fraction. Raise for cleaner scenes, lower for blur/heavy distortion. |
| `max_rms_residual` | `1.0` | Maximum RMS Sampson residual. Tighten to reject sloppy inner fits. |
| `max_center_shift_px` | `12.0` | Largest allowed shift between outer and inner centers. Raise only if strong perspective or distortion genuinely moves the fitted center more. |
| `max_ratio_abs_error` | `0.15` | Maximum deviation from the expected inner/outer ratio. Tighten for well-calibrated, fixed targets. |
| `local_peak_halfwidth_idx` | `3` | Radial index half-width around the predicted inner peak. |
| `ransac.max_iters` | `200` | More iterations help when inner edges are noisy. |
| `ransac.inlier_threshold` | `1.5` | Inner-fit Sampson threshold in pixels. |
| `ransac.min_inliers` | `8` | Minimum inliers required for the inner fit. |
| `ransac.seed` | `43` | Deterministic RNG seed for the inner fit. |
| `miss_confidence_factor` | `0.7` | Confidence multiplier when the inner fit is missing. Lower values punish missing inner rings more strongly. |
| `max_angular_gap_rad` | `1.5707963267948966` | Largest missing inner-edge arc accepted before rejection. |
| `require_inner_fit` | `False` | Soft by default. Set to `True` when you want to reject any marker that lacks a trustworthy inner ellipse. |

### `decode`

Controls code-band sampling and codebook matching.

| Field | Default | Practical notes |
|---|---|---|
| `codebook_profile` | `"base"` | String selector: `"base"` keeps the stable shipped IDs `0..892`; `"extended"` opts into the additive larger profile. |
| `code_band_ratio` | derived -> `0.74404764` | Sampling radius inside the outer ellipse. Derived from board geometry and usually not tuned directly. |
| `samples_per_sector` | `5` | Angular intensity samples per bit sector. Raise for blur, lower only for aggressive speed tradeoffs. |
| `n_radial_rings` | `3` | Radial samples across the code band. More rings improve robustness on soft edges. |
| `max_decode_dist` | `3` | Maximum Hamming distance accepted. Lower it to reject noisy decodes more aggressively. |
| `min_decode_confidence` | `0.3` | Overall decode-confidence floor. Raise this first when you want stricter decoding. |
| `min_decode_margin` | `1` | Rejects ambiguous ties by default. Raising it is a strong way to prefer only very clear decodes. |
| `min_decode_contrast` | `0.03` | Minimum sampled code-band contrast before decoding. Lower only for low-contrast images. |
| `threshold_max_iters` | `10` | Iteration cap for the internal 2-means threshold refinement. |
| `threshold_convergence_eps` | `0.0001` | Convergence epsilon for that refinement loop. |

Typical decode tuning:

- Too many ambiguous IDs: raise `cfg.decode_min_margin` or lower `cfg.decode_max_dist`.
- Good geometry but noisy contrast: lower `min_decode_contrast` slightly before widening geometry gates.
- Need IDs beyond the stable shipped set: set `codebook_profile` to `"extended"` explicitly.

### `seed_proposals`

Controls proposal injection from already-known seed centers during multi-pass
or guided workflows.

| Field | Default | Practical notes |
|---|---|---|
| `merge_radius_px` | `3.0` | Seed/proposal merge distance. Raise only if seed centers are systematically off by several pixels. |
| `seed_score` | `1000000000000.0` | Score assigned to injected seeds so they survive proposal ranking. |
| `max_seeds` | `512` | Optional cap on consumed seeds. Use it to bound runtime when external seed lists get large. |

### `projective_center`

Controls the center-refinement stage used when
`cfg.circle_refinement == ringgrid.CircleRefinementMethod.PROJECTIVE_CENTER`.

| Field | Default | Practical notes |
|---|---|---|
| `use_expected_ratio` | `True` | Uses the board-driven inner/outer ratio as a prior in the selector. Usually leave this on. |
| `ratio_penalty_weight` | `1.0` | Strength of that ratio prior. Lower it if you need the selector to trust raw conic evidence more. |
| `max_center_shift_px` | derived -> `40.0` | Maximum accepted correction jump. Re-derived from `cfg.marker_scale`. |
| `max_selected_residual` | `0.25` | Rejects unstable projective-center candidates. Raise only if valid markers are failing this gate. |
| `min_eig_separation` | `1e-06` | Guards against unstable conic-pencil eigenpairs. Lower only if you have evidence the default is too strict. |

### `completion`

Completion tries to recover missing IDs at homography-projected board
locations. It only runs when a valid homography is available.

| Field | Default | Practical notes |
|---|---|---|
| `enable` | `True` | Set to `False` to inspect only directly fit-decoded markers. |
| `roi_radius_px` | derived -> `30.0` | Radius of the completion search ROI. Re-derived from `cfg.marker_scale`. |
| `reproj_gate_px` | `3.0` | Maximum allowed distance between the fitted center and the projected board position. |
| `min_fit_confidence` | `0.45` | Minimum confidence for a recovered completion marker. |
| `min_arc_coverage` | `0.35` | Minimum fraction of rays that found both edges. Lower only for heavy occlusion. |
| `max_attempts` | `None` | Optional cap on attempted missing IDs. |
| `image_margin_px` | `10.0` | Skip projected centers too close to the image boundary. |
| `require_perfect_decode` | `False` | Strong safety gate for distortion-heavy scenes without a trusted mapper. |
| `max_radii_std_ratio` | `0.35` | Rejects fits with highly inconsistent outer radii across rays. |

### `ransac_homography`

Controls global homography fitting from decoded markers.

| Field | Default | Practical notes |
|---|---|---|
| `max_iters` | `2000` | Iteration budget for the global RANSAC loop. |
| `inlier_threshold` | `5.0` | Pixel reprojection threshold for inliers. Tighten for cleaner data; loosen for more distortion or weak initial geometry. |
| `min_inliers` | `6` | Minimum correspondences accepted for a homography. |
| `seed` | `0` | Deterministic RNG seed for repeatable fitting. |

### `self_undistort`

Controls the optional one-parameter division-model self-undistort flow.
`cfg.self_undistort_enable = True` affects `Detector.detect(...)`, but
`Detector.detect_with_mapper(...)` always uses the mapper you pass in instead.

| Field | Default | Practical notes |
|---|---|---|
| `enable` | `False` | Turn on only when you want `detect()` to estimate distortion first. |
| `lambda_range` | `[-8e-07, 8e-07]` | Search interval for the division-model parameter. Widen only if you know distortion is stronger. |
| `max_evals` | `40` | Maximum objective evaluations during optimization. |
| `min_markers` | `6` | Minimum markers with usable edge data before self-undistort is attempted. |
| `improvement_threshold` | `0.01` | Relative improvement required before the estimate is considered useful. |
| `min_abs_improvement` | `0.0001` | Absolute improvement floor. Prevents tiny numerical wins from activating the model. |
| `trim_fraction` | `0.1` | Fraction of worst residuals trimmed in robust scoring. |
| `min_lambda_abs` | `5e-09` | Rejects near-zero solutions that do not meaningfully change the model. |
| `reject_range_edge` | `True` | Rejects solutions that land too close to the search-interval edge. |
| `range_edge_margin_frac` | `0.02` | Edge margin used by that range-edge rejection. |
| `validation_min_markers` | `24` | Marker count required for the homography validation pass. |
| `validation_abs_improvement_px` | `0.05` | Absolute reprojection improvement required by validation. |
| `validation_rel_improvement` | `0.03` | Relative reprojection improvement required by validation. |

### `id_correction`

Runs after local fit/decode to verify or recover IDs from the board's hex
lattice structure.

| Field | Default | Practical notes |
|---|---|---|
| `enable` | `True` | Leave on unless you are explicitly debugging raw decoder output. |
| `auto_search_radius_outer_muls` | `[2.4, 2.9, 3.5, 4.2, 5.0]` | Staged neighborhood radii for local search. Tighten only if incorrect neighbors dominate. |
| `consistency_outer_mul` | `3.2` | Neighborhood radius for structural consistency checks. |
| `consistency_min_neighbors` | `1` | Minimum neighbors required before a consistency check runs. |
| `consistency_min_support_edges` | `1` | Minimum supporting board-neighbor edges required to keep an ID. |
| `consistency_max_contradiction_frac` | `0.5` | Maximum allowed contradiction fraction before clearing an ID. |
| `soft_lock_exact_decode` | `True` | Protects exact decodes unless structure strongly contradicts them. |
| `min_votes` | `2` | Votes required to change an already-assigned ID. |
| `min_votes_recover` | `1` | Votes required to recover a missing ID. |
| `min_vote_weight_frac` | `0.55` | Minimum weighted-vote share for the winning candidate. |
| `h_reproj_gate_px` | `30.0` | Loose reprojection gate used by the fallback homography assignment. |
| `homography_fallback_enable` | `True` | Enables fallback ID recovery from a rough homography when local evidence is weak. |
| `homography_min_trusted` | `24` | Minimum trusted markers before that fallback is attempted. |
| `homography_min_inliers` | `12` | Minimum inliers required for the fallback homography. |
| `max_iters` | `5` | Maximum iterative correction passes. |
| `remove_unverified` | `False` | Default keeps the detection but clears the ID. Set to `True` to drop unverifiable markers entirely. |
| `seed_min_decode_confidence` | `0.7` | Minimum decode confidence used when bootstrapping trusted seeds without a homography. |

### `inner_as_outer_recovery`

Post-processing stage that tries to fix markers whose outer fit locked onto the
inner ring.

| Field | Default | Practical notes |
|---|---|---|
| `enable` | `True` | Leave on for the default blur-tolerant behavior. |
| `ratio_threshold` | `0.75` | Neighbor-radius ratio below which a marker is considered suspicious. |
| `k_neighbors` | `6` | Number of nearest neighbors used to estimate the expected outer radius. |
| `min_theta_consistency` | `0.18` | Lower-than-normal consistency gate used by this recovery path. |
| `min_theta_coverage` | `0.4` | Minimum angular coverage required during the re-fit. |
| `min_ring_depth` | `0.02` | Relaxed edge-depth gate used for blurry outer edges. |
| `refine_halfwidth_px` | `2.5` | Wider local radius refinement window for the recovery re-fit. |
| `size_gate_tolerance` | `0.25` | Prevents the relaxed recovery fit from re-locking onto the inner ring. |

Typical tuning sequence:

- Known scale: tighten `cfg.marker_scale` before touching low-level proposal thresholds.
- Too many weak IDs: raise `cfg.decode_min_confidence` or `cfg.decode_min_margin`.
- Good local fits but unstable global cleanup: tighten `cfg.homography_inlier_threshold_px`.
- Distorted scenes without calibration: try `cfg.self_undistort_enable = True`.
- Want direct detections only: set `cfg.completion_enable = False`.

## Adaptive Detection

Use adaptive detection when marker diameter varies substantially across the
image (near/far perspective, zoom changes, mixed target scales).

### Which Method Should I Use?

| Situation | Recommended call | Why |
|---|---|---|
| You do not know marker size in advance | `detector.detect_adaptive(image)` | Probes scale and auto-selects tiers |
| You know approximate marker diameter (px) | `detector.detect_adaptive(image, nominal_diameter_px=d)` | Skips probe and uses focused two-tier bracket around `d` |
| You need fixed/reproducible tier policy | `detector.detect_multiscale(image, tiers)` | Full explicit control over tiers |
| Marker size range is tight and runtime is priority | `detector.detect(image)` | Single-pass (fastest) |

Canonical adaptive entry point is:
- `Detector.detect_adaptive(image, nominal_diameter_px: float | None = None)`

Compatibility alias (deprecated, still supported):
- `Detector.detect_adaptive_with_hint(image, nominal_diameter_px=...)`

Tier objects:
- `ScaleTier(diameter_min_px, diameter_max_px)`
- `ScaleTiers([...])`
- Presets: `ScaleTiers.four_tier_wide()`, `ScaleTiers.two_tier_standard()`
- Single-pass equivalent: `ScaleTiers.single(MarkerScalePrior(...))`

### Practical Recipes

Unknown scene scale:

```python
from pathlib import Path
import ringgrid

board = ringgrid.BoardLayout.default()
detector = ringgrid.Detector.from_board(board)
image = Path("testdata/target_3_split_00.png")

result = detector.detect_adaptive(image)
```

Known nominal diameter (for example, ~32 px):

```python
result = detector.detect_adaptive(image, nominal_diameter_px=32.0)
```

Inspect tiers used by adaptive logic (debug/repro):

```python
tiers = detector.adaptive_tiers(image, nominal_diameter_px=32.0)
for tier in tiers.tiers:
    print(tier.diameter_min_px, tier.diameter_max_px)

# Re-run exactly those tiers
result = detector.detect_multiscale(image, tiers)
```

## Examples

Run from repository root:

```bash
python crates/ringgrid-py/examples/basic_detect.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_py.json

python crates/ringgrid-py/examples/detect_with_camera.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_cam_py.json

python crates/ringgrid-py/examples/detect_adaptive.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_det_adaptive_py.json

python crates/ringgrid-py/examples/detect_multiscale.py \
  --image testdata/target_3_split_00.png \
  --tiers four_tier_wide \
  --out testdata/target_3_split_00_det_multiscale_py.json
```

Plotting example:

```bash
python crates/ringgrid-py/examples/plot_detection.py \
  --image testdata/target_3_split_00.png \
  --out testdata/target_3_split_00_overlay_py.png
```
