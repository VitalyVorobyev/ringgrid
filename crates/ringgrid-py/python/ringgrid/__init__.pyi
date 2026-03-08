from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Mapping
import numpy as np

class DetectionFrame(str, Enum):
    IMAGE: DetectionFrame
    WORKING: DetectionFrame

class CircleRefinementMethod(str, Enum):
    NONE: CircleRefinementMethod
    PROJECTIVE_CENTER: CircleRefinementMethod

class BoardMarker:
    id: int
    xy_mm: list[float]
    q: int | None
    r: int | None
    def __init__(self, id: int, xy_mm: list[float], q: int | None = ..., r: int | None = ...) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoardMarker: ...
    def to_dict(self) -> dict[str, Any]: ...

class BoardLayout:
    schema: str
    name: str
    pitch_mm: float
    rows: int
    long_row_cols: int
    marker_outer_radius_mm: float
    marker_inner_radius_mm: float
    markers: list[BoardMarker]
    @classmethod
    def default(cls) -> BoardLayout: ...
    @classmethod
    def from_json_file(cls, path: str | Path) -> BoardLayout: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoardLayout: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_spec_dict(self) -> dict[str, Any]: ...

class MarkerScalePrior:
    diameter_min_px: float
    diameter_max_px: float
    def __init__(self, diameter_min_px: float, diameter_max_px: float) -> None: ...
    @classmethod
    def from_nominal_diameter_px(cls, diameter_px: float) -> MarkerScalePrior: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MarkerScalePrior: ...
    def to_dict(self) -> dict[str, Any]: ...

class ScaleTier:
    diameter_min_px: float
    diameter_max_px: float
    def __init__(self, diameter_min_px: float, diameter_max_px: float) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScaleTier: ...
    def to_dict(self) -> dict[str, Any]: ...

class ScaleTiers:
    tiers: list[ScaleTier]
    def __init__(self, tiers: list[ScaleTier]) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScaleTiers: ...
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def four_tier_wide(cls) -> ScaleTiers: ...
    @classmethod
    def two_tier_standard(cls) -> ScaleTiers: ...
    @classmethod
    def single(cls, prior: MarkerScalePrior) -> ScaleTiers: ...

class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CameraIntrinsics: ...
    def to_dict(self) -> dict[str, Any]: ...

class RadialTangentialDistortion:
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    def __init__(self, k1: float = ..., k2: float = ..., p1: float = ..., p2: float = ..., k3: float = ...) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RadialTangentialDistortion: ...
    def to_dict(self) -> dict[str, Any]: ...

class CameraModel:
    intrinsics: CameraIntrinsics
    distortion: RadialTangentialDistortion
    def __init__(self, intrinsics: CameraIntrinsics, distortion: RadialTangentialDistortion = ...) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CameraModel: ...
    def to_dict(self) -> dict[str, Any]: ...

class DivisionModel:
    lambda_: float
    cx: float
    cy: float
    def __init__(self, lambda_: float, cx: float, cy: float) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DivisionModel: ...
    def to_dict(self) -> dict[str, Any]: ...

class Ellipse:
    cx: float
    cy: float
    a: float
    b: float
    angle: float
    def __init__(self, cx: float, cy: float, a: float, b: float, angle: float) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Ellipse: ...
    def to_dict(self) -> dict[str, Any]: ...

class DecodeMetrics:
    observed_word: int
    best_id: int
    best_rotation: int
    best_dist: int
    margin: int
    decode_confidence: float
    def __init__(self, observed_word: int, best_id: int, best_rotation: int, best_dist: int, margin: int, decode_confidence: float) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecodeMetrics: ...
    def to_dict(self) -> dict[str, Any]: ...

class FitMetrics:
    n_angles_total: int
    n_angles_with_both_edges: int
    n_points_outer: int
    n_points_inner: int
    ransac_inlier_ratio_outer: float | None
    ransac_inlier_ratio_inner: float | None
    rms_residual_outer: float | None
    rms_residual_inner: float | None
    max_angular_gap_outer: float | None
    max_angular_gap_inner: float | None
    inner_fit_status: str | None
    inner_fit_reason: str | None
    neighbor_radius_ratio: float | None
    inner_theta_consistency: float | None
    def __init__(
        self,
        n_angles_total: int,
        n_angles_with_both_edges: int,
        n_points_outer: int,
        n_points_inner: int,
        ransac_inlier_ratio_outer: float | None = ...,
        ransac_inlier_ratio_inner: float | None = ...,
        rms_residual_outer: float | None = ...,
        rms_residual_inner: float | None = ...,
        max_angular_gap_outer: float | None = ...,
        max_angular_gap_inner: float | None = ...,
        inner_fit_status: str | None = ...,
        inner_fit_reason: str | None = ...,
        neighbor_radius_ratio: float | None = ...,
        inner_theta_consistency: float | None = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FitMetrics: ...
    def to_dict(self) -> dict[str, Any]: ...

class DetectedMarker:
    id: int | None
    confidence: float
    center: list[float]
    center_mapped: list[float] | None
    board_xy_mm: list[float] | None
    ellipse_outer: Ellipse | None
    ellipse_inner: Ellipse | None
    edge_points_outer: list[list[float]] | None
    edge_points_inner: list[list[float]] | None
    fit: FitMetrics
    decode: DecodeMetrics | None
    def __init__(
        self,
        confidence: float,
        center: list[float],
        fit: FitMetrics,
        id: int | None = ...,
        center_mapped: list[float] | None = ...,
        board_xy_mm: list[float] | None = ...,
        ellipse_outer: Ellipse | None = ...,
        ellipse_inner: Ellipse | None = ...,
        edge_points_outer: list[list[float]] | None = ...,
        edge_points_inner: list[list[float]] | None = ...,
        decode: DecodeMetrics | None = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectedMarker: ...
    def to_dict(self) -> dict[str, Any]: ...

class RansacStats:
    n_candidates: int
    n_inliers: int
    threshold_px: float
    mean_err_px: float
    p95_err_px: float
    def __init__(self, n_candidates: int, n_inliers: int, threshold_px: float, mean_err_px: float, p95_err_px: float) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RansacStats: ...
    def to_dict(self) -> dict[str, Any]: ...

class SelfUndistortResult:
    model: DivisionModel
    objective_at_lambda: float
    objective_at_zero: float
    n_markers_used: int
    applied: bool
    def __init__(self, model: DivisionModel, objective_at_lambda: float, objective_at_zero: float, n_markers_used: int, applied: bool) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelfUndistortResult: ...
    def to_dict(self) -> dict[str, Any]: ...

class DetectionResult:
    detected_markers: list[DetectedMarker]
    center_frame: DetectionFrame
    homography_frame: DetectionFrame
    image_size: list[int]
    homography: list[list[float]] | None
    ransac: RansacStats | None
    self_undistort: SelfUndistortResult | None
    def __init__(
        self,
        detected_markers: list[DetectedMarker],
        center_frame: DetectionFrame,
        homography_frame: DetectionFrame,
        image_size: list[int],
        homography: list[list[float]] | None = ...,
        ransac: RansacStats | None = ...,
        self_undistort: SelfUndistortResult | None = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectionResult: ...
    @classmethod
    def from_json(cls, path_or_json: str | Path) -> DetectionResult: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_json(self, path: str | Path | None = ...) -> str | None: ...
    def plot(
        self,
        *,
        image: np.ndarray | str | Path,
        out: str | Path | None = ...,
        marker_id: int | None = ...,
        zoom: float | None = ...,
        show_ellipses: bool = ...,
        show_confidence: bool = ...,
        alpha: float = ...,
    ) -> None: ...

class RansacFitConfig:
    max_iters: int
    inlier_threshold: float
    min_inliers: int
    seed: int
    def __init__(self, max_iters: int = ..., inlier_threshold: float = ..., min_inliers: int = ..., seed: int = ...) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RansacFitConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class InnerFitConfig:
    min_points: int
    min_inlier_ratio: float
    max_rms_residual: float
    max_center_shift_px: float
    max_ratio_abs_error: float
    local_peak_halfwidth_idx: int
    ransac: RansacFitConfig
    miss_confidence_factor: float
    max_angular_gap_rad: float
    require_inner_fit: bool
    def __init__(
        self,
        min_points: int = ...,
        min_inlier_ratio: float = ...,
        max_rms_residual: float = ...,
        max_center_shift_px: float = ...,
        max_ratio_abs_error: float = ...,
        local_peak_halfwidth_idx: int = ...,
        ransac: RansacFitConfig = ...,
        miss_confidence_factor: float = ...,
        max_angular_gap_rad: float = ...,
        require_inner_fit: bool = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InnerFitConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class OuterFitConfig:
    min_direct_fit_points: int
    min_ransac_points: int
    ransac: RansacFitConfig
    size_score_weight: float
    max_angular_gap_rad: float
    def __init__(
        self,
        min_direct_fit_points: int = ...,
        min_ransac_points: int = ...,
        ransac: RansacFitConfig = ...,
        size_score_weight: float = ...,
        max_angular_gap_rad: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OuterFitConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class CompletionParams:
    enable: bool
    roi_radius_px: float
    reproj_gate_px: float
    min_fit_confidence: float
    min_arc_coverage: float
    max_attempts: int | None
    image_margin_px: float
    require_perfect_decode: bool
    max_radii_std_ratio: float
    def __init__(
        self,
        enable: bool = ...,
        roi_radius_px: float = ...,
        reproj_gate_px: float = ...,
        min_fit_confidence: float = ...,
        min_arc_coverage: float = ...,
        max_attempts: int | None = ...,
        image_margin_px: float = ...,
        require_perfect_decode: bool = ...,
        max_radii_std_ratio: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompletionParams: ...
    def to_dict(self) -> dict[str, Any]: ...

class ProjectiveCenterParams:
    use_expected_ratio: bool
    ratio_penalty_weight: float
    max_center_shift_px: float | None
    max_selected_residual: float
    min_eig_separation: float
    def __init__(
        self,
        use_expected_ratio: bool = ...,
        ratio_penalty_weight: float = ...,
        max_center_shift_px: float | None = ...,
        max_selected_residual: float = ...,
        min_eig_separation: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectiveCenterParams: ...
    def to_dict(self) -> dict[str, Any]: ...

class SeedProposalParams:
    merge_radius_px: float
    seed_score: float
    max_seeds: int | None
    def __init__(self, merge_radius_px: float = ..., seed_score: float = ..., max_seeds: int | None = ...) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeedProposalParams: ...
    def to_dict(self) -> dict[str, Any]: ...

class ProposalConfig:
    r_min: float
    r_max: float
    grad_threshold: float
    nms_radius: float
    min_vote_frac: float
    accum_sigma: float
    max_candidates: int | None
    def __init__(
        self,
        r_min: float = ...,
        r_max: float = ...,
        grad_threshold: float = ...,
        nms_radius: float = ...,
        min_vote_frac: float = ...,
        accum_sigma: float = ...,
        max_candidates: int | None = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProposalConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class EdgeSampleConfig:
    n_rays: int
    r_max: float
    r_min: float
    r_step: float
    min_ring_depth: float
    min_rays_with_ring: int
    def __init__(
        self,
        n_rays: int = ...,
        r_max: float = ...,
        r_min: float = ...,
        r_step: float = ...,
        min_ring_depth: float = ...,
        min_rays_with_ring: int = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EdgeSampleConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class DecodeConfig:
    code_band_ratio: float
    samples_per_sector: int
    n_radial_rings: int
    max_decode_dist: int
    min_decode_confidence: float
    min_decode_margin: int
    min_decode_contrast: float
    threshold_max_iters: int
    threshold_convergence_eps: float
    def __init__(
        self,
        code_band_ratio: float = ...,
        samples_per_sector: int = ...,
        n_radial_rings: int = ...,
        max_decode_dist: int = ...,
        min_decode_confidence: float = ...,
        min_decode_margin: int = ...,
        min_decode_contrast: float = ...,
        threshold_max_iters: int = ...,
        threshold_convergence_eps: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DecodeConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class MarkerSpec:
    r_inner_expected: float
    inner_search_halfwidth: float
    inner_grad_polarity: str
    radial_samples: int
    theta_samples: int
    aggregator: str
    min_theta_coverage: float
    min_theta_consistency: float
    def __init__(
        self,
        r_inner_expected: float = ...,
        inner_search_halfwidth: float = ...,
        inner_grad_polarity: str = ...,
        radial_samples: int = ...,
        theta_samples: int = ...,
        aggregator: str = ...,
        min_theta_coverage: float = ...,
        min_theta_consistency: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MarkerSpec: ...
    def to_dict(self) -> dict[str, Any]: ...

class OuterEstimationConfig:
    search_halfwidth_px: float
    radial_samples: int
    aggregator: str
    grad_polarity: str
    min_theta_coverage: float
    min_theta_consistency: float
    allow_two_hypotheses: bool
    second_peak_min_rel: float
    refine_halfwidth_px: float
    def __init__(
        self,
        search_halfwidth_px: float = ...,
        radial_samples: int = ...,
        aggregator: str = ...,
        grad_polarity: str = ...,
        min_theta_coverage: float = ...,
        min_theta_consistency: float = ...,
        allow_two_hypotheses: bool = ...,
        second_peak_min_rel: float = ...,
        refine_halfwidth_px: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OuterEstimationConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class RansacHomographyConfig:
    max_iters: int
    inlier_threshold: float
    min_inliers: int
    seed: int
    def __init__(self, max_iters: int = ..., inlier_threshold: float = ..., min_inliers: int = ..., seed: int = ...) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RansacHomographyConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class SelfUndistortConfig:
    enable: bool
    lambda_range: list[float]
    max_evals: int
    min_markers: int
    improvement_threshold: float
    min_abs_improvement: float
    trim_fraction: float
    min_lambda_abs: float
    reject_range_edge: bool
    range_edge_margin_frac: float
    validation_min_markers: int
    validation_abs_improvement_px: float
    validation_rel_improvement: float
    def __init__(
        self,
        enable: bool = ...,
        lambda_range: list[float] = ...,
        max_evals: int = ...,
        min_markers: int = ...,
        improvement_threshold: float = ...,
        min_abs_improvement: float = ...,
        trim_fraction: float = ...,
        min_lambda_abs: float = ...,
        reject_range_edge: bool = ...,
        range_edge_margin_frac: float = ...,
        validation_min_markers: int = ...,
        validation_abs_improvement_px: float = ...,
        validation_rel_improvement: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelfUndistortConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class IdCorrectionConfig:
    enable: bool
    auto_search_radius_outer_muls: list[float]
    consistency_outer_mul: float
    consistency_min_neighbors: int
    consistency_min_support_edges: int
    consistency_max_contradiction_frac: float
    soft_lock_exact_decode: bool
    min_votes: int
    min_votes_recover: int
    min_vote_weight_frac: float
    h_reproj_gate_px: float
    homography_fallback_enable: bool
    homography_min_trusted: int
    homography_min_inliers: int
    max_iters: int
    remove_unverified: bool
    seed_min_decode_confidence: float
    def __init__(
        self,
        enable: bool = ...,
        auto_search_radius_outer_muls: list[float] = ...,
        consistency_outer_mul: float = ...,
        consistency_min_neighbors: int = ...,
        consistency_min_support_edges: int = ...,
        consistency_max_contradiction_frac: float = ...,
        soft_lock_exact_decode: bool = ...,
        min_votes: int = ...,
        min_votes_recover: int = ...,
        min_vote_weight_frac: float = ...,
        h_reproj_gate_px: float = ...,
        homography_fallback_enable: bool = ...,
        homography_min_trusted: int = ...,
        homography_min_inliers: int = ...,
        max_iters: int = ...,
        remove_unverified: bool = ...,
        seed_min_decode_confidence: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IdCorrectionConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class InnerAsOuterRecoveryConfig:
    enable: bool
    ratio_threshold: float
    k_neighbors: int
    min_theta_consistency: float
    min_theta_coverage: float
    min_ring_depth: float
    refine_halfwidth_px: float
    size_gate_tolerance: float
    def __init__(
        self,
        enable: bool = ...,
        ratio_threshold: float = ...,
        k_neighbors: int = ...,
        min_theta_consistency: float = ...,
        min_theta_coverage: float = ...,
        min_ring_depth: float = ...,
        refine_halfwidth_px: float = ...,
        size_gate_tolerance: float = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InnerAsOuterRecoveryConfig: ...
    def to_dict(self) -> dict[str, Any]: ...

class DetectConfig:
    def __init__(self, board: BoardLayout) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @property
    def board(self) -> BoardLayout: ...
    @property
    def marker_scale(self) -> MarkerScalePrior: ...
    @marker_scale.setter
    def marker_scale(self, value: MarkerScalePrior) -> None: ...
    @property
    def inner_fit(self) -> InnerFitConfig: ...
    @inner_fit.setter
    def inner_fit(self, value: InnerFitConfig) -> None: ...
    @property
    def outer_fit(self) -> OuterFitConfig: ...
    @outer_fit.setter
    def outer_fit(self, value: OuterFitConfig) -> None: ...
    @property
    def completion(self) -> CompletionParams: ...
    @completion.setter
    def completion(self, value: CompletionParams) -> None: ...
    @property
    def projective_center(self) -> ProjectiveCenterParams: ...
    @projective_center.setter
    def projective_center(self, value: ProjectiveCenterParams) -> None: ...
    @property
    def seed_proposals(self) -> SeedProposalParams: ...
    @seed_proposals.setter
    def seed_proposals(self, value: SeedProposalParams) -> None: ...
    @property
    def proposal(self) -> ProposalConfig: ...
    @proposal.setter
    def proposal(self, value: ProposalConfig) -> None: ...
    @property
    def edge_sample(self) -> EdgeSampleConfig: ...
    @edge_sample.setter
    def edge_sample(self, value: EdgeSampleConfig) -> None: ...
    @property
    def decode(self) -> DecodeConfig: ...
    @decode.setter
    def decode(self, value: DecodeConfig) -> None: ...
    @property
    def marker_spec(self) -> MarkerSpec: ...
    @marker_spec.setter
    def marker_spec(self, value: MarkerSpec) -> None: ...
    @property
    def outer_estimation(self) -> OuterEstimationConfig: ...
    @outer_estimation.setter
    def outer_estimation(self, value: OuterEstimationConfig) -> None: ...
    @property
    def ransac_homography(self) -> RansacHomographyConfig: ...
    @ransac_homography.setter
    def ransac_homography(self, value: RansacHomographyConfig) -> None: ...
    @property
    def self_undistort(self) -> SelfUndistortConfig: ...
    @self_undistort.setter
    def self_undistort(self, value: SelfUndistortConfig) -> None: ...
    @property
    def id_correction(self) -> IdCorrectionConfig: ...
    @id_correction.setter
    def id_correction(self, value: IdCorrectionConfig) -> None: ...
    @property
    def inner_as_outer_recovery(self) -> InnerAsOuterRecoveryConfig: ...
    @inner_as_outer_recovery.setter
    def inner_as_outer_recovery(self, value: InnerAsOuterRecoveryConfig) -> None: ...
    @property
    def dedup_radius(self) -> float: ...
    @dedup_radius.setter
    def dedup_radius(self, value: float) -> None: ...
    @property
    def max_aspect_ratio(self) -> float: ...
    @max_aspect_ratio.setter
    def max_aspect_ratio(self, value: float) -> None: ...
    @property
    def completion_enable(self) -> bool: ...
    @completion_enable.setter
    def completion_enable(self, value: bool) -> None: ...
    @property
    def use_global_filter(self) -> bool: ...
    @use_global_filter.setter
    def use_global_filter(self, value: bool) -> None: ...
    @property
    def self_undistort_enable(self) -> bool: ...
    @self_undistort_enable.setter
    def self_undistort_enable(self, value: bool) -> None: ...
    @property
    def decode_min_margin(self) -> int: ...
    @decode_min_margin.setter
    def decode_min_margin(self, value: int) -> None: ...
    @property
    def decode_max_dist(self) -> int: ...
    @decode_max_dist.setter
    def decode_max_dist(self, value: int) -> None: ...
    @property
    def decode_min_confidence(self) -> float: ...
    @decode_min_confidence.setter
    def decode_min_confidence(self, value: float) -> None: ...
    @property
    def circle_refinement(self) -> CircleRefinementMethod: ...
    @circle_refinement.setter
    def circle_refinement(self, value: CircleRefinementMethod) -> None: ...
    @property
    def inner_fit_required(self) -> bool: ...
    @inner_fit_required.setter
    def inner_fit_required(self, value: bool) -> None: ...
    @property
    def homography_inlier_threshold_px(self) -> float: ...
    @homography_inlier_threshold_px.setter
    def homography_inlier_threshold_px(self, value: float) -> None: ...

class Detector:
    def __init__(self, config: DetectConfig) -> None: ...
    @classmethod
    def from_board(cls, board: BoardLayout) -> Detector: ...
    @classmethod
    def with_config(cls, config: DetectConfig) -> Detector: ...
    @property
    def board(self) -> BoardLayout: ...
    @property
    def config(self) -> DetectConfig: ...
    def detect(self, image: np.ndarray | str | Path) -> DetectionResult: ...
    def detect_with_mapper(self, image: np.ndarray | str | Path, mapper: CameraModel | DivisionModel) -> DetectionResult: ...
    def detect_adaptive(self, image: np.ndarray | str | Path, nominal_diameter_px: float | None = ...) -> DetectionResult: ...
    def detect_adaptive_with_hint(self, image: np.ndarray | str | Path, nominal_diameter_px: float | None = ...) -> DetectionResult: ...
    def detect_multiscale(self, image: np.ndarray | str | Path, tiers: ScaleTiers) -> DetectionResult: ...
    def adaptive_tiers(self, image: np.ndarray | str | Path, nominal_diameter_px: float | None = ...) -> ScaleTiers: ...

__version__: str

__all__: list[str]
