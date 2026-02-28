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

class DetectConfig:
    def __init__(self, board: BoardLayout, data: Mapping[str, Any] | None = ...) -> None: ...
    @classmethod
    def from_dict(cls, board: BoardLayout, data: Mapping[str, Any]) -> DetectConfig: ...
    def to_dict(self) -> dict[str, Any]: ...
    def update_from_dict(self, data: Mapping[str, Any]) -> None: ...
    @property
    def board(self) -> BoardLayout: ...
    @property
    def marker_scale(self) -> MarkerScalePrior: ...
    @marker_scale.setter
    def marker_scale(self, value: MarkerScalePrior) -> None: ...
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
    def __init__(self, board: BoardLayout, config: DetectConfig | None = ...) -> None: ...
    @property
    def board(self) -> BoardLayout: ...
    @property
    def config(self) -> DetectConfig: ...
    def detect(self, image: np.ndarray | str | Path) -> DetectionResult: ...
    def detect_with_mapper(self, image: np.ndarray | str | Path, mapper: CameraModel | DivisionModel) -> DetectionResult: ...

__version__: str

__all__: list[str]
