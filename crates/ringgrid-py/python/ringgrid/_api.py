"""Public Python API for the ringgrid detector.

This module exposes a native detector backed by Rust (`PyO3` extension) and a
typed Python surface that feels idiomatic in notebooks and applications.

Typical flow:
1. Build/load a :class:`BoardLayout`.
2. Construct :class:`DetectConfig` and :class:`Detector`.
3. Call :meth:`Detector.detect` / :meth:`Detector.detect_adaptive` /
   :meth:`Detector.detect_multiscale`.
4. Consume :class:`DetectionResult` in memory, JSON, or via plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

from ._ringgrid import (
    DetectConfigCore as _DetectConfigCore,
    DetectorCore as _DetectorCore,
    board_snapshot_json as _board_snapshot_json,
    default_board_spec_json as _default_board_spec_json,
    load_board_spec_json as _load_board_spec_json,
    package_version as _package_version,
    scale_tiers_four_tier_wide_json as _scale_tiers_four_tier_wide_json,
    scale_tiers_two_tier_standard_json as _scale_tiers_two_tier_standard_json,
)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _coerce_path(path: str | Path) -> str:
    return str(Path(path))


def _require_finite_positive_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return parsed


def _json_loads_path_or_text(path_or_json: str | Path) -> dict[str, Any]:
    if isinstance(path_or_json, Path):
        return json.loads(path_or_json.read_text(encoding="utf-8"))

    text = str(path_or_json)
    if text.lstrip().startswith("{"):
        return json.loads(text)

    path = Path(text)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    return json.loads(text)


class DetectionFrame(str, Enum):
    """Coordinate frame used in detection outputs."""

    IMAGE = "image"
    WORKING = "working"


class CircleRefinementMethod(str, Enum):
    """Center refinement method used after local fits."""

    NONE = "none"
    PROJECTIVE_CENTER = "projective_center"


@dataclass(slots=True)
class BoardMarker:
    """Marker metadata on the physical board."""

    id: int
    xy_mm: list[float]
    q: int | None = None
    r: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoardMarker":
        return cls(
            id=int(data["id"]),
            xy_mm=[float(data["xy_mm"][0]), float(data["xy_mm"][1])],
            q=None if data.get("q") is None else int(data["q"]),
            r=None if data.get("r") is None else int(data["r"]),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": int(self.id),
            "xy_mm": [float(self.xy_mm[0]), float(self.xy_mm[1])],
        }
        if self.q is not None:
            out["q"] = int(self.q)
        if self.r is not None:
            out["r"] = int(self.r)
        return out


@dataclass(slots=True)
class BoardLayout:
    """Board layout specification and generated marker list.

    The board object carries both:
    - the compact board spec (`schema`, `rows`, `pitch_mm`, radii), and
    - the generated marker list (`markers`) with board coordinates.
    """

    schema: str
    name: str
    pitch_mm: float
    rows: int
    long_row_cols: int
    marker_outer_radius_mm: float
    marker_inner_radius_mm: float
    markers: list[BoardMarker] = field(default_factory=list)
    _spec_json: str = field(default="", repr=False)

    @classmethod
    def default(cls) -> "BoardLayout":
        """Construct the built-in default board layout."""
        spec_json = _default_board_spec_json()
        snapshot = json.loads(_board_snapshot_json(spec_json))
        return cls._from_snapshot(snapshot, spec_json)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BoardLayout":
        """Load a board layout from a `ringgrid.target.v3` JSON file."""
        spec_json = _load_board_spec_json(_coerce_path(path))
        snapshot = json.loads(_board_snapshot_json(spec_json))
        return cls._from_snapshot(snapshot, spec_json)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoardLayout":
        """Construct a board layout from a board-spec mapping."""
        data = _require_mapping(data, name="data")
        spec = {
            "schema": str(data["schema"]),
            "name": str(data["name"]),
            "pitch_mm": float(data["pitch_mm"]),
            "rows": int(data["rows"]),
            "long_row_cols": int(data["long_row_cols"]),
            "marker_outer_radius_mm": float(data["marker_outer_radius_mm"]),
            "marker_inner_radius_mm": float(data["marker_inner_radius_mm"]),
        }
        spec_json = json.dumps(spec)
        snapshot = json.loads(_board_snapshot_json(spec_json))
        return cls._from_snapshot(snapshot, spec_json)

    @classmethod
    def _from_snapshot(cls, snapshot: Mapping[str, Any], spec_json: str) -> "BoardLayout":
        markers = [BoardMarker.from_dict(m) for m in snapshot.get("markers", [])]
        return cls(
            schema=str(snapshot["schema"]),
            name=str(snapshot["name"]),
            pitch_mm=float(snapshot["pitch_mm"]),
            rows=int(snapshot["rows"]),
            long_row_cols=int(snapshot["long_row_cols"]),
            marker_outer_radius_mm=float(snapshot["marker_outer_radius_mm"]),
            marker_inner_radius_mm=float(snapshot["marker_inner_radius_mm"]),
            markers=markers,
            _spec_json=spec_json,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full board payload including generated marker list."""
        return {
            "schema": self.schema,
            "name": self.name,
            "pitch_mm": float(self.pitch_mm),
            "rows": int(self.rows),
            "long_row_cols": int(self.long_row_cols),
            "marker_outer_radius_mm": float(self.marker_outer_radius_mm),
            "marker_inner_radius_mm": float(self.marker_inner_radius_mm),
            "markers": [m.to_dict() for m in self.markers],
        }

    def to_spec_dict(self) -> dict[str, Any]:
        """Serialize only the board spec fields (no expanded marker list)."""
        return {
            "schema": self.schema,
            "name": self.name,
            "pitch_mm": float(self.pitch_mm),
            "rows": int(self.rows),
            "long_row_cols": int(self.long_row_cols),
            "marker_outer_radius_mm": float(self.marker_outer_radius_mm),
            "marker_inner_radius_mm": float(self.marker_inner_radius_mm),
        }


@dataclass(slots=True)
class MarkerScalePrior:
    """Marker diameter range prior in pixels."""

    diameter_min_px: float
    diameter_max_px: float

    @classmethod
    def from_nominal_diameter_px(cls, diameter_px: float) -> "MarkerScalePrior":
        """Build a fixed-size prior from one diameter hint."""
        d = float(diameter_px)
        return cls(diameter_min_px=d, diameter_max_px=d)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarkerScalePrior":
        data = _require_mapping(data, name="data")
        return cls(
            diameter_min_px=float(data["diameter_min_px"]),
            diameter_max_px=float(data["diameter_max_px"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diameter_min_px": float(self.diameter_min_px),
            "diameter_max_px": float(self.diameter_max_px),
        }


@dataclass(slots=True)
class ScaleTier:
    """One adaptive detection tier, parameterized by marker diameter range."""

    diameter_min_px: float
    diameter_max_px: float

    def __post_init__(self) -> None:
        self.diameter_min_px = _require_finite_positive_float(
            self.diameter_min_px, name="diameter_min_px"
        )
        self.diameter_max_px = _require_finite_positive_float(
            self.diameter_max_px, name="diameter_max_px"
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScaleTier":
        data = _require_mapping(data, name="data")
        return cls(
            diameter_min_px=float(data["diameter_min_px"]),
            diameter_max_px=float(data["diameter_max_px"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diameter_min_px": float(self.diameter_min_px),
            "diameter_max_px": float(self.diameter_max_px),
        }

    @classmethod
    def _from_wire(cls, data: Mapping[str, Any]) -> "ScaleTier":
        return cls.from_dict(data)

    def _to_wire(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(slots=True)
class ScaleTiers:
    """Ordered scale tiers for explicit multi-scale adaptive detection."""

    tiers: list[ScaleTier]

    def __post_init__(self) -> None:
        normalized: list[ScaleTier] = []
        for idx, tier in enumerate(self.tiers):
            if not isinstance(tier, ScaleTier):
                raise TypeError(f"tiers[{idx}] must be ScaleTier")
            normalized.append(tier)
        if not normalized:
            raise ValueError("tiers must contain at least one ScaleTier")
        self.tiers = normalized

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScaleTiers":
        data = _require_mapping(data, name="data")
        raw_tiers = data.get("tiers")
        if not isinstance(raw_tiers, list):
            raise TypeError("data['tiers'] must be a list")
        return cls(
            tiers=[ScaleTier.from_dict(_require_mapping(tier, name="tier")) for tier in raw_tiers]
        )

    def to_dict(self) -> dict[str, Any]:
        return {"tiers": [tier.to_dict() for tier in self.tiers]}

    @classmethod
    def four_tier_wide(cls) -> "ScaleTiers":
        return cls._from_wire(json.loads(_scale_tiers_four_tier_wide_json()))

    @classmethod
    def two_tier_standard(cls) -> "ScaleTiers":
        return cls._from_wire(json.loads(_scale_tiers_two_tier_standard_json()))

    @classmethod
    def single(cls, prior: MarkerScalePrior) -> "ScaleTiers":
        if not isinstance(prior, MarkerScalePrior):
            raise TypeError("prior must be MarkerScalePrior")
        return cls(
            tiers=[
                ScaleTier(
                    diameter_min_px=float(prior.diameter_min_px),
                    diameter_max_px=float(prior.diameter_max_px),
                )
            ]
        )

    @classmethod
    def _from_wire(cls, data: Mapping[str, Any]) -> "ScaleTiers":
        return cls.from_dict(data)

    def _to_wire(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(slots=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraIntrinsics":
        data = _require_mapping(data, name="data")
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
        }


@dataclass(slots=True)
class RadialTangentialDistortion:
    """Brown-Conrady radial-tangential distortion coefficients."""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RadialTangentialDistortion":
        data = _require_mapping(data, name="data")
        return cls(
            k1=float(data.get("k1", 0.0)),
            k2=float(data.get("k2", 0.0)),
            p1=float(data.get("p1", 0.0)),
            p2=float(data.get("p2", 0.0)),
            k3=float(data.get("k3", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "k3": float(self.k3),
        }


@dataclass(slots=True)
class CameraModel:
    """Calibrated camera model for distortion-aware detection."""

    intrinsics: CameraIntrinsics
    distortion: RadialTangentialDistortion = field(default_factory=RadialTangentialDistortion)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraModel":
        data = _require_mapping(data, name="data")
        return cls(
            intrinsics=CameraIntrinsics.from_dict(data["intrinsics"]),
            distortion=RadialTangentialDistortion.from_dict(data["distortion"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "distortion": self.distortion.to_dict(),
        }

    def _to_mapper_payload(self) -> dict[str, Any]:
        return {
            "kind": "camera",
            "intrinsics": self.intrinsics.to_dict(),
            "distortion": self.distortion.to_dict(),
        }


@dataclass(slots=True)
class DivisionModel:
    """Single-parameter division distortion model."""

    lambda_: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DivisionModel":
        data = _require_mapping(data, name="data")
        lambda_val = data.get("lambda")
        if lambda_val is None:
            lambda_val = data.get("lambda_")
        if lambda_val is None:
            raise ValueError("DivisionModel requires 'lambda'")
        return cls(
            lambda_=float(lambda_val),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda": float(self.lambda_),
            "cx": float(self.cx),
            "cy": float(self.cy),
        }

    def _to_mapper_payload(self) -> dict[str, Any]:
        return {
            "kind": "division",
            "lambda": float(self.lambda_),
            "cx": float(self.cx),
            "cy": float(self.cy),
        }


@dataclass(slots=True)
class Ellipse:
    """Geometric ellipse parameters in pixel coordinates."""

    cx: float
    cy: float
    a: float
    b: float
    angle: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ellipse":
        return cls(
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            a=float(data["a"]),
            b=float(data["b"]),
            angle=float(data["angle"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cx": float(self.cx),
            "cy": float(self.cy),
            "a": float(self.a),
            "b": float(self.b),
            "angle": float(self.angle),
        }


@dataclass(slots=True)
class DecodeMetrics:
    """Decode confidence and codeword matching details."""

    observed_word: int
    best_id: int
    best_rotation: int
    best_dist: int
    margin: int
    decode_confidence: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecodeMetrics":
        return cls(
            observed_word=int(data["observed_word"]),
            best_id=int(data["best_id"]),
            best_rotation=int(data["best_rotation"]),
            best_dist=int(data["best_dist"]),
            margin=int(data["margin"]),
            decode_confidence=float(data["decode_confidence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_word": int(self.observed_word),
            "best_id": int(self.best_id),
            "best_rotation": int(self.best_rotation),
            "best_dist": int(self.best_dist),
            "margin": int(self.margin),
            "decode_confidence": float(self.decode_confidence),
        }


@dataclass(slots=True)
class FitMetrics:
    """Per-marker fit quality metrics."""

    n_angles_total: int
    n_angles_with_both_edges: int
    n_points_outer: int
    n_points_inner: int
    ransac_inlier_ratio_outer: float | None = None
    ransac_inlier_ratio_inner: float | None = None
    rms_residual_outer: float | None = None
    rms_residual_inner: float | None = None
    max_angular_gap_outer: float | None = None
    max_angular_gap_inner: float | None = None
    inner_fit_status: str | None = None
    inner_fit_reason: str | None = None
    neighbor_radius_ratio: float | None = None
    inner_theta_consistency: float | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FitMetrics":
        return cls(
            n_angles_total=int(data["n_angles_total"]),
            n_angles_with_both_edges=int(data["n_angles_with_both_edges"]),
            n_points_outer=int(data["n_points_outer"]),
            n_points_inner=int(data["n_points_inner"]),
            ransac_inlier_ratio_outer=_optional_float(data.get("ransac_inlier_ratio_outer")),
            ransac_inlier_ratio_inner=_optional_float(data.get("ransac_inlier_ratio_inner")),
            rms_residual_outer=_optional_float(data.get("rms_residual_outer")),
            rms_residual_inner=_optional_float(data.get("rms_residual_inner")),
            max_angular_gap_outer=_optional_float(data.get("max_angular_gap_outer")),
            max_angular_gap_inner=_optional_float(data.get("max_angular_gap_inner")),
            inner_fit_status=_optional_str(data.get("inner_fit_status")),
            inner_fit_reason=_optional_str(data.get("inner_fit_reason")),
            neighbor_radius_ratio=_optional_float(data.get("neighbor_radius_ratio")),
            inner_theta_consistency=_optional_float(data.get("inner_theta_consistency")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_angles_total": int(self.n_angles_total),
            "n_angles_with_both_edges": int(self.n_angles_with_both_edges),
            "n_points_outer": int(self.n_points_outer),
            "n_points_inner": int(self.n_points_inner),
        }
        _set_optional(out, "ransac_inlier_ratio_outer", self.ransac_inlier_ratio_outer)
        _set_optional(out, "ransac_inlier_ratio_inner", self.ransac_inlier_ratio_inner)
        _set_optional(out, "rms_residual_outer", self.rms_residual_outer)
        _set_optional(out, "rms_residual_inner", self.rms_residual_inner)
        _set_optional(out, "max_angular_gap_outer", self.max_angular_gap_outer)
        _set_optional(out, "max_angular_gap_inner", self.max_angular_gap_inner)
        _set_optional(out, "inner_fit_status", self.inner_fit_status)
        _set_optional(out, "inner_fit_reason", self.inner_fit_reason)
        _set_optional(out, "neighbor_radius_ratio", self.neighbor_radius_ratio)
        _set_optional(out, "inner_theta_consistency", self.inner_theta_consistency)
        return out


@dataclass(slots=True)
class DetectedMarker:
    """Single detected marker with geometry, ID, and metrics."""

    confidence: float
    center: list[float]
    fit: FitMetrics
    id: int | None = None
    center_mapped: list[float] | None = None
    board_xy_mm: list[float] | None = None
    ellipse_outer: Ellipse | None = None
    ellipse_inner: Ellipse | None = None
    edge_points_outer: list[list[float]] | None = None
    edge_points_inner: list[list[float]] | None = None
    decode: DecodeMetrics | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectedMarker":
        return cls(
            id=None if data.get("id") is None else int(data["id"]),
            confidence=float(data["confidence"]),
            center=[float(data["center"][0]), float(data["center"][1])],
            center_mapped=_optional_vec2(data.get("center_mapped")),
            board_xy_mm=_optional_vec2(data.get("board_xy_mm")),
            ellipse_outer=None
            if data.get("ellipse_outer") is None
            else Ellipse.from_dict(data["ellipse_outer"]),
            ellipse_inner=None
            if data.get("ellipse_inner") is None
            else Ellipse.from_dict(data["ellipse_inner"]),
            edge_points_outer=_optional_points(data.get("edge_points_outer")),
            edge_points_inner=_optional_points(data.get("edge_points_inner")),
            fit=FitMetrics.from_dict(data["fit"]),
            decode=None
            if data.get("decode") is None
            else DecodeMetrics.from_dict(data["decode"]),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "confidence": float(self.confidence),
            "center": [float(self.center[0]), float(self.center[1])],
            "fit": self.fit.to_dict(),
        }
        _set_optional(out, "id", self.id)
        _set_optional(out, "center_mapped", self.center_mapped)
        _set_optional(out, "board_xy_mm", self.board_xy_mm)
        _set_optional(out, "ellipse_outer", None if self.ellipse_outer is None else self.ellipse_outer.to_dict())
        _set_optional(out, "ellipse_inner", None if self.ellipse_inner is None else self.ellipse_inner.to_dict())
        _set_optional(out, "edge_points_outer", self.edge_points_outer)
        _set_optional(out, "edge_points_inner", self.edge_points_inner)
        _set_optional(out, "decode", None if self.decode is None else self.decode.to_dict())
        return out


@dataclass(slots=True)
class RansacStats:
    """Homography RANSAC diagnostics."""

    n_candidates: int
    n_inliers: int
    threshold_px: float
    mean_err_px: float
    p95_err_px: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RansacStats":
        return cls(
            n_candidates=int(data["n_candidates"]),
            n_inliers=int(data["n_inliers"]),
            threshold_px=float(data["threshold_px"]),
            mean_err_px=float(data["mean_err_px"]),
            p95_err_px=float(data["p95_err_px"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": int(self.n_candidates),
            "n_inliers": int(self.n_inliers),
            "threshold_px": float(self.threshold_px),
            "mean_err_px": float(self.mean_err_px),
            "p95_err_px": float(self.p95_err_px),
        }


@dataclass(slots=True)
class SelfUndistortResult:
    """Self-undistort estimation output."""

    model: DivisionModel
    objective_at_lambda: float
    objective_at_zero: float
    n_markers_used: int
    applied: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelfUndistortResult":
        return cls(
            model=DivisionModel.from_dict(data["model"]),
            objective_at_lambda=float(data["objective_at_lambda"]),
            objective_at_zero=float(data["objective_at_zero"]),
            n_markers_used=int(data["n_markers_used"]),
            applied=bool(data["applied"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "objective_at_lambda": float(self.objective_at_lambda),
            "objective_at_zero": float(self.objective_at_zero),
            "n_markers_used": int(self.n_markers_used),
            "applied": bool(self.applied),
        }


@dataclass(slots=True)
class DetectionResult:
    """Full detector output for one image.

    All geometry and quality fields match the Rust `DetectionResult` schema.
    Use :meth:`to_dict`, :meth:`to_json`, and :meth:`plot` for downstream tasks.
    """

    detected_markers: list[DetectedMarker]
    center_frame: DetectionFrame
    homography_frame: DetectionFrame
    image_size: list[int]
    homography: list[list[float]] | None = None
    ransac: RansacStats | None = None
    self_undistort: SelfUndistortResult | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectionResult":
        markers = [DetectedMarker.from_dict(m) for m in data.get("detected_markers", [])]
        return cls(
            detected_markers=markers,
            center_frame=DetectionFrame(str(data["center_frame"])),
            homography_frame=DetectionFrame(str(data["homography_frame"])),
            image_size=[int(data["image_size"][0]), int(data["image_size"][1])],
            homography=_optional_h(data.get("homography")),
            ransac=None if data.get("ransac") is None else RansacStats.from_dict(data["ransac"]),
            self_undistort=None
            if data.get("self_undistort") is None
            else SelfUndistortResult.from_dict(data["self_undistort"]),
        )

    @classmethod
    def from_json(cls, path_or_json: str | Path) -> "DetectionResult":
        """Load from JSON text or a JSON file path."""
        return cls.from_dict(_json_loads_path_or_text(path_or_json))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "detected_markers": [m.to_dict() for m in self.detected_markers],
            "center_frame": self.center_frame.value,
            "homography_frame": self.homography_frame.value,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
        }
        _set_optional(out, "homography", self.homography)
        _set_optional(out, "ransac", None if self.ransac is None else self.ransac.to_dict())
        _set_optional(
            out,
            "self_undistort",
            None if self.self_undistort is None else self.self_undistort.to_dict(),
        )
        return out

    def to_json(self, path: str | Path | None = None) -> str | None:
        """Serialize to pretty JSON text or write JSON to `path`."""
        text = json.dumps(self.to_dict(), indent=2)
        if path is None:
            return text
        Path(path).write_text(text, encoding="utf-8")
        return None

    def plot(
        self,
        *,
        image: np.ndarray | str | Path,
        out: str | Path | None = None,
        marker_id: int | None = None,
        zoom: float | None = None,
        show_ellipses: bool = True,
        show_confidence: bool = True,
        alpha: float = 0.8,
    ) -> None:
        """Plot this result over `image` using :mod:`ringgrid.viz`."""
        from .viz import plot_detection

        plot_detection(
            image=image,
            detection=self,
            out=out,
            marker_id=marker_id,
            zoom=zoom,
            show_ellipses=show_ellipses,
            show_confidence=show_confidence,
            alpha=alpha,
        )


@dataclass(slots=True)
class RansacFitConfig:
    """Shared RANSAC fit parameters used by inner/outer ellipse fitting."""

    max_iters: int = 200
    inlier_threshold: float = 1.5
    min_inliers: int = 6
    seed: int = 42

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RansacFitConfig":
        data = _require_mapping(data, name="data")
        return cls(
            max_iters=int(data["max_iters"]),
            inlier_threshold=float(data["inlier_threshold"]),
            min_inliers=int(data["min_inliers"]),
            seed=int(data["seed"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iters": int(self.max_iters),
            "inlier_threshold": float(self.inlier_threshold),
            "min_inliers": int(self.min_inliers),
            "seed": int(self.seed),
        }


@dataclass(slots=True)
class InnerFitConfig:
    min_points: int = 20
    min_inlier_ratio: float = 0.5
    max_rms_residual: float = 1.0
    max_center_shift_px: float = 12.0
    max_ratio_abs_error: float = 0.15
    local_peak_halfwidth_idx: int = 3
    ransac: RansacFitConfig = field(
        default_factory=lambda: RansacFitConfig(min_inliers=8, seed=43)
    )
    miss_confidence_factor: float = 0.7
    max_angular_gap_rad: float = float(np.pi / 2.0)
    require_inner_fit: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InnerFitConfig":
        data = _require_mapping(data, name="data")
        return cls(
            min_points=int(data["min_points"]),
            min_inlier_ratio=float(data["min_inlier_ratio"]),
            max_rms_residual=float(data["max_rms_residual"]),
            max_center_shift_px=float(data["max_center_shift_px"]),
            max_ratio_abs_error=float(data["max_ratio_abs_error"]),
            local_peak_halfwidth_idx=int(data["local_peak_halfwidth_idx"]),
            ransac=RansacFitConfig.from_dict(_require_mapping(data["ransac"], name="ransac")),
            miss_confidence_factor=float(data["miss_confidence_factor"]),
            max_angular_gap_rad=float(data["max_angular_gap_rad"]),
            require_inner_fit=bool(data["require_inner_fit"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_points": int(self.min_points),
            "min_inlier_ratio": float(self.min_inlier_ratio),
            "max_rms_residual": float(self.max_rms_residual),
            "max_center_shift_px": float(self.max_center_shift_px),
            "max_ratio_abs_error": float(self.max_ratio_abs_error),
            "local_peak_halfwidth_idx": int(self.local_peak_halfwidth_idx),
            "ransac": self.ransac.to_dict(),
            "miss_confidence_factor": float(self.miss_confidence_factor),
            "max_angular_gap_rad": float(self.max_angular_gap_rad),
            "require_inner_fit": bool(self.require_inner_fit),
        }


@dataclass(slots=True)
class OuterFitConfig:
    min_direct_fit_points: int = 6
    min_ransac_points: int = 8
    ransac: RansacFitConfig = field(default_factory=RansacFitConfig)
    size_score_weight: float = 0.15
    max_angular_gap_rad: float = float(np.pi / 2.0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OuterFitConfig":
        data = _require_mapping(data, name="data")
        return cls(
            min_direct_fit_points=int(data["min_direct_fit_points"]),
            min_ransac_points=int(data["min_ransac_points"]),
            ransac=RansacFitConfig.from_dict(_require_mapping(data["ransac"], name="ransac")),
            size_score_weight=float(data["size_score_weight"]),
            max_angular_gap_rad=float(data["max_angular_gap_rad"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_direct_fit_points": int(self.min_direct_fit_points),
            "min_ransac_points": int(self.min_ransac_points),
            "ransac": self.ransac.to_dict(),
            "size_score_weight": float(self.size_score_weight),
            "max_angular_gap_rad": float(self.max_angular_gap_rad),
        }


@dataclass(slots=True)
class CompletionParams:
    enable: bool = True
    roi_radius_px: float = 30.0
    reproj_gate_px: float = 3.0
    min_fit_confidence: float = 0.45
    min_arc_coverage: float = 0.35
    max_attempts: int | None = None
    image_margin_px: float = 10.0
    require_perfect_decode: bool = False
    max_radii_std_ratio: float = 0.35

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompletionParams":
        data = _require_mapping(data, name="data")
        return cls(
            enable=bool(data["enable"]),
            roi_radius_px=float(data["roi_radius_px"]),
            reproj_gate_px=float(data["reproj_gate_px"]),
            min_fit_confidence=float(data["min_fit_confidence"]),
            min_arc_coverage=float(data["min_arc_coverage"]),
            max_attempts=_optional_int(data.get("max_attempts")),
            image_margin_px=float(data["image_margin_px"]),
            require_perfect_decode=bool(data["require_perfect_decode"]),
            max_radii_std_ratio=float(data["max_radii_std_ratio"]),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "enable": bool(self.enable),
            "roi_radius_px": float(self.roi_radius_px),
            "reproj_gate_px": float(self.reproj_gate_px),
            "min_fit_confidence": float(self.min_fit_confidence),
            "min_arc_coverage": float(self.min_arc_coverage),
            "image_margin_px": float(self.image_margin_px),
            "require_perfect_decode": bool(self.require_perfect_decode),
            "max_radii_std_ratio": float(self.max_radii_std_ratio),
        }
        _set_optional(out, "max_attempts", None if self.max_attempts is None else int(self.max_attempts))
        return out


@dataclass(slots=True)
class ProjectiveCenterParams:
    use_expected_ratio: bool = True
    ratio_penalty_weight: float = 1.0
    max_center_shift_px: float | None = 40.0
    max_selected_residual: float = 0.25
    min_eig_separation: float = 1e-6

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectiveCenterParams":
        data = _require_mapping(data, name="data")
        return cls(
            use_expected_ratio=bool(data["use_expected_ratio"]),
            ratio_penalty_weight=float(data["ratio_penalty_weight"]),
            max_center_shift_px=_optional_float(data.get("max_center_shift_px")),
            max_selected_residual=float(data["max_selected_residual"]),
            min_eig_separation=float(data["min_eig_separation"]),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "use_expected_ratio": bool(self.use_expected_ratio),
            "ratio_penalty_weight": float(self.ratio_penalty_weight),
            "max_selected_residual": float(self.max_selected_residual),
            "min_eig_separation": float(self.min_eig_separation),
        }
        _set_optional(out, "max_center_shift_px", self.max_center_shift_px)
        return out


@dataclass(slots=True)
class SeedProposalParams:
    merge_radius_px: float = 3.0
    seed_score: float = 1_000_000_000_000.0
    max_seeds: int | None = 512

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeedProposalParams":
        data = _require_mapping(data, name="data")
        return cls(
            merge_radius_px=float(data["merge_radius_px"]),
            seed_score=float(data["seed_score"]),
            max_seeds=_optional_int(data.get("max_seeds")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "merge_radius_px": float(self.merge_radius_px),
            "seed_score": float(self.seed_score),
        }
        _set_optional(out, "max_seeds", None if self.max_seeds is None else int(self.max_seeds))
        return out


@dataclass(slots=True)
class ProposalConfig:
    r_min: float = 2.8
    r_max: float = 56.100002
    grad_threshold: float = 0.05
    nms_radius: float = 5.6
    min_vote_frac: float = 0.1
    accum_sigma: float = 2.0
    max_candidates: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProposalConfig":
        data = _require_mapping(data, name="data")
        return cls(
            r_min=float(data["r_min"]),
            r_max=float(data["r_max"]),
            grad_threshold=float(data["grad_threshold"]),
            nms_radius=float(data["nms_radius"]),
            min_vote_frac=float(data["min_vote_frac"]),
            accum_sigma=float(data["accum_sigma"]),
            max_candidates=_optional_int(data.get("max_candidates")),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "r_min": float(self.r_min),
            "r_max": float(self.r_max),
            "grad_threshold": float(self.grad_threshold),
            "nms_radius": float(self.nms_radius),
            "min_vote_frac": float(self.min_vote_frac),
            "accum_sigma": float(self.accum_sigma),
        }
        _set_optional(
            out, "max_candidates", None if self.max_candidates is None else int(self.max_candidates)
        )
        return out


@dataclass(slots=True)
class EdgeSampleConfig:
    n_rays: int = 48
    r_max: float = 66.0
    r_min: float = 1.5
    r_step: float = 0.5
    min_ring_depth: float = 0.08
    min_rays_with_ring: int = 16

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EdgeSampleConfig":
        data = _require_mapping(data, name="data")
        return cls(
            n_rays=int(data["n_rays"]),
            r_max=float(data["r_max"]),
            r_min=float(data["r_min"]),
            r_step=float(data["r_step"]),
            min_ring_depth=float(data["min_ring_depth"]),
            min_rays_with_ring=int(data["min_rays_with_ring"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rays": int(self.n_rays),
            "r_max": float(self.r_max),
            "r_min": float(self.r_min),
            "r_step": float(self.r_step),
            "min_ring_depth": float(self.min_ring_depth),
            "min_rays_with_ring": int(self.min_rays_with_ring),
        }


@dataclass(slots=True)
class DecodeConfig:
    code_band_ratio: float = 0.74404764
    samples_per_sector: int = 5
    n_radial_rings: int = 3
    max_decode_dist: int = 3
    min_decode_confidence: float = 0.3
    min_decode_margin: int = 1
    min_decode_contrast: float = 0.03
    threshold_max_iters: int = 10
    threshold_convergence_eps: float = 0.0001

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecodeConfig":
        data = _require_mapping(data, name="data")
        return cls(
            code_band_ratio=float(data["code_band_ratio"]),
            samples_per_sector=int(data["samples_per_sector"]),
            n_radial_rings=int(data["n_radial_rings"]),
            max_decode_dist=int(data["max_decode_dist"]),
            min_decode_confidence=float(data["min_decode_confidence"]),
            min_decode_margin=int(data["min_decode_margin"]),
            min_decode_contrast=float(data["min_decode_contrast"]),
            threshold_max_iters=int(data["threshold_max_iters"]),
            threshold_convergence_eps=float(data["threshold_convergence_eps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_band_ratio": float(self.code_band_ratio),
            "samples_per_sector": int(self.samples_per_sector),
            "n_radial_rings": int(self.n_radial_rings),
            "max_decode_dist": int(self.max_decode_dist),
            "min_decode_confidence": float(self.min_decode_confidence),
            "min_decode_margin": int(self.min_decode_margin),
            "min_decode_contrast": float(self.min_decode_contrast),
            "threshold_max_iters": int(self.threshold_max_iters),
            "threshold_convergence_eps": float(self.threshold_convergence_eps),
        }


@dataclass(slots=True)
class MarkerSpec:
    r_inner_expected: float = 0.48809522
    inner_search_halfwidth: float = 0.08
    inner_grad_polarity: str = "light_to_dark"
    radial_samples: int = 64
    theta_samples: int = 96
    aggregator: str = "median"
    min_theta_coverage: float = 0.6
    min_theta_consistency: float = 0.25

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarkerSpec":
        data = _require_mapping(data, name="data")
        return cls(
            r_inner_expected=float(data["r_inner_expected"]),
            inner_search_halfwidth=float(data["inner_search_halfwidth"]),
            inner_grad_polarity=str(data["inner_grad_polarity"]),
            radial_samples=int(data["radial_samples"]),
            theta_samples=int(data["theta_samples"]),
            aggregator=str(data["aggregator"]),
            min_theta_coverage=float(data["min_theta_coverage"]),
            min_theta_consistency=float(data["min_theta_consistency"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_inner_expected": float(self.r_inner_expected),
            "inner_search_halfwidth": float(self.inner_search_halfwidth),
            "inner_grad_polarity": str(self.inner_grad_polarity),
            "radial_samples": int(self.radial_samples),
            "theta_samples": int(self.theta_samples),
            "aggregator": str(self.aggregator),
            "min_theta_coverage": float(self.min_theta_coverage),
            "min_theta_consistency": float(self.min_theta_consistency),
        }


@dataclass(slots=True)
class OuterEstimationConfig:
    search_halfwidth_px: float = 13.0
    radial_samples: int = 64
    aggregator: str = "median"
    grad_polarity: str = "dark_to_light"
    min_theta_coverage: float = 0.6
    min_theta_consistency: float = 0.35
    allow_two_hypotheses: bool = True
    second_peak_min_rel: float = 0.85
    refine_halfwidth_px: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OuterEstimationConfig":
        data = _require_mapping(data, name="data")
        return cls(
            search_halfwidth_px=float(data["search_halfwidth_px"]),
            radial_samples=int(data["radial_samples"]),
            aggregator=str(data["aggregator"]),
            grad_polarity=str(data["grad_polarity"]),
            min_theta_coverage=float(data["min_theta_coverage"]),
            min_theta_consistency=float(data["min_theta_consistency"]),
            allow_two_hypotheses=bool(data["allow_two_hypotheses"]),
            second_peak_min_rel=float(data["second_peak_min_rel"]),
            refine_halfwidth_px=float(data["refine_halfwidth_px"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_halfwidth_px": float(self.search_halfwidth_px),
            "radial_samples": int(self.radial_samples),
            "aggregator": str(self.aggregator),
            "grad_polarity": str(self.grad_polarity),
            "min_theta_coverage": float(self.min_theta_coverage),
            "min_theta_consistency": float(self.min_theta_consistency),
            "allow_two_hypotheses": bool(self.allow_two_hypotheses),
            "second_peak_min_rel": float(self.second_peak_min_rel),
            "refine_halfwidth_px": float(self.refine_halfwidth_px),
        }


@dataclass(slots=True)
class RansacHomographyConfig:
    max_iters: int = 2000
    inlier_threshold: float = 5.0
    min_inliers: int = 6
    seed: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RansacHomographyConfig":
        data = _require_mapping(data, name="data")
        return cls(
            max_iters=int(data["max_iters"]),
            inlier_threshold=float(data["inlier_threshold"]),
            min_inliers=int(data["min_inliers"]),
            seed=int(data["seed"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iters": int(self.max_iters),
            "inlier_threshold": float(self.inlier_threshold),
            "min_inliers": int(self.min_inliers),
            "seed": int(self.seed),
        }


@dataclass(slots=True)
class SelfUndistortConfig:
    enable: bool = False
    lambda_range: list[float] = field(default_factory=lambda: [-8e-7, 8e-7])
    max_evals: int = 40
    min_markers: int = 6
    improvement_threshold: float = 0.01
    min_abs_improvement: float = 1e-4
    trim_fraction: float = 0.1
    min_lambda_abs: float = 5e-9
    reject_range_edge: bool = True
    range_edge_margin_frac: float = 0.02
    validation_min_markers: int = 24
    validation_abs_improvement_px: float = 0.05
    validation_rel_improvement: float = 0.03

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelfUndistortConfig":
        data = _require_mapping(data, name="data")
        return cls(
            enable=bool(data["enable"]),
            lambda_range=[float(data["lambda_range"][0]), float(data["lambda_range"][1])],
            max_evals=int(data["max_evals"]),
            min_markers=int(data["min_markers"]),
            improvement_threshold=float(data["improvement_threshold"]),
            min_abs_improvement=float(data["min_abs_improvement"]),
            trim_fraction=float(data["trim_fraction"]),
            min_lambda_abs=float(data["min_lambda_abs"]),
            reject_range_edge=bool(data["reject_range_edge"]),
            range_edge_margin_frac=float(data["range_edge_margin_frac"]),
            validation_min_markers=int(data["validation_min_markers"]),
            validation_abs_improvement_px=float(data["validation_abs_improvement_px"]),
            validation_rel_improvement=float(data["validation_rel_improvement"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable": bool(self.enable),
            "lambda_range": [float(self.lambda_range[0]), float(self.lambda_range[1])],
            "max_evals": int(self.max_evals),
            "min_markers": int(self.min_markers),
            "improvement_threshold": float(self.improvement_threshold),
            "min_abs_improvement": float(self.min_abs_improvement),
            "trim_fraction": float(self.trim_fraction),
            "min_lambda_abs": float(self.min_lambda_abs),
            "reject_range_edge": bool(self.reject_range_edge),
            "range_edge_margin_frac": float(self.range_edge_margin_frac),
            "validation_min_markers": int(self.validation_min_markers),
            "validation_abs_improvement_px": float(self.validation_abs_improvement_px),
            "validation_rel_improvement": float(self.validation_rel_improvement),
        }


@dataclass(slots=True)
class IdCorrectionConfig:
    enable: bool = True
    auto_search_radius_outer_muls: list[float] = field(
        default_factory=lambda: [2.4, 2.9, 3.5, 4.2, 5.0]
    )
    consistency_outer_mul: float = 3.2
    consistency_min_neighbors: int = 1
    consistency_min_support_edges: int = 1
    consistency_max_contradiction_frac: float = 0.5
    soft_lock_exact_decode: bool = True
    min_votes: int = 2
    min_votes_recover: int = 1
    min_vote_weight_frac: float = 0.55
    h_reproj_gate_px: float = 30.0
    homography_fallback_enable: bool = True
    homography_min_trusted: int = 24
    homography_min_inliers: int = 12
    max_iters: int = 5
    remove_unverified: bool = False
    seed_min_decode_confidence: float = 0.7

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IdCorrectionConfig":
        data = _require_mapping(data, name="data")
        return cls(
            enable=bool(data["enable"]),
            auto_search_radius_outer_muls=[float(v) for v in data["auto_search_radius_outer_muls"]],
            consistency_outer_mul=float(data["consistency_outer_mul"]),
            consistency_min_neighbors=int(data["consistency_min_neighbors"]),
            consistency_min_support_edges=int(data["consistency_min_support_edges"]),
            consistency_max_contradiction_frac=float(data["consistency_max_contradiction_frac"]),
            soft_lock_exact_decode=bool(data["soft_lock_exact_decode"]),
            min_votes=int(data["min_votes"]),
            min_votes_recover=int(data["min_votes_recover"]),
            min_vote_weight_frac=float(data["min_vote_weight_frac"]),
            h_reproj_gate_px=float(data["h_reproj_gate_px"]),
            homography_fallback_enable=bool(data["homography_fallback_enable"]),
            homography_min_trusted=int(data["homography_min_trusted"]),
            homography_min_inliers=int(data["homography_min_inliers"]),
            max_iters=int(data["max_iters"]),
            remove_unverified=bool(data["remove_unverified"]),
            seed_min_decode_confidence=float(data["seed_min_decode_confidence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable": bool(self.enable),
            "auto_search_radius_outer_muls": [float(v) for v in self.auto_search_radius_outer_muls],
            "consistency_outer_mul": float(self.consistency_outer_mul),
            "consistency_min_neighbors": int(self.consistency_min_neighbors),
            "consistency_min_support_edges": int(self.consistency_min_support_edges),
            "consistency_max_contradiction_frac": float(self.consistency_max_contradiction_frac),
            "soft_lock_exact_decode": bool(self.soft_lock_exact_decode),
            "min_votes": int(self.min_votes),
            "min_votes_recover": int(self.min_votes_recover),
            "min_vote_weight_frac": float(self.min_vote_weight_frac),
            "h_reproj_gate_px": float(self.h_reproj_gate_px),
            "homography_fallback_enable": bool(self.homography_fallback_enable),
            "homography_min_trusted": int(self.homography_min_trusted),
            "homography_min_inliers": int(self.homography_min_inliers),
            "max_iters": int(self.max_iters),
            "remove_unverified": bool(self.remove_unverified),
            "seed_min_decode_confidence": float(self.seed_min_decode_confidence),
        }


@dataclass(slots=True)
class InnerAsOuterRecoveryConfig:
    enable: bool = True
    ratio_threshold: float = 0.75
    k_neighbors: int = 6
    min_theta_consistency: float = 0.18
    min_theta_coverage: float = 0.4
    min_ring_depth: float = 0.02
    refine_halfwidth_px: float = 2.5
    size_gate_tolerance: float = 0.25

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InnerAsOuterRecoveryConfig":
        data = _require_mapping(data, name="data")
        return cls(
            enable=bool(data["enable"]),
            ratio_threshold=float(data["ratio_threshold"]),
            k_neighbors=int(data["k_neighbors"]),
            min_theta_consistency=float(data["min_theta_consistency"]),
            min_theta_coverage=float(data["min_theta_coverage"]),
            min_ring_depth=float(data["min_ring_depth"]),
            refine_halfwidth_px=float(data["refine_halfwidth_px"]),
            size_gate_tolerance=float(data["size_gate_tolerance"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable": bool(self.enable),
            "ratio_threshold": float(self.ratio_threshold),
            "k_neighbors": int(self.k_neighbors),
            "min_theta_consistency": float(self.min_theta_consistency),
            "min_theta_coverage": float(self.min_theta_coverage),
            "min_ring_depth": float(self.min_ring_depth),
            "refine_halfwidth_px": float(self.refine_halfwidth_px),
            "size_gate_tolerance": float(self.size_gate_tolerance),
        }


class DetectConfig:
    """High-level detector configuration with typed section properties.

    Parameters
    ----------
    board:
        Board layout used to derive default scale-coupled settings.
    """

    def __init__(self, board: BoardLayout) -> None:
        if not isinstance(board, BoardLayout):
            raise TypeError("board must be BoardLayout")
        self._board = board
        self._core = _DetectConfigCore(board._spec_json)
        self._version = 0

    def _snapshot(self) -> dict[str, Any]:
        return json.loads(self._core.dump_json())

    def _config_json(self) -> str:
        return self._core.dump_json()

    def to_dict(self) -> dict[str, Any]:
        """Return a snapshot of the fully resolved config dictionary."""
        return self._snapshot()

    def _apply_overlay(self, overlay: dict[str, Any]) -> None:
        self._core.apply_overlay_json(json.dumps(dict(overlay)))
        self._version += 1

    @property
    def board(self) -> BoardLayout:
        return self._board

    @property
    def marker_scale(self) -> MarkerScalePrior:
        resolved = self._snapshot()
        return MarkerScalePrior.from_dict(resolved["marker_scale"])

    @marker_scale.setter
    def marker_scale(self, value: MarkerScalePrior) -> None:
        if not isinstance(value, MarkerScalePrior):
            raise TypeError("marker_scale must be MarkerScalePrior")
        self._apply_overlay({"marker_scale": value.to_dict()})

    @property
    def inner_fit(self) -> InnerFitConfig:
        resolved = self._snapshot()
        return InnerFitConfig.from_dict(resolved["inner_fit"])

    @inner_fit.setter
    def inner_fit(self, value: InnerFitConfig) -> None:
        if not isinstance(value, InnerFitConfig):
            raise TypeError("inner_fit must be InnerFitConfig")
        self._apply_overlay({"inner_fit": value.to_dict()})

    @property
    def outer_fit(self) -> OuterFitConfig:
        resolved = self._snapshot()
        return OuterFitConfig.from_dict(resolved["outer_fit"])

    @outer_fit.setter
    def outer_fit(self, value: OuterFitConfig) -> None:
        if not isinstance(value, OuterFitConfig):
            raise TypeError("outer_fit must be OuterFitConfig")
        self._apply_overlay({"outer_fit": value.to_dict()})

    @property
    def completion(self) -> CompletionParams:
        resolved = self._snapshot()
        return CompletionParams.from_dict(resolved["completion"])

    @completion.setter
    def completion(self, value: CompletionParams) -> None:
        if not isinstance(value, CompletionParams):
            raise TypeError("completion must be CompletionParams")
        self._apply_overlay({"completion": value.to_dict()})

    @property
    def projective_center(self) -> ProjectiveCenterParams:
        resolved = self._snapshot()
        return ProjectiveCenterParams.from_dict(resolved["projective_center"])

    @projective_center.setter
    def projective_center(self, value: ProjectiveCenterParams) -> None:
        if not isinstance(value, ProjectiveCenterParams):
            raise TypeError("projective_center must be ProjectiveCenterParams")
        self._apply_overlay({"projective_center": value.to_dict()})

    @property
    def seed_proposals(self) -> SeedProposalParams:
        resolved = self._snapshot()
        return SeedProposalParams.from_dict(resolved["seed_proposals"])

    @seed_proposals.setter
    def seed_proposals(self, value: SeedProposalParams) -> None:
        if not isinstance(value, SeedProposalParams):
            raise TypeError("seed_proposals must be SeedProposalParams")
        self._apply_overlay({"seed_proposals": value.to_dict()})

    @property
    def proposal(self) -> ProposalConfig:
        resolved = self._snapshot()
        return ProposalConfig.from_dict(resolved["proposal"])

    @proposal.setter
    def proposal(self, value: ProposalConfig) -> None:
        if not isinstance(value, ProposalConfig):
            raise TypeError("proposal must be ProposalConfig")
        self._apply_overlay({"proposal": value.to_dict()})

    @property
    def edge_sample(self) -> EdgeSampleConfig:
        resolved = self._snapshot()
        return EdgeSampleConfig.from_dict(resolved["edge_sample"])

    @edge_sample.setter
    def edge_sample(self, value: EdgeSampleConfig) -> None:
        if not isinstance(value, EdgeSampleConfig):
            raise TypeError("edge_sample must be EdgeSampleConfig")
        self._apply_overlay({"edge_sample": value.to_dict()})

    @property
    def decode(self) -> DecodeConfig:
        resolved = self._snapshot()
        return DecodeConfig.from_dict(resolved["decode"])

    @decode.setter
    def decode(self, value: DecodeConfig) -> None:
        if not isinstance(value, DecodeConfig):
            raise TypeError("decode must be DecodeConfig")
        self._apply_overlay({"decode": value.to_dict()})

    @property
    def marker_spec(self) -> MarkerSpec:
        resolved = self._snapshot()
        return MarkerSpec.from_dict(resolved["marker_spec"])

    @marker_spec.setter
    def marker_spec(self, value: MarkerSpec) -> None:
        if not isinstance(value, MarkerSpec):
            raise TypeError("marker_spec must be MarkerSpec")
        self._apply_overlay({"marker_spec": value.to_dict()})

    @property
    def outer_estimation(self) -> OuterEstimationConfig:
        resolved = self._snapshot()
        return OuterEstimationConfig.from_dict(resolved["outer_estimation"])

    @outer_estimation.setter
    def outer_estimation(self, value: OuterEstimationConfig) -> None:
        if not isinstance(value, OuterEstimationConfig):
            raise TypeError("outer_estimation must be OuterEstimationConfig")
        self._apply_overlay({"outer_estimation": value.to_dict()})

    @property
    def ransac_homography(self) -> RansacHomographyConfig:
        resolved = self._snapshot()
        return RansacHomographyConfig.from_dict(resolved["ransac_homography"])

    @ransac_homography.setter
    def ransac_homography(self, value: RansacHomographyConfig) -> None:
        if not isinstance(value, RansacHomographyConfig):
            raise TypeError("ransac_homography must be RansacHomographyConfig")
        self._apply_overlay({"ransac_homography": value.to_dict()})

    @property
    def self_undistort(self) -> SelfUndistortConfig:
        resolved = self._snapshot()
        return SelfUndistortConfig.from_dict(resolved["self_undistort"])

    @self_undistort.setter
    def self_undistort(self, value: SelfUndistortConfig) -> None:
        if not isinstance(value, SelfUndistortConfig):
            raise TypeError("self_undistort must be SelfUndistortConfig")
        self._apply_overlay({"self_undistort": value.to_dict()})

    @property
    def id_correction(self) -> IdCorrectionConfig:
        resolved = self._snapshot()
        return IdCorrectionConfig.from_dict(resolved["id_correction"])

    @id_correction.setter
    def id_correction(self, value: IdCorrectionConfig) -> None:
        if not isinstance(value, IdCorrectionConfig):
            raise TypeError("id_correction must be IdCorrectionConfig")
        self._apply_overlay({"id_correction": value.to_dict()})

    @property
    def inner_as_outer_recovery(self) -> InnerAsOuterRecoveryConfig:
        resolved = self._snapshot()
        return InnerAsOuterRecoveryConfig.from_dict(resolved["inner_as_outer_recovery"])

    @inner_as_outer_recovery.setter
    def inner_as_outer_recovery(self, value: InnerAsOuterRecoveryConfig) -> None:
        if not isinstance(value, InnerAsOuterRecoveryConfig):
            raise TypeError("inner_as_outer_recovery must be InnerAsOuterRecoveryConfig")
        self._apply_overlay({"inner_as_outer_recovery": value.to_dict()})

    @property
    def circle_refinement(self) -> CircleRefinementMethod:
        resolved = self._snapshot()
        return _circle_refinement_from_wire(str(resolved["circle_refinement"]))

    @circle_refinement.setter
    def circle_refinement(self, value: CircleRefinementMethod) -> None:
        self._apply_overlay({"circle_refinement": _circle_refinement_to_wire(value)})

    @property
    def dedup_radius(self) -> float:
        resolved = self._snapshot()
        return float(resolved["dedup_radius"])

    @dedup_radius.setter
    def dedup_radius(self, value: float) -> None:
        self._apply_overlay({"dedup_radius": float(value)})

    @property
    def max_aspect_ratio(self) -> float:
        resolved = self._snapshot()
        return float(resolved["max_aspect_ratio"])

    @max_aspect_ratio.setter
    def max_aspect_ratio(self, value: float) -> None:
        self._apply_overlay({"max_aspect_ratio": float(value)})

    @property
    def inner_fit_required(self) -> bool:
        return bool(self.inner_fit.require_inner_fit)

    @inner_fit_required.setter
    def inner_fit_required(self, value: bool) -> None:
        section = self.inner_fit
        section.require_inner_fit = bool(value)
        self.inner_fit = section

    @property
    def homography_inlier_threshold_px(self) -> float:
        return float(self.ransac_homography.inlier_threshold)

    @homography_inlier_threshold_px.setter
    def homography_inlier_threshold_px(self, value: float) -> None:
        section = self.ransac_homography
        section.inlier_threshold = float(value)
        self.ransac_homography = section

    @property
    def completion_enable(self) -> bool:
        return bool(self.completion.enable)

    @completion_enable.setter
    def completion_enable(self, value: bool) -> None:
        section = self.completion
        section.enable = bool(value)
        self.completion = section

    @property
    def use_global_filter(self) -> bool:
        resolved = self._snapshot()
        return bool(resolved["use_global_filter"])

    @use_global_filter.setter
    def use_global_filter(self, value: bool) -> None:
        self._apply_overlay({"use_global_filter": bool(value)})

    @property
    def self_undistort_enable(self) -> bool:
        return bool(self.self_undistort.enable)

    @self_undistort_enable.setter
    def self_undistort_enable(self, value: bool) -> None:
        section = self.self_undistort
        section.enable = bool(value)
        self.self_undistort = section

    @property
    def decode_min_margin(self) -> int:
        return int(self.decode.min_decode_margin)

    @decode_min_margin.setter
    def decode_min_margin(self, value: int) -> None:
        section = self.decode
        section.min_decode_margin = int(value)
        self.decode = section

    @property
    def decode_max_dist(self) -> int:
        return int(self.decode.max_decode_dist)

    @decode_max_dist.setter
    def decode_max_dist(self, value: int) -> None:
        section = self.decode
        section.max_decode_dist = int(value)
        self.decode = section

    @property
    def decode_min_confidence(self) -> float:
        return float(self.decode.min_decode_confidence)

    @decode_min_confidence.setter
    def decode_min_confidence(self, value: float) -> None:
        section = self.decode
        section.min_decode_confidence = float(value)
        self.decode = section


class Detector:
    """High-level detector wrapper.

    Parameters
    ----------
    config:
        Detection configuration.
    """

    def __init__(self, config: DetectConfig) -> None:
        if not isinstance(config, DetectConfig):
            raise TypeError("config must be DetectConfig")
        self._config = config
        self._core = _DetectorCore(config.board._spec_json, config._config_json())
        self._core_config_version = config._version

    @classmethod
    def from_board(cls, board: BoardLayout) -> "Detector":
        if not isinstance(board, BoardLayout):
            raise TypeError("board must be BoardLayout")
        return cls(DetectConfig(board))

    @classmethod
    def with_config(cls, config: DetectConfig) -> "Detector":
        return cls(config)

    @property
    def board(self) -> BoardLayout:
        return self._config.board

    @property
    def config(self) -> DetectConfig:
        return self._config

    def detect(self, image: np.ndarray | str | Path) -> DetectionResult:
        """Run detection on a NumPy image or image path.

        Accepted array inputs:
        - grayscale: `(H, W)`, `dtype=uint8`
        - RGB/RGBA: `(H, W, 3|4)`, `dtype=uint8`
        """
        result_json = self._detect_impl(image=image, mapper_payload=None)
        return DetectionResult.from_dict(json.loads(result_json))

    def detect_with_mapper(
        self,
        image: np.ndarray | str | Path,
        mapper: CameraModel | DivisionModel,
    ) -> DetectionResult:
        """Run two-pass detection with an explicit pixel mapper."""
        if isinstance(mapper, CameraModel):
            payload = mapper._to_mapper_payload()
        elif isinstance(mapper, DivisionModel):
            payload = mapper._to_mapper_payload()
        else:
            raise TypeError("mapper must be CameraModel or DivisionModel")

        result_json = self._detect_impl(image=image, mapper_payload=payload)
        return DetectionResult.from_dict(json.loads(result_json))

    def detect_adaptive(
        self,
        image: np.ndarray | str | Path,
        nominal_diameter_px: float | None = None,
    ) -> DetectionResult:
        """Run robust adaptive detection.

        Parameters
        ----------
        image:
            NumPy image array or image path.
        nominal_diameter_px:
            Optional expected marker diameter in pixels. When provided, adaptive
            mode includes a focused two-tier bracket around the hint as its
            primary candidate, while still keeping fallback candidates enabled.
            Must be finite and > 0.
        """
        parsed_hint = _parse_nominal_diameter_hint(nominal_diameter_px)
        if parsed_hint is None:
            result_json = self._detect_adaptive_impl(image=image)
        else:
            result_json = self._detect_adaptive_with_hint_impl(
                image=image, nominal_diameter_px=parsed_hint
            )
        return DetectionResult.from_dict(json.loads(result_json))

    def detect_adaptive_with_hint(
        self,
        image: np.ndarray | str | Path,
        nominal_diameter_px: float | None = None,
    ) -> DetectionResult:
        """Compatibility alias for :meth:`detect_adaptive`.

        Deprecated: prefer :meth:`detect_adaptive(image, nominal_diameter_px=...)`.
        """
        warnings.warn(
            "Detector.detect_adaptive_with_hint is deprecated; "
            "use Detector.detect_adaptive(image, nominal_diameter_px=...)",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.detect_adaptive(image, nominal_diameter_px=nominal_diameter_px)

    def detect_multiscale(
        self,
        image: np.ndarray | str | Path,
        tiers: ScaleTiers,
    ) -> DetectionResult:
        """Run explicit multi-scale detection with caller-provided tiers."""
        if not isinstance(tiers, ScaleTiers):
            raise TypeError("tiers must be ScaleTiers")
        result_json = self._detect_multiscale_impl(image=image, tiers_payload=tiers._to_wire())
        return DetectionResult.from_dict(json.loads(result_json))

    def adaptive_tiers(
        self,
        image: np.ndarray | str | Path,
        nominal_diameter_px: float | None = None,
    ) -> ScaleTiers:
        """Return the tier set adaptive detection would use for `image`.

        Useful for debugging and reproducible experiments: inspect or persist
        selected tiers, then run :meth:`detect_multiscale` with that exact set.
        """
        parsed_hint = _parse_nominal_diameter_hint(nominal_diameter_px)
        tiers_json = self._adaptive_tiers_impl(image=image, nominal_diameter_px=parsed_hint)
        return ScaleTiers._from_wire(json.loads(tiers_json))

    def _detect_impl(
        self,
        *,
        image: np.ndarray | str | Path,
        mapper_payload: Mapping[str, Any] | None,
    ) -> str:
        self._refresh_core_if_needed()
        if isinstance(image, np.ndarray):
            _validate_image_array(image)
            if mapper_payload is None:
                return self._core.detect_array(image)
            return self._core.detect_with_mapper_array(image, json.dumps(dict(mapper_payload)))

        image_path = _coerce_path(image)
        if mapper_payload is None:
            return self._core.detect_path(image_path)
        return self._core.detect_with_mapper_path(image_path, json.dumps(dict(mapper_payload)))

    def _detect_adaptive_impl(
        self,
        *,
        image: np.ndarray | str | Path,
    ) -> str:
        self._refresh_core_if_needed()
        if isinstance(image, np.ndarray):
            _validate_image_array(image)
            return self._core.detect_adaptive_array(image)

        image_path = _coerce_path(image)
        return self._core.detect_adaptive_path(image_path)

    def _detect_adaptive_with_hint_impl(
        self,
        *,
        image: np.ndarray | str | Path,
        nominal_diameter_px: float | None,
    ) -> str:
        self._refresh_core_if_needed()
        if isinstance(image, np.ndarray):
            _validate_image_array(image)
            return self._core.detect_adaptive_with_hint_array(image, nominal_diameter_px)

        image_path = _coerce_path(image)
        return self._core.detect_adaptive_with_hint_path(image_path, nominal_diameter_px)

    def _detect_multiscale_impl(
        self,
        *,
        image: np.ndarray | str | Path,
        tiers_payload: Mapping[str, Any],
    ) -> str:
        self._refresh_core_if_needed()
        if isinstance(image, np.ndarray):
            _validate_image_array(image)
            return self._core.detect_multiscale_array(image, json.dumps(dict(tiers_payload)))

        image_path = _coerce_path(image)
        return self._core.detect_multiscale_path(image_path, json.dumps(dict(tiers_payload)))

    def _adaptive_tiers_impl(
        self,
        *,
        image: np.ndarray | str | Path,
        nominal_diameter_px: float | None,
    ) -> str:
        self._refresh_core_if_needed()
        if isinstance(image, np.ndarray):
            _validate_image_array(image)
            return self._core.adaptive_tiers_array(image, nominal_diameter_px)

        image_path = _coerce_path(image)
        return self._core.adaptive_tiers_path(image_path, nominal_diameter_px)

    def _refresh_core_if_needed(self) -> None:
        if self._core_config_version == self._config._version:
            return
        self._core = _DetectorCore(self._config.board._spec_json, self._config._config_json())
        self._core_config_version = self._config._version


__version__ = _package_version()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_vec2(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(value[0]), float(value[1])]


def _optional_points(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    return [[float(p[0]), float(p[1])] for p in value]


def _optional_h(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    return [
        [float(value[0][0]), float(value[0][1]), float(value[0][2])],
        [float(value[1][0]), float(value[1][1]), float(value[1][2])],
        [float(value[2][0]), float(value[2][1]), float(value[2][2])],
    ]


def _set_optional(out: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        out[key] = value


def _validate_image_array(image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise TypeError("image ndarray must have dtype=uint8")

    if image.ndim == 2:
        return

    if image.ndim == 3 and image.shape[2] in (3, 4):
        return

    raise TypeError("image ndarray must have shape (H, W) or (H, W, 3|4)")


def _parse_nominal_diameter_hint(value: float | None) -> float | None:
    if value is None:
        return None
    return _require_finite_positive_float(value, name="nominal_diameter_px")


def _circle_refinement_from_wire(value: str) -> CircleRefinementMethod:
    lowered = value.strip().lower()
    if lowered in {"none"}:
        return CircleRefinementMethod.NONE
    if lowered in {"projectivecenter", "projective_center"}:
        return CircleRefinementMethod.PROJECTIVE_CENTER
    raise ValueError(f"unknown circle refinement value: {value!r}")


def _circle_refinement_to_wire(value: CircleRefinementMethod) -> str:
    if value == CircleRefinementMethod.NONE:
        return "None"
    if value == CircleRefinementMethod.PROJECTIVE_CENTER:
        return "ProjectiveCenter"
    raise ValueError(f"unsupported circle refinement value: {value!r}")


__all__ = [
    "BoardLayout",
    "BoardMarker",
    "CircleRefinementMethod",
    "MarkerSpec",
    "MarkerScalePrior",
    "RansacFitConfig",
    "InnerFitConfig",
    "OuterFitConfig",
    "CompletionParams",
    "ProjectiveCenterParams",
    "SeedProposalParams",
    "ProposalConfig",
    "EdgeSampleConfig",
    "DecodeConfig",
    "OuterEstimationConfig",
    "RansacHomographyConfig",
    "SelfUndistortConfig",
    "IdCorrectionConfig",
    "InnerAsOuterRecoveryConfig",
    "ScaleTier",
    "ScaleTiers",
    "DetectConfig",
    "Detector",
    "CameraIntrinsics",
    "RadialTangentialDistortion",
    "CameraModel",
    "DivisionModel",
    "DetectionResult",
    "DetectedMarker",
    "FitMetrics",
    "DecodeMetrics",
    "RansacStats",
    "SelfUndistortResult",
    "Ellipse",
    "DetectionFrame",
    "__version__",
]
