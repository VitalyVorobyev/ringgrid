"""Public Python API for the ringgrid detector.

This module exposes a native detector backed by Rust (`PyO3` extension) and a
typed Python surface that feels idiomatic in notebooks and applications.

Typical flow:
1. Build/load a :class:`BoardLayout`.
2. Construct :class:`DetectConfig` and :class:`Detector`.
3. Call :meth:`Detector.detect` (or :meth:`Detector.detect_with_mapper`).
4. Consume :class:`DetectionResult` in memory, JSON, or via plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._ringgrid import (
    DetectorCore as _DetectorCore,
    board_snapshot_json as _board_snapshot_json,
    default_board_spec_json as _default_board_spec_json,
    load_board_spec_json as _load_board_spec_json,
    package_version as _package_version,
    resolve_config_json as _resolve_config_json,
    update_config_json as _update_config_json,
)


def _deepcopy_jsonable(value: Any) -> Any:
    return copy.deepcopy(value)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _coerce_path(path: str | Path) -> str:
    return str(Path(path))


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


class DetectConfig:
    """High-level detector configuration with curated properties and dict escape hatch.

    Parameters
    ----------
    board:
        Board layout used to derive default scale-coupled settings.
    data:
        Optional config overlay mapping. Any omitted fields keep defaults.

    Notes
    -----
    For advanced tuning beyond curated properties, use:
    - :meth:`to_dict` to inspect the full resolved config
    - :meth:`update_from_dict` to patch nested settings
    """

    def __init__(self, board: BoardLayout, data: Mapping[str, Any] | None = None) -> None:
        self._board = board
        overlay_json = None if data is None else json.dumps(dict(data))
        self._resolved = json.loads(_resolve_config_json(board._spec_json, overlay_json))

    @classmethod
    def from_dict(cls, board: BoardLayout, data: Mapping[str, Any]) -> "DetectConfig":
        """Construct config from an overlay mapping."""
        return cls(board=board, data=data)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the fully resolved config dictionary."""
        return _deepcopy_jsonable(self._resolved)

    def update_from_dict(self, data: Mapping[str, Any]) -> None:
        """Apply an overlay mapping onto the current resolved config."""
        data = _require_mapping(data, name="data")
        updated = _update_config_json(
            self._board._spec_json,
            json.dumps(self._resolved),
            json.dumps(dict(data)),
        )
        self._resolved = json.loads(updated)

    @property
    def board(self) -> BoardLayout:
        return self._board

    @property
    def marker_scale(self) -> MarkerScalePrior:
        return MarkerScalePrior.from_dict(self._resolved["marker_scale"])

    @marker_scale.setter
    def marker_scale(self, value: MarkerScalePrior) -> None:
        self.update_from_dict({"marker_scale": value.to_dict()})

    @property
    def completion_enable(self) -> bool:
        return bool(self._resolved["completion"]["enable"])

    @completion_enable.setter
    def completion_enable(self, value: bool) -> None:
        section = dict(self._resolved["completion"])
        section["enable"] = bool(value)
        self.update_from_dict({"completion": section})

    @property
    def use_global_filter(self) -> bool:
        return bool(self._resolved["use_global_filter"])

    @use_global_filter.setter
    def use_global_filter(self, value: bool) -> None:
        self.update_from_dict({"use_global_filter": bool(value)})

    @property
    def self_undistort_enable(self) -> bool:
        return bool(self._resolved["self_undistort"]["enable"])

    @self_undistort_enable.setter
    def self_undistort_enable(self, value: bool) -> None:
        section = dict(self._resolved["self_undistort"])
        section["enable"] = bool(value)
        self.update_from_dict({"self_undistort": section})

    @property
    def decode_min_margin(self) -> int:
        return int(self._resolved["decode"]["min_decode_margin"])

    @decode_min_margin.setter
    def decode_min_margin(self, value: int) -> None:
        section = dict(self._resolved["decode"])
        section["min_decode_margin"] = int(value)
        self.update_from_dict({"decode": section})

    @property
    def decode_max_dist(self) -> int:
        return int(self._resolved["decode"]["max_decode_dist"])

    @decode_max_dist.setter
    def decode_max_dist(self, value: int) -> None:
        section = dict(self._resolved["decode"])
        section["max_decode_dist"] = int(value)
        self.update_from_dict({"decode": section})

    @property
    def decode_min_confidence(self) -> float:
        return float(self._resolved["decode"]["min_decode_confidence"])

    @decode_min_confidence.setter
    def decode_min_confidence(self, value: float) -> None:
        section = dict(self._resolved["decode"])
        section["min_decode_confidence"] = float(value)
        self.update_from_dict({"decode": section})

    @property
    def circle_refinement(self) -> CircleRefinementMethod:
        return _circle_refinement_from_wire(str(self._resolved["circle_refinement"]))

    @circle_refinement.setter
    def circle_refinement(self, value: CircleRefinementMethod) -> None:
        self.update_from_dict({"circle_refinement": _circle_refinement_to_wire(value)})

    @property
    def inner_fit_required(self) -> bool:
        return bool(self._resolved["inner_fit"]["require_inner_fit"])

    @inner_fit_required.setter
    def inner_fit_required(self, value: bool) -> None:
        section = dict(self._resolved["inner_fit"])
        section["require_inner_fit"] = bool(value)
        self.update_from_dict({"inner_fit": section})

    @property
    def homography_inlier_threshold_px(self) -> float:
        return float(self._resolved["ransac_homography"]["inlier_threshold"])

    @homography_inlier_threshold_px.setter
    def homography_inlier_threshold_px(self, value: float) -> None:
        section = dict(self._resolved["ransac_homography"])
        section["inlier_threshold"] = float(value)
        self.update_from_dict({"ransac_homography": section})


class Detector:
    """High-level detector wrapper.

    Parameters
    ----------
    board:
        Board layout to use.
    config:
        Optional detection configuration. If omitted, defaults are used.
    """

    def __init__(self, board: BoardLayout, config: DetectConfig | None = None) -> None:
        if config is None:
            config = DetectConfig(board)
        self._board = board
        self._config = config
        self._core = _DetectorCore(board._spec_json, json.dumps(config.to_dict()))

    @property
    def board(self) -> BoardLayout:
        return self._board

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

    def _detect_impl(
        self,
        *,
        image: np.ndarray | str | Path,
        mapper_payload: Mapping[str, Any] | None,
    ) -> str:
        if isinstance(image, np.ndarray):
            _validate_image_array(image)
            if mapper_payload is None:
                return self._core.detect_array(image)
            return self._core.detect_with_mapper_array(image, json.dumps(dict(mapper_payload)))

        image_path = _coerce_path(image)
        if mapper_payload is None:
            return self._core.detect_path(image_path)
        return self._core.detect_with_mapper_path(image_path, json.dumps(dict(mapper_payload)))


__version__ = _package_version()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


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
    "MarkerScalePrior",
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
    "CircleRefinementMethod",
    "__version__",
]
