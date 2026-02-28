"""ringgrid Python bindings.

High-level, typed API for dense coded ring marker detection backed by the
native Rust detector.
"""

from ._api import (
    BoardLayout,
    BoardMarker,
    CameraIntrinsics,
    CameraModel,
    CircleRefinementMethod,
    DecodeMetrics,
    DetectConfig,
    DetectedMarker,
    DetectionFrame,
    DetectionResult,
    Detector,
    DivisionModel,
    Ellipse,
    FitMetrics,
    MarkerScalePrior,
    RadialTangentialDistortion,
    RansacStats,
    SelfUndistortResult,
    __version__,
)

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
