//! External pixel-mapper support for distortion-aware detection. The mapper
//! crosses the boundary as JSON with the same `MapperSpec` schema the Python
//! binding uses.

use image::GrayImage;
use serde::Deserialize;

use crate::status::RinggridStatus;

/// JSON spec for an external pixel mapper, tagged by `kind`:
///
/// - `{"kind": "camera", "intrinsics": {...}, "distortion": {...}}`
/// - `{"kind": "division", "lambda": ..., "cx": ..., "cy": ...}`
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum MapperSpec {
    Camera {
        intrinsics: ringgrid::CameraIntrinsics,
        distortion: ringgrid::RadialTangentialDistortion,
    },
    Division {
        lambda: f64,
        cx: f64,
        cy: f64,
    },
}

pub(crate) fn parse_mapper(json: &str) -> Result<MapperSpec, RinggridStatus> {
    serde_json::from_str(json).map_err(|_| RinggridStatus::ErrBadJson)
}

pub(crate) fn detect_with_mapper(
    detector: &ringgrid::Detector,
    gray: &GrayImage,
    spec: &MapperSpec,
) -> Result<ringgrid::DetectionResult, RinggridStatus> {
    match spec {
        MapperSpec::Camera {
            intrinsics,
            distortion,
        } => {
            let camera = ringgrid::CameraModel {
                intrinsics: *intrinsics,
                distortion: *distortion,
            };
            detector.detect_with_mapper(gray, &camera)
        }
        MapperSpec::Division { lambda, cx, cy } => {
            let mapper = ringgrid::DivisionModel::new(*lambda, *cx, *cy);
            detector.detect_with_mapper(gray, &mapper)
        }
    }
    .map_err(|_| RinggridStatus::ErrDetect)
}

pub(crate) fn detect_with_mapper_diagnostics(
    detector: &ringgrid::Detector,
    gray: &GrayImage,
    spec: &MapperSpec,
) -> Result<
    (
        ringgrid::DetectionResult,
        ringgrid::diagnostics::DetectionDiagnostics,
    ),
    RinggridStatus,
> {
    match spec {
        MapperSpec::Camera {
            intrinsics,
            distortion,
        } => {
            let camera = ringgrid::CameraModel {
                intrinsics: *intrinsics,
                distortion: *distortion,
            };
            detector.detect_with_mapper_diagnostics(gray, &camera)
        }
        MapperSpec::Division { lambda, cx, cy } => {
            let mapper = ringgrid::DivisionModel::new(*lambda, *cx, *cy);
            detector.detect_with_mapper_diagnostics(gray, &mapper)
        }
    }
    .map_err(|_| RinggridStatus::ErrDetect)
}
