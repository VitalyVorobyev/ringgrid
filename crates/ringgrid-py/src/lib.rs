use image::GrayImage;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MapperSpec {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleTierWire {
    diameter_min_px: f32,
    diameter_max_px: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleTiersWire {
    tiers: Vec<ScaleTierWire>,
}

#[derive(Serialize)]
struct ProposalResultPayload {
    image_size: [u32; 2],
    proposals: Vec<ringgrid::Proposal>,
}

/// Combined payload for the diagnostics-returning detection entry points.
///
/// The slim [`ringgrid::DetectionResult`] and the opt-in
/// [`ringgrid::diagnostics::DetectionDiagnostics`] are nested under `result` and
/// `diagnostics`; `diagnostics.markers` aligns 1:1 with
/// `result.detected_markers`.
#[derive(Serialize)]
struct DetectionWithDiagnostics {
    result: ringgrid::DetectionResult,
    diagnostics: ringgrid::diagnostics::DetectionDiagnostics,
}

fn detection_with_diagnostics_json(
    result: ringgrid::DetectionResult,
    diagnostics: ringgrid::diagnostics::DetectionDiagnostics,
) -> PyResult<String> {
    serde_json::to_string(&DetectionWithDiagnostics {
        result,
        diagnostics,
    })
    .map_err(py_value_error)
}

fn py_value_error<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn py_target_generation_error(err: ringgrid::TargetGenerationError) -> PyErr {
    match err {
        ringgrid::TargetGenerationError::InvalidMargin { .. }
        | ringgrid::TargetGenerationError::InvalidDpi { .. } => {
            PyValueError::new_err(err.to_string())
        }
        ringgrid::TargetGenerationError::Io(io) => PyOSError::new_err(io.to_string()),
        ringgrid::TargetGenerationError::Image(_)
        | ringgrid::TargetGenerationError::PngEncoding(_) => {
            PyRuntimeError::new_err(err.to_string())
        }
    }
}

/// Parse a target spec (compositional `ringgrid.target.v6`, or legacy
/// `ringgrid.target.v4` auto-migrated).
fn target_from_spec_json(spec_json: &str) -> PyResult<ringgrid::TargetLayout> {
    ringgrid::TargetLayout::from_json_str(spec_json).map_err(py_value_error)
}

/// Reconstitute a [`ringgrid::DetectConfig`] from its JSON representation,
/// attaching `target` and re-deriving the target-coupled / scale-coupled fields.
fn config_from_json(
    target: ringgrid::TargetLayout,
    config_json: &serde_json::Value,
) -> PyResult<ringgrid::DetectConfig> {
    let config: ringgrid::DetectConfig =
        serde_json::from_value(config_json.clone()).map_err(py_value_error)?;
    Ok(config.with_target(target))
}

fn parse_overlay_object(overlay_json: &str) -> PyResult<serde_json::Value> {
    let overlay =
        serde_json::from_str::<serde_json::Value>(overlay_json).map_err(py_value_error)?;
    match overlay {
        obj @ serde_json::Value::Object(_) => Ok(obj),
        _ => Err(PyValueError::new_err(
            "config overlay must be a JSON object",
        )),
    }
}

/// Apply a partial config overlay onto `config`.
///
/// The overlay is a (possibly partial) [`ringgrid::DetectConfig`] JSON object;
/// stage tuning nests under `"advanced"`. It is recursively merged onto the
/// current config's JSON form so that a deeply nested partial section (e.g.
/// `{"advanced": {"completion": {"enable": false}}}`) overrides only the named
/// leaves. After the merge the config is deserialized and the target is
/// re-attached, which re-derives all target- and scale-coupled fields.
fn apply_overlay_json(config: &mut ringgrid::DetectConfig, overlay_json: &str) -> PyResult<()> {
    let overlay = parse_overlay_object(overlay_json)?;
    *config = config.with_json_overlay(overlay).map_err(py_value_error)?;
    Ok(())
}

fn detector_from_json(board_spec_json: &str, config_json: &str) -> PyResult<ringgrid::Detector> {
    let target = target_from_spec_json(board_spec_json)?;
    let config_value =
        serde_json::from_str::<serde_json::Value>(config_json).map_err(py_value_error)?;
    let config = config_from_json(target, &config_value)?;
    Ok(ringgrid::Detector::with_config(config))
}

fn detect_with_core_mapper(
    detector: &ringgrid::Detector,
    gray: &GrayImage,
    mapper_spec: &MapperSpec,
) -> PyResult<ringgrid::DetectionResult> {
    match mapper_spec {
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
    .map_err(py_value_error)
}

fn detect_with_core_mapper_diagnostics(
    detector: &ringgrid::Detector,
    gray: &GrayImage,
    mapper_spec: &MapperSpec,
) -> PyResult<(ringgrid::DetectionResult, ringgrid::diagnostics::DetectionDiagnostics)> {
    match mapper_spec {
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
    .map_err(py_value_error)
}

fn validate_nominal_diameter_hint(nominal_diameter_px: Option<f32>) -> PyResult<Option<f32>> {
    if let Some(diameter_px) = nominal_diameter_px
        && (!diameter_px.is_finite() || diameter_px <= 0.0)
    {
        return Err(PyValueError::new_err(
            "nominal_diameter_px must be finite and > 0 when provided",
        ));
    }
    Ok(nominal_diameter_px)
}

fn scale_tiers_to_wire(tiers: &ringgrid::ScaleTiers) -> ScaleTiersWire {
    ScaleTiersWire {
        tiers: tiers
            .tiers()
            .iter()
            .map(|tier| ScaleTierWire {
                diameter_min_px: tier.prior.diameter_min_px,
                diameter_max_px: tier.prior.diameter_max_px,
            })
            .collect(),
    }
}

fn scale_tiers_from_wire(wire: ScaleTiersWire) -> PyResult<ringgrid::ScaleTiers> {
    if wire.tiers.is_empty() {
        return Err(PyValueError::new_err(
            "scale tiers cannot be empty; provide at least one tier",
        ));
    }

    let mut tiers = Vec::with_capacity(wire.tiers.len());
    for tier in wire.tiers {
        if !tier.diameter_min_px.is_finite()
            || !tier.diameter_max_px.is_finite()
            || tier.diameter_min_px <= 0.0
            || tier.diameter_max_px <= 0.0
        {
            return Err(PyValueError::new_err(
                "tier diameters must be finite and > 0",
            ));
        }
        tiers.push(ringgrid::ScaleTier::new(
            tier.diameter_min_px,
            tier.diameter_max_px,
        ));
    }
    Ok(ringgrid::ScaleTiers::new(tiers))
}

fn parse_scale_tiers(tiers_json: &str) -> PyResult<ringgrid::ScaleTiers> {
    let wire = serde_json::from_str::<ScaleTiersWire>(tiers_json).map_err(py_value_error)?;
    scale_tiers_from_wire(wire)
}

fn parse_proposal_config(config_json: Option<&str>) -> PyResult<ringgrid::ProposalConfig> {
    match config_json {
        Some(config_json) => serde_json::from_str(config_json).map_err(py_value_error),
        None => Ok(ringgrid::ProposalConfig::default()),
    }
}

fn parse_marker_scale(marker_scale_json: &str) -> PyResult<ringgrid::MarkerScalePrior> {
    serde_json::from_str(marker_scale_json).map_err(py_value_error)
}

fn load_gray_image(path: &str) -> PyResult<GrayImage> {
    image::open(path)
        .map(|img| img.to_luma8())
        .map_err(|e| py_value_error(format!("failed to open image '{path}': {e}")))
}

fn rgb_to_luma_u8(r: u8, g: u8, b: u8) -> u8 {
    let y = 77u32 * r as u32 + 150u32 * g as u32 + 29u32 * b as u32 + 128u32;
    (y >> 8) as u8
}

fn gray_image_from_array(array: PyReadonlyArrayDyn<'_, u8>) -> PyResult<GrayImage> {
    let shape = array.shape();
    let view = array.as_array();

    match shape {
        [h, w] => {
            let mut out = Vec::with_capacity(h.saturating_mul(*w));
            for y in 0..*h {
                for x in 0..*w {
                    out.push(view[[y, x]]);
                }
            }
            GrayImage::from_raw(*w as u32, *h as u32, out)
                .ok_or_else(|| PyRuntimeError::new_err("failed to build grayscale image"))
        }
        [h, w, c] => {
            if *c != 3 && *c != 4 {
                return Err(PyTypeError::new_err(
                    "expected image array with shape (H, W) or (H, W, 3|4)",
                ));
            }
            let mut out = Vec::with_capacity(h.saturating_mul(*w));
            for y in 0..*h {
                for x in 0..*w {
                    let r = view[[y, x, 0]];
                    let g = view[[y, x, 1]];
                    let b = view[[y, x, 2]];
                    out.push(rgb_to_luma_u8(r, g, b));
                }
            }
            GrayImage::from_raw(*w as u32, *h as u32, out)
                .ok_or_else(|| PyRuntimeError::new_err("failed to build grayscale image"))
        }
        _ => Err(PyTypeError::new_err(
            "expected image array with shape (H, W) or (H, W, 3|4)",
        )),
    }
}

fn proposal_json(proposals: &[ringgrid::Proposal]) -> PyResult<String> {
    serde_json::to_string(proposals).map_err(py_value_error)
}

fn proposal_result_payload<'py>(
    py: Python<'py>,
    result: ringgrid::ProposalResult,
) -> PyResult<(String, Py<PyArray2<f32>>)> {
    let ringgrid::ProposalResult {
        image_size,
        proposals,
        heatmap,
    } = result;
    let [width, height] = image_size;
    let payload_json = serde_json::to_string(&ProposalResultPayload {
        image_size,
        proposals,
    })
    .map_err(py_value_error)?;
    let accumulator = Array2::from_shape_vec((height as usize, width as usize), heatmap)
        .expect("proposal accumulator shape matches image size");
    let accumulator = accumulator.into_pyarray(py).unbind();
    Ok((payload_json, accumulator))
}

#[pyclass(module = "ringgrid._ringgrid")]
struct DetectConfigCore {
    config: ringgrid::DetectConfig,
}

#[pymethods]
impl DetectConfigCore {
    #[new]
    fn new(board_spec_json: String) -> PyResult<Self> {
        let target = target_from_spec_json(&board_spec_json)?;
        let config = ringgrid::DetectConfig::from_target(target);
        Ok(Self { config })
    }

    fn dump_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.config).map_err(py_value_error)
    }

    fn apply_overlay_json(&mut self, overlay_json: &str) -> PyResult<()> {
        apply_overlay_json(&mut self.config, overlay_json)
    }
}

#[pyclass(module = "ringgrid._ringgrid")]
struct DetectorCore {
    board_spec_json: String,
    config_json: String,
}

#[pymethods]
impl DetectorCore {
    #[new]
    fn new(board_spec_json: String, config_json: String) -> PyResult<Self> {
        let _ = detector_from_json(&board_spec_json, &config_json)?;
        Ok(Self {
            board_spec_json,
            config_json,
        })
    }

    fn detect_path(&self, image_path: &str) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let result = detector.detect(&gray).map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_array(&self, image: PyReadonlyArrayDyn<'_, u8>) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let result = detector.detect(&gray).map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_with_diagnostics_path(&self, image_path: &str) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let (result, diagnostics) = detector
            .detect_with_diagnostics(&gray)
            .map_err(py_value_error)?;
        detection_with_diagnostics_json(result, diagnostics)
    }

    fn detect_with_diagnostics_array(&self, image: PyReadonlyArrayDyn<'_, u8>) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let (result, diagnostics) = detector
            .detect_with_diagnostics(&gray)
            .map_err(py_value_error)?;
        detection_with_diagnostics_json(result, diagnostics)
    }

    fn detect_with_mapper_diagnostics_path(
        &self,
        image_path: &str,
        mapper_json: &str,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let mapper = serde_json::from_str::<MapperSpec>(mapper_json).map_err(py_value_error)?;
        let (result, diagnostics) = detect_with_core_mapper_diagnostics(&detector, &gray, &mapper)?;
        detection_with_diagnostics_json(result, diagnostics)
    }

    fn detect_with_mapper_diagnostics_array(
        &self,
        image: PyReadonlyArrayDyn<'_, u8>,
        mapper_json: &str,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let mapper = serde_json::from_str::<MapperSpec>(mapper_json).map_err(py_value_error)?;
        let (result, diagnostics) = detect_with_core_mapper_diagnostics(&detector, &gray, &mapper)?;
        detection_with_diagnostics_json(result, diagnostics)
    }

    fn propose_path(&self, image_path: &str) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let proposals = detector.propose(&gray);
        proposal_json(&proposals)
    }

    fn propose_array(&self, image: PyReadonlyArrayDyn<'_, u8>) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let proposals = detector.propose(&gray);
        proposal_json(&proposals)
    }

    fn propose_with_heatmap_path<'py>(
        &self,
        py: Python<'py>,
        image_path: &str,
    ) -> PyResult<(String, Py<PyArray2<f32>>)> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        proposal_result_payload(py, detector.propose_with_heatmap(&gray))
    }

    fn propose_with_heatmap_array<'py>(
        &self,
        py: Python<'py>,
        image: PyReadonlyArrayDyn<'py, u8>,
    ) -> PyResult<(String, Py<PyArray2<f32>>)> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        proposal_result_payload(py, detector.propose_with_heatmap(&gray))
    }

    fn detect_with_mapper_path(&self, image_path: &str, mapper_json: &str) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let mapper = serde_json::from_str::<MapperSpec>(mapper_json).map_err(py_value_error)?;
        let result = detect_with_core_mapper(&detector, &gray, &mapper)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_with_mapper_array(
        &self,
        image: PyReadonlyArrayDyn<'_, u8>,
        mapper_json: &str,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let mapper = serde_json::from_str::<MapperSpec>(mapper_json).map_err(py_value_error)?;
        let result = detect_with_core_mapper(&detector, &gray, &mapper)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_adaptive_path(&self, image_path: &str) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let result = detector.detect_adaptive(&gray).map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_adaptive_array(&self, image: PyReadonlyArrayDyn<'_, u8>) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let result = detector.detect_adaptive(&gray).map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    #[pyo3(signature = (image_path, nominal_diameter_px=None))]
    fn detect_adaptive_with_hint_path(
        &self,
        image_path: &str,
        nominal_diameter_px: Option<f32>,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let nominal_diameter_px = validate_nominal_diameter_hint(nominal_diameter_px)?;
        let result = detector
            .detect_adaptive_with_hint(&gray, nominal_diameter_px)
            .map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    #[pyo3(signature = (image, nominal_diameter_px=None))]
    fn detect_adaptive_with_hint_array(
        &self,
        image: PyReadonlyArrayDyn<'_, u8>,
        nominal_diameter_px: Option<f32>,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let nominal_diameter_px = validate_nominal_diameter_hint(nominal_diameter_px)?;
        let result = detector
            .detect_adaptive_with_hint(&gray, nominal_diameter_px)
            .map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_multiscale_path(&self, image_path: &str, tiers_json: &str) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let tiers = parse_scale_tiers(tiers_json)?;
        let result = detector
            .detect_multiscale(&gray, &tiers)
            .map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_multiscale_array(
        &self,
        image: PyReadonlyArrayDyn<'_, u8>,
        tiers_json: &str,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let tiers = parse_scale_tiers(tiers_json)?;
        let result = detector
            .detect_multiscale(&gray, &tiers)
            .map_err(py_value_error)?;
        serde_json::to_string(&result).map_err(py_value_error)
    }

    #[pyo3(signature = (image_path, nominal_diameter_px=None))]
    fn adaptive_tiers_path(
        &self,
        image_path: &str,
        nominal_diameter_px: Option<f32>,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let nominal_diameter_px = validate_nominal_diameter_hint(nominal_diameter_px)?;
        let tiers = detector.adaptive_tiers(&gray, nominal_diameter_px);
        serde_json::to_string(&scale_tiers_to_wire(&tiers)).map_err(py_value_error)
    }

    #[pyo3(signature = (image, nominal_diameter_px=None))]
    fn adaptive_tiers_array(
        &self,
        image: PyReadonlyArrayDyn<'_, u8>,
        nominal_diameter_px: Option<f32>,
    ) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let nominal_diameter_px = validate_nominal_diameter_hint(nominal_diameter_px)?;
        let tiers = detector.adaptive_tiers(&gray, nominal_diameter_px);
        serde_json::to_string(&scale_tiers_to_wire(&tiers)).map_err(py_value_error)
    }
}

#[pyfunction]
fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Canonical `ringgrid.target.v6` JSON for a named [`ringgrid::TargetLayout`]
/// preset. Presets are the single source of truth for their geometry — the
/// typed Python `TargetLayout` constructors read this JSON rather than
/// duplicating constants.
#[pyfunction]
fn target_preset_json(name: &str) -> PyResult<String> {
    let target = match name {
        "default_hex" => ringgrid::TargetLayout::default_hex(),
        "rect_24x24" => ringgrid::TargetLayout::rect_24x24(),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown target preset '{other}' (expected 'default_hex' or 'rect_24x24')"
            )));
        }
    };
    Ok(target.to_json_string())
}

/// Canonical `ringgrid.target.v6` JSON for a 16-sector coded hex target built
/// from direct geometry arguments (mirrors [`ringgrid::TargetLayout::coded_hex`]).
#[pyfunction]
#[pyo3(signature = (
    pitch_mm,
    rows,
    long_row_cols,
    outer_radius_mm,
    inner_radius_mm,
    ring_width_mm
))]
fn coded_hex_target_json(
    pitch_mm: f32,
    rows: usize,
    long_row_cols: usize,
    outer_radius_mm: f32,
    inner_radius_mm: f32,
    ring_width_mm: f32,
) -> PyResult<String> {
    let target = ringgrid::TargetLayout::coded_hex(
        pitch_mm,
        rows,
        long_row_cols,
        outer_radius_mm,
        inner_radius_mm,
        ring_width_mm,
    )
    .map_err(py_value_error)?;
    Ok(target.to_json_string())
}

/// Canonical `ringgrid.target.v6` JSON for a 16-sector coded rect target built
/// from direct geometry arguments (mirrors [`ringgrid::TargetLayout::coded_rect`]).
#[pyfunction]
#[pyo3(signature = (pitch_mm, rows, cols, outer_radius_mm, inner_radius_mm, ring_width_mm))]
fn coded_rect_target_json(
    pitch_mm: f32,
    rows: usize,
    cols: usize,
    outer_radius_mm: f32,
    inner_radius_mm: f32,
    ring_width_mm: f32,
) -> PyResult<String> {
    let target = ringgrid::TargetLayout::coded_rect(
        pitch_mm,
        rows,
        cols,
        outer_radius_mm,
        inner_radius_mm,
        ring_width_mm,
    )
    .map_err(py_value_error)?;
    Ok(target.to_json_string())
}

/// Canonical `ringgrid.target.v6` JSON for a plain hex target built from direct
/// geometry arguments (mirrors [`ringgrid::TargetLayout::plain_hex`]).
///
/// `dots` selects auto-placed origin fiducials — the placement math (lattice
/// gaps + rotational-symmetry validation) lives in the native library, so
/// Python callers never hand-author dot coordinates.
#[pyfunction]
#[pyo3(signature = (pitch_mm, rows, long_row_cols, outer_radius_mm, inner_radius_mm, dots=true))]
fn plain_hex_target_json(
    pitch_mm: f32,
    rows: usize,
    long_row_cols: usize,
    outer_radius_mm: f32,
    inner_radius_mm: f32,
    dots: bool,
) -> PyResult<String> {
    let target = ringgrid::TargetLayout::plain_hex(
        pitch_mm,
        rows,
        long_row_cols,
        outer_radius_mm,
        inner_radius_mm,
        origin_dots(dots),
    )
    .map_err(py_value_error)?;
    Ok(target.to_json_string())
}

/// Canonical `ringgrid.target.v6` JSON for a plain rect target built from direct
/// geometry arguments (mirrors [`ringgrid::TargetLayout::plain_rect`]).
#[pyfunction]
#[pyo3(signature = (pitch_mm, rows, cols, outer_radius_mm, inner_radius_mm, dots=true))]
fn plain_rect_target_json(
    pitch_mm: f32,
    rows: usize,
    cols: usize,
    outer_radius_mm: f32,
    inner_radius_mm: f32,
    dots: bool,
) -> PyResult<String> {
    let target = ringgrid::TargetLayout::plain_rect(
        pitch_mm,
        rows,
        cols,
        outer_radius_mm,
        inner_radius_mm,
        origin_dots(dots),
    )
    .map_err(py_value_error)?;
    Ok(target.to_json_string())
}

/// Origin-dot centers of a target spec, in board millimeters.
///
/// Positions are derived from the lattice rather than stored in the spec, so
/// callers drawing an overlay must ask the library instead of reading the JSON.
#[pyfunction]
fn target_fiducial_dots_mm(spec_json: &str) -> PyResult<Vec<[f32; 2]>> {
    Ok(target_from_spec_json(spec_json)?.fiducial_dots_mm().to_vec())
}

/// Python exposes the two-state `OriginDots` selector as a plain `bool`.
fn origin_dots(dots: bool) -> ringgrid::OriginDots {
    if dots {
        ringgrid::OriginDots::Auto
    } else {
        ringgrid::OriginDots::None
    }
}

/// Validate a target spec (`ringgrid.target.v6`, or legacy `v5` / `v4`, auto-migrated)
/// and return its canonical `v5` JSON. Validation failures surface as
/// `ValueError` carrying the Rust error message.
#[pyfunction]
fn canonical_target_spec_json(spec_json: &str) -> PyResult<String> {
    let target = target_from_spec_json(spec_json)?;
    Ok(target.to_json_string())
}

#[pyfunction]
#[pyo3(signature = (spec_json, path, margin_mm=0.0, include_scale_bar=true))]
fn write_target_svg(
    spec_json: &str,
    path: &str,
    margin_mm: f32,
    include_scale_bar: bool,
) -> PyResult<()> {
    let target = target_from_spec_json(spec_json)?;
    target
        .write_target_svg(
            std::path::Path::new(path),
            &ringgrid::SvgTargetOptions {
                margin_mm,
                include_scale_bar,
            },
        )
        .map_err(py_target_generation_error)
}

#[pyfunction]
#[pyo3(signature = (spec_json, path, dpi=300.0, margin_mm=0.0, include_scale_bar=true))]
fn write_target_png(
    spec_json: &str,
    path: &str,
    dpi: f32,
    margin_mm: f32,
    include_scale_bar: bool,
) -> PyResult<()> {
    let target = target_from_spec_json(spec_json)?;
    target
        .write_target_png(
            std::path::Path::new(path),
            &ringgrid::PngTargetOptions {
                dpi,
                margin_mm,
                include_scale_bar,
            },
        )
        .map_err(py_target_generation_error)
}

#[pyfunction]
fn write_target_dxf(spec_json: &str, path: &str) -> PyResult<()> {
    let target = target_from_spec_json(spec_json)?;
    target
        .write_target_dxf(std::path::Path::new(path))
        .map_err(py_target_generation_error)
}

#[pyfunction]
#[pyo3(signature = (image_path, config_json=None))]
fn proposal_json_path(image_path: &str, config_json: Option<&str>) -> PyResult<String> {
    let gray = load_gray_image(image_path)?;
    let config = parse_proposal_config(config_json)?;
    let proposals = ringgrid::find_ellipse_centers(&gray, &config);
    proposal_json(&proposals)
}

#[pyfunction]
#[pyo3(signature = (image, config_json=None))]
fn proposal_json_array(
    image: PyReadonlyArrayDyn<'_, u8>,
    config_json: Option<&str>,
) -> PyResult<String> {
    let gray = gray_image_from_array(image)?;
    let config = parse_proposal_config(config_json)?;
    let proposals = ringgrid::find_ellipse_centers(&gray, &config);
    proposal_json(&proposals)
}

#[pyfunction]
#[pyo3(signature = (image_path, config_json=None))]
fn proposal_result_payload_path<'py>(
    py: Python<'py>,
    image_path: &str,
    config_json: Option<&str>,
) -> PyResult<(String, Py<PyArray2<f32>>)> {
    let gray = load_gray_image(image_path)?;
    let config = parse_proposal_config(config_json)?;
    proposal_result_payload(
        py,
        ringgrid::find_ellipse_centers_with_heatmap(&gray, &config),
    )
}

#[pyfunction]
#[pyo3(signature = (image, config_json=None))]
fn proposal_result_payload_array<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    config_json: Option<&str>,
) -> PyResult<(String, Py<PyArray2<f32>>)> {
    let gray = gray_image_from_array(image)?;
    let config = parse_proposal_config(config_json)?;
    proposal_result_payload(
        py,
        ringgrid::find_ellipse_centers_with_heatmap(&gray, &config),
    )
}

#[pyfunction]
fn proposal_with_scale_json_path(
    image_path: &str,
    board_spec_json: &str,
    marker_scale_json: &str,
) -> PyResult<String> {
    let gray = load_gray_image(image_path)?;
    let target = target_from_spec_json(board_spec_json)?;
    let marker_scale = parse_marker_scale(marker_scale_json)?;
    let proposals = ringgrid::propose_with_marker_scale(&gray, &target, marker_scale);
    proposal_json(&proposals)
}

#[pyfunction]
fn proposal_with_scale_json_array(
    image: PyReadonlyArrayDyn<'_, u8>,
    board_spec_json: &str,
    marker_scale_json: &str,
) -> PyResult<String> {
    let gray = gray_image_from_array(image)?;
    let target = target_from_spec_json(board_spec_json)?;
    let marker_scale = parse_marker_scale(marker_scale_json)?;
    let proposals = ringgrid::propose_with_marker_scale(&gray, &target, marker_scale);
    proposal_json(&proposals)
}

#[pyfunction]
fn proposal_result_with_scale_payload_path<'py>(
    py: Python<'py>,
    image_path: &str,
    board_spec_json: &str,
    marker_scale_json: &str,
) -> PyResult<(String, Py<PyArray2<f32>>)> {
    let gray = load_gray_image(image_path)?;
    let target = target_from_spec_json(board_spec_json)?;
    let marker_scale = parse_marker_scale(marker_scale_json)?;
    proposal_result_payload(
        py,
        ringgrid::propose_with_heatmap_and_marker_scale(&gray, &target, marker_scale),
    )
}

#[pyfunction]
fn proposal_result_with_scale_payload_array<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    board_spec_json: &str,
    marker_scale_json: &str,
) -> PyResult<(String, Py<PyArray2<f32>>)> {
    let gray = gray_image_from_array(image)?;
    let target = target_from_spec_json(board_spec_json)?;
    let marker_scale = parse_marker_scale(marker_scale_json)?;
    proposal_result_payload(
        py,
        ringgrid::propose_with_heatmap_and_marker_scale(&gray, &target, marker_scale),
    )
}

#[pyfunction]
#[pyo3(signature = (board_spec_json, overlay_json=None))]
fn resolve_config_json(board_spec_json: &str, overlay_json: Option<&str>) -> PyResult<String> {
    let target = target_from_spec_json(board_spec_json)?;
    let mut config = ringgrid::DetectConfig::from_target(target);
    apply_overlay_json(&mut config, overlay_json.unwrap_or("{}"))?;
    serde_json::to_string(&config).map_err(py_value_error)
}

#[pyfunction]
fn update_config_json(
    board_spec_json: &str,
    base_config_json: &str,
    overlay_json: &str,
) -> PyResult<String> {
    let target = target_from_spec_json(board_spec_json)?;
    let base_value =
        serde_json::from_str::<serde_json::Value>(base_config_json).map_err(py_value_error)?;
    let mut config = config_from_json(target, &base_value)?;
    apply_overlay_json(&mut config, overlay_json)?;
    serde_json::to_string(&config).map_err(py_value_error)
}

#[pyfunction]
fn scale_tiers_four_tier_wide_json() -> PyResult<String> {
    let tiers = ringgrid::ScaleTiers::four_tier_wide();
    serde_json::to_string(&scale_tiers_to_wire(&tiers)).map_err(py_value_error)
}

#[pyfunction]
fn scale_tiers_two_tier_standard_json() -> PyResult<String> {
    let tiers = ringgrid::ScaleTiers::two_tier_standard();
    serde_json::to_string(&scale_tiers_to_wire(&tiers)).map_err(py_value_error)
}

#[pymodule]
fn _ringgrid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DetectConfigCore>()?;
    m.add_class::<DetectorCore>()?;

    m.add_function(wrap_pyfunction!(package_version, m)?)?;
    m.add_function(wrap_pyfunction!(target_preset_json, m)?)?;
    m.add_function(wrap_pyfunction!(coded_hex_target_json, m)?)?;
    m.add_function(wrap_pyfunction!(coded_rect_target_json, m)?)?;
    m.add_function(wrap_pyfunction!(plain_hex_target_json, m)?)?;
    m.add_function(wrap_pyfunction!(plain_rect_target_json, m)?)?;
    m.add_function(wrap_pyfunction!(target_fiducial_dots_mm, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_target_spec_json, m)?)?;
    m.add_function(wrap_pyfunction!(write_target_svg, m)?)?;
    m.add_function(wrap_pyfunction!(write_target_png, m)?)?;
    m.add_function(wrap_pyfunction!(write_target_dxf, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_json_path, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_json_array, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_result_payload_path, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_result_payload_array, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_with_scale_json_path, m)?)?;
    m.add_function(wrap_pyfunction!(proposal_with_scale_json_array, m)?)?;
    m.add_function(wrap_pyfunction!(
        proposal_result_with_scale_payload_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        proposal_result_with_scale_payload_array,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(resolve_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(update_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(scale_tiers_four_tier_wide_json, m)?)?;
    m.add_function(wrap_pyfunction!(scale_tiers_two_tier_standard_json, m)?)?;
    Ok(())
}
