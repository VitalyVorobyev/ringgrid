use image::GrayImage;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

const TARGET_SCHEMA_V3: &str = "ringgrid.target.v3";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoardSpecV3 {
    schema: String,
    name: String,
    pitch_mm: f32,
    rows: usize,
    long_row_cols: usize,
    marker_outer_radius_mm: f32,
    marker_inner_radius_mm: f32,
}

#[derive(Debug, Clone, Serialize)]
struct BoardSnapshot {
    schema: String,
    name: String,
    pitch_mm: f32,
    rows: usize,
    long_row_cols: usize,
    marker_outer_radius_mm: f32,
    marker_inner_radius_mm: f32,
    markers: Vec<ringgrid::BoardMarker>,
}

/// JSON-loadable detection configuration overlay.
#[derive(Deserialize, Default)]
#[serde(default)]
struct DetectConfigFile {
    marker_scale: Option<ringgrid::MarkerScalePrior>,
    circle_refinement: Option<ringgrid::CircleRefinementMethod>,
    inner_fit: Option<ringgrid::InnerFitConfig>,
    outer_fit: Option<ringgrid::OuterFitConfig>,
    completion: Option<ringgrid::CompletionParams>,
    projective_center: Option<ringgrid::ProjectiveCenterParams>,
    seed_proposals: Option<ringgrid::SeedProposalParams>,
    proposal: Option<ringgrid::ProposalConfig>,
    edge_sample: Option<ringgrid::EdgeSampleConfig>,
    decode: Option<ringgrid::DecodeConfig>,
    marker_spec: Option<ringgrid::MarkerSpec>,
    outer_estimation: Option<ringgrid::OuterEstimationConfig>,
    ransac_homography: Option<ringgrid::RansacHomographyConfig>,
    self_undistort: Option<ringgrid::SelfUndistortConfig>,
    id_correction: Option<ringgrid::IdCorrectionConfig>,
    inner_as_outer_recovery: Option<ringgrid::InnerAsOuterRecoveryConfig>,
    dedup_radius: Option<f64>,
    max_aspect_ratio: Option<f64>,
    use_global_filter: Option<bool>,
}

/// Serializable snapshot of all tunable detection parameters.
#[derive(Serialize, Deserialize, Clone)]
struct DetectConfigDump {
    marker_scale: ringgrid::MarkerScalePrior,
    circle_refinement: ringgrid::CircleRefinementMethod,
    inner_fit: ringgrid::InnerFitConfig,
    outer_fit: ringgrid::OuterFitConfig,
    completion: ringgrid::CompletionParams,
    projective_center: ringgrid::ProjectiveCenterParams,
    seed_proposals: ringgrid::SeedProposalParams,
    proposal: ringgrid::ProposalConfig,
    edge_sample: ringgrid::EdgeSampleConfig,
    decode: ringgrid::DecodeConfig,
    marker_spec: ringgrid::MarkerSpec,
    outer_estimation: ringgrid::OuterEstimationConfig,
    ransac_homography: ringgrid::RansacHomographyConfig,
    self_undistort: ringgrid::SelfUndistortConfig,
    id_correction: ringgrid::IdCorrectionConfig,
    inner_as_outer_recovery: ringgrid::InnerAsOuterRecoveryConfig,
    dedup_radius: f64,
    max_aspect_ratio: f64,
    use_global_filter: bool,
}

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

fn py_value_error<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn board_to_spec(board: &ringgrid::BoardLayout) -> BoardSpecV3 {
    BoardSpecV3 {
        schema: TARGET_SCHEMA_V3.to_string(),
        name: board.name.clone(),
        pitch_mm: board.pitch_mm,
        rows: board.rows,
        long_row_cols: board.long_row_cols,
        marker_outer_radius_mm: board.marker_outer_radius_mm,
        marker_inner_radius_mm: board.marker_inner_radius_mm,
    }
}

fn board_snapshot(board: &ringgrid::BoardLayout) -> BoardSnapshot {
    BoardSnapshot {
        schema: TARGET_SCHEMA_V3.to_string(),
        name: board.name.clone(),
        pitch_mm: board.pitch_mm,
        rows: board.rows,
        long_row_cols: board.long_row_cols,
        marker_outer_radius_mm: board.marker_outer_radius_mm,
        marker_inner_radius_mm: board.marker_inner_radius_mm,
        markers: board.markers().to_vec(),
    }
}

fn board_from_spec_json(spec_json: &str) -> PyResult<ringgrid::BoardLayout> {
    ringgrid::BoardLayout::from_json_str(spec_json).map_err(py_value_error)
}

fn config_to_dump(config: &ringgrid::DetectConfig) -> DetectConfigDump {
    DetectConfigDump {
        marker_scale: config.marker_scale,
        circle_refinement: config.circle_refinement,
        inner_fit: config.inner_fit.clone(),
        outer_fit: config.outer_fit.clone(),
        completion: config.completion.clone(),
        projective_center: config.projective_center.clone(),
        seed_proposals: config.seed_proposals.clone(),
        proposal: config.proposal.clone(),
        edge_sample: config.edge_sample.clone(),
        decode: config.decode.clone(),
        marker_spec: config.marker_spec.clone(),
        outer_estimation: config.outer_estimation.clone(),
        ransac_homography: config.ransac_homography.clone(),
        self_undistort: config.self_undistort.clone(),
        id_correction: config.id_correction.clone(),
        inner_as_outer_recovery: config.inner_as_outer_recovery.clone(),
        dedup_radius: config.dedup_radius,
        max_aspect_ratio: config.max_aspect_ratio,
        use_global_filter: config.use_global_filter,
    }
}

fn dump_to_config(board: ringgrid::BoardLayout, dump: &DetectConfigDump) -> ringgrid::DetectConfig {
    let mut config =
        ringgrid::DetectConfig::from_target_and_scale_prior(board, dump.marker_scale);
    config.circle_refinement = dump.circle_refinement;
    config.inner_fit = dump.inner_fit.clone();
    config.outer_fit = dump.outer_fit.clone();
    config.completion = dump.completion.clone();
    config.projective_center = dump.projective_center.clone();
    config.seed_proposals = dump.seed_proposals.clone();
    config.proposal = dump.proposal.clone();
    config.edge_sample = dump.edge_sample.clone();
    config.decode = dump.decode.clone();
    config.marker_spec = dump.marker_spec.clone();
    config.outer_estimation = dump.outer_estimation.clone();
    config.ransac_homography = dump.ransac_homography.clone();
    config.self_undistort = dump.self_undistort.clone();
    config.id_correction = dump.id_correction.clone();
    config.inner_as_outer_recovery = dump.inner_as_outer_recovery.clone();
    config.dedup_radius = dump.dedup_radius;
    config.max_aspect_ratio = dump.max_aspect_ratio;
    config.use_global_filter = dump.use_global_filter;
    config
}

fn apply_config_overlay(config: &mut ringgrid::DetectConfig, overlay: DetectConfigFile) {
    if let Some(marker_scale) = overlay.marker_scale {
        config.set_marker_scale_prior(marker_scale);
    }
    if let Some(v) = overlay.circle_refinement {
        config.circle_refinement = v;
    }
    if let Some(v) = overlay.inner_fit {
        config.inner_fit = v;
    }
    if let Some(v) = overlay.outer_fit {
        config.outer_fit = v;
    }
    if let Some(v) = overlay.completion {
        config.completion = v;
    }
    if let Some(v) = overlay.projective_center {
        config.projective_center = v;
    }
    if let Some(v) = overlay.seed_proposals {
        config.seed_proposals = v;
    }
    if let Some(v) = overlay.proposal {
        config.proposal = v;
    }
    if let Some(v) = overlay.edge_sample {
        config.edge_sample = v;
    }
    if let Some(v) = overlay.decode {
        config.decode = v;
    }
    if let Some(v) = overlay.marker_spec {
        config.marker_spec = v;
    }
    if let Some(v) = overlay.outer_estimation {
        config.outer_estimation = v;
    }
    if let Some(v) = overlay.ransac_homography {
        config.ransac_homography = v;
    }
    if let Some(v) = overlay.self_undistort {
        config.self_undistort = v;
    }
    if let Some(v) = overlay.id_correction {
        config.id_correction = v;
    }
    if let Some(v) = overlay.inner_as_outer_recovery {
        config.inner_as_outer_recovery = v;
    }
    if let Some(v) = overlay.dedup_radius {
        config.dedup_radius = v;
    }
    if let Some(v) = overlay.max_aspect_ratio {
        config.max_aspect_ratio = v;
    }
    if let Some(v) = overlay.use_global_filter {
        config.use_global_filter = v;
    }
}

fn parse_overlay(overlay_json: Option<&str>) -> PyResult<DetectConfigFile> {
    let overlay_json = overlay_json.unwrap_or("{}");
    serde_json::from_str::<DetectConfigFile>(overlay_json).map_err(py_value_error)
}

fn parse_dump(config_json: &str) -> PyResult<DetectConfigDump> {
    serde_json::from_str::<DetectConfigDump>(config_json).map_err(py_value_error)
}

fn detector_from_json(board_spec_json: &str, config_json: &str) -> PyResult<ringgrid::Detector> {
    let board = board_from_spec_json(board_spec_json)?;
    let dump = parse_dump(config_json)?;
    let config = dump_to_config(board, &dump);
    Ok(ringgrid::Detector::with_config(config))
}

fn detect_with_core_mapper(
    detector: &ringgrid::Detector,
    gray: &GrayImage,
    mapper_spec: &MapperSpec,
) -> ringgrid::DetectionResult {
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
        let result = detector.detect(&gray);
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_array(&self, image: PyReadonlyArrayDyn<'_, u8>) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = gray_image_from_array(image)?;
        let result = detector.detect(&gray);
        serde_json::to_string(&result).map_err(py_value_error)
    }

    fn detect_with_mapper_path(&self, image_path: &str, mapper_json: &str) -> PyResult<String> {
        let detector = detector_from_json(&self.board_spec_json, &self.config_json)?;
        let gray = load_gray_image(image_path)?;
        let mapper = serde_json::from_str::<MapperSpec>(mapper_json).map_err(py_value_error)?;
        let result = detect_with_core_mapper(&detector, &gray, &mapper);
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
        let result = detect_with_core_mapper(&detector, &gray, &mapper);
        serde_json::to_string(&result).map_err(py_value_error)
    }
}

#[pyfunction]
fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn default_board_spec_json() -> PyResult<String> {
    let board = ringgrid::BoardLayout::default();
    serde_json::to_string(&board_to_spec(&board)).map_err(py_value_error)
}

#[pyfunction]
fn load_board_spec_json(path: &str) -> PyResult<String> {
    let board = ringgrid::BoardLayout::from_json_file(std::path::Path::new(path))
        .map_err(py_value_error)?;
    serde_json::to_string(&board_to_spec(&board)).map_err(py_value_error)
}

#[pyfunction]
fn board_snapshot_json(spec_json: &str) -> PyResult<String> {
    let board = board_from_spec_json(spec_json)?;
    serde_json::to_string(&board_snapshot(&board)).map_err(py_value_error)
}

#[pyfunction]
#[pyo3(signature = (board_spec_json, overlay_json=None))]
fn resolve_config_json(board_spec_json: &str, overlay_json: Option<&str>) -> PyResult<String> {
    let board = board_from_spec_json(board_spec_json)?;
    let overlay = parse_overlay(overlay_json)?;

    let marker_scale = overlay.marker_scale.unwrap_or_default();
    let mut config = ringgrid::DetectConfig::from_target_and_scale_prior(board, marker_scale);
    apply_config_overlay(&mut config, overlay);

    serde_json::to_string(&config_to_dump(&config)).map_err(py_value_error)
}

#[pyfunction]
fn update_config_json(
    board_spec_json: &str,
    base_config_json: &str,
    overlay_json: &str,
) -> PyResult<String> {
    let board = board_from_spec_json(board_spec_json)?;
    let base = parse_dump(base_config_json)?;
    let overlay = parse_overlay(Some(overlay_json))?;

    let mut config = dump_to_config(board, &base);
    apply_config_overlay(&mut config, overlay);

    serde_json::to_string(&config_to_dump(&config)).map_err(py_value_error)
}

#[pymodule]
fn _ringgrid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DetectorCore>()?;

    m.add_function(wrap_pyfunction!(package_version, m)?)?;
    m.add_function(wrap_pyfunction!(default_board_spec_json, m)?)?;
    m.add_function(wrap_pyfunction!(load_board_spec_json, m)?)?;
    m.add_function(wrap_pyfunction!(board_snapshot_json, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(update_config_json, m)?)?;
    Ok(())
}
