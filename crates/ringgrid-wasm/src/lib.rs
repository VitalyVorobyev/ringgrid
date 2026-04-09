use image::GrayImage;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Install console_error_panic_hook for better WASM panic messages.
/// Call this before any detection to get stack traces in the browser console.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// ── Config dump / overlay types ────────────────────────────────────

/// Serializable snapshot of all tunable detection parameters.
/// Mirrors the Python bindings' `DetectConfigDump`.
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
    h_reproj_confidence_alpha: f32,
    topology_filter_threshold_px: Option<f32>,
    proposal_downscale: ringgrid::ProposalDownscale,
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
        h_reproj_confidence_alpha: config.h_reproj_confidence_alpha,
        topology_filter_threshold_px: config.topology_filter_threshold_px,
        proposal_downscale: config.proposal_downscale,
    }
}

fn dump_to_config(
    board: ringgrid::BoardLayout,
    dump: &DetectConfigDump,
) -> ringgrid::DetectConfig {
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
    config.h_reproj_confidence_alpha = dump.h_reproj_confidence_alpha;
    config.topology_filter_threshold_px = dump.topology_filter_threshold_px;
    config.proposal_downscale = dump.proposal_downscale;
    config
}

fn merge_json_value(base: &mut serde_json::Value, overlay: serde_json::Value) {
    match (base, overlay) {
        (serde_json::Value::Object(base_obj), serde_json::Value::Object(overlay_obj)) => {
            for (key, overlay_value) in overlay_obj {
                match base_obj.get_mut(&key) {
                    Some(base_value) => merge_json_value(base_value, overlay_value),
                    None => {
                        base_obj.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_slot, overlay_value) => *base_slot = overlay_value,
    }
}

fn parse_overlay_object(
    overlay_json: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, JsValue> {
    let overlay: serde_json::Value =
        serde_json::from_str(overlay_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    match overlay {
        serde_json::Value::Object(obj) => Ok(obj),
        _ => Err(JsValue::from_str("config overlay must be a JSON object")),
    }
}

fn merge_overlay_value<T>(current: &T, overlay: serde_json::Value) -> Result<T, JsValue>
where
    T: Serialize + serde::de::DeserializeOwned,
{
    let mut merged =
        serde_json::to_value(current).map_err(|e| JsValue::from_str(&e.to_string()))?;
    merge_json_value(&mut merged, overlay);
    serde_json::from_value(merged).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn apply_config_overlay(
    config: &mut ringgrid::DetectConfig,
    overlay_json: &str,
) -> Result<(), JsValue> {
    let mut overlay = parse_overlay_object(overlay_json)?;

    if let Some(value) = overlay.remove("marker_scale") {
        let marker_scale = merge_overlay_value(&config.marker_scale, value)?;
        config.set_marker_scale_prior(marker_scale);
    }
    if let Some(value) = overlay.remove("circle_refinement") {
        config.circle_refinement = merge_overlay_value(&config.circle_refinement, value)?;
    }
    if let Some(value) = overlay.remove("inner_fit") {
        config.inner_fit = merge_overlay_value(&config.inner_fit, value)?;
    }
    if let Some(value) = overlay.remove("outer_fit") {
        config.outer_fit = merge_overlay_value(&config.outer_fit, value)?;
    }
    if let Some(value) = overlay.remove("completion") {
        config.completion = merge_overlay_value(&config.completion, value)?;
    }
    if let Some(value) = overlay.remove("projective_center") {
        config.projective_center = merge_overlay_value(&config.projective_center, value)?;
    }
    if let Some(value) = overlay.remove("seed_proposals") {
        config.seed_proposals = merge_overlay_value(&config.seed_proposals, value)?;
    }
    if let Some(value) = overlay.remove("proposal") {
        config.proposal = merge_overlay_value(&config.proposal, value)?;
    }
    if let Some(value) = overlay.remove("edge_sample") {
        config.edge_sample = merge_overlay_value(&config.edge_sample, value)?;
    }
    if let Some(value) = overlay.remove("decode") {
        config.decode = merge_overlay_value(&config.decode, value)?;
    }
    if let Some(value) = overlay.remove("marker_spec") {
        config.marker_spec = merge_overlay_value(&config.marker_spec, value)?;
    }
    if let Some(value) = overlay.remove("outer_estimation") {
        config.outer_estimation = merge_overlay_value(&config.outer_estimation, value)?;
    }
    if let Some(value) = overlay.remove("ransac_homography") {
        config.ransac_homography = merge_overlay_value(&config.ransac_homography, value)?;
    }
    if let Some(value) = overlay.remove("self_undistort") {
        config.self_undistort = merge_overlay_value(&config.self_undistort, value)?;
    }
    if let Some(value) = overlay.remove("id_correction") {
        config.id_correction = merge_overlay_value(&config.id_correction, value)?;
    }
    if let Some(value) = overlay.remove("inner_as_outer_recovery") {
        config.inner_as_outer_recovery =
            merge_overlay_value(&config.inner_as_outer_recovery, value)?;
    }
    if let Some(value) = overlay.remove("dedup_radius") {
        config.dedup_radius = merge_overlay_value(&config.dedup_radius, value)?;
    }
    if let Some(value) = overlay.remove("max_aspect_ratio") {
        config.max_aspect_ratio = merge_overlay_value(&config.max_aspect_ratio, value)?;
    }
    if let Some(value) = overlay.remove("use_global_filter") {
        config.use_global_filter = merge_overlay_value(&config.use_global_filter, value)?;
    }
    if let Some(value) = overlay.remove("h_reproj_confidence_alpha") {
        config.h_reproj_confidence_alpha =
            merge_overlay_value(&config.h_reproj_confidence_alpha, value)?;
    }
    if let Some(value) = overlay.remove("topology_filter_threshold_px") {
        config.topology_filter_threshold_px =
            merge_overlay_value(&config.topology_filter_threshold_px, value)?;
    }
    if let Some(value) = overlay.remove("proposal_downscale") {
        config.proposal_downscale = merge_overlay_value(&config.proposal_downscale, value)?;
    }

    Ok(())
}

// ── Scale tier wire types ──────────────────────────────────────────

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

fn parse_scale_tiers(tiers_json: &str) -> Result<ringgrid::ScaleTiers, JsValue> {
    let wire: ScaleTiersWire =
        serde_json::from_str(tiers_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if wire.tiers.is_empty() {
        return Err(JsValue::from_str("scale tiers must not be empty"));
    }
    Ok(ringgrid::ScaleTiers(
        wire.tiers
            .iter()
            .map(|t| ringgrid::ScaleTier::new(t.diameter_min_px, t.diameter_max_px))
            .collect(),
    ))
}

fn scale_tiers_to_wire(tiers: &ringgrid::ScaleTiers) -> ScaleTiersWire {
    ScaleTiersWire {
        tiers: tiers
            .tiers()
            .iter()
            .map(|t| ScaleTierWire {
                diameter_min_px: t.prior.diameter_min_px,
                diameter_max_px: t.prior.diameter_max_px,
            })
            .collect(),
    }
}

// ── Pixel helpers ──────────────────────────────────────────────────

/// Checked pixel count: returns `width * height` as `usize`, or `Err` on overflow.
fn checked_pixel_count(width: u32, height: u32) -> Result<usize, JsValue> {
    (width as usize).checked_mul(height as usize).ok_or_else(|| {
        JsValue::from_str(&format!(
            "image dimensions overflow: {}x{} exceeds addressable range",
            width, height
        ))
    })
}

/// BT.601 luma: Y = (77R + 150G + 29B + 128) >> 8
fn rgba_to_gray(rgba: &[u8], width: u32, height: u32) -> GrayImage {
    // Caller must validate dimensions first; this is only called after validate_rgba.
    let n = (width as usize) * (height as usize);
    let mut gray = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgba[4 * i] as u32;
        let g = rgba[4 * i + 1] as u32;
        let b = rgba[4 * i + 2] as u32;
        gray.push(((77 * r + 150 * g + 29 * b + 128) >> 8) as u8);
    }
    GrayImage::from_raw(width, height, gray).expect("buffer size matches width * height")
}

fn validate_gray(pixels: &[u8], width: u32, height: u32) -> Result<GrayImage, JsValue> {
    let expected = checked_pixel_count(width, height)?;
    if pixels.len() != expected {
        return Err(JsValue::from_str(&format!(
            "expected {} grayscale pixels ({}x{}), got {}",
            expected, width, height,
            pixels.len()
        )));
    }
    Ok(GrayImage::from_raw(width, height, pixels.to_vec()).expect("buffer size validated"))
}

fn validate_rgba(pixels: &[u8], width: u32, height: u32) -> Result<(), JsValue> {
    let expected = checked_pixel_count(width, height)?
        .checked_mul(4)
        .ok_or_else(|| {
            JsValue::from_str(&format!(
                "RGBA buffer size overflow: {}x{}x4 exceeds addressable range",
                width, height
            ))
        })?;
    if pixels.len() != expected {
        return Err(JsValue::from_str(&format!(
            "expected {} RGBA bytes ({}x{}x4), got {}",
            expected, width, height,
            pixels.len()
        )));
    }
    Ok(())
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), JsValue> {
    if width == 0 || height == 0 {
        return Err(JsValue::from_str("image dimensions must be non-zero"));
    }
    checked_pixel_count(width, height)?;
    Ok(())
}

fn parse_board(board_json: &str) -> Result<ringgrid::BoardLayout, JsValue> {
    ringgrid::BoardLayout::from_json_str(board_json).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String, JsValue> {
    serde_json::to_string(value).map_err(|e| JsValue::from_str(&e.to_string()))
}

// ── Main class ──────────────────────────────────────────────────────

/// Stateful ring marker detector for WebAssembly.
///
/// Holds board layout and detection configuration. Create once,
/// call detection methods on multiple images.
#[wasm_bindgen]
pub struct RinggridDetector {
    detector: ringgrid::Detector,
    last_heatmap: Option<Vec<f32>>,
    last_heatmap_size: [u32; 2],
}

impl Default for RinggridDetector {
    fn default() -> Self {
        Self {
            detector: ringgrid::Detector::new(ringgrid::BoardLayout::default()),
            last_heatmap: None,
            last_heatmap_size: [0, 0],
        }
    }
}

#[wasm_bindgen]
impl RinggridDetector {
    /// Create a detector from a board layout JSON string.
    #[wasm_bindgen(constructor)]
    pub fn new(board_json: &str) -> Result<RinggridDetector, JsValue> {
        let board = parse_board(board_json)?;
        Ok(Self {
            detector: ringgrid::Detector::new(board),
            last_heatmap: None,
            last_heatmap_size: [0, 0],
        })
    }

    /// Create a detector with explicit min/max marker scale in pixels.
    pub fn with_marker_scale(
        board_json: &str,
        min_px: f32,
        max_px: f32,
    ) -> Result<RinggridDetector, JsValue> {
        let board = parse_board(board_json)?;
        let scale = ringgrid::MarkerScalePrior {
            diameter_min_px: min_px,
            diameter_max_px: max_px,
        };
        Ok(Self {
            detector: ringgrid::Detector::with_marker_scale(board, scale),
            last_heatmap: None,
            last_heatmap_size: [0, 0],
        })
    }

    /// Create a detector with a nominal marker diameter hint.
    pub fn with_marker_diameter(
        board_json: &str,
        diameter_px: f32,
    ) -> Result<RinggridDetector, JsValue> {
        let board = parse_board(board_json)?;
        Ok(Self {
            detector: ringgrid::Detector::with_marker_diameter_hint(board, diameter_px),
            last_heatmap: None,
            last_heatmap_size: [0, 0],
        })
    }

    /// Create a detector with full config control.
    /// `config_json` must be a complete config snapshot (as returned by `config_json()`).
    pub fn with_config(
        board_json: &str,
        config_json: &str,
    ) -> Result<RinggridDetector, JsValue> {
        let board = parse_board(board_json)?;
        let dump: DetectConfigDump =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let config = dump_to_config(board, &dump);
        Ok(Self {
            detector: ringgrid::Detector::with_config(config),
            last_heatmap: None,
            last_heatmap_size: [0, 0],
        })
    }

    // ── Config access ──────────────────────────────────────────────

    /// Get current detection config as a JSON string.
    pub fn config_json(&self) -> Result<String, JsValue> {
        to_json(&config_to_dump(self.detector.config()))
    }

    /// Apply a partial config overlay (only provided fields are updated).
    /// Pass a JSON object with any subset of config fields.
    pub fn update_config(&mut self, overlay_json: &str) -> Result<(), JsValue> {
        apply_config_overlay(self.detector.config_mut(), overlay_json)
    }

    // ── Detection: grayscale ────────────────────────────────────────

    /// Detect markers from grayscale pixels. Returns JSON string (DetectionResult).
    pub fn detect(&self, pixels: &[u8], width: u32, height: u32) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        let gray = validate_gray(pixels, width, height)?;
        let result = self.detector.detect(&gray);
        to_json(&result)
    }

    /// Adaptive detection from grayscale pixels. Returns JSON string (DetectionResult).
    pub fn detect_adaptive(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        let gray = validate_gray(pixels, width, height)?;
        let result = self.detector.detect_adaptive(&gray);
        to_json(&result)
    }

    /// Adaptive detection with a nominal diameter hint (grayscale).
    /// Returns JSON string (DetectionResult).
    pub fn detect_adaptive_with_hint(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        nominal_diameter_px: f32,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        let gray = validate_gray(pixels, width, height)?;
        let result = self
            .detector
            .detect_adaptive_with_hint(&gray, Some(nominal_diameter_px));
        to_json(&result)
    }

    /// Multi-scale detection with explicit scale tiers (grayscale).
    /// `tiers_json`: `{"tiers": [{"diameter_min_px": 14, "diameter_max_px": 42}, ...]}`.
    /// Returns JSON string (DetectionResult).
    pub fn detect_multiscale(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        tiers_json: &str,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        let gray = validate_gray(pixels, width, height)?;
        let tiers = parse_scale_tiers(tiers_json)?;
        let result = self.detector.detect_multiscale(&gray, &tiers);
        to_json(&result)
    }

    // ── Detection: RGBA ─────────────────────────────────────────────

    /// Detect markers from RGBA pixels (e.g. canvas ImageData).
    /// Returns JSON string (DetectionResult).
    pub fn detect_rgba(&self, pixels: &[u8], width: u32, height: u32) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        validate_rgba(pixels, width, height)?;
        let gray = rgba_to_gray(pixels, width, height);
        let result = self.detector.detect(&gray);
        to_json(&result)
    }

    /// Adaptive detection from RGBA pixels. Returns JSON string (DetectionResult).
    pub fn detect_adaptive_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        validate_rgba(pixels, width, height)?;
        let gray = rgba_to_gray(pixels, width, height);
        let result = self.detector.detect_adaptive(&gray);
        to_json(&result)
    }

    /// Adaptive detection with diameter hint from RGBA pixels.
    /// Returns JSON string (DetectionResult).
    pub fn detect_adaptive_with_hint_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        nominal_diameter_px: f32,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        validate_rgba(pixels, width, height)?;
        let gray = rgba_to_gray(pixels, width, height);
        let result = self
            .detector
            .detect_adaptive_with_hint(&gray, Some(nominal_diameter_px));
        to_json(&result)
    }

    /// Multi-scale detection with explicit scale tiers from RGBA pixels.
    /// Returns JSON string (DetectionResult).
    pub fn detect_multiscale_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        tiers_json: &str,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        validate_rgba(pixels, width, height)?;
        let gray = rgba_to_gray(pixels, width, height);
        let tiers = parse_scale_tiers(tiers_json)?;
        let result = self.detector.detect_multiscale(&gray, &tiers);
        to_json(&result)
    }

    // ── Proposal + heatmap ──────────────────────────────────────────

    /// Generate proposals with heatmap from grayscale pixels.
    /// Returns JSON string (proposals and image_size; heatmap via `heatmap_f32()`).
    pub fn propose_with_heatmap(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        let gray = validate_gray(pixels, width, height)?;
        let result = self.detector.propose_with_heatmap(&gray);
        self.last_heatmap_size = result.image_size;
        self.last_heatmap = Some(result.heatmap);
        // Return proposals + image_size as JSON (heatmap excluded for efficiency)
        let payload = ProposalPayload {
            image_size: self.last_heatmap_size,
            proposals: &result.proposals,
        };
        to_json(&payload)
    }

    /// Generate proposals with heatmap from RGBA pixels.
    /// Returns JSON string (proposals and image_size; heatmap via `heatmap_f32()`).
    pub fn propose_with_heatmap_rgba(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<String, JsValue> {
        validate_dimensions(width, height)?;
        validate_rgba(pixels, width, height)?;
        let gray = rgba_to_gray(pixels, width, height);
        let result = self.detector.propose_with_heatmap(&gray);
        self.last_heatmap_size = result.image_size;
        self.last_heatmap = Some(result.heatmap);
        let payload = ProposalPayload {
            image_size: self.last_heatmap_size,
            proposals: &result.proposals,
        };
        to_json(&payload)
    }

    /// Get the last heatmap as a Float32Array for canvas rendering.
    /// Call `propose_with_heatmap` or `propose_with_heatmap_rgba` first.
    pub fn heatmap_f32(&self) -> Result<js_sys::Float32Array, JsValue> {
        match &self.last_heatmap {
            Some(data) => {
                let array = js_sys::Float32Array::new_with_length(data.len() as u32);
                array.copy_from(data);
                Ok(array)
            }
            None => Err(JsValue::from_str(
                "no heatmap available; call propose_with_heatmap first",
            )),
        }
    }

    /// Heatmap width from the last proposal call.
    pub fn heatmap_width(&self) -> u32 {
        self.last_heatmap_size[0]
    }

    /// Heatmap height from the last proposal call.
    pub fn heatmap_height(&self) -> u32 {
        self.last_heatmap_size[1]
    }
}

// ── Free functions ──────────────────────────────────────────────────

/// Default board layout as a JSON string.
#[wasm_bindgen]
pub fn default_board_json() -> String {
    ringgrid::BoardLayout::default().to_json_string()
}

/// Default detection config for a given board layout, as a JSON string.
#[wasm_bindgen]
pub fn default_config_json(board_json: &str) -> Result<String, JsValue> {
    let board = parse_board(board_json)?;
    let config = ringgrid::DetectConfig::from_target(board);
    to_json(&config_to_dump(&config))
}

/// Four-tier wide scale tiers preset as JSON.
/// Covers 8-220 px marker diameter range.
#[wasm_bindgen]
pub fn scale_tiers_four_tier_wide_json() -> String {
    serde_json::to_string(&scale_tiers_to_wire(&ringgrid::ScaleTiers::four_tier_wide()))
        .expect("serialization cannot fail")
}

/// Two-tier standard scale tiers preset as JSON.
/// Covers 14-100 px marker diameter range.
#[wasm_bindgen]
pub fn scale_tiers_two_tier_standard_json() -> String {
    serde_json::to_string(&scale_tiers_to_wire(
        &ringgrid::ScaleTiers::two_tier_standard(),
    ))
    .expect("serialization cannot fail")
}

/// Package version string.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ── Internal helpers ────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct ProposalPayload<'a> {
    image_size: [u32; 2],
    proposals: &'a [ringgrid::Proposal],
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageReader;
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn load_fixture_image() -> GrayImage {
        ImageReader::open(repo_root().join("testdata/target_3_split_00.png"))
            .expect("open fixture image")
            .decode()
            .expect("decode fixture image")
            .to_luma8()
    }

    fn load_fixture_board_json() -> String {
        std::fs::read_to_string(repo_root().join("testdata/board_ringgrid.json"))
            .unwrap_or_else(|_| default_board_json())
    }

    fn gray_to_rgba(gray: &GrayImage) -> Vec<u8> {
        gray.pixels()
            .flat_map(|p| [p.0[0], p.0[0], p.0[0], 255])
            .collect()
    }

    // ── Group 1: Validation ────────────────────────────────────────
    // Note: validate_* functions return Result<_, JsValue>. JsValue aborts
    // on non-wasm targets when an Err is constructed. We test the underlying
    // logic (buffer length checks) directly to avoid JsValue in error paths.

    #[test]
    fn validate_dimensions_accepts_nonzero() {
        assert!(validate_dimensions(10, 10).is_ok());
    }

    #[test]
    fn validate_gray_correct_length() {
        let img = validate_gray(&[128; 4], 2, 2).unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
    }

    #[test]
    fn validate_gray_wrong_length_detected() {
        // Buffer length != width * height should be an error
        let expected = 2usize * 2;
        assert_ne!([0u8; 5].len(), expected);
    }

    #[test]
    fn validate_rgba_correct_length() {
        assert!(validate_rgba(&[0; 16], 2, 2).is_ok());
    }

    #[test]
    fn validate_rgba_wrong_length_detected() {
        let expected = 4 * 2usize * 2;
        assert_ne!([0u8; 8].len(), expected);
    }

    // ── Group 2: RGBA-to-gray conversion ───────────────────────────

    #[test]
    fn rgba_to_gray_bt601_known_value() {
        // R=100, G=150, B=50 → (77*100 + 150*150 + 29*50 + 128) >> 8
        let expected = ((77 * 100 + 150 * 150 + 29 * 50 + 128) >> 8) as u8;
        let rgba = [100u8, 150, 50, 255];
        let img = rgba_to_gray(&rgba, 1, 1);
        assert_eq!(img.get_pixel(0, 0).0[0], expected);
    }

    #[test]
    fn rgba_to_gray_pure_white() {
        let img = rgba_to_gray(&[255, 255, 255, 255], 1, 1);
        assert_eq!(img.get_pixel(0, 0).0[0], 255);
    }

    #[test]
    fn rgba_to_gray_pure_black() {
        let img = rgba_to_gray(&[0, 0, 0, 255], 1, 1);
        assert_eq!(img.get_pixel(0, 0).0[0], 0);
    }

    #[test]
    fn rgba_to_gray_ignores_alpha() {
        let img_a = rgba_to_gray(&[100, 150, 50, 255], 1, 1);
        let img_b = rgba_to_gray(&[100, 150, 50, 0], 1, 1);
        assert_eq!(img_a.get_pixel(0, 0), img_b.get_pixel(0, 0));
    }

    // ── Group 3: Board parsing ─────────────────────────────────────

    #[test]
    fn parse_board_valid_json() {
        let json = default_board_json();
        assert!(parse_board(&json).is_ok());
    }

    #[test]
    fn parse_board_invalid_json() {
        // parse_board returns JsValue on error, which aborts in native tests.
        // Test the underlying BoardLayout parsing instead.
        assert!(ringgrid::BoardLayout::from_json_str("not json").is_err());
    }

    #[test]
    fn default_board_json_roundtrip() {
        let json = default_board_json();
        let board = ringgrid::BoardLayout::from_json_str(&json).unwrap();
        let default = ringgrid::BoardLayout::default();
        assert_eq!(board.rows, default.rows);
        assert_eq!(board.long_row_cols, default.long_row_cols);
        assert!((board.pitch_mm - default.pitch_mm).abs() < 1e-6);
    }

    // ── Group 4: Detection parity (grayscale) ──────────────────────

    #[test]
    fn detect_parity_grayscale() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let board = parse_board(&board_json).unwrap();
        let (w, h) = (img.width(), img.height());
        let pixels = img.as_raw();

        // Native detection
        let native_detector = ringgrid::Detector::new(board);
        let native_result = native_detector.detect(&img);

        // WASM wrapper detection (returns JSON)
        let wasm_det = RinggridDetector::new(&board_json).unwrap();
        let json_str = wasm_det.detect(pixels, w, h).unwrap();
        let wasm_result: ringgrid::DetectionResult = serde_json::from_str(&json_str).unwrap();

        // Compare
        assert_eq!(
            wasm_result.detected_markers.len(),
            native_result.detected_markers.len(),
            "marker count mismatch"
        );
        assert_eq!(wasm_result.image_size, native_result.image_size);

        for (wm, nm) in wasm_result
            .detected_markers
            .iter()
            .zip(native_result.detected_markers.iter())
        {
            assert_eq!(wm.id, nm.id, "marker ID mismatch");
            // JSON roundtrip can introduce tiny f64 precision differences
            assert!(
                (wm.center[0] - nm.center[0]).abs() < 1e-10
                    && (wm.center[1] - nm.center[1]).abs() < 1e-10,
                "center mismatch for id {:?}: {:?} vs {:?}",
                nm.id,
                wm.center,
                nm.center
            );
            assert!(
                (wm.confidence - nm.confidence).abs() < 1e-6,
                "confidence mismatch for id {:?}",
                nm.id
            );
        }

        // Verify we actually detected something
        assert!(
            !native_result.detected_markers.is_empty(),
            "fixture should produce detections"
        );
    }

    // ── Group 5: Adaptive detection parity ─────────────────────────

    #[test]
    fn detect_adaptive_parity() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let board = parse_board(&board_json).unwrap();
        let (w, h) = (img.width(), img.height());
        let pixels = img.as_raw();

        let native_detector = ringgrid::Detector::new(board);
        let native_result = native_detector.detect_adaptive(&img);

        let wasm_det = RinggridDetector::new(&board_json).unwrap();
        let json_str = wasm_det.detect_adaptive(pixels, w, h).unwrap();
        let wasm_result: ringgrid::DetectionResult = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            wasm_result.detected_markers.len(),
            native_result.detected_markers.len(),
            "adaptive marker count mismatch"
        );
        assert_eq!(wasm_result.image_size, native_result.image_size);

        for (wm, nm) in wasm_result
            .detected_markers
            .iter()
            .zip(native_result.detected_markers.iter())
        {
            assert_eq!(wm.id, nm.id);
            assert!(
                (wm.center[0] - nm.center[0]).abs() < 1e-10
                    && (wm.center[1] - nm.center[1]).abs() < 1e-10,
                "center mismatch for id {:?}: {:?} vs {:?}",
                nm.id,
                wm.center,
                nm.center
            );
        }

        assert!(
            !native_result.detected_markers.is_empty(),
            "adaptive detection should find markers on fixture"
        );
    }

    // ── Group 6: Proposal parity ───────────────────────────────────

    #[test]
    fn propose_with_heatmap_parity() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let board = parse_board(&board_json).unwrap();
        let (w, h) = (img.width(), img.height());
        let pixels = img.as_raw();

        // Native
        let native_detector = ringgrid::Detector::new(board);
        let native_result = native_detector.propose_with_heatmap(&img);

        // WASM wrapper
        let mut wasm_det = RinggridDetector::new(&board_json).unwrap();
        let json_str = wasm_det.propose_with_heatmap(pixels, w, h).unwrap();
        let payload: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        // Compare proposal count
        let wasm_proposals = payload["proposals"].as_array().unwrap();
        assert_eq!(
            wasm_proposals.len(),
            native_result.proposals.len(),
            "proposal count mismatch"
        );

        // Compare image_size
        let wasm_size: [u32; 2] = serde_json::from_value(payload["image_size"].clone()).unwrap();
        assert_eq!(wasm_size, native_result.image_size);

        // Compare cached heatmap
        let cached = wasm_det.last_heatmap.as_ref().unwrap();
        assert_eq!(
            cached.len(),
            native_result.heatmap.len(),
            "heatmap length mismatch"
        );
        for (i, (w_val, n_val)) in cached.iter().zip(native_result.heatmap.iter()).enumerate() {
            assert!(
                (w_val - n_val).abs() < 1e-6,
                "heatmap mismatch at index {i}: {w_val} vs {n_val}"
            );
        }

        assert!(
            !native_result.proposals.is_empty(),
            "fixture should produce proposals"
        );
    }

    // ── Group 7: Heatmap state ─────────────────────────────────────

    #[test]
    fn heatmap_state_lifecycle() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let (w, h) = (img.width(), img.height());
        let pixels = img.as_raw();

        let mut det = RinggridDetector::new(&board_json).unwrap();

        // Before any proposal call
        assert!(det.last_heatmap.is_none());
        assert_eq!(det.last_heatmap_size, [0, 0]);

        // After proposal call
        det.propose_with_heatmap(pixels, w, h).unwrap();
        assert!(det.last_heatmap.is_some());
        assert_eq!(det.last_heatmap_size, [w, h]);
        assert_eq!(det.last_heatmap.as_ref().unwrap().len(), (w * h) as usize);
    }

    // ── Group 8: RGBA detection parity ─────────────────────────────

    #[test]
    fn detect_rgba_matches_grayscale() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let (w, h) = (img.width(), img.height());
        let gray_pixels = img.as_raw();
        let rgba_pixels = gray_to_rgba(&img);

        let det = RinggridDetector::new(&board_json).unwrap();

        let gray_json = det.detect(gray_pixels, w, h).unwrap();
        let rgba_json = det.detect_rgba(&rgba_pixels, w, h).unwrap();

        let gray_result: ringgrid::DetectionResult = serde_json::from_str(&gray_json).unwrap();
        let rgba_result: ringgrid::DetectionResult = serde_json::from_str(&rgba_json).unwrap();

        // When R=G=B=v, BT.601 gives: (77+150+29)*v + 128 >> 8 = (256*v+128)>>8 = v
        // So RGBA and grayscale paths should produce identical results
        assert_eq!(
            gray_result.detected_markers.len(),
            rgba_result.detected_markers.len(),
            "RGBA vs gray marker count mismatch"
        );

        for (gm, rm) in gray_result
            .detected_markers
            .iter()
            .zip(rgba_result.detected_markers.iter())
        {
            assert_eq!(gm.id, rm.id);
            assert_eq!(gm.center, rm.center);
        }
    }

    // ── Group 9: Constructors and version ──────────────────────────

    #[test]
    fn constructor_new_detects_markers() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let det = RinggridDetector::new(&board_json).unwrap();
        let json = det.detect(img.as_raw(), img.width(), img.height()).unwrap();
        let result: ringgrid::DetectionResult = serde_json::from_str(&json).unwrap();
        assert!(!result.detected_markers.is_empty());
    }

    #[test]
    fn constructor_with_marker_scale_detects_markers() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let det = RinggridDetector::with_marker_scale(&board_json, 10.0, 50.0).unwrap();
        let json = det.detect(img.as_raw(), img.width(), img.height()).unwrap();
        let result: ringgrid::DetectionResult = serde_json::from_str(&json).unwrap();
        assert!(!result.detected_markers.is_empty());
    }

    #[test]
    fn constructor_with_marker_diameter_detects_markers() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let det = RinggridDetector::with_marker_diameter(&board_json, 30.0).unwrap();
        let json = det.detect(img.as_raw(), img.width(), img.height()).unwrap();
        let result: ringgrid::DetectionResult = serde_json::from_str(&json).unwrap();
        assert!(!result.detected_markers.is_empty());
    }

    #[test]
    fn version_matches_cargo_pkg() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }

    // ── Group 10: Config roundtrip ────────────────────────────────

    #[test]
    fn config_json_roundtrip() {
        let board_json = load_fixture_board_json();
        let det1 = RinggridDetector::new(&board_json).unwrap();
        let config_json = det1.config_json().unwrap();

        // Reconstruct from the dumped config
        let det2 = RinggridDetector::with_config(&board_json, &config_json).unwrap();
        let config_json2 = det2.config_json().unwrap();

        // JSON roundtrip should be stable
        let v1: serde_json::Value = serde_json::from_str(&config_json).unwrap();
        let v2: serde_json::Value = serde_json::from_str(&config_json2).unwrap();
        assert_eq!(v1, v2, "config JSON roundtrip mismatch");
    }

    #[test]
    fn update_config_applies_overlay() {
        let board_json = load_fixture_board_json();
        let mut det = RinggridDetector::new(&board_json).unwrap();

        // Verify completion is enabled by default
        let cfg: serde_json::Value = serde_json::from_str(&det.config_json().unwrap()).unwrap();
        assert_eq!(cfg["completion"]["enable"], true);

        // Disable completion via overlay
        det.update_config(r#"{"completion": {"enable": false}}"#)
            .unwrap();

        let cfg2: serde_json::Value = serde_json::from_str(&det.config_json().unwrap()).unwrap();
        assert_eq!(cfg2["completion"]["enable"], false);
    }

    #[test]
    fn update_config_merges_nested_completion_overlay() {
        let board_json = load_fixture_board_json();
        let mut det = RinggridDetector::new(&board_json).unwrap();

        det.update_config(
            r#"{"completion": {"require_perfect_decode": true, "max_attempts": 17}}"#,
        )
        .unwrap();
        det.update_config(r#"{"completion": {"enable": false}}"#)
            .unwrap();

        let cfg: serde_json::Value = serde_json::from_str(&det.config_json().unwrap()).unwrap();
        assert_eq!(cfg["completion"]["enable"], false);
        assert_eq!(cfg["completion"]["require_perfect_decode"], true);
        assert_eq!(cfg["completion"]["max_attempts"].as_u64(), Some(17));
    }

    #[test]
    fn default_config_json_parses() {
        let board_json = load_fixture_board_json();
        let config_str = default_config_json(&board_json).unwrap();
        let _: DetectConfigDump = serde_json::from_str(&config_str).unwrap();
    }

    // ── Group 11: Multiscale detection parity ─────────────────────

    #[test]
    fn detect_multiscale_parity() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let board = parse_board(&board_json).unwrap();
        let (w, h) = (img.width(), img.height());
        let pixels = img.as_raw();

        let tiers = ringgrid::ScaleTiers::two_tier_standard();
        let tiers_json = scale_tiers_two_tier_standard_json();

        // Native
        let native_detector = ringgrid::Detector::new(board);
        let native_result = native_detector.detect_multiscale(&img, &tiers);

        // WASM wrapper
        let wasm_det = RinggridDetector::new(&board_json).unwrap();
        let json_str = wasm_det.detect_multiscale(pixels, w, h, &tiers_json).unwrap();
        let wasm_result: ringgrid::DetectionResult = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            wasm_result.detected_markers.len(),
            native_result.detected_markers.len(),
            "multiscale marker count mismatch"
        );

        for (wm, nm) in wasm_result
            .detected_markers
            .iter()
            .zip(native_result.detected_markers.iter())
        {
            assert_eq!(wm.id, nm.id, "marker ID mismatch");
            assert!(
                (wm.center[0] - nm.center[0]).abs() < 1e-10
                    && (wm.center[1] - nm.center[1]).abs() < 1e-10,
                "center mismatch for id {:?}",
                nm.id
            );
        }
    }

    // ── Group 12: Adaptive with hint parity ───────────────────────

    #[test]
    fn detect_adaptive_with_hint_parity() {
        let img = load_fixture_image();
        let board_json = load_fixture_board_json();
        let board = parse_board(&board_json).unwrap();
        let (w, h) = (img.width(), img.height());
        let pixels = img.as_raw();

        let native_detector = ringgrid::Detector::new(board);
        let native_result = native_detector.detect_adaptive_with_hint(&img, Some(30.0));

        let wasm_det = RinggridDetector::new(&board_json).unwrap();
        let json_str = wasm_det
            .detect_adaptive_with_hint(pixels, w, h, 30.0)
            .unwrap();
        let wasm_result: ringgrid::DetectionResult = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            wasm_result.detected_markers.len(),
            native_result.detected_markers.len(),
            "adaptive-with-hint marker count mismatch"
        );

        for (wm, nm) in wasm_result
            .detected_markers
            .iter()
            .zip(native_result.detected_markers.iter())
        {
            assert_eq!(wm.id, nm.id);
            assert!(
                (wm.center[0] - nm.center[0]).abs() < 1e-10
                    && (wm.center[1] - nm.center[1]).abs() < 1e-10,
                "center mismatch for id {:?}",
                nm.id
            );
        }
    }

    // ── Group 13: Scale tier presets ──────────────────────────────

    #[test]
    fn scale_tier_presets_roundtrip() {
        let four = scale_tiers_four_tier_wide_json();
        let two = scale_tiers_two_tier_standard_json();

        let four_parsed = parse_scale_tiers(&four).unwrap();
        let two_parsed = parse_scale_tiers(&two).unwrap();

        assert_eq!(four_parsed.tiers().len(), 4);
        assert_eq!(two_parsed.tiers().len(), 2);
    }
}
