use image::GrayImage;
use wasm_bindgen::prelude::*;

/// BT.601 luma: Y = (77R + 150G + 29B + 128) >> 8
fn rgba_to_gray(rgba: &[u8], width: u32, height: u32) -> GrayImage {
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
    let expected = (width as usize) * (height as usize);
    if pixels.len() != expected {
        return Err(JsValue::from_str(&format!(
            "expected {} grayscale pixels ({}x{}), got {}",
            expected,
            width,
            height,
            pixels.len()
        )));
    }
    Ok(GrayImage::from_raw(width, height, pixels.to_vec()).expect("buffer size validated"))
}

fn validate_rgba(pixels: &[u8], width: u32, height: u32) -> Result<(), JsValue> {
    let expected = 4 * (width as usize) * (height as usize);
    if pixels.len() != expected {
        return Err(JsValue::from_str(&format!(
            "expected {} RGBA bytes ({}x{}x4), got {}",
            expected,
            width,
            height,
            pixels.len()
        )));
    }
    Ok(())
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), JsValue> {
    if width == 0 || height == 0 {
        return Err(JsValue::from_str("image dimensions must be non-zero"));
    }
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
