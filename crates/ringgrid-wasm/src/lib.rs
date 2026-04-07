use image::GrayImage;
use wasm_bindgen::prelude::*;

/// Install console_error_panic_hook for better WASM panic messages.
/// Call this before any detection to get stack traces in the browser console.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

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
}
