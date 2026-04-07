use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// Embed the test image at compile time so the WASM test doesn't need file I/O
const FIXTURE_PNG: &[u8] = include_bytes!("../../../testdata/target_3_split_00.png");

fn decode_fixture_gray() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(FIXTURE_PNG)
        .expect("decode embedded fixture PNG")
        .to_luma8();
    let w = img.width();
    let h = img.height();
    let pixels = img.into_raw();
    (pixels, w, h)
}

fn decode_fixture_rgba() -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(FIXTURE_PNG)
        .expect("decode embedded fixture PNG")
        .to_rgba8();
    let w = img.width();
    let h = img.height();
    let pixels = img.into_raw();
    (pixels, w, h)
}

#[wasm_bindgen_test]
fn wasm_version() {
    let v = ringgrid_wasm::version();
    assert!(!v.is_empty());
}

#[wasm_bindgen_test]
fn wasm_default_board_json() {
    let json = ringgrid_wasm::default_board_json();
    assert!(json.contains("ringgrid"));
}

#[wasm_bindgen_test]
fn wasm_detect_grayscale() {
    ringgrid_wasm::init_panic_hook();
    let (pixels, w, h) = decode_fixture_gray();
    let board_json = ringgrid_wasm::default_board_json();
    let detector = ringgrid_wasm::RinggridDetector::new(&board_json).unwrap();
    let result_json = detector.detect(&pixels, w, h).unwrap();
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    let markers = result["detected_markers"].as_array().unwrap();
    assert!(
        !markers.is_empty(),
        "WASM detect should find markers on fixture image"
    );
}

#[wasm_bindgen_test]
fn wasm_detect_rgba() {
    ringgrid_wasm::init_panic_hook();
    let (pixels, w, h) = decode_fixture_rgba();
    let board_json = ringgrid_wasm::default_board_json();
    let detector = ringgrid_wasm::RinggridDetector::new(&board_json).unwrap();
    let result_json = detector.detect_rgba(&pixels, w, h).unwrap();
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    let markers = result["detected_markers"].as_array().unwrap();
    assert!(
        !markers.is_empty(),
        "WASM detect_rgba should find markers on fixture image"
    );
}

#[wasm_bindgen_test]
fn wasm_detect_adaptive() {
    ringgrid_wasm::init_panic_hook();
    let (pixels, w, h) = decode_fixture_gray();
    let board_json = ringgrid_wasm::default_board_json();
    let detector = ringgrid_wasm::RinggridDetector::new(&board_json).unwrap();
    let result_json = detector.detect_adaptive(&pixels, w, h).unwrap();
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    let markers = result["detected_markers"].as_array().unwrap();
    assert!(
        !markers.is_empty(),
        "WASM detect_adaptive should find markers on fixture image"
    );
}

#[wasm_bindgen_test]
fn wasm_propose_with_heatmap() {
    ringgrid_wasm::init_panic_hook();
    let (pixels, w, h) = decode_fixture_gray();
    let board_json = ringgrid_wasm::default_board_json();
    let mut detector = ringgrid_wasm::RinggridDetector::new(&board_json).unwrap();
    let result_json = detector.propose_with_heatmap(&pixels, w, h).unwrap();
    let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();
    let proposals = result["proposals"].as_array().unwrap();
    assert!(
        !proposals.is_empty(),
        "WASM propose_with_heatmap should find proposals on fixture image"
    );

    let heatmap = detector.heatmap_f32().unwrap();
    assert!(heatmap.length() > 0);
    assert_eq!(detector.heatmap_width(), w);
    assert_eq!(detector.heatmap_height(), h);
}
