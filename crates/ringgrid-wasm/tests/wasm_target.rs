//! Browser-side checks for the target-rendering exports.
//!
//! The native tests in `src/lib.rs` cover the library call underneath; these
//! exercise the part only a browser can: that the bundle actually crosses the
//! wasm boundary as a JS object with a real `Uint8Array` of PNG bytes.

#![cfg(target_arch = "wasm32")]

use ringgrid_wasm::{
    coded_hex_target_json, default_board_json, rect_24x24_target_json, render_target_bundle_json,
    target_page_size_mm,
};
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn field(bundle: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(bundle, &JsValue::from_str(key)).expect("bundle field")
}

#[wasm_bindgen_test]
fn bundle_crosses_the_boundary_with_every_format() {
    let bundle = render_target_bundle_json(&default_board_json(), "{}").expect("render bundle");

    let svg = field(&bundle, "svg_text").as_string().expect("svg string");
    assert!(
        svg.starts_with("<?xml"),
        "svg: {}",
        &svg[..40.min(svg.len())]
    );

    let json = field(&bundle, "json_text")
        .as_string()
        .expect("json string");
    assert!(json.contains("ringgrid.target.v6"));

    let dxf = field(&bundle, "dxf_text").as_string().expect("dxf string");
    assert!(dxf.ends_with("\nEOF\n"));

    let png = js_sys::Uint8Array::new(&field(&bundle, "png_bytes"));
    assert!(png.length() > 1000, "png length {}", png.length());
    let head = png.subarray(0, 4).to_vec();
    assert_eq!(head, b"\x89PNG", "png magic bytes");
}

#[wasm_bindgen_test]
fn a4_page_options_are_honoured() {
    let target = coded_hex_target_json(8.0, 5, 5, 4.8, 3.2, 1.152).expect("target");
    let options = r#"{"page":{"size":{"kind":"a4"},"orientation":"landscape"}}"#;

    let size: Vec<f32> =
        serde_json::from_str(&target_page_size_mm(&target, options).expect("page size"))
            .expect("size json");
    assert!((size[0] - 297.0).abs() < 1e-3, "width {}", size[0]);
    assert!((size[1] - 210.0).abs() < 1e-3, "height {}", size[1]);

    let bundle = render_target_bundle_json(&target, options).expect("render");
    let svg = field(&bundle, "svg_text").as_string().expect("svg");
    assert!(svg.contains("width=\"297mm\""), "landscape A4 width");
    assert!(svg.contains("height=\"210mm\""), "landscape A4 height");
}

#[wasm_bindgen_test]
fn a_target_too_large_for_the_page_throws() {
    // The 24x24 rect board prints 343 mm square — wider than A4's 210 mm.
    let err = render_target_bundle_json(
        &rect_24x24_target_json(),
        r#"{"page":{"size":{"kind":"a4"}}}"#,
    )
    .expect_err("a 343 mm board does not fit A4");
    let message = err.as_string().expect("string error");
    assert!(message.contains("printable area"), "{message}");
}
