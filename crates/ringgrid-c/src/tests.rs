//! Ownership / marshalling unit tests and parity tests against the Rust
//! reference. Because the crate is only `cdylib`/`staticlib` (no `rlib`), tests
//! live inline rather than in `tests/`.

use super::*;
use crate::detect::{
    ringgrid_detect, ringgrid_detect_adaptive, ringgrid_detect_multiscale,
    ringgrid_detect_with_diagnostics,
};
use crate::detector::{
    ringgrid_detector_config_json, ringgrid_detector_free, ringgrid_detector_new,
    ringgrid_detector_update_config,
};
use crate::introspect::{
    ringgrid_abi_version, ringgrid_default_target_json, ringgrid_rect_24x24_target_json,
    ringgrid_scale_tiers_two_tier_standard_json, ringgrid_version,
};
use crate::propose::{ringgrid_heatmap_data, ringgrid_propose_with_heatmap};
use crate::status::{RinggridStatus, ringgrid_status_str};

use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};

use image::{GrayImage, ImageReader};

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

fn load_fixture_target_json() -> String {
    std::fs::read_to_string(repo_root().join("testdata/board_ringgrid.json"))
        .unwrap_or_else(|_| ringgrid::TargetLayout::default_hex().to_json_string())
}

/// Assert OK, copy the owned string, and free it.
fn take_string(status: RinggridStatus, ptr: *mut c_char) -> String {
    assert_eq!(status, RinggridStatus::Ok);
    assert!(!ptr.is_null());
    let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned();
    unsafe { ringgrid_string_free(ptr) };
    s
}

fn new_handle(target_json: &str) -> *mut RinggridDetector {
    let c = CString::new(target_json).unwrap();
    let mut handle: *mut RinggridDetector = std::ptr::null_mut();
    let st = unsafe { ringgrid_detector_new(c.as_ptr(), &mut handle) };
    assert_eq!(st, RinggridStatus::Ok);
    assert!(!handle.is_null());
    handle
}

fn detect_json(handle: *const RinggridDetector, img: &GrayImage) -> String {
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe {
        ringgrid_detect(
            handle,
            img.as_raw().as_ptr(),
            img.width(),
            img.height(),
            &mut out,
        )
    };
    take_string(st, out)
}

// ── Introspection ──────────────────────────────────────────────────

#[test]
fn version_matches_crate() {
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe { ringgrid_version(&mut out) };
    assert_eq!(take_string(st, out), env!("CARGO_PKG_VERSION"));
}

#[test]
fn abi_version_is_one() {
    assert_eq!(ringgrid_abi_version(), RINGGRID_ABI_VERSION);
    assert_eq!(RINGGRID_ABI_VERSION, 1);
}

#[test]
fn status_str_is_static_and_readable() {
    for st in [
        RinggridStatus::Ok,
        RinggridStatus::ErrNullArg,
        RinggridStatus::ErrPanic,
    ] {
        let ptr = ringgrid_status_str(st);
        assert!(!ptr.is_null());
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert!(!s.is_empty());
    }
}

#[test]
fn presets_are_valid_v5_json() {
    let mut a: *mut c_char = std::ptr::null_mut();
    let mut b: *mut c_char = std::ptr::null_mut();
    let hex = take_string(unsafe { ringgrid_default_target_json(&mut a) }, a);
    let rect = take_string(unsafe { ringgrid_rect_24x24_target_json(&mut b) }, b);
    assert!(ringgrid::TargetLayout::from_json_str(&hex).is_ok());
    assert!(ringgrid::TargetLayout::from_json_str(&rect).is_ok());
}

#[test]
fn scale_tier_preset_parses() {
    let mut out: *mut c_char = std::ptr::null_mut();
    let json = take_string(
        unsafe { ringgrid_scale_tiers_two_tier_standard_json(&mut out) },
        out,
    );
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["tiers"].as_array().unwrap().len(), 2);
}

// ── Error / ownership behavior ─────────────────────────────────────

#[test]
fn null_target_returns_null_arg() {
    let mut handle: *mut RinggridDetector = std::ptr::null_mut();
    let st = unsafe { ringgrid_detector_new(std::ptr::null(), &mut handle) };
    assert_eq!(st, RinggridStatus::ErrNullArg);
    assert!(handle.is_null());
}

#[test]
fn invalid_target_json_returns_bad_json() {
    let c = CString::new("not json").unwrap();
    let mut handle: *mut RinggridDetector = std::ptr::null_mut();
    let st = unsafe { ringgrid_detector_new(c.as_ptr(), &mut handle) };
    assert_eq!(st, RinggridStatus::ErrBadJson);
}

#[test]
fn null_pixels_returns_null_arg() {
    let handle = new_handle(&load_fixture_target_json());
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe { ringgrid_detect(handle, std::ptr::null(), 8, 8, &mut out) };
    assert_eq!(st, RinggridStatus::ErrNullArg);
    unsafe { ringgrid_detector_free(handle) };
}

#[test]
fn zero_dimensions_return_bad_image() {
    let handle = new_handle(&load_fixture_target_json());
    let pixels = [0u8; 4];
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe { ringgrid_detect(handle, pixels.as_ptr(), 0, 4, &mut out) };
    assert_eq!(st, RinggridStatus::ErrBadImage);
    unsafe { ringgrid_detector_free(handle) };
}

#[test]
fn free_null_handle_is_noop() {
    unsafe { ringgrid_detector_free(std::ptr::null_mut()) };
    unsafe { ringgrid_string_free(std::ptr::null_mut()) };
}

#[test]
fn heatmap_before_propose_errors() {
    let handle = new_handle(&load_fixture_target_json());
    let mut ptr: *const f32 = std::ptr::null();
    let mut len: usize = 0;
    let st = unsafe { ringgrid_heatmap_data(handle, &mut ptr, &mut len) };
    assert_eq!(st, RinggridStatus::ErrNoHeatmap);
    unsafe { ringgrid_detector_free(handle) };
}

#[test]
fn detect_blank_image_returns_json() {
    let handle = new_handle(&ringgrid::TargetLayout::default_hex().to_json_string());
    let pixels = vec![0u8; 64 * 48];
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe { ringgrid_detect(handle, pixels.as_ptr(), 64, 48, &mut out) };
    let json = take_string(st, out);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value.get("detected_markers").is_some());
    unsafe { ringgrid_detector_free(handle) };
}

// ── Config ─────────────────────────────────────────────────────────

#[test]
fn config_json_and_update_overlay() {
    let handle = new_handle(&load_fixture_target_json());

    let mut out: *mut c_char = std::ptr::null_mut();
    let cfg = take_string(
        unsafe { ringgrid_detector_config_json(handle, &mut out) },
        out,
    );
    let value: serde_json::Value = serde_json::from_str(&cfg).unwrap();
    assert_eq!(value["advanced"]["completion"]["enable"], true);

    let overlay = CString::new(r#"{"advanced": {"completion": {"enable": false}}}"#).unwrap();
    let st = unsafe { ringgrid_detector_update_config(handle, overlay.as_ptr()) };
    assert_eq!(st, RinggridStatus::Ok);

    let mut out2: *mut c_char = std::ptr::null_mut();
    let cfg2 = take_string(
        unsafe { ringgrid_detector_config_json(handle, &mut out2) },
        out2,
    );
    let value2: serde_json::Value = serde_json::from_str(&cfg2).unwrap();
    assert_eq!(value2["advanced"]["completion"]["enable"], false);

    unsafe { ringgrid_detector_free(handle) };
}

#[test]
fn update_config_rejects_non_object() {
    let handle = new_handle(&load_fixture_target_json());
    let overlay = CString::new("42").unwrap();
    let st = unsafe { ringgrid_detector_update_config(handle, overlay.as_ptr()) };
    assert_eq!(st, RinggridStatus::ErrBadJson);
    unsafe { ringgrid_detector_free(handle) };
}

// ── Parity against the Rust reference ──────────────────────────────

fn assert_marker_parity(c: &ringgrid::DetectionResult, native: &ringgrid::DetectionResult) {
    assert_eq!(
        c.detected_markers.len(),
        native.detected_markers.len(),
        "marker count mismatch"
    );
    assert_eq!(c.image_size, native.image_size);
    for (cm, nm) in c
        .detected_markers
        .iter()
        .zip(native.detected_markers.iter())
    {
        assert_eq!(cm.id, nm.id, "marker ID mismatch");
        assert!(
            (cm.center[0] - nm.center[0]).abs() < 1e-10
                && (cm.center[1] - nm.center[1]).abs() < 1e-10,
            "center mismatch for id {:?}: {:?} vs {:?}",
            nm.id,
            cm.center,
            nm.center
        );
        assert!(
            (cm.confidence - nm.confidence).abs() < 1e-6,
            "confidence mismatch for id {:?}",
            nm.id
        );
    }
}

#[test]
fn detect_parity() {
    let img = load_fixture_image();
    let target_json = load_fixture_target_json();
    let target = ringgrid::TargetLayout::from_json_str(&target_json).unwrap();
    let native = ringgrid::Detector::new(target).detect(&img).unwrap();

    let handle = new_handle(&target_json);
    let json = detect_json(handle, &img);
    let c_result: ringgrid::DetectionResult = serde_json::from_str(&json).unwrap();
    unsafe { ringgrid_detector_free(handle) };

    assert!(
        !native.detected_markers.is_empty(),
        "fixture should detect markers"
    );
    assert_marker_parity(&c_result, &native);
}

#[test]
fn detect_adaptive_parity() {
    let img = load_fixture_image();
    let target_json = load_fixture_target_json();
    let target = ringgrid::TargetLayout::from_json_str(&target_json).unwrap();
    let native = ringgrid::Detector::new(target)
        .detect_adaptive(&img)
        .unwrap();

    let handle = new_handle(&target_json);
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe {
        ringgrid_detect_adaptive(
            handle,
            img.as_raw().as_ptr(),
            img.width(),
            img.height(),
            &mut out,
        )
    };
    let json = take_string(st, out);
    let c_result: ringgrid::DetectionResult = serde_json::from_str(&json).unwrap();
    unsafe { ringgrid_detector_free(handle) };

    assert_marker_parity(&c_result, &native);
}

#[test]
fn detect_multiscale_parity() {
    let img = load_fixture_image();
    let target_json = load_fixture_target_json();
    let target = ringgrid::TargetLayout::from_json_str(&target_json).unwrap();
    let tiers = ringgrid::ScaleTiers::two_tier_standard();
    let native = ringgrid::Detector::new(target)
        .detect_multiscale(&img, &tiers)
        .unwrap();

    let mut tiers_out: *mut c_char = std::ptr::null_mut();
    let tiers_json = take_string(
        unsafe { ringgrid_scale_tiers_two_tier_standard_json(&mut tiers_out) },
        tiers_out,
    );
    let tiers_c = CString::new(tiers_json).unwrap();

    let handle = new_handle(&target_json);
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe {
        ringgrid_detect_multiscale(
            handle,
            img.as_raw().as_ptr(),
            img.width(),
            img.height(),
            tiers_c.as_ptr(),
            &mut out,
        )
    };
    let json = take_string(st, out);
    let c_result: ringgrid::DetectionResult = serde_json::from_str(&json).unwrap();
    unsafe { ringgrid_detector_free(handle) };

    assert_marker_parity(&c_result, &native);
}

#[test]
fn propose_with_heatmap_parity_and_borrow() {
    let img = load_fixture_image();
    let target_json = load_fixture_target_json();
    let target = ringgrid::TargetLayout::from_json_str(&target_json).unwrap();
    let native = ringgrid::Detector::new(target).propose_with_heatmap(&img);

    let handle = new_handle(&target_json);
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe {
        ringgrid_propose_with_heatmap(
            handle,
            img.as_raw().as_ptr(),
            img.width(),
            img.height(),
            &mut out,
        )
    };
    let json = take_string(st, out);
    let payload: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        payload["proposals"].as_array().unwrap().len(),
        native.proposals.len(),
        "proposal count mismatch"
    );

    // Borrowed heatmap matches the native buffer.
    let mut ptr: *const f32 = std::ptr::null();
    let mut len: usize = 0;
    let st = unsafe { ringgrid_heatmap_data(handle, &mut ptr, &mut len) };
    assert_eq!(st, RinggridStatus::Ok);
    assert_eq!(len, native.heatmap.len());
    let borrowed = unsafe { std::slice::from_raw_parts(ptr, len) };
    for (b, n) in borrowed.iter().zip(native.heatmap.iter()) {
        assert!((b - n).abs() < 1e-6);
    }
    unsafe { ringgrid_detector_free(handle) };
}

#[test]
fn detect_with_diagnostics_is_aligned() {
    let img = load_fixture_image();
    let target_json = load_fixture_target_json();

    let handle = new_handle(&target_json);
    let mut out: *mut c_char = std::ptr::null_mut();
    let st = unsafe {
        ringgrid_detect_with_diagnostics(
            handle,
            img.as_raw().as_ptr(),
            img.width(),
            img.height(),
            &mut out,
        )
    };
    let json = take_string(st, out);
    let combined: serde_json::Value = serde_json::from_str(&json).unwrap();
    let result: ringgrid::DetectionResult =
        serde_json::from_value(combined["result"].clone()).unwrap();
    let diagnostics: ringgrid::diagnostics::DetectionDiagnostics =
        serde_json::from_value(combined["diagnostics"].clone()).unwrap();
    assert_eq!(diagnostics.markers.len(), result.detected_markers.len());
    unsafe { ringgrid_detector_free(handle) };
}
