//! Detection entry points. Each has a grayscale and an `_rgba` variant; the
//! `_rgba` variants BT.601-convert to grayscale first. Bodies are literal (so
//! cbindgen sees every signature) but delegate to shared runners and ops.

use std::ffi::c_char;

use image::GrayImage;

use crate::detector::{RinggridDetector, as_ref_handle};
use crate::mapper::{detect_with_mapper, detect_with_mapper_diagnostics, parse_mapper};
use crate::status::RinggridStatus;
use crate::util::{gray_from_raw, gray_from_rgba, guard, read_cstr, to_json, write_out};
use crate::wire::{DetectionWithDiagnostics, parse_scale_tiers};

/// Run `op` against a grayscale buffer and write its JSON to `out`.
///
/// # Safety
/// `handle` must be a live handle; `pixels` must point to `width*height` bytes;
/// `out` must be a writable `char*` slot.
unsafe fn run_gray<F>(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
    op: F,
) -> Result<(), RinggridStatus>
where
    F: FnOnce(&ringgrid::Detector, &GrayImage) -> Result<String, RinggridStatus>,
{
    let det = unsafe { as_ref_handle(handle) }?;
    let gray = unsafe { gray_from_raw(pixels, width, height) }?;
    unsafe { write_out(out, op(&det.detector, &gray)?) }
}

/// Run `op` against an RGBA buffer (BT.601-converted) and write its JSON to `out`.
///
/// # Safety
/// As [`run_gray`], but `pixels` must point to `width*height*4` bytes.
unsafe fn run_rgba<F>(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
    op: F,
) -> Result<(), RinggridStatus>
where
    F: FnOnce(&ringgrid::Detector, &GrayImage) -> Result<String, RinggridStatus>,
{
    let det = unsafe { as_ref_handle(handle) }?;
    let gray = unsafe { gray_from_rgba(pixels, width, height) }?;
    unsafe { write_out(out, op(&det.detector, &gray)?) }
}

// ── Ops (shared logic) ──────────────────────────────────────────────

fn op_detect(d: &ringgrid::Detector, g: &GrayImage) -> Result<String, RinggridStatus> {
    to_json(&d.detect(g).map_err(|_| RinggridStatus::ErrDetect)?)
}

fn op_detect_diagnostics(d: &ringgrid::Detector, g: &GrayImage) -> Result<String, RinggridStatus> {
    let (result, diagnostics) = d
        .detect_with_diagnostics(g)
        .map_err(|_| RinggridStatus::ErrDetect)?;
    to_json(&DetectionWithDiagnostics {
        result,
        diagnostics,
    })
}

fn op_detect_adaptive(d: &ringgrid::Detector, g: &GrayImage) -> Result<String, RinggridStatus> {
    to_json(
        &d.detect_adaptive(g)
            .map_err(|_| RinggridStatus::ErrDetect)?,
    )
}

fn op_detect_adaptive_hint(
    d: &ringgrid::Detector,
    g: &GrayImage,
    hint: f32,
) -> Result<String, RinggridStatus> {
    to_json(
        &d.detect_adaptive_with_hint(g, Some(hint))
            .map_err(|_| RinggridStatus::ErrDetect)?,
    )
}

fn op_detect_multiscale(
    d: &ringgrid::Detector,
    g: &GrayImage,
    tiers_json: &str,
) -> Result<String, RinggridStatus> {
    let tiers = parse_scale_tiers(tiers_json)?;
    to_json(
        &d.detect_multiscale(g, &tiers)
            .map_err(|_| RinggridStatus::ErrDetect)?,
    )
}

fn op_detect_mapper(
    d: &ringgrid::Detector,
    g: &GrayImage,
    mapper_json: &str,
) -> Result<String, RinggridStatus> {
    let spec = parse_mapper(mapper_json)?;
    to_json(&detect_with_mapper(d, g, &spec)?)
}

fn op_detect_mapper_diagnostics(
    d: &ringgrid::Detector,
    g: &GrayImage,
    mapper_json: &str,
) -> Result<String, RinggridStatus> {
    let spec = parse_mapper(mapper_json)?;
    let (result, diagnostics) = detect_with_mapper_diagnostics(d, g, &spec)?;
    to_json(&DetectionWithDiagnostics {
        result,
        diagnostics,
    })
}

// ── Entry points: single-pass ───────────────────────────────────────

/// Detect markers in an 8-bit grayscale image. Writes a `DetectionResult` JSON
/// string to `out` (owned; free with `ringgrid_string_free`).
///
/// # Safety
/// `handle` must be a live handle; `pixels` must point to at least
/// `width * height` readable bytes; `out` must point to a writable `char*` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe { run_gray(handle, pixels, width, height, out, op_detect) })
}

/// Detect markers in an RGBA image (BT.601-converted to grayscale first).
///
/// # Safety
/// As [`ringgrid_detect`], but `pixels` must point to `width * height * 4` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_rgba(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe { run_rgba(handle, pixels, width, height, out, op_detect) })
}

/// Detect markers (grayscale), writing `{"result": ..., "diagnostics": ...}` to
/// `out`. `diagnostics.markers` aligns 1:1 with `result.detected_markers`.
///
/// # Safety
/// See [`ringgrid_detect`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_with_diagnostics(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe { run_gray(handle, pixels, width, height, out, op_detect_diagnostics) })
}

/// Detect markers (RGBA) with diagnostics. See
/// [`ringgrid_detect_with_diagnostics`].
///
/// # Safety
/// See [`ringgrid_detect_rgba`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_with_diagnostics_rgba(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe { run_rgba(handle, pixels, width, height, out, op_detect_diagnostics) })
}

// ── Entry points: adaptive ──────────────────────────────────────────

/// Adaptive multi-tier detection (grayscale).
///
/// # Safety
/// See [`ringgrid_detect`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_adaptive(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe { run_gray(handle, pixels, width, height, out, op_detect_adaptive) })
}

/// Adaptive multi-tier detection (RGBA).
///
/// # Safety
/// See [`ringgrid_detect_rgba`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_adaptive_rgba(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe { run_rgba(handle, pixels, width, height, out, op_detect_adaptive) })
}

/// Adaptive detection with a nominal marker diameter hint in pixels (grayscale).
///
/// # Safety
/// See [`ringgrid_detect`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_adaptive_with_hint(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    nominal_diameter_px: f32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe {
        run_gray(handle, pixels, width, height, out, |d, g| {
            op_detect_adaptive_hint(d, g, nominal_diameter_px)
        })
    })
}

/// Adaptive detection with a nominal marker diameter hint in pixels (RGBA).
///
/// # Safety
/// See [`ringgrid_detect_rgba`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_adaptive_with_hint_rgba(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    nominal_diameter_px: f32,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| unsafe {
        run_rgba(handle, pixels, width, height, out, |d, g| {
            op_detect_adaptive_hint(d, g, nominal_diameter_px)
        })
    })
}

// ── Entry points: multi-scale ───────────────────────────────────────

/// Multi-scale detection with explicit tiers `{"tiers": [{"diameter_min_px":
/// .., "diameter_max_px": ..}, ...]}` (grayscale).
///
/// # Safety
/// As [`ringgrid_detect`]; `tiers_json` must be a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_multiscale(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    tiers_json: *const c_char,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let tiers = unsafe { read_cstr(tiers_json) }?;
        unsafe {
            run_gray(handle, pixels, width, height, out, |d, g| {
                op_detect_multiscale(d, g, tiers)
            })
        }
    })
}

/// Multi-scale detection with explicit tiers (RGBA).
///
/// # Safety
/// As [`ringgrid_detect_rgba`]; `tiers_json` must be a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_multiscale_rgba(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    tiers_json: *const c_char,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let tiers = unsafe { read_cstr(tiers_json) }?;
        unsafe {
            run_rgba(handle, pixels, width, height, out, |d, g| {
                op_detect_multiscale(d, g, tiers)
            })
        }
    })
}

// ── Entry points: external mapper ───────────────────────────────────

/// Two-pass detection through an external pixel mapper (grayscale).
/// `mapper_json` is a `MapperSpec` (`{"kind":"camera",...}` /
/// `{"kind":"division",...}`).
///
/// # Safety
/// As [`ringgrid_detect`]; `mapper_json` must be a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_with_mapper(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    mapper_json: *const c_char,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let mapper = unsafe { read_cstr(mapper_json) }?;
        unsafe {
            run_gray(handle, pixels, width, height, out, |d, g| {
                op_detect_mapper(d, g, mapper)
            })
        }
    })
}

/// Two-pass detection through an external pixel mapper (RGBA).
///
/// # Safety
/// As [`ringgrid_detect_rgba`]; `mapper_json` must be a valid NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_with_mapper_rgba(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    mapper_json: *const c_char,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let mapper = unsafe { read_cstr(mapper_json) }?;
        unsafe {
            run_rgba(handle, pixels, width, height, out, |d, g| {
                op_detect_mapper(d, g, mapper)
            })
        }
    })
}

/// Two-pass detection through an external mapper, with diagnostics (grayscale).
///
/// # Safety
/// As [`ringgrid_detect_with_mapper`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_with_mapper_diagnostics(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    mapper_json: *const c_char,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let mapper = unsafe { read_cstr(mapper_json) }?;
        unsafe {
            run_gray(handle, pixels, width, height, out, |d, g| {
                op_detect_mapper_diagnostics(d, g, mapper)
            })
        }
    })
}

/// Two-pass detection through an external mapper, with diagnostics (RGBA).
///
/// # Safety
/// As [`ringgrid_detect_with_mapper_rgba`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ringgrid_detect_with_mapper_diagnostics_rgba(
    handle: *const RinggridDetector,
    pixels: *const u8,
    width: u32,
    height: u32,
    mapper_json: *const c_char,
    out: *mut *mut c_char,
) -> RinggridStatus {
    guard(|| {
        let mapper = unsafe { read_cstr(mapper_json) }?;
        unsafe {
            run_rgba(handle, pixels, width, height, out, |d, g| {
                op_detect_mapper_diagnostics(d, g, mapper)
            })
        }
    })
}
