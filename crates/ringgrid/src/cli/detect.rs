//! Detection helpers for the CLI: load a target (spec or recipe), detect one
//! image, and drive a directory batch. Detection logic and serialization live
//! here so the published binary and dev CLI share one implementation.

use std::path::{Path, PathBuf};

use image::GrayImage;

use crate::DetectionResult;
use crate::api::{DetectError, Detector};
use crate::detector::DetectConfig;
use crate::target::TargetLayout;

use super::recipe::TargetRecipe;

/// Failure during a CLI detection run.
#[derive(Debug)]
pub enum DetectRunError {
    /// Filesystem error.
    Io {
        /// Path being read/written.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Failed to decode an image.
    Image {
        /// Image path.
        path: PathBuf,
        /// Underlying decode error.
        message: String,
    },
    /// Failed to load a target (neither a valid v5/v4 spec nor a recipe).
    TargetLoad {
        /// Target path.
        path: PathBuf,
        /// Explanation.
        message: String,
    },
    /// Failed to parse a detection-config overlay.
    Config(String),
    /// Detection itself failed (e.g. the strict complete-board gate).
    Detect(DetectError),
}

impl std::fmt::Display for DetectRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Image { path, message } => {
                write!(f, "failed to read image {}: {message}", path.display())
            }
            Self::TargetLoad { path, message } => {
                write!(f, "failed to load target {}: {message}", path.display())
            }
            Self::Config(message) => write!(f, "invalid detect config: {message}"),
            Self::Detect(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DetectRunError {}

/// Load a target from a v5/v4 JSON spec **or** a TOML/JSON recipe.
///
/// `.toml` files are always recipes; `.json` files are tried as a canonical
/// spec first (v5/v4 auto-migration), then as a recipe.
pub fn load_target(path: &Path) -> Result<TargetLayout, DetectRunError> {
    let text = std::fs::read_to_string(path).map_err(|source| DetectRunError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let is_toml = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("toml"));

    if is_toml {
        let recipe = TargetRecipe::parse_toml(&text).map_err(|e| DetectRunError::TargetLoad {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        return recipe.to_target().map_err(|e| DetectRunError::TargetLoad {
            path: path.to_path_buf(),
            message: e.to_string(),
        });
    }

    // JSON: canonical spec first, then recipe.
    match TargetLayout::from_json_str(&text) {
        Ok(target) => Ok(target),
        Err(spec_err) => match TargetRecipe::parse_json(&text).ok().map(|r| r.to_target()) {
            Some(Ok(target)) => Ok(target),
            _ => Err(DetectRunError::TargetLoad {
                path: path.to_path_buf(),
                message: spec_err.to_string(),
            }),
        },
    }
}

/// Build a detector from a target and CLI options.
///
/// Precedence for the strict complete-board gate: `--strict` (`strict = true`)
/// OR a `require_complete_board: true` in the config overlay enables it.
pub fn build_detector(
    target: TargetLayout,
    marker_diameter_px: Option<f32>,
    strict: bool,
    overlay: Option<serde_json::Value>,
) -> Result<Detector, DetectRunError> {
    let mut config = match marker_diameter_px {
        Some(d) => DetectConfig::from_target_and_marker_diameter(target, d),
        None => DetectConfig::from_target(target),
    };
    if let Some(overlay) = overlay {
        config = config
            .with_json_overlay(overlay)
            .map_err(|e| DetectRunError::Config(e.to_string()))?;
    }
    config.require_complete_board = config.require_complete_board || strict;
    Ok(Detector::with_config(config))
}

/// Detect markers in a single already-loaded grayscale image.
///
/// Uses single-pass detection when a diameter hint is supplied (fast, focused
/// scale) and adaptive multi-scale detection otherwise (robust to unknown
/// marker size).
pub fn detect_image(
    detector: &Detector,
    image: &GrayImage,
    has_diameter_hint: bool,
) -> Result<DetectionResult, DetectError> {
    if has_diameter_hint {
        detector.detect(image)
    } else {
        detector.detect_adaptive(image)
    }
}

/// Load and detect a single image file.
pub fn detect_file(
    detector: &Detector,
    image_path: &Path,
    has_diameter_hint: bool,
) -> Result<DetectionResult, DetectRunError> {
    let image = image::open(image_path)
        .map_err(|e| DetectRunError::Image {
            path: image_path.to_path_buf(),
            message: e.to_string(),
        })?
        .to_luma8();
    detect_image(detector, &image, has_diameter_hint).map_err(DetectRunError::Detect)
}

/// Outcome for one image in a batch run.
#[derive(Debug)]
pub struct BatchOutcome {
    /// Source image path.
    pub image: PathBuf,
    /// Detection result, or `None` if the image failed.
    pub result: Option<DetectionResult>,
    /// Failure message when `result` is `None`.
    pub error: Option<String>,
}

/// List image files in a directory (png/jpg/jpeg/bmp/tif/tiff), sorted.
pub fn image_files_in_dir(dir: &Path) -> Result<Vec<PathBuf>, DetectRunError> {
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|source| DetectRunError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| DetectRunError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let is_image = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .is_some_and(|e| matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff"));
        if is_image {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Detect every image with a shared detector, collecting per-image outcomes.
pub fn run_batch(
    detector: &Detector,
    images: &[PathBuf],
    has_diameter_hint: bool,
) -> Vec<BatchOutcome> {
    images
        .iter()
        .map(
            |path| match detect_file(detector, path, has_diameter_hint) {
                Ok(result) => BatchOutcome {
                    image: path.clone(),
                    result: Some(result),
                    error: None,
                },
                Err(e) => BatchOutcome {
                    image: path.clone(),
                    result: None,
                    error: Some(e.to_string()),
                },
            },
        )
        .collect()
}

/// Number of markers with a decoded ID in a result (a compact quality signal).
pub fn decoded_count(result: &DetectionResult) -> usize {
    result
        .detected_markers
        .iter()
        .filter(|m| m.id.is_some())
        .count()
}
