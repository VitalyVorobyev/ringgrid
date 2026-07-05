//! Write target artifacts (v5 JSON spec + printable SVG/PNG/DXF) per a render
//! recipe. Thin orchestration over the library's `write_target_*` methods —
//! the single source of truth for target output across the published CLI and
//! the in-repo dev CLI.

use std::path::{Path, PathBuf};

use crate::target::TargetLayout;
use crate::target_generation::{PngTargetOptions, SvgTargetOptions, TargetGenerationError};

use super::recipe::{Format, RenderRecipe};

/// Failure writing a target artifact.
#[derive(Debug)]
pub enum ArtifactError {
    /// Filesystem error writing a file.
    Io {
        /// Path being written.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// Rendering error producing an artifact.
    Render {
        /// Path being written.
        path: PathBuf,
        /// Underlying rendering error.
        source: TargetGenerationError,
    },
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "failed to write {}: {source}", path.display()),
            Self::Render { path, source } => {
                write!(f, "failed to render {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for ArtifactError {}

/// Write the requested artifacts for `target` into `out_dir`, returning the
/// written paths in emission order.
///
/// The canonical v5 spec is always named `target_spec.json` (the name detection
/// expects); printable artifacts use `basename`.
pub fn write_target_artifacts(
    target: &TargetLayout,
    out_dir: &Path,
    basename: &str,
    render: &RenderRecipe,
) -> Result<Vec<PathBuf>, ArtifactError> {
    std::fs::create_dir_all(out_dir).map_err(|source| ArtifactError::Io {
        path: out_dir.to_path_buf(),
        source,
    })?;

    let mut written = Vec::with_capacity(render.formats.len());
    for format in &render.formats {
        match format {
            Format::Json => {
                let path = out_dir.join("target_spec.json");
                target
                    .write_json_file(&path)
                    .map_err(|source| ArtifactError::Io {
                        path: path.clone(),
                        source,
                    })?;
                written.push(path);
            }
            Format::Svg => {
                let path = out_dir.join(format!("{basename}.svg"));
                target
                    .write_target_svg(
                        &path,
                        &SvgTargetOptions {
                            margin_mm: render.margin_mm,
                            include_scale_bar: render.scale_bar,
                        },
                    )
                    .map_err(|source| ArtifactError::Render {
                        path: path.clone(),
                        source,
                    })?;
                written.push(path);
            }
            Format::Png => {
                let path = out_dir.join(format!("{basename}.png"));
                target
                    .write_target_png(
                        &path,
                        &PngTargetOptions {
                            dpi: render.dpi,
                            margin_mm: render.margin_mm,
                            include_scale_bar: render.scale_bar,
                        },
                    )
                    .map_err(|source| ArtifactError::Render {
                        path: path.clone(),
                        source,
                    })?;
                written.push(path);
            }
            Format::Dxf => {
                let path = out_dir.join(format!("{basename}.dxf"));
                target
                    .write_target_dxf(&path)
                    .map_err(|source| ArtifactError::Render {
                        path: path.clone(),
                        source,
                    })?;
                written.push(path);
            }
        }
    }
    Ok(written)
}
