//! Write target artifacts (the canonical v6 JSON spec plus printable
//! SVG/PNG/DXF) per a render recipe. Thin orchestration over
//! [`TargetLayout::render_target_artifacts`] — the single source of truth for
//! target output across the published CLI and the in-repo dev CLI.

use std::path::{Path, PathBuf};

use crate::target::TargetLayout;
use crate::target_generation::TargetGenerationError;

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
    /// Rendering error producing the artifacts.
    Render(TargetGenerationError),
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "failed to write {}: {source}", path.display()),
            Self::Render(source) => write!(f, "failed to render target: {source}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

/// Write the requested artifacts for `target` into `out_dir`, returning the
/// written paths in emission order.
///
/// The canonical v6 spec is always named `target_spec.json` (the name detection
/// expects); printable artifacts use `basename`.
///
/// Every format is rendered once, up front, so a target that cannot be drawn on
/// the requested page fails before any file is created rather than leaving a
/// half-written output directory behind.
pub fn write_target_artifacts(
    target: &TargetLayout,
    out_dir: &Path,
    basename: &str,
    render: &RenderRecipe,
) -> Result<Vec<PathBuf>, ArtifactError> {
    let artifacts = target
        .render_target_artifacts(&render.to_render_options())
        .map_err(ArtifactError::Render)?;

    std::fs::create_dir_all(out_dir).map_err(|source| ArtifactError::Io {
        path: out_dir.to_path_buf(),
        source,
    })?;

    let mut written = Vec::with_capacity(render.formats.len());
    for format in &render.formats {
        let (path, bytes): (PathBuf, &[u8]) = match format {
            Format::Json => (
                out_dir.join("target_spec.json"),
                artifacts.json_text.as_bytes(),
            ),
            Format::Svg => (
                out_dir.join(format!("{basename}.svg")),
                artifacts.svg_text.as_bytes(),
            ),
            Format::Png => (
                out_dir.join(format!("{basename}.png")),
                &artifacts.png_bytes,
            ),
            Format::Dxf => (
                out_dir.join(format!("{basename}.dxf")),
                artifacts.dxf_text.as_bytes(),
            ),
        };
        std::fs::write(&path, bytes).map_err(|source| ArtifactError::Io {
            path: path.clone(),
            source,
        })?;
        written.push(path);
    }
    Ok(written)
}
