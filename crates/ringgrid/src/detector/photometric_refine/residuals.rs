//! Per-marker fit residuals and covariance diagnostics, for studying where
//! the intensity model fails (`study` feature).

use super::{BLUR_GRADIENT_BIAS_PER_SIGMA, FitMode, PhotometricFit, PrintRadii, Stage, Task};
use crate::ring::photometric_center::{PhotometricResidualSample, residual_samples};

/// Where the stage sends each fitted marker's residuals, when anywhere.
pub(crate) type ResidualSink<'a> = Option<&'a mut Vec<PhotometricResidualMarker>>;

/// Spatial autocorrelation of one marker's photometric fit residuals.
///
/// Entry `k` is the correlation coefficient of the (robustly weighted)
/// residuals of pixels `k + 1` px apart along the image x or y axis. White
/// pixel noise gives values near 0; a pattern model that does not match the
/// image (PSF shape, response nonlinearity, print defects) or correlated
/// sensor noise gives positive values over a few pixels. The centre
/// covariance accounts for this correlation (see
/// [`PhotometricResidualMarker::covariance_inflation`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualAutocorrelation {
    /// Correlation at lags 1 to 6 px along image x.
    pub x: [f64; 6],
    /// Correlation at lags 1 to 6 px along image y.
    pub y: [f64; 6],
}

/// The per-pixel fit residuals and noise diagnostics of one marker's
/// photometric fit, whether or not the fit was accepted. Obtained through
/// `Detector::detect_with_photometric_residuals`.
#[derive(Debug, Clone)]
pub struct PhotometricResidualMarker {
    /// Index into `DetectionResult::detected_markers`.
    pub marker_index: usize,
    /// Fitted centre (image pixels), whether or not the fit was accepted.
    pub center: [f64; 2],
    /// Fitted PSF sigma (pixels).
    pub blur_sigma_px: f64,
    /// Fitted PSF anisotropy `[a, b]`.
    pub aniso: [f64; 2],
    /// Covariance of `(blur_sigma_px, aniso[0], aniso[1])`, from the same
    /// residual-correlation-aware sandwich as the centre covariance.
    pub blur_covariance: [[f64; 3]; 3],
    /// The same covariance assuming spatially white residuals.
    pub blur_covariance_white: [[f64; 3]; 3],
    /// Outer print radius used (board mm); the unit of the samples' `radius`.
    pub outer_radius_mm: f64,
    /// Robust residual standard deviation of the fit (grey levels).
    pub residual_sigma: f64,
    /// Ratio of the reported centre standard deviation to the one the fit
    /// would report for spatially white residuals (`sqrt` of the trace ratio
    /// of the two covariances). Near 1 when the model explains the pixels up
    /// to white noise; above 1 when correlated misfit or correlated noise
    /// inflates the centre uncertainty.
    pub covariance_inflation: Option<f64>,
    /// Spatial autocorrelation of the fit residuals.
    pub residual_autocorrelation: ResidualAutocorrelation,
    /// Predicted blur-gradient bias of an accepted centre (px, image frame):
    /// `-1.24 σ ∇σ`, towards the sharper side, `∇σ` the neighbour-plane blur
    /// gradient (its magnitude enters the confidence). Subtracting it from
    /// `center` is the first-order law correction. `None` for a fit that was
    /// not accepted or whose neighbourhood does not determine the plane.
    pub blur_gradient_bias: Option<[f64; 2]>,
    /// The patch pixels in row-major order.
    pub samples: Vec<PhotometricResidualSample>,
}

/// Re-evaluate one finished fit over its patch and record it. `seed` is the
/// centre the fit started from (it fixes the patch).
pub(super) fn record_residuals(
    sink: &mut Vec<PhotometricResidualMarker>,
    stage: &Stage<'_>,
    task: &Task,
    seed: [f64; 2],
    radii: PrintRadii,
    fit: &PhotometricFit,
) {
    let mode = FitMode {
        fit_radii: false,
        cubic_background: false,
    };
    let problem = stage.problem(task, seed, radii, mode);
    let Some(samples) = residual_samples(&problem, fit) else {
        return;
    };
    sink.push(PhotometricResidualMarker {
        marker_index: task.index,
        center: fit.center,
        blur_sigma_px: fit.blur.sigma_px,
        aniso: fit.blur.anisotropy,
        blur_covariance: fit.blur.cov,
        blur_covariance_white: fit.blur_cov_white,
        outer_radius_mm: fit.radii[3],
        residual_sigma: fit.residual_sigma,
        covariance_inflation: fit.covariance_inflation(),
        residual_autocorrelation: ResidualAutocorrelation {
            x: fit.residual_acf.x,
            y: fit.residual_acf.y,
        },
        blur_gradient_bias: None,
        samples,
    });
}

/// Record the blur-gradient bias vector `-1.24 σ ∇σ` of every accepted fit
/// (`accepted` and `gradients` aligned, as in the stage).
pub(super) fn record_blur_gradient_bias(
    sink: &mut [PhotometricResidualMarker],
    accepted: &[(usize, f64)],
    gradients: &[Option<[f64; 2]>],
) {
    for (&(i, sigma), g) in accepted.iter().zip(gradients) {
        if let (Some(r), Some(g)) = (sink.iter_mut().find(|r| r.marker_index == i), g) {
            let k = -BLUR_GRADIENT_BIAS_PER_SIGMA * sigma;
            r.blur_gradient_bias = Some([k * g[0], k * g[1]]);
        }
    }
}
