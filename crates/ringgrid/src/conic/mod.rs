//! Ellipse / conic fitting primitives.
//!
//! Implements:
//! - Direct least-squares conic fit (Fitzgibbon et al., "Direct Least Square Fitting of Ellipses", 1999).
//! - Conversion between general conic coefficients and geometric ellipse parameters.
//! - Algebraic and geometric residual computation.
//! - RANSAC wrapper for outlier-robust fitting.

mod eigen;
mod fit;
mod ransac;
mod types;

pub use fit::{fit_conic_direct, fit_ellipse_direct, rms_sampson_distance};
pub use ransac::try_fit_ellipse_ransac;
pub use types::{Conic2D, ConicCoeffs, Ellipse, RansacConfig, RansacResult};
