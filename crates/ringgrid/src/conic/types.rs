//! Core conic and ellipse types with conversions.

use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

// ── Error type ─────────────────────────────────────────────────────────────

/// Errors that can occur during conic/ellipse fitting.
#[derive(Debug, Clone, PartialEq)]
pub enum ConicError {
    /// Too few points for the requested operation.
    TooFewPoints {
        /// Required minimum number of points.
        needed: usize,
        /// Provided number of points.
        got: usize,
    },
    /// RANSAC could not find enough inliers.
    InsufficientInliers {
        /// Required minimum number of inliers.
        needed: usize,
        /// Number of inliers found by RANSAC.
        found: usize,
    },
}

impl std::fmt::Display for ConicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewPoints { needed, got } => {
                write!(f, "too few points: need {}, got {}", needed, got)
            }
            Self::InsufficientInliers { needed, found } => {
                write!(f, "insufficient inliers: need {}, found {}", needed, found)
            }
        }
    }
}

impl std::error::Error for ConicError {}

// ── Types ──────────────────────────────────────────────────────────────────

/// General conic: A x² + B xy + C y² + D x + E y + F = 0
/// Stored as [A, B, C, D, E, F].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConicCoeffs(pub [f64; 6]);

/// Geometric ellipse parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ellipse {
    /// Center x.
    /// This is the undistorted pixel frame when camera intrinsics are provided,
    /// otherwise it is the raw image pixel frame.
    pub cx: f64,
    /// Center y.
    pub cy: f64,
    /// Semi-major axis length.
    pub a: f64,
    /// Semi-minor axis length.
    pub b: f64,
    /// Rotation angle of the major axis from +x, in radians (−π/2, π/2].
    pub angle: f64,
}

/// 2D conic in homogeneous image coordinates: `x^T Q x = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Conic2D {
    /// Symmetric conic matrix `Q` such that `x^T Q x = 0`.
    pub mat: Matrix3<f64>,
}

impl Conic2D {
    /// Build from general quadratic coefficients:
    /// `A x^2 + B xy + C y^2 + D x + E y + F = 0`.
    pub fn from_quadratic_coeffs(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        Self {
            mat: Matrix3::new(
                a,
                b * 0.5,
                d * 0.5,
                b * 0.5,
                c,
                e * 0.5,
                d * 0.5,
                e * 0.5,
                f,
            ),
        }
    }

    /// Build from conic coefficients.
    pub fn from_coeffs(c: &ConicCoeffs) -> Self {
        let [a, b, cc, d, e, f] = c.0;
        Self::from_quadratic_coeffs(a, b, cc, d, e, f)
    }

    /// Build from fitted geometric ellipse representation.
    pub fn from_ellipse(e: &Ellipse) -> Self {
        Self::from_coeffs(&ellipse_to_conic(e))
    }

    /// Normalize conic scale to unit Frobenius norm.
    pub fn normalize_frobenius(&self) -> Option<Self> {
        let n = self.mat.norm();
        if !n.is_finite() || n <= 1e-15 {
            return None;
        }
        Some(Self { mat: self.mat / n })
    }
}

/// Configuration for RANSAC ellipse fitting.
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum number of RANSAC iterations.
    pub max_iters: usize,
    /// Inlier threshold (algebraic distance, after normalization).
    pub inlier_threshold: f64,
    /// Minimum number of inliers for a valid model.
    pub min_inliers: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iters: 500,
            inlier_threshold: 1.0, // Sampson distance in pixels
            min_inliers: 10,
            seed: 42,
        }
    }
}

/// Result of a RANSAC fit.
#[derive(Debug, Clone)]
pub struct RansacResult {
    /// Final geometric ellipse fitted on the inlier set.
    pub ellipse: Ellipse,
    /// Number of inliers under the configured Sampson threshold.
    pub num_inliers: usize,
}

// ── Conic normalization ────────────────────────────────────────────────────

impl ConicCoeffs {
    /// Normalize conic coefficients so that A + C = 1 (trace of quadratic part).
    /// This makes the algebraic distance comparable across different conics.
    /// Returns `None` if A + C ≈ 0 (degenerate / hyperbola-like).
    pub fn normalized(&self) -> Option<Self> {
        let [a, b, c, d, e, f] = self.0;
        let trace = a + c;
        if trace.abs() < 1e-15 {
            return None;
        }
        let s = 1.0 / trace;
        Some(Self([a * s, b * s, c * s, d * s, e * s, f * s]))
    }

    /// Algebraic distance of a point (x, y) to this conic.
    pub fn algebraic_distance(&self, x: f64, y: f64) -> f64 {
        let [a, b, c, d, e, f] = self.0;
        a * x * x + b * x * y + c * y * y + d * x + e * y + f
    }

    /// Check whether the conic represents an ellipse (discriminant B²−4AC < 0).
    pub fn is_ellipse(&self) -> bool {
        let [a, b, c, ..] = self.0;
        b * b - 4.0 * a * c < 0.0
    }

    /// Convert to geometric ellipse parameters.
    /// Returns `None` if the conic is not an ellipse.
    pub fn to_ellipse(self) -> Option<Ellipse> {
        conic_to_ellipse(&self)
    }
}

// ── Ellipse utilities ──────────────────────────────────────────────────────

impl Ellipse {
    /// Check basic validity: positive semi-axes, finite values.
    pub fn is_valid(&self) -> bool {
        self.a > 0.0
            && self.b > 0.0
            && self.a.is_finite()
            && self.b.is_finite()
            && self.cx.is_finite()
            && self.cy.is_finite()
            && self.angle.is_finite()
    }

    /// Aspect ratio a/b (always >= 1 when canonicalized).
    pub fn aspect_ratio(&self) -> f64 {
        if self.a >= self.b {
            self.a / self.b
        } else {
            self.b / self.a
        }
    }

    /// Convert back to conic coefficients.
    pub fn to_conic(self) -> ConicCoeffs {
        ellipse_to_conic(&self)
    }

    /// Sample `n` points on the ellipse boundary.
    pub fn sample_points(&self, n: usize) -> Vec<[f64; 2]> {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        (0..n)
            .map(|i| {
                let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
                let px = self.a * t.cos();
                let py = self.b * t.sin();
                let x = self.cx + cos_a * px - sin_a * py;
                let y = self.cy + sin_a * px + cos_a * py;
                [x, y]
            })
            .collect()
    }

    /// Approximate geometric distance from a point to the ellipse boundary.
    /// Uses the algebraic distance divided by the gradient magnitude as a
    /// first-order approximation (Sampson distance).
    pub fn sampson_distance(&self, x: f64, y: f64) -> f64 {
        let c = self.to_conic();
        let [ca, cb, cc, cd, ce, _cf] = c.0;
        let alg = c.algebraic_distance(x, y);
        let gx = 2.0 * ca * x + cb * y + cd;
        let gy = cb * x + 2.0 * cc * y + ce;
        let grad_mag_sq = gx * gx + gy * gy;
        if grad_mag_sq < 1e-30 {
            return alg.abs();
        }
        alg.abs() / grad_mag_sq.sqrt()
    }
}

// ── Conversion: conic ↔ ellipse ────────────────────────────────────────────

/// Convert general conic coefficients to geometric ellipse parameters.
///
/// The general conic is A x² + B xy + C y² + D x + E y + F = 0.
/// Returns `None` if the conic does not represent a proper ellipse.
pub fn conic_to_ellipse(c: &ConicCoeffs) -> Option<Ellipse> {
    let [a, b, c_coeff, d, e, f] = c.0;

    // Discriminant check: must be an ellipse
    let disc = b * b - 4.0 * a * c_coeff;
    if disc >= 0.0 {
        return None;
    }

    // Matrix form: M = [[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]]
    let m = Matrix3::new(
        a,
        b / 2.0,
        d / 2.0,
        b / 2.0,
        c_coeff,
        e / 2.0,
        d / 2.0,
        e / 2.0,
        f,
    );
    let det_m = m.determinant();

    // For a proper ellipse, det(M) must be nonzero and have opposite sign to A+C
    // (so that the ellipse encloses a finite area).
    if det_m.abs() < 1e-15 {
        return None;
    }

    // Center by solving the 2x2 system:
    //   2A·cx + B·cy + D = 0
    //   B·cx + 2C·cy + E = 0
    let denom = 4.0 * a * c_coeff - b * b; // = -disc > 0
    let cx = (b * e - 2.0 * c_coeff * d) / denom;
    let cy = (b * d - 2.0 * a * e) / denom;

    // Rotation angle
    let angle = if (a - c_coeff).abs() < 1e-15 {
        if b > 0.0 {
            std::f64::consts::FRAC_PI_4
        } else if b < 0.0 {
            -std::f64::consts::FRAC_PI_4
        } else {
            0.0
        }
    } else {
        0.5 * (b).atan2(a - c_coeff)
    };

    // Semi-axes from eigenvalues of the 2x2 quadratic part
    let sum = a + c_coeff;
    let diff = ((a - c_coeff).powi(2) + b * b).sqrt();
    let lambda1 = (sum + diff) / 2.0;
    let lambda2 = (sum - diff) / 2.0;

    // F' = value of the conic at the center
    let f_prime = a * cx * cx + b * cx * cy + c_coeff * cy * cy + d * cx + e * cy + f;

    if f_prime.abs() < 1e-15 {
        return None;
    }

    let a_sq = -f_prime / lambda1;
    let b_sq = -f_prime / lambda2;

    if a_sq <= 0.0 || b_sq <= 0.0 {
        return None;
    }

    let semi_a = a_sq.sqrt();
    let semi_b = b_sq.sqrt();

    // Canonicalize so that a >= b, adjusting angle accordingly
    let (semi_a, semi_b, angle) = if semi_a >= semi_b {
        (semi_a, semi_b, angle)
    } else {
        (semi_b, semi_a, angle + std::f64::consts::FRAC_PI_2)
    };

    // Normalize angle to (−π/2, π/2]
    let angle = normalize_angle(angle);

    Some(Ellipse {
        cx,
        cy,
        a: semi_a,
        b: semi_b,
        angle,
    })
}

/// Convert geometric ellipse parameters to general conic coefficients.
pub fn ellipse_to_conic(e: &Ellipse) -> ConicCoeffs {
    let cos_a = e.angle.cos();
    let sin_a = e.angle.sin();
    let a2 = e.a * e.a;
    let b2 = e.b * e.b;

    let ca = cos_a * cos_a / a2 + sin_a * sin_a / b2;
    let cb = 2.0 * cos_a * sin_a * (1.0 / a2 - 1.0 / b2);
    let cc = sin_a * sin_a / a2 + cos_a * cos_a / b2;
    let cd = -2.0 * ca * e.cx - cb * e.cy;
    let ce = -cb * e.cx - 2.0 * cc * e.cy;
    let cf = ca * e.cx * e.cx + cb * e.cx * e.cy + cc * e.cy * e.cy - 1.0;

    ConicCoeffs([ca, cb, cc, cd, ce, cf])
}

/// Normalize angle to (−π/2, π/2].
fn normalize_angle(mut angle: f64) -> f64 {
    let pi = std::f64::consts::PI;
    while angle > pi / 2.0 {
        angle -= pi;
    }
    while angle <= -pi / 2.0 {
        angle += pi;
    }
    angle
}
