//! Circular 16-sector sampling and codebook matching.
//!
//! Samples sector intensities along a circle at the code band radius
//! (derived from the fitted outer ellipse). Uses the ellipse center and
//! mean semi-axis but NOT the ellipse angle — the fitted angle reflects
//! perspective distortion rather than the board-to-image rotation.
//! The codebook matcher handles the unknown rotation via cyclic matching
//! over all 16 sector offsets. Inverted polarity is also tried as fallback.

use image::GrayImage;

use crate::conic::Ellipse;
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::DistortionAwareSampler;

use super::codec::Codebook;

/// Decode quality metrics for a detected marker.
///
/// Reports how confidently a 16-sector binary code was matched against the
/// 893-codeword codebook. A `best_dist` of 0 means an exact match; `margin`
/// measures how far the second-best codeword is (higher = more confident).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeMetrics {
    /// Raw 16-bit word sampled from the code band.
    pub observed_word: u16,
    /// Best-matching codebook entry index (0–892).
    pub best_id: usize,
    /// Cyclic rotation (0–15 sectors) that produced the best match.
    pub best_rotation: u8,
    /// Hamming distance (bit errors) to the best-matching codeword.
    pub best_dist: u8,
    /// Margin: `second_best_dist - best_dist`. Higher values indicate
    /// a more reliable decode. A margin of 3 or more is typically very confident.
    pub margin: u8,
    /// Combined confidence heuristic in \[0, 1\].
    ///
    /// Equals `clamp(1 - dist/6) * clamp(margin / CODEBOOK_MIN_CYCLIC_DIST)`.
    /// A margin equal to `CODEBOOK_MIN_CYCLIC_DIST` (the minimum attainable for
    /// a correct decode) yields `conf_margin = 1.0`, so a perfect decode scores
    /// 1.0 regardless of the codebook density.
    pub decode_confidence: f32,
}

/// Configuration for sector decoding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeConfig {
    /// Ratio of code band center radius to outer ellipse semi-major axis.
    /// The code band is sampled at `code_band_ratio * (a, b)` in the
    /// ellipse coordinate frame.
    /// Default: [`DecodeConfig::DEFAULT_CODE_BAND_RATIO`].
    pub code_band_ratio: f32,
    /// Number of angular samples per sector.
    /// Default: [`DecodeConfig::DEFAULT_SAMPLES_PER_SECTOR`].
    pub samples_per_sector: usize,
    /// Number of radial rings to sample within the code band.
    /// Default: [`DecodeConfig::DEFAULT_N_RADIAL_RINGS`].
    pub n_radial_rings: usize,
    /// Maximum Hamming distance for a valid decode.
    /// Default: [`DecodeConfig::DEFAULT_MAX_DECODE_DIST`].
    pub max_decode_dist: u8,
    /// Minimum confidence for a valid decode.
    /// Default: [`DecodeConfig::DEFAULT_MIN_DECODE_CONFIDENCE`].
    pub min_decode_confidence: f32,
    /// Minimum Hamming margin (`second_best_dist - best_dist`) for a valid decode.
    ///
    /// A margin of 0 means the two closest codewords are equidistant from the
    /// observed word — genuinely ambiguous. Default rejects ties (margin = 0).
    /// For setups without homography validation (e.g. no camera intrinsics),
    /// setting this to `CODEBOOK_MIN_CYCLIC_DIST` (= 2) accepts only matches
    /// that are unambiguous within the codebook's minimum distance guarantee.
    ///
    /// Default: [`DecodeConfig::DEFAULT_MIN_DECODE_MARGIN`].
    #[serde(default = "DecodeConfig::default_min_decode_margin")]
    pub min_decode_margin: u8,
    /// Minimum accepted sector-intensity contrast (`max - min`) before decode.
    /// Default: [`DecodeConfig::DEFAULT_MIN_DECODE_CONTRAST`].
    #[serde(default = "DecodeConfig::default_min_decode_contrast")]
    pub min_decode_contrast: f32,
    /// Maximum iterations for iterative 2-means threshold refinement.
    /// Default: [`DecodeConfig::DEFAULT_THRESHOLD_MAX_ITERS`].
    #[serde(default = "DecodeConfig::default_threshold_max_iters")]
    pub threshold_max_iters: usize,
    /// Convergence epsilon for iterative 2-means threshold refinement.
    /// Stop when `|new_threshold - old_threshold| <= eps`.
    /// Default: [`DecodeConfig::DEFAULT_THRESHOLD_CONVERGENCE_EPS`].
    #[serde(default = "DecodeConfig::default_threshold_convergence_eps")]
    pub threshold_convergence_eps: f32,
}

impl DecodeConfig {
    pub const DEFAULT_CODE_BAND_RATIO: f32 = 0.76;
    pub const DEFAULT_SAMPLES_PER_SECTOR: usize = 5;
    pub const DEFAULT_N_RADIAL_RINGS: usize = 3;
    pub const DEFAULT_MAX_DECODE_DIST: u8 = 3;
    /// Minimum decode confidence with the corrected formula
    /// `clamp(1-dist/6) * clamp(margin/CODEBOOK_MIN_CYCLIC_DIST)`.
    /// A perfect decode (dist=0, margin≥2) scores 1.0; this threshold accepts
    /// matches down to dist=2 with margin≥1.
    pub const DEFAULT_MIN_DECODE_CONFIDENCE: f32 = 0.3;
    /// Minimum Hamming margin required for a valid decode.
    ///
    /// A margin of 0 means two codewords are equidistant from the observed word
    /// (genuinely ambiguous). Setting this to 1 (default) rejects such ties.
    pub const DEFAULT_MIN_DECODE_MARGIN: u8 = 1;
    pub const DEFAULT_MIN_DECODE_CONTRAST: f32 = 0.03;
    pub const DEFAULT_THRESHOLD_MAX_ITERS: usize = 10;
    pub const DEFAULT_THRESHOLD_CONVERGENCE_EPS: f32 = 1e-4;

    fn default_min_decode_contrast() -> f32 {
        Self::DEFAULT_MIN_DECODE_CONTRAST
    }

    fn default_min_decode_margin() -> u8 {
        Self::DEFAULT_MIN_DECODE_MARGIN
    }

    fn default_threshold_max_iters() -> usize {
        Self::DEFAULT_THRESHOLD_MAX_ITERS
    }

    fn default_threshold_convergence_eps() -> f32 {
        Self::DEFAULT_THRESHOLD_CONVERGENCE_EPS
    }
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            code_band_ratio: Self::DEFAULT_CODE_BAND_RATIO,
            samples_per_sector: Self::DEFAULT_SAMPLES_PER_SECTOR,
            n_radial_rings: Self::DEFAULT_N_RADIAL_RINGS,
            max_decode_dist: Self::DEFAULT_MAX_DECODE_DIST,
            min_decode_confidence: Self::DEFAULT_MIN_DECODE_CONFIDENCE,
            min_decode_margin: Self::DEFAULT_MIN_DECODE_MARGIN,
            min_decode_contrast: Self::DEFAULT_MIN_DECODE_CONTRAST,
            threshold_max_iters: Self::DEFAULT_THRESHOLD_MAX_ITERS,
            threshold_convergence_eps: Self::DEFAULT_THRESHOLD_CONVERGENCE_EPS,
        }
    }
}

/// Result of decoding a marker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeResult {
    /// Matched marker ID (codebook index).
    pub id: usize,
    /// Overall confidence [0, 1].
    pub confidence: f32,
    /// Raw 16-bit word sampled from image (before codebook matching).
    pub raw_word: u16,
    /// Hamming distance to best codebook match.
    pub dist: u8,
    /// Margin (second_best - best distance).
    pub margin: u8,
    /// Cyclic rotation applied to the codeword.
    pub rotation: u8,
}

/// Stable reject code for a decode attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecodeRejectReason {
    InvalidEllipse,
    LowContrast,
    DistTooHigh,
    MarginTooLow,
    ConfidenceTooLow,
}

impl DecodeRejectReason {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::InvalidEllipse => "invalid_ellipse",
            Self::LowContrast => "low_contrast",
            Self::DistTooHigh => "dist_too_high",
            Self::MarginTooLow => "margin_too_low",
            Self::ConfidenceTooLow => "confidence_too_low",
        }
    }
}

impl std::fmt::Display for DecodeRejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

/// Structured reject context for decode diagnostics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum DecodeRejectContext {
    InvalidEllipse {
        is_valid: bool,
        semi_axis_a: f64,
        semi_axis_b: f64,
        min_allowed_semi_axis: f64,
    },
    LowContrast {
        contrast: f32,
        min_required_contrast: f32,
    },
    DistTooHigh {
        observed_dist: u8,
        max_allowed_dist: u8,
    },
    MarginTooLow {
        observed_margin: u8,
        min_required_margin: u8,
    },
    ConfidenceTooLow {
        observed_confidence: f32,
        min_required_confidence: f32,
    },
}

/// Debug/diagnostic information about a decode attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeDiagnostics {
    /// Per-sector average intensities sampled from the code band.
    pub sector_intensities: [f32; 16],
    /// Threshold used to binarize sector intensities.
    pub threshold: f32,
    /// Word actually matched against the codebook (possibly inverted).
    pub used_word: u16,
    /// Whether the matched word was polarity-inverted before matching.
    pub inverted_used: bool,
    /// Best-matching codebook id.
    pub best_id: usize,
    /// Rotation (in sectors) of the best match.
    pub best_rotation: u8,
    /// Hamming distance of the best match.
    pub best_dist: u8,
    /// Distance margin to the second-best match.
    pub margin: u8,
    /// Decode confidence in `[0, 1]`.
    pub decode_confidence: f32,
    /// Stable reject code, if rejected.
    pub reject_reason: Option<DecodeRejectReason>,
    /// Structured reject diagnostics, if rejected.
    pub reject_context: Option<DecodeRejectContext>,
}

/// Decode a marker and return `(accepted_result, diagnostics)`.
///
/// Uses an optional working<->image mapper for distortion-aware sampling.
pub fn decode_marker_with_diagnostics_and_mapper(
    gray: &GrayImage,
    outer_ellipse: &Ellipse,
    config: &DecodeConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (Option<DecodeResult>, DecodeDiagnostics) {
    decode_marker_impl(gray, outer_ellipse, config, mapper)
}

#[inline]
fn intensity_range(values: &[f32; 16]) -> (f32, f32) {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in values {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    (min_v, max_v)
}

/// Compute a binary threshold from 16 sector intensities via 1D Lloyd updates.
///
/// This is equivalent to iterative 2-means with an explicit iteration cap and
/// convergence epsilon. The bounded update keeps decode deterministic and
/// prevents unbounded loops on pathological intensity distributions.
fn compute_iterative_two_means_threshold(
    sector_intensities: &[f32; 16],
    max_iters: usize,
    convergence_eps: f32,
) -> f32 {
    let (min_v, max_v) = intensity_range(sector_intensities);
    let mut threshold = 0.5 * (min_v + max_v);
    let eps = if convergence_eps.is_finite() {
        convergence_eps.abs()
    } else {
        DecodeConfig::DEFAULT_THRESHOLD_CONVERGENCE_EPS
    };

    for _ in 0..max_iters {
        let (mut sum_lo, mut cnt_lo) = (0.0f32, 0u32);
        let (mut sum_hi, mut cnt_hi) = (0.0f32, 0u32);
        for &v in sector_intensities {
            if v <= threshold {
                sum_lo += v;
                cnt_lo += 1;
            } else {
                sum_hi += v;
                cnt_hi += 1;
            }
        }
        if cnt_lo == 0 || cnt_hi == 0 {
            break;
        }
        let new_threshold = (sum_lo / cnt_lo as f32 + sum_hi / cnt_hi as f32) * 0.5;
        if (new_threshold - threshold).abs() <= eps {
            threshold = new_threshold;
            break;
        }
        threshold = new_threshold;
    }

    threshold
}

fn decode_marker_impl(
    gray: &GrayImage,
    outer_ellipse: &Ellipse,
    config: &DecodeConfig,
    mapper: Option<&dyn PixelMapper>,
) -> (Option<DecodeResult>, DecodeDiagnostics) {
    // Validate ellipse
    if !outer_ellipse.is_valid() || outer_ellipse.a < 2.0 || outer_ellipse.b < 2.0 {
        return (
            None,
            DecodeDiagnostics {
                sector_intensities: [0.0; 16],
                threshold: 0.0,
                used_word: 0,
                inverted_used: false,
                best_id: 0,
                best_rotation: 0,
                best_dist: u8::MAX,
                margin: 0,
                decode_confidence: 0.0,
                reject_reason: Some(DecodeRejectReason::InvalidEllipse),
                reject_context: Some(DecodeRejectContext::InvalidEllipse {
                    is_valid: outer_ellipse.is_valid(),
                    semi_axis_a: outer_ellipse.a,
                    semi_axis_b: outer_ellipse.b,
                    min_allowed_semi_axis: 2.0,
                }),
            },
        );
    }

    let cx = outer_ellipse.cx;
    let cy = outer_ellipse.cy;

    // Use mean semi-axis as sampling radius (markers are nearly circular).
    // The ellipse angle is NOT used: it reflects the perspective distortion
    // axis, not the board-to-image rotation. Sector angular alignment is
    // handled by the codebook's cyclic rotation matching.
    let r_mean = (outer_ellipse.a + outer_ellipse.b) / 2.0;

    // Sample sector intensities in image coordinates.
    // The absolute angular reference doesn't matter: the codebook matcher
    // tries all 16 cyclic rotations (each 22.5°) to find the best match.
    let sampler = DistortionAwareSampler::new(gray, mapper);
    let mut sector_intensities = [0.0f32; 16];

    for s in 0..16u32 {
        let mut sum = 0.0f32;
        let mut count = 0u32;

        for j in 0..config.samples_per_sector {
            // Angular position within the sector
            let t = (j as f64 + 0.5) / config.samples_per_sector as f64;
            let theta = (s as f64 + t) / 16.0 * 2.0 * std::f64::consts::PI;

            for k in 0..config.n_radial_rings {
                // Radial position within code band (±10% of center ratio)
                let r_ratio = if config.n_radial_rings == 1 {
                    config.code_band_ratio as f64
                } else {
                    let t_r = k as f64 / (config.n_radial_rings - 1) as f64;
                    config.code_band_ratio as f64 * (0.90 + 0.20 * t_r)
                };
                let r = r_ratio * r_mean;

                // Sample in working coordinates (distortion-aware when camera is provided).
                let x_img = cx + r * theta.cos();
                let y_img = cy + r * theta.sin();

                if let Some(intensity) = sampler.sample_checked(x_img as f32, y_img as f32) {
                    sum += intensity;
                    count += 1;
                }
            }
        }

        sector_intensities[s as usize] = if count > 0 { sum / count as f32 } else { 0.5 };
    }

    // Check that there is reasonable contrast
    let (min_intensity, max_intensity) = intensity_range(&sector_intensities);
    let contrast = max_intensity - min_intensity;
    let min_decode_contrast = if config.min_decode_contrast.is_finite() {
        config.min_decode_contrast.max(0.0)
    } else {
        DecodeConfig::DEFAULT_MIN_DECODE_CONTRAST
    };
    if contrast < min_decode_contrast {
        return (
            None,
            DecodeDiagnostics {
                sector_intensities,
                threshold: 0.5 * (min_intensity + max_intensity),
                used_word: 0,
                inverted_used: false,
                best_id: 0,
                best_rotation: 0,
                best_dist: u8::MAX,
                margin: 0,
                decode_confidence: 0.0,
                reject_reason: Some(DecodeRejectReason::LowContrast),
                reject_context: Some(DecodeRejectContext::LowContrast {
                    contrast,
                    min_required_contrast: min_decode_contrast,
                }),
            },
        );
    }

    // Threshold using iterative 2-means clustering (more robust than pure median).
    let threshold = compute_iterative_two_means_threshold(
        &sector_intensities,
        config.threshold_max_iters,
        config.threshold_convergence_eps,
    );

    // Form word: bit=1 if intensity > threshold, bit=0 otherwise
    let mut word: u16 = 0;
    for (s, &intensity) in sector_intensities.iter().enumerate() {
        if intensity > threshold {
            word |= 1 << s;
        }
    }

    // Match against codebook (normal and inverted polarity)
    let cb = Codebook::default();
    let m_normal = cb.match_word(word);
    let m_inverted = cb.match_word(!word);

    // Pick the better match
    let (best_match, used_word, inverted) = if m_normal.confidence >= m_inverted.confidence {
        (m_normal, word, false)
    } else {
        (m_inverted, !word, true)
    };

    // Reject if quality too low
    if best_match.dist > config.max_decode_dist
        || best_match.margin < config.min_decode_margin
        || best_match.confidence < config.min_decode_confidence
    {
        let (reason, context) = if best_match.dist > config.max_decode_dist {
            (
                DecodeRejectReason::DistTooHigh,
                DecodeRejectContext::DistTooHigh {
                    observed_dist: best_match.dist,
                    max_allowed_dist: config.max_decode_dist,
                },
            )
        } else if best_match.margin < config.min_decode_margin {
            (
                DecodeRejectReason::MarginTooLow,
                DecodeRejectContext::MarginTooLow {
                    observed_margin: best_match.margin,
                    min_required_margin: config.min_decode_margin,
                },
            )
        } else {
            (
                DecodeRejectReason::ConfidenceTooLow,
                DecodeRejectContext::ConfidenceTooLow {
                    observed_confidence: best_match.confidence,
                    min_required_confidence: config.min_decode_confidence,
                },
            )
        };
        return (
            None,
            DecodeDiagnostics {
                sector_intensities,
                threshold,
                used_word,
                inverted_used: inverted,
                best_id: best_match.id,
                best_rotation: best_match.rotation,
                best_dist: best_match.dist,
                margin: best_match.margin,
                decode_confidence: best_match.confidence,
                reject_reason: Some(reason),
                reject_context: Some(context),
            },
        );
    }

    let result = DecodeResult {
        id: best_match.id,
        confidence: best_match.confidence,
        raw_word: used_word,
        dist: best_match.dist,
        margin: best_match.margin,
        rotation: best_match.rotation,
    };

    let diag = DecodeDiagnostics {
        sector_intensities,
        threshold,
        used_word,
        inverted_used: inverted,
        best_id: best_match.id,
        best_rotation: best_match.rotation,
        best_dist: best_match.dist,
        margin: best_match.margin,
        decode_confidence: best_match.confidence,
        reject_reason: None,
        reject_context: None,
    };

    (Some(result), diag)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::super::codebook::CODEBOOK;
    use super::*;

    /// Paint a synthetic ring marker with known code on a small image.
    fn make_coded_ring_image(
        w: u32,
        h: u32,
        ellipse: &Ellipse,
        codeword: u16,
        inverted: bool,
    ) -> GrayImage {
        let mut img = GrayImage::new(w, h);
        let cx = ellipse.cx;
        let cy = ellipse.cy;
        let cos_a = ellipse.angle.cos();
        let sin_a = ellipse.angle.sin();

        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;

                // Map to canonical coords
                let xc = cos_a * dx + sin_a * dy;
                let yc = -sin_a * dx + cos_a * dy;
                let xn = xc / ellipse.a;
                let yn = yc / ellipse.b;
                let r = (xn * xn + yn * yn).sqrt();

                let val = if (0.90..=1.10).contains(&r) {
                    // Outer ring: dark
                    30u8
                } else if (0.55..=0.75).contains(&r) {
                    // Inner ring: dark
                    30u8
                } else if (0.76..=0.89).contains(&r) {
                    // Code band
                    let angle = yn.atan2(xn);
                    let sector = ((angle / (2.0 * std::f64::consts::PI) + 0.5) * 16.0) as i32 % 16;
                    let sector = if sector < 0 { sector + 16 } else { sector } as u32;
                    let bit = (codeword >> sector) & 1;
                    let bright = if inverted { bit == 0 } else { bit == 1 };
                    if bright {
                        220u8
                    } else {
                        40u8
                    }
                } else {
                    200u8 // background
                };
                img.put_pixel(x, y, image::Luma([val]));
            }
        }
        img
    }

    #[test]
    fn decode_config_defaults_are_stable() {
        let cfg = DecodeConfig::default();
        assert_abs_diff_eq!(
            cfg.code_band_ratio,
            DecodeConfig::DEFAULT_CODE_BAND_RATIO,
            epsilon = 1e-6
        );
        assert_eq!(
            cfg.samples_per_sector,
            DecodeConfig::DEFAULT_SAMPLES_PER_SECTOR
        );
        assert_eq!(cfg.n_radial_rings, DecodeConfig::DEFAULT_N_RADIAL_RINGS);
        assert_eq!(cfg.max_decode_dist, DecodeConfig::DEFAULT_MAX_DECODE_DIST);
        assert_abs_diff_eq!(
            cfg.min_decode_confidence,
            DecodeConfig::DEFAULT_MIN_DECODE_CONFIDENCE,
            epsilon = 1e-6
        );
        assert_eq!(
            cfg.min_decode_margin,
            DecodeConfig::DEFAULT_MIN_DECODE_MARGIN
        );
        assert_abs_diff_eq!(
            cfg.min_decode_contrast,
            DecodeConfig::DEFAULT_MIN_DECODE_CONTRAST,
            epsilon = 1e-6
        );
        assert_eq!(
            cfg.threshold_max_iters,
            DecodeConfig::DEFAULT_THRESHOLD_MAX_ITERS
        );
        assert_abs_diff_eq!(
            cfg.threshold_convergence_eps,
            DecodeConfig::DEFAULT_THRESHOLD_CONVERGENCE_EPS,
            epsilon = 1e-8
        );
    }

    #[test]
    fn decode_config_deserialize_missing_hidden_threshold_fields_uses_defaults() {
        let json = r#"{
            "code_band_ratio": 0.76,
            "samples_per_sector": 5,
            "n_radial_rings": 3,
            "max_decode_dist": 3,
            "min_decode_confidence": 0.15
        }"#;

        let cfg: DecodeConfig =
            serde_json::from_str(json).expect("decode config json should parse");
        assert_abs_diff_eq!(
            cfg.min_decode_contrast,
            DecodeConfig::DEFAULT_MIN_DECODE_CONTRAST,
            epsilon = 1e-6
        );
        assert_eq!(
            cfg.min_decode_margin,
            DecodeConfig::DEFAULT_MIN_DECODE_MARGIN
        );
        assert_eq!(
            cfg.threshold_max_iters,
            DecodeConfig::DEFAULT_THRESHOLD_MAX_ITERS
        );
        assert_abs_diff_eq!(
            cfg.threshold_convergence_eps,
            DecodeConfig::DEFAULT_THRESHOLD_CONVERGENCE_EPS,
            epsilon = 1e-8
        );
    }

    #[test]
    fn test_decode_known_codeword() {
        let cw = CODEBOOK[42];
        let ellipse = Ellipse {
            cx: 40.0,
            cy: 40.0,
            a: 15.0,
            b: 15.0,
            angle: 0.0,
        };
        let img = make_coded_ring_image(80, 80, &ellipse, cw, false);
        let config = DecodeConfig::default();

        let (result, diag) =
            decode_marker_with_diagnostics_and_mapper(&img, &ellipse, &config, None);
        assert!(result.is_some(), "should decode successfully");
        let result = result.unwrap();
        assert_eq!(result.id, 42, "decoded id should be 42, got {}", result.id);
        assert!(!diag.inverted_used, "should not use inverted polarity");
    }

    #[test]
    fn test_decode_inverted_polarity() {
        // Use codeword 0 whose complement is NOT a rotation of any codebook entry
        let cw = CODEBOOK[0];
        let ellipse = Ellipse {
            cx: 40.0,
            cy: 40.0,
            a: 15.0,
            b: 15.0,
            angle: 0.0,
        };
        // Paint with inverted polarity
        let img = make_coded_ring_image(80, 80, &ellipse, cw, true);
        let config = DecodeConfig::default();

        let (result, diag) =
            decode_marker_with_diagnostics_and_mapper(&img, &ellipse, &config, None);
        assert!(
            result.is_some(),
            "should decode successfully with inverted polarity"
        );
        let result = result.unwrap();
        assert_eq!(result.id, 0, "decoded id should be 0, got {}", result.id);
        assert!(diag.inverted_used, "should use inverted polarity");
    }

    #[test]
    fn decode_reject_reason_serialization_is_stable() {
        let reason = DecodeRejectReason::LowContrast;
        assert_eq!(reason.to_string(), "low_contrast");
        let json = serde_json::to_string(&reason).expect("serialize reject reason");
        assert_eq!(json, "\"low_contrast\"");
    }

    #[test]
    fn decode_low_contrast_reports_typed_reason_with_context() {
        let img = GrayImage::from_pixel(64, 64, image::Luma([128u8]));
        let ellipse = Ellipse {
            cx: 32.0,
            cy: 32.0,
            a: 14.0,
            b: 14.0,
            angle: 0.0,
        };
        let config = DecodeConfig {
            min_decode_contrast: 0.2,
            ..DecodeConfig::default()
        };
        let (result, diag) =
            decode_marker_with_diagnostics_and_mapper(&img, &ellipse, &config, None);
        assert!(result.is_none());
        assert_eq!(diag.reject_reason, Some(DecodeRejectReason::LowContrast));
        match diag.reject_context {
            Some(DecodeRejectContext::LowContrast {
                contrast,
                min_required_contrast,
            }) => {
                assert!(contrast < min_required_contrast);
                assert_abs_diff_eq!(contrast, 0.0, epsilon = 1e-6);
                assert_abs_diff_eq!(min_required_contrast, 0.2, epsilon = 1e-6);
            }
            other => panic!("unexpected reject context: {other:?}"),
        }
    }

    #[test]
    fn threshold_loop_respects_iteration_guard() {
        let sector_intensities = [
            0.898, 0.923, 0.541, 0.391, 0.705, 0.276, 0.812, 0.849, 0.895, 0.59, 0.95, 0.58, 0.451,
            0.66, 0.996, 0.917,
        ];

        let t_one_iter = compute_iterative_two_means_threshold(&sector_intensities, 1, 1e-6);
        let t_converged = compute_iterative_two_means_threshold(&sector_intensities, 10, 1e-6);

        assert_abs_diff_eq!(t_one_iter, 0.666, epsilon = 1e-6);
        assert_abs_diff_eq!(t_converged, 0.6906032, epsilon = 1e-6);
        assert!(t_converged > t_one_iter);
    }

    #[test]
    fn threshold_loop_respects_convergence_epsilon_guard() {
        let sector_intensities = [
            0.898, 0.923, 0.541, 0.391, 0.705, 0.276, 0.812, 0.849, 0.895, 0.59, 0.95, 0.58, 0.451,
            0.66, 0.996, 0.917,
        ];

        let t_loose = compute_iterative_two_means_threshold(&sector_intensities, 10, 0.05);
        let t_tight = compute_iterative_two_means_threshold(&sector_intensities, 10, 1e-6);

        assert_abs_diff_eq!(t_loose, 0.666, epsilon = 1e-6);
        assert_abs_diff_eq!(t_tight, 0.6906032, epsilon = 1e-6);
        assert!(t_tight > t_loose);
    }

    #[test]
    fn decode_confidence_gate_reports_typed_reason_with_context() {
        let cw = CODEBOOK[7];
        let ellipse = Ellipse {
            cx: 40.0,
            cy: 40.0,
            a: 15.0,
            b: 15.0,
            angle: 0.0,
        };
        let img = make_coded_ring_image(80, 80, &ellipse, cw, false);
        let config = DecodeConfig {
            min_decode_confidence: 1.1,
            ..DecodeConfig::default()
        };

        let (result, diag) =
            decode_marker_with_diagnostics_and_mapper(&img, &ellipse, &config, None);
        assert!(result.is_none(), "confidence gate should reject");
        assert_eq!(
            diag.reject_reason,
            Some(DecodeRejectReason::ConfidenceTooLow)
        );
        match diag.reject_context {
            Some(DecodeRejectContext::ConfidenceTooLow {
                observed_confidence,
                min_required_confidence,
            }) => {
                assert!(observed_confidence < min_required_confidence);
            }
            other => panic!("unexpected reject context: {other:?}"),
        }
    }
}
