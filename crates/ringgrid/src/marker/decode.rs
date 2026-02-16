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
    pub decode_confidence: f32,
}

/// Configuration for sector decoding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeConfig {
    /// Ratio of code band center radius to outer ellipse semi-major axis.
    /// The code band is sampled at `code_band_ratio * (a, b)` in the
    /// ellipse coordinate frame. Default: 0.82
    pub code_band_ratio: f32,
    /// Number of angular samples per sector. Default: 3.
    pub samples_per_sector: usize,
    /// Number of radial rings to sample within the code band. Default: 2.
    pub n_radial_rings: usize,
    /// Maximum Hamming distance for a valid decode. Default: 3.
    pub max_decode_dist: u8,
    /// Minimum confidence for a valid decode. Default: 0.2.
    pub min_decode_confidence: f32,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            code_band_ratio: 0.76,
            samples_per_sector: 5,
            n_radial_rings: 3,
            max_decode_dist: 3,
            min_decode_confidence: 0.15,
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
    /// If rejected, why.
    pub reject_reason: Option<String>,
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
                reject_reason: Some("invalid_ellipse".to_string()),
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

    // Threshold using iterative 2-means clustering (more robust than pure median)
    let mut sorted = sector_intensities;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Check that there is reasonable contrast
    let contrast = sorted[15] - sorted[0];
    if contrast < 0.03 {
        return (
            None,
            DecodeDiagnostics {
                sector_intensities,
                threshold: (sorted[0] + sorted[15]) / 2.0,
                used_word: 0,
                inverted_used: false,
                best_id: 0,
                best_rotation: 0,
                best_dist: u8::MAX,
                margin: 0,
                decode_confidence: 0.0,
                reject_reason: Some("low_contrast".to_string()),
            },
        );
    }

    // Initial threshold: midpoint of range
    let mut threshold = (sorted[0] + sorted[15]) / 2.0;
    for _ in 0..10 {
        let (mut sum_lo, mut cnt_lo) = (0.0f32, 0u32);
        let (mut sum_hi, mut cnt_hi) = (0.0f32, 0u32);
        for &v in &sector_intensities {
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
        let new_threshold = (sum_lo / cnt_lo as f32 + sum_hi / cnt_hi as f32) / 2.0;
        if (new_threshold - threshold).abs() < 1e-4 {
            break;
        }
        threshold = new_threshold;
    }

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
        || best_match.confidence < config.min_decode_confidence
    {
        let reason = if best_match.dist > config.max_decode_dist {
            format!(
                "dist_too_high({}>{})",
                best_match.dist, config.max_decode_dist
            )
        } else {
            format!(
                "confidence_too_low({:.3}<{:.3})",
                best_match.confidence, config.min_decode_confidence
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
    };

    (Some(result), diag)
}

#[cfg(test)]
mod tests {
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
}
