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
use crate::codec::Codebook;

use super::edge_sample::bilinear_sample_u8;

/// Configuration for sector decoding.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
    /// Per-sector average intensities (for debug).
    pub sector_intensities: [f32; 16],
    /// Whether the inverted polarity was used.
    pub inverted: bool,
}

/// Decode a marker ID from the image using the fitted outer ellipse.
///
/// Samples sector intensities along a circle at the code band radius,
/// centered on the ellipse center. Uses circular (not elliptical) sampling
/// because the ellipse orientation axis does not correspond to the
/// board-to-image rotation — it reflects perspective distortion instead.
/// The codebook matcher handles the unknown rotation via cyclic matching
/// over all 16 sector offsets.
///
/// Returns `None` if decoding quality is insufficient.
pub fn decode_marker(
    gray: &GrayImage,
    outer_ellipse: &Ellipse,
    config: &DecodeConfig,
) -> Option<DecodeResult> {
    // Validate ellipse
    if !outer_ellipse.is_valid() || outer_ellipse.a < 2.0 || outer_ellipse.b < 2.0 {
        return None;
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

                // Sample directly in image coordinates (circular frame)
                let x_img = cx + r * theta.cos();
                let y_img = cy + r * theta.sin();

                let intensity = bilinear_sample_u8(gray, x_img as f32, y_img as f32);
                if intensity > 0.0 {
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
        return None;
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
    for s in 0..16 {
        if sector_intensities[s] > threshold {
            word |= 1 << s;
        }
    }

    // Match against codebook (normal and inverted polarity)
    let cb = Codebook::default();
    let m_normal = cb.match_word(word);
    let m_inverted = cb.match_word(!word);

    // Pick the better match
    let (best_match, used_word, inverted) =
        if m_normal.confidence >= m_inverted.confidence {
            (m_normal, word, false)
        } else {
            (m_inverted, !word, true)
        };

    // Reject if quality too low
    if best_match.dist > config.max_decode_dist
        || best_match.confidence < config.min_decode_confidence
    {
        return None;
    }

    Some(DecodeResult {
        id: best_match.id,
        confidence: best_match.confidence,
        raw_word: used_word,
        dist: best_match.dist,
        margin: best_match.margin,
        rotation: best_match.rotation,
        sector_intensities,
        inverted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::CODEBOOK;

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

                let val = if r >= 0.90 && r <= 1.10 {
                    // Outer ring: dark
                    30u8
                } else if r >= 0.55 && r <= 0.75 {
                    // Inner ring: dark
                    30u8
                } else if r >= 0.76 && r <= 0.89 {
                    // Code band
                    let angle = yn.atan2(xn);
                    let sector =
                        ((angle / (2.0 * std::f64::consts::PI) + 0.5) * 16.0) as i32 % 16;
                    let sector = if sector < 0 { sector + 16 } else { sector } as u32;
                    let bit = (codeword >> sector) & 1;
                    let bright = if inverted { bit == 0 } else { bit == 1 };
                    if bright { 220u8 } else { 40u8 }
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

        let result = decode_marker(&img, &ellipse, &config);
        assert!(result.is_some(), "should decode successfully");
        let result = result.unwrap();
        assert_eq!(result.id, 42, "decoded id should be 42, got {}", result.id);
        assert!(!result.inverted, "should not use inverted polarity");
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

        let result = decode_marker(&img, &ellipse, &config);
        assert!(result.is_some(), "should decode successfully with inverted polarity");
        let result = result.unwrap();
        assert_eq!(result.id, 0, "decoded id should be 0, got {}", result.id);
        assert!(result.inverted, "should use inverted polarity");
    }
}
