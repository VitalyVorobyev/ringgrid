//! Elliptical 16-sector sampling and exact matching against the target's
//! codewords.
//!
//! **Geometry.** The board's sectors are equal-angle in board space; under the
//! affine approximation of the fitted outer ellipse that corresponds to
//! uniform *parametric* angle on the ellipse, so the code band is sampled at
//! uniform parametric angle on the ellipse scaled by the code-band ratio (a
//! circle of mean radius drifts off the elliptical band on tilted views).
//!
//! **Rotation phase.** Where the printed sector boundaries land in parametric
//! angle is a *continuous* unknown: board rotation plus the ellipse's own
//! angle origin. Cyclic codebook matching absorbs only whole sectors. A
//! fractional offset makes every fixed sampling sector straddle two printed
//! ones — on the rtv3d captures 89 % of the decodes accepted at an offset of
//! 0.4–0.5 sector were wrong (ADR-027). The band is therefore sampled densely
//! and the sub-sector phase is estimated first: the phase whose sector means,
//! each taken over the central half of its sector, vary the most is the one
//! aligned with the printed boundaries. The reported rotation is continuous
//! (whole sectors plus that phase).
//!
//! **Polarity.** Targets are printed dark on light, so the sampled word is
//! matched as is. Trying the complement as a fallback only ever produced
//! misreads (all 508 inverted decodes on rtv3d were wrong).
//!
//! **Matching.** The word is matched against the codewords of the target's own
//! cells and accepted only on an exact match. The embedded profiles have
//! minimum cyclic Hamming distance 2 (base) or 1 (extended): 99.9 % of all
//! 16-bit words lie within one bit of a base codeword, so a word one bit off
//! a codeword is no evidence for it, and a distance tolerance only admits
//! noise.

use image::GrayImage;

use crate::conic::Ellipse;
use crate::pixelmap::PixelMapper;
use crate::ring::edge_sample::DistortionAwareSampler;
use crate::target::{MarkerCoding, TargetLayout};

use super::codec::{Codebook, Match};

/// Number of code sectors (bits) per marker.
const SECTORS: usize = 16;

/// Decode metrics for a decoded marker.
///
/// Only exact matches against the target's codewords are accepted, so a
/// decode carries no distance or margin: the observed word *is* the codeword
/// of `best_id`, rotated.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct DecodeMetrics {
    /// 16-bit word sampled from the code band (bit `s` = sector `s` of the
    /// phase-aligned sampling grid; 1 = bright).
    pub observed_word: u16,
    /// Codebook ID of the matched codeword.
    pub best_id: usize,
    /// Rotation of the printed code in sectors, in `[0, 16)`: printed sector
    /// `j` starts at parametric angle `(j + rotation) · 2π/16` on the fitted
    /// outer ellipse (whose parametric angle 0 is its major axis). The whole
    /// part is the cyclic shift that maps the codeword onto `observed_word`;
    /// the fraction is the estimated sub-sector phase.
    pub rotation: f32,
}

/// Configuration for sector decoding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default)]
#[non_exhaustive]
pub struct DecodeConfig {
    /// Ratio of code band center radius to outer ellipse semi-major axis.
    /// The code band is sampled at `code_band_ratio * (a, b)` in the
    /// ellipse coordinate frame.
    /// Derived from the target's ring radii and stroke width; a value supplied in JSON is overwritten when the config is loaded.
    /// Default: `0.76` (see the `Default` impl).
    #[cfg_attr(feature = "schemars", schemars(extend("readOnly" = true)))]
    pub code_band_ratio: f32,
    /// Angular samples per sector. Also the number of sub-sector phases tried
    /// when aligning the sampling grid with the printed sector boundaries
    /// (phase step `1 / samples_per_sector` sector). Clamped to `[2, 32]`.
    /// Default: `8` (see the `Default` impl).
    pub samples_per_sector: usize,
    /// Number of radial rings to sample within the code band.
    /// Default: `3` (see the `Default` impl).
    pub n_radial_rings: usize,
    /// Minimum accepted sector-intensity contrast (`max - min`, in normalized 0 to 1 intensity units) before decode; negative values are treated as 0.
    /// Default: `0.03` (see the `Default` impl).
    #[serde(default = "DecodeConfig::default_min_decode_contrast")]
    #[cfg_attr(feature = "schemars", schemars(range(min = 0.0)))]
    pub min_decode_contrast: f32,
    /// Maximum iterations for iterative 2-means threshold refinement.
    /// Default: `10` (see the `Default` impl).
    #[serde(default = "DecodeConfig::default_threshold_max_iters")]
    pub threshold_max_iters: usize,
    /// Convergence epsilon for iterative 2-means threshold refinement.
    /// Stop when `|new_threshold - old_threshold| <= eps`.
    /// Default: `1e-4` (see the `Default` impl).
    #[serde(default = "DecodeConfig::default_threshold_convergence_eps")]
    pub threshold_convergence_eps: f32,
}

impl DecodeConfig {
    /// Default radial location of the code band as a fraction of outer radius.
    pub(crate) const DEFAULT_CODE_BAND_RATIO: f32 = 0.76;
    /// Default number of angular intensity samples (and phases) per sector.
    pub(crate) const DEFAULT_SAMPLES_PER_SECTOR: usize = 8;
    /// Default number of concentric rings used for radial sampling.
    pub(crate) const DEFAULT_N_RADIAL_RINGS: usize = 3;
    /// Default minimum sector-intensity contrast (`max - min`) before decode.
    pub(crate) const DEFAULT_MIN_DECODE_CONTRAST: f32 = 0.03;
    /// Default iteration cap for iterative 2-means threshold refinement.
    pub(crate) const DEFAULT_THRESHOLD_MAX_ITERS: usize = 10;
    /// Default convergence epsilon for iterative 2-means threshold refinement.
    pub(crate) const DEFAULT_THRESHOLD_CONVERGENCE_EPS: f32 = 1e-4;
    /// Bounds of the effective `samples_per_sector`: at least two phases, and
    /// a dense ring of at most 512 samples.
    const SAMPLES_PER_SECTOR_RANGE: (usize, usize) = (2, 32);

    fn default_min_decode_contrast() -> f32 {
        Self::DEFAULT_MIN_DECODE_CONTRAST
    }

    fn default_threshold_max_iters() -> usize {
        Self::DEFAULT_THRESHOLD_MAX_ITERS
    }

    fn default_threshold_convergence_eps() -> f32 {
        Self::DEFAULT_THRESHOLD_CONVERGENCE_EPS
    }

    fn effective_samples_per_sector(&self) -> usize {
        let (lo, hi) = Self::SAMPLES_PER_SECTOR_RANGE;
        self.samples_per_sector.clamp(lo, hi)
    }
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            code_band_ratio: Self::DEFAULT_CODE_BAND_RATIO,
            samples_per_sector: Self::DEFAULT_SAMPLES_PER_SECTOR,
            n_radial_rings: Self::DEFAULT_N_RADIAL_RINGS,
            min_decode_contrast: Self::DEFAULT_MIN_DECODE_CONTRAST,
            threshold_max_iters: Self::DEFAULT_THRESHOLD_MAX_ITERS,
            threshold_convergence_eps: Self::DEFAULT_THRESHOLD_CONVERGENCE_EPS,
        }
    }
}

/// The codewords a coded target carries: the codebook profile its coding
/// names, restricted to the IDs of its cells. Exists only for coded targets.
pub(crate) struct TargetCodes<'a> {
    codebook: Codebook,
    target: &'a TargetLayout,
}

impl<'a> TargetCodes<'a> {
    /// The target's codewords, or `None` for a plain target.
    pub(crate) fn of(target: &'a TargetLayout) -> Option<Self> {
        match target.coding() {
            MarkerCoding::Coded16(spec) => Some(Self {
                codebook: Codebook::from_profile(spec.codebook_profile),
                target,
            }),
            MarkerCoding::Plain => None,
        }
    }

    /// Nearest of the target's codewords (over all rotations) to `word`.
    fn match_word(&self, word: u16) -> Match {
        self.codebook
            .match_word_among(word, self.target.marker_ids())
    }
}

/// Result of decoding a marker: an exact match against one of the target's
/// codewords.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeResult {
    /// Matched marker ID (codebook index).
    pub id: usize,
    /// 16-bit word sampled from the image (phase-aligned grid).
    pub raw_word: u16,
    /// Continuous rotation in sectors (see [`DecodeMetrics::rotation`]).
    pub rotation: f32,
}

/// Stable reject code for a decode attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecodeRejectReason {
    InvalidEllipse,
    LowContrast,
    /// The sampled word is not (a rotation of) any of the target's codewords.
    NotACodeword,
}

impl DecodeRejectReason {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::InvalidEllipse => "invalid_ellipse",
            Self::LowContrast => "low_contrast",
            Self::NotACodeword => "not_a_codeword",
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
    NotACodeword {
        nearest_dist: u8,
    },
}

/// Debug/diagnostic information about a decode attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecodeDiagnostics {
    /// Per-sector average intensities sampled from the code band (at `phase`).
    pub sector_intensities: [f32; 16],
    /// Sub-sector phase of the sampling grid: where its sector 0 starts, in
    /// sectors of parametric angle.
    pub phase: f32,
    /// Threshold used to binarize sector intensities.
    pub threshold: f32,
    /// Binarized word matched against the target's codewords.
    pub word: u16,
    /// Nearest target codeword ID.
    pub best_id: usize,
    /// Rotation (in whole sectors) of the nearest codeword.
    pub best_rotation: u8,
    /// Hamming distance to the nearest target codeword (0 when accepted).
    pub best_dist: u8,
    /// Graded match quality in `[0, 1]` (distance and margin to the nearest
    /// target codeword): evidence for ranking outer-fit hypotheses, also for
    /// rejected words.
    pub decode_confidence: f32,
    /// Stable reject code, if rejected.
    pub reject_reason: Option<DecodeRejectReason>,
    /// Structured reject diagnostics, if rejected.
    pub reject_context: Option<DecodeRejectContext>,
}

impl DecodeDiagnostics {
    fn rejected(
        sector_intensities: [f32; 16],
        phase: f32,
        threshold: f32,
        reason: DecodeRejectReason,
        context: DecodeRejectContext,
    ) -> Self {
        Self {
            sector_intensities,
            phase,
            threshold,
            word: 0,
            best_id: 0,
            best_rotation: 0,
            best_dist: u8::MAX,
            decode_confidence: 0.0,
            reject_reason: Some(reason),
            reject_context: Some(context),
        }
    }
}

/// Decode a marker against the target's codewords and return
/// `(accepted_result, diagnostics)`.
///
/// Uses an optional working<->image mapper for distortion-aware sampling.
pub(crate) fn decode_marker_with_diagnostics_and_mapper(
    gray: &GrayImage,
    outer_ellipse: &Ellipse,
    config: &DecodeConfig,
    codes: &TargetCodes<'_>,
    mapper: Option<&dyn PixelMapper>,
) -> (Option<DecodeResult>, DecodeDiagnostics) {
    if !outer_ellipse.is_valid() || outer_ellipse.a < 2.0 || outer_ellipse.b < 2.0 {
        let diag = DecodeDiagnostics::rejected(
            [0.0; 16],
            0.0,
            0.0,
            DecodeRejectReason::InvalidEllipse,
            DecodeRejectContext::InvalidEllipse {
                is_valid: outer_ellipse.is_valid(),
                semi_axis_a: outer_ellipse.a,
                semi_axis_b: outer_ellipse.b,
                min_allowed_semi_axis: 2.0,
            },
        );
        return (None, diag);
    }
    let sampler = DistortionAwareSampler::new(gray, mapper);
    let band = sample_code_band(&sampler, outer_ellipse, config);
    threshold_and_match(band, config, codes)
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

/// Phase-aligned sector means of the code band.
struct CodeBand {
    /// Mean intensity per sector of the phase-aligned grid.
    sectors: [f32; 16],
    /// Where grid sector 0 starts, in sectors of parametric angle (the
    /// sub-sector phase; within `(-1/2, 1)`).
    phase: f32,
}

/// Sample the code band densely and align the sector grid with the printed
/// sector boundaries.
///
/// `n = samples_per_sector` samples per sector at uniform parametric angle
/// (each averaged over `n_radial_rings` radii across the band) form a ring of
/// `16·n` values. For each of the `n` sub-sector phases the sector means are
/// taken over the central half of each sector; the phase with the largest
/// variance of the 16 means is the one whose sectors do not straddle printed
/// boundaries (a straddling sector averages a bright and a dark sector toward
/// the middle).
fn sample_code_band(
    sampler: &DistortionAwareSampler<'_>,
    ellipse: &Ellipse,
    config: &DecodeConfig,
) -> CodeBand {
    let n = config.effective_samples_per_sector();
    let n_rad = config.n_radial_rings.max(1);
    let (sin_rot, cos_rot) = ellipse.angle.sin_cos();
    let radial_ratio = |k: usize| -> f64 {
        let r = f64::from(config.code_band_ratio);
        if n_rad == 1 {
            r
        } else {
            r * (0.90 + 0.20 * k as f64 / (n_rad - 1) as f64)
        }
    };

    // (sum, count) of the valid samples at each dense angular position.
    let n_dense = SECTORS * n;
    let dense: Vec<(f32, u32)> = (0..n_dense)
        .map(|i| {
            let theta = (i as f64 + 0.5) / n_dense as f64 * std::f64::consts::TAU;
            let px = ellipse.a * theta.cos();
            let py = ellipse.b * theta.sin();
            (0..n_rad).fold((0.0f32, 0u32), |(sum, count), k| {
                let r = radial_ratio(k);
                let x = ellipse.cx + r * (cos_rot * px - sin_rot * py);
                let y = ellipse.cy + r * (sin_rot * px + cos_rot * py);
                match sampler.sample_checked(x as f32, y as f32) {
                    Some(v) => (sum + v, count + 1),
                    None => (sum, count),
                }
            })
        })
        .collect();

    let core = (n / 2).max(1);
    let core_start = (n - core) / 2;
    let sectors_at = |p: usize| -> [f32; 16] {
        std::array::from_fn(|s| {
            let (sum, count) = (0..core)
                .map(|j| dense[(s * n + p + core_start + j) % n_dense])
                .fold((0.0f32, 0u32), |(a, c), (v, k)| (a + v, c + k));
            if count > 0 { sum / count as f32 } else { 0.5 }
        })
    };
    let variance = |v: &[f32; 16]| -> f32 {
        let mean = v.iter().sum::<f32>() / 16.0;
        v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>()
    };

    // Every phase whose central half stays inside the printed sectors scores
    // the same on a sharp band; smoothing the score over neighbouring phases
    // (cyclic: phase `n` is phase 0 with the sectors relabelled) picks the
    // middle of that plateau, the phase farthest from the boundaries.
    let var: Vec<f32> = (0..n).map(|p| variance(&sectors_at(p))).collect();
    let smoothed = |p: usize| var[(p + n - 1) % n] + 2.0 * var[p] + var[(p + 1) % n];
    let best_p = (1..n).fold(0usize, |best, p| {
        if smoothed(p) > smoothed(best) {
            p
        } else {
            best
        }
    });
    // Grid sector `s` averages samples centred on `s + centre` sectors, so it
    // starts at `s + centre - 1/2`.
    let centre = (best_p as f32 + core_start as f32 + 0.5 * core as f32) / n as f32;
    CodeBand {
        sectors: sectors_at(best_p),
        phase: centre - 0.5,
    }
}

/// Binarize the sector means, match the word against the target's codewords
/// and accept an exact match.
fn threshold_and_match(
    band: CodeBand,
    config: &DecodeConfig,
    codes: &TargetCodes<'_>,
) -> (Option<DecodeResult>, DecodeDiagnostics) {
    let CodeBand { sectors, phase } = band;
    let (min_intensity, max_intensity) = intensity_range(&sectors);
    let contrast = max_intensity - min_intensity;
    let min_decode_contrast = if config.min_decode_contrast.is_finite() {
        config.min_decode_contrast.max(0.0)
    } else {
        DecodeConfig::DEFAULT_MIN_DECODE_CONTRAST
    };
    if contrast < min_decode_contrast {
        let diag = DecodeDiagnostics::rejected(
            sectors,
            phase,
            0.5 * (min_intensity + max_intensity),
            DecodeRejectReason::LowContrast,
            DecodeRejectContext::LowContrast {
                contrast,
                min_required_contrast: min_decode_contrast,
            },
        );
        return (None, diag);
    }

    let threshold = compute_iterative_two_means_threshold(
        &sectors,
        config.threshold_max_iters,
        config.threshold_convergence_eps,
    );
    let word = sectors
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v > threshold)
        .fold(0u16, |w, (s, _)| w | (1 << s));

    let m = codes.match_word(word);
    let exact = m.dist == 0;
    let diag = DecodeDiagnostics {
        sector_intensities: sectors,
        phase,
        threshold,
        word,
        best_id: m.id,
        best_rotation: m.rotation,
        best_dist: m.dist,
        decode_confidence: m.confidence,
        reject_reason: (!exact).then_some(DecodeRejectReason::NotACodeword),
        reject_context: (!exact).then_some(DecodeRejectContext::NotACodeword {
            nearest_dist: m.dist,
        }),
    };
    let result = exact.then(|| DecodeResult {
        id: m.id,
        raw_word: word,
        rotation: (f32::from(m.rotation) + phase).rem_euclid(SECTORS as f32),
    });
    (result, diag)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::super::codebook::{CODEBOOK, CODEBOOK_EXTENDED, CODEBOOK_N};
    use super::super::codec::CodebookProfile;
    use super::*;
    use crate::target::CodedRingSpec;

    /// Paint a ring marker with `codeword` on its code band. Printed sector
    /// `j` spans parametric angle `[-π + (j + offset)·2π/16, …)`, i.e. the
    /// decoder should report rotation `8 + offset`.
    fn paint_marker(ellipse: &Ellipse, codeword: u16, offset: f64, inverted: bool) -> GrayImage {
        let (w, h) = (80, 80);
        let mut img = GrayImage::new(w, h);
        let (sin_a, cos_a) = ellipse.angle.sin_cos();
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - ellipse.cx;
                let dy = y as f64 - ellipse.cy;
                let xn = (cos_a * dx + sin_a * dy) / ellipse.a;
                let yn = (-sin_a * dx + cos_a * dy) / ellipse.b;
                let r = xn.hypot(yn);
                let val = if (0.90..=1.10).contains(&r) || (0.55..=0.75).contains(&r) {
                    30u8
                } else if (0.75..0.90).contains(&r) {
                    let u = (yn.atan2(xn) / std::f64::consts::TAU + 0.5) * 16.0 - offset;
                    let sector = (u.floor() as i64).rem_euclid(16) as u32;
                    let bright = ((codeword >> sector) & 1 == 1) != inverted;
                    if bright { 220 } else { 40 }
                } else {
                    200
                };
                img.put_pixel(x, y, image::Luma([val]));
            }
        }
        img
    }

    fn circle() -> Ellipse {
        Ellipse {
            cx: 40.0,
            cy: 40.0,
            a: 15.0,
            b: 15.0,
            angle: 0.0,
        }
    }

    /// Sampling at the painted band's centre (0.825 of the outer radius).
    fn config() -> DecodeConfig {
        DecodeConfig {
            code_band_ratio: 0.825,
            ..DecodeConfig::default()
        }
    }

    fn decode(
        img: &GrayImage,
        ellipse: &Ellipse,
        target: &TargetLayout,
    ) -> (Option<DecodeResult>, DecodeDiagnostics) {
        let codes = TargetCodes::of(target).expect("coded target");
        decode_marker_with_diagnostics_and_mapper(img, ellipse, &config(), &codes, None)
    }

    fn rotation_error(rotation: f32, expected: f64) -> f64 {
        let d = (f64::from(rotation) - expected).rem_euclid(16.0);
        d.min(16.0 - d)
    }

    #[test]
    fn decode_config_defaults_are_stable() {
        let cfg = DecodeConfig::default();
        assert_abs_diff_eq!(cfg.code_band_ratio, 0.76, epsilon = 1e-6);
        assert_eq!(cfg.samples_per_sector, 8);
        assert_eq!(cfg.n_radial_rings, 3);
        assert_abs_diff_eq!(cfg.min_decode_contrast, 0.03, epsilon = 1e-6);
        assert_eq!(cfg.threshold_max_iters, 10);
        assert_abs_diff_eq!(cfg.threshold_convergence_eps, 1e-4, epsilon = 1e-8);
    }

    #[test]
    fn decode_config_deserialize_fills_missing_fields() {
        let json = r#"{ "code_band_ratio": 0.76, "n_radial_rings": 3 }"#;
        let cfg: DecodeConfig = serde_json::from_str(json).expect("decode config json");
        assert_eq!(
            cfg.samples_per_sector,
            DecodeConfig::DEFAULT_SAMPLES_PER_SECTOR
        );
        assert_abs_diff_eq!(
            cfg.min_decode_contrast,
            DecodeConfig::DEFAULT_MIN_DECODE_CONTRAST,
            epsilon = 1e-6
        );
    }

    #[test]
    fn decodes_known_codeword_with_its_rotation() {
        let target = TargetLayout::default();
        let img = paint_marker(&circle(), CODEBOOK[42], 0.0, false);
        let (result, diag) = decode(&img, &circle(), &target);
        let result = result.expect("exact codeword decodes");
        assert_eq!(result.id, 42);
        assert_eq!(diag.best_dist, 0);
        assert!(
            rotation_error(result.rotation, 8.0) < 0.15,
            "{}",
            result.rotation
        );
    }

    /// The printed sector boundaries may land anywhere in parametric angle.
    /// A fixed sampling grid half a sector off averages every sector with its
    /// neighbour; the phase-aligned grid decodes every offset and reports the
    /// continuous rotation.
    #[test]
    fn decodes_at_every_sub_sector_phase_offset() {
        let target = TargetLayout::default();
        for id in [0usize, 42, 101, 202] {
            for step in 0..=10 {
                let offset = step as f64 * 0.05;
                let img = paint_marker(&circle(), CODEBOOK[id], offset, false);
                let (result, diag) = decode(&img, &circle(), &target);
                let result = result.unwrap_or_else(|| {
                    panic!("id {id} offset {offset}: rejected {:?}", diag.reject_reason)
                });
                assert_eq!(result.id, id, "id {id} offset {offset}");
                assert!(
                    rotation_error(result.rotation, 8.0 + offset) < 0.2,
                    "id {id} offset {offset}: rotation {}",
                    result.rotation
                );
            }
        }
    }

    #[test]
    fn eccentric_marker_decodes() {
        // A circle of mean radius would leave the code band near both axes of
        // this strongly eccentric ellipse; parametric sampling stays on it.
        let ellipse = Ellipse {
            cx: 40.0,
            cy: 40.0,
            a: 18.0,
            b: 10.0,
            angle: 0.5,
        };
        let target = TargetLayout::default();
        let img = paint_marker(&ellipse, CODEBOOK[123], 0.3, false);
        let (result, _) = decode(&img, &ellipse, &target);
        assert_eq!(result.expect("eccentric marker decodes").id, 123);
    }

    /// Targets are printed dark on light: a light-on-dark code band is never
    /// read as the painted ID (its complement may be another board codeword).
    /// A few codewords are their own complement up to rotation (e.g. ID 84);
    /// no polarity rule can tell those apart, so they are skipped.
    #[test]
    fn inverted_code_band_is_not_decoded_as_the_painted_id() {
        let target = TargetLayout::default();
        let self_complementary =
            |w: u16| (0..16).any(|k| super::super::codec::rotate_left_16(w, k) == !w);
        let mut n_checked = 0;
        for (id, &cw) in CODEBOOK.iter().enumerate().take(target.n_cells()) {
            if self_complementary(cw) {
                continue;
            }
            n_checked += 1;
            let img = paint_marker(&circle(), cw, 0.0, true);
            let (result, _) = decode(&img, &circle(), &target);
            assert_ne!(result.map(|r| r.id), Some(id), "id {id}");
        }
        assert!(n_checked > 190, "only {n_checked} codewords checked");
    }

    /// Only the target's own codewords match: a profile codeword that no cell
    /// carries is rejected, not decoded to itself.
    #[test]
    fn codewords_off_the_target_are_rejected() {
        let target = TargetLayout::default();
        assert!(target.coord_of_id(500).is_none());
        let img = paint_marker(&circle(), CODEBOOK[500], 0.0, false);
        let (result, diag) = decode(&img, &circle(), &target);
        assert!(result.is_none());
        assert_eq!(diag.reject_reason, Some(DecodeRejectReason::NotACodeword));
        assert!(diag.best_dist > 0);
    }

    #[test]
    fn appended_extended_codeword_needs_an_extended_target() {
        let cw = CODEBOOK_EXTENDED[CODEBOOK_N];
        let img = paint_marker(&circle(), cw, 0.0, false);
        let base = TargetLayout::default();
        assert!(decode(&img, &circle(), &base).0.is_none());

        // The default board's geometry, carrying the first appended IDs.
        let extended = TargetLayout::new(
            "extended_test",
            *base.lattice(),
            base.ring(),
            MarkerCoding::Coded16(CodedRingSpec {
                ring_width_mm: 1.152,
                codebook_profile: CodebookProfile::Extended,
                id_assignment: Some((CODEBOOK_N..CODEBOOK_N + base.n_cells()).collect()),
            }),
            None,
        )
        .expect("extended target");
        let (result, _) = decode(&img, &circle(), &extended);
        assert_eq!(result.expect("appended word decodes").id, CODEBOOK_N);
    }

    #[test]
    fn decode_reject_reason_serialization_is_stable() {
        let reason = DecodeRejectReason::NotACodeword;
        assert_eq!(reason.to_string(), "not_a_codeword");
        let json = serde_json::to_string(&reason).expect("serialize reject reason");
        assert_eq!(json, "\"not_a_codeword\"");
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
        let target = TargetLayout::default();
        let codes = TargetCodes::of(&target).unwrap();
        let (result, diag) =
            decode_marker_with_diagnostics_and_mapper(&img, &ellipse, &config, &codes, None);
        assert!(result.is_none());
        assert_eq!(diag.reject_reason, Some(DecodeRejectReason::LowContrast));
        match diag.reject_context {
            Some(DecodeRejectContext::LowContrast {
                contrast,
                min_required_contrast,
            }) => {
                assert_abs_diff_eq!(contrast, 0.0, epsilon = 1e-6);
                assert_abs_diff_eq!(min_required_contrast, 0.2, epsilon = 1e-6);
            }
            other => panic!("unexpected reject context: {other:?}"),
        }
    }

    #[test]
    fn plain_targets_have_no_codes() {
        assert!(TargetCodes::of(&TargetLayout::rect_24x24()).is_none());
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
}
