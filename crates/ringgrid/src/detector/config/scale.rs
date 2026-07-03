//! Marker scale priors, multi-scale tiers, center-refinement selection, and
//! proposal downscaling.

/// Center-correction strategy used after local fits are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CircleRefinementMethod {
    /// Disable center correction.
    None,
    /// Run projective-center recovery from inner/outer conics.
    #[default]
    ProjectiveCenter,
}

impl CircleRefinementMethod {
    /// Returns `true` when this method includes projective-center recovery.
    pub fn uses_projective_center(self) -> bool {
        matches!(self, Self::ProjectiveCenter)
    }
}

/// Scale prior for marker diameter in detector working pixels.
///
/// The detector uses this range to derive proposal radii, outer-edge search
/// windows, ellipse validation bounds, and completion ROI. When the marker
/// scale prior is set via [`DetectConfig::set_marker_scale_prior`] or a
/// constructor, all scale-dependent parameters are auto-derived.
///
/// A single known size can be expressed with
/// [`MarkerScalePrior::from_nominal_diameter_px`]. For scenes where markers
/// vary in apparent size, use [`MarkerScalePrior::new`] with a range.
///
/// [`DetectConfig::set_marker_scale_prior`]: super::DetectConfig::set_marker_scale_prior
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MarkerScalePrior {
    /// Minimum expected marker outer diameter in pixels.
    pub diameter_min_px: f32,
    /// Maximum expected marker outer diameter in pixels.
    pub diameter_max_px: f32,
}

impl MarkerScalePrior {
    const MIN_DIAMETER_FLOOR_PX: f32 = 4.0;

    /// Construct a scale prior from a diameter range in pixels.
    pub fn new(diameter_min_px: f32, diameter_max_px: f32) -> Self {
        let mut out = Self {
            diameter_min_px,
            diameter_max_px,
        };
        out.normalize_in_place();
        out
    }

    /// Construct a fixed-size scale prior from one diameter hint.
    pub fn from_nominal_diameter_px(diameter_px: f32) -> Self {
        Self::new(diameter_px, diameter_px)
    }

    /// Return the normalized diameter range `[min, max]` in pixels.
    pub fn diameter_range_px(self) -> [f32; 2] {
        let n = self.normalized();
        [n.diameter_min_px, n.diameter_max_px]
    }

    /// Return nominal diameter (midpoint of `[min, max]`) in pixels.
    pub fn nominal_diameter_px(self) -> f32 {
        let [d_min, d_max] = self.diameter_range_px();
        0.5 * (d_min + d_max)
    }

    /// Return nominal outer radius in pixels.
    pub fn nominal_outer_radius_px(self) -> f32 {
        self.nominal_diameter_px() * 0.5
    }

    /// Return a normalized copy with finite, ordered, non-degenerate bounds.
    pub fn normalized(self) -> Self {
        let mut out = self;
        out.normalize_in_place();
        out
    }

    fn normalize_in_place(&mut self) {
        let defaults = MarkerScalePrior::default();
        let mut d_min = if self.diameter_min_px.is_finite() {
            self.diameter_min_px
        } else {
            defaults.diameter_min_px
        };
        let mut d_max = if self.diameter_max_px.is_finite() {
            self.diameter_max_px
        } else {
            defaults.diameter_max_px
        };
        if d_min > d_max {
            std::mem::swap(&mut d_min, &mut d_max);
        }
        d_min = d_min.max(Self::MIN_DIAMETER_FLOOR_PX);
        d_max = d_max.max(d_min);
        self.diameter_min_px = d_min;
        self.diameter_max_px = d_max;
    }
}

impl Default for MarkerScalePrior {
    fn default() -> Self {
        Self {
            diameter_min_px: 14.0,
            diameter_max_px: 66.0,
        }
    }
}

/// One scale band for multi-scale adaptive detection.
///
/// Each tier covers a diameter range with a ratio of at most 3:1. Combine
/// multiple tiers with [`ScaleTiers`] to cover scenes where markers span a
/// wide range of apparent sizes.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ScaleTier {
    /// Marker outer diameter range for this tier (pixels).
    pub prior: MarkerScalePrior,
}

impl ScaleTier {
    /// Create a tier covering `[diameter_min_px, diameter_max_px]`.
    ///
    /// For best results keep the ratio `diameter_max_px / diameter_min_px ≤ 3`.
    pub fn new(diameter_min_px: f32, diameter_max_px: f32) -> Self {
        Self {
            prior: MarkerScalePrior::new(diameter_min_px, diameter_max_px),
        }
    }
}

/// An ordered set of scale tiers for multi-scale adaptive detection.
///
/// Each tier runs one full detection pass (fit + decode + projective centers).
/// Results from all tiers are merged with size-consistency-aware dedup, then
/// global filter, completion, and final H refit run once on the merged pool.
///
/// Use the preset constructors for common scenarios:
///
/// - [`ScaleTiers::four_tier_wide`] — 8–220 px, full range (27:1 ratio)
/// - [`ScaleTiers::two_tier_standard`] — 14–100 px, moderate variation (7:1)
/// - [`ScaleTiers::single`] — single-pass equivalent, no merge overhead
/// - [`ScaleTiers::from_detected_radii`] — built from a scale-probe result
///
/// See [`crate::Detector::detect_adaptive`] and
/// [`crate::Detector::detect_multiscale`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScaleTiers(Vec<ScaleTier>);

impl ScaleTiers {
    /// Construct tiers from an explicit ordered tier list.
    ///
    /// This is the general-purpose constructor for callers building custom
    /// tier sets; the preset constructors ([`four_tier_wide`](Self::four_tier_wide),
    /// [`two_tier_standard`](Self::two_tier_standard), [`single`](Self::single),
    /// [`from_detected_radii`](Self::from_detected_radii)) cover the common cases.
    ///
    /// The vec is accepted as-is; an empty vec yields tiers with no scales to
    /// probe. Callers that require a non-empty set should validate before
    /// constructing.
    pub fn new(tiers: Vec<ScaleTier>) -> Self {
        Self(tiers)
    }

    /// Four overlapping tiers covering 8–220 px (27:1 diameter ratio).
    ///
    /// Tier boundaries: `[8,24]`, `[20,60]`, `[50,130]`, `[110,220]` px.
    /// Each tier has ratio ≤ 3:1. Overlapping boundaries ensure markers in
    /// the 20–24, 50–60, and 110–130 px range are detected by two tiers.
    ///
    /// Use when marker apparent size is unknown or spans a very wide range.
    pub fn four_tier_wide() -> Self {
        Self(vec![
            ScaleTier::new(8.0, 24.0),
            ScaleTier::new(20.0, 60.0),
            ScaleTier::new(50.0, 130.0),
            ScaleTier::new(110.0, 220.0),
        ])
    }

    /// Two overlapping tiers covering 14–100 px (~7:1 diameter ratio).
    ///
    /// Tier boundaries: `[14,42]`, `[36,100]` px. Faster than
    /// [`four_tier_wide`](Self::four_tier_wide) for moderate scale variation.
    pub fn two_tier_standard() -> Self {
        Self(vec![
            ScaleTier::new(14.0, 42.0),
            ScaleTier::new(36.0, 100.0),
        ])
    }

    /// Single tier wrapping the given prior — no multi-scale overhead.
    ///
    /// Equivalent to single-pass detection. Use when marker scale is
    /// approximately known.
    pub fn single(prior: MarkerScalePrior) -> Self {
        Self(vec![ScaleTier { prior }])
    }

    /// Construct tiers from dominant code-band radii estimated by the scale probe.
    ///
    /// Clusters `probe_radii` into groups where `max/min ≤ 3.0`, converts each
    /// cluster to an outer-ring diameter estimate (probe radii land in the code
    /// band at ~0.8× the outer ring radius), and pads each tier by ±30 %.
    ///
    /// Falls back to `single(MarkerScalePrior::default())` when `probe_radii`
    /// is empty or contains no finite positive values.
    pub fn from_detected_radii(probe_radii: &[f32]) -> Self {
        let mut sorted: Vec<f32> = probe_radii
            .iter()
            .copied()
            .filter(|r| r.is_finite() && *r > 0.0)
            .collect();

        if sorted.is_empty() {
            return Self::single(MarkerScalePrior::default());
        }

        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Code-band midpoint sits at ~0.8× the outer ring radius for the default
        // inner/outer ratio of 0.6: midpoint = 0.5*(0.6+1.0) = 0.8.
        const PROBE_TO_OUTER: f32 = 1.0 / 0.8; // ≈ 1.25

        let mut tiers = Vec::new();
        let mut cluster_min = sorted[0];
        let mut cluster_max = sorted[0];

        for &r in &sorted[1..] {
            if r / cluster_min <= 3.0 {
                cluster_max = r;
            } else {
                let r_outer_min = cluster_min * PROBE_TO_OUTER;
                let r_outer_max = cluster_max * PROBE_TO_OUTER;
                let d_min = (2.0 * r_outer_min * 0.70).max(4.0);
                let d_max = (2.0 * r_outer_max * 1.35).max(d_min);
                tiers.push(ScaleTier::new(d_min, d_max));
                cluster_min = r;
                cluster_max = r;
            }
        }
        // Final cluster.
        let r_outer_min = cluster_min * PROBE_TO_OUTER;
        let r_outer_max = cluster_max * PROBE_TO_OUTER;
        let d_min = (2.0 * r_outer_min * 0.70).max(4.0);
        let d_max = (2.0 * r_outer_max * 1.35).max(d_min);
        tiers.push(ScaleTier::new(d_min, d_max));

        Self(tiers)
    }

    /// Access the ordered tier list.
    pub fn tiers(&self) -> &[ScaleTier] {
        &self.0
    }
}

/// Controls optional image downscaling before proposal generation.
///
/// When markers are large in the image, the proposal stage can be run on a
/// downscaled copy for significant speedup. All downstream stages (outer fit,
/// decode, inner fit) still operate at full resolution.
///
/// Proposal coordinates are automatically scaled back to original image space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalDownscale {
    /// Auto-select downscale factor from `marker_scale.diameter_min_px`.
    ///
    /// `factor = clamp(floor(d_min / 14.0), 1, 4)`. Ensures markers remain
    /// at least ~7 px diameter after downscaling, which is sufficient for
    /// RSD center detection.
    Auto,
    /// No downscaling (full resolution proposals).
    #[default]
    Off,
    /// Explicit integer downscale factor (clamped to `[1, 4]`).
    Factor(u32),
}

impl ProposalDownscale {
    /// Resolve the concrete integer factor given the current marker scale prior.
    pub fn resolve(&self, marker_scale: MarkerScalePrior) -> u32 {
        match self {
            Self::Auto => {
                let d_min = marker_scale.diameter_range_px()[0];
                (d_min / 14.0).floor().clamp(1.0, 4.0) as u32
            }
            Self::Off => 1,
            Self::Factor(f) => (*f).clamp(1, 4),
        }
    }
}
