//! Marker geometry/specification used by the detector.
//!
//! This is the single source of truth for expected inner/outer geometry in
//! normalized coordinates (outer radius == 1.0).

/// Expected polarity of the radial intensity derivative `dI/dr` at an edge.
///
/// Used by both inner and outer edge estimators to constrain the search
/// direction. `Auto` tries both polarities and picks the more coherent peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradPolarity {
    /// Intensity increases as radius increases (dark → light).
    DarkToLight,
    /// Intensity decreases as radius increases (light → dark).
    LightToDark,
    /// Try both and pick the more coherent peak.
    Auto,
}

/// Aggregation method across theta samples.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AngularAggregator {
    /// Median across theta responses.
    Median,
    /// Trimmed mean (drops `trim_fraction` on each side).
    TrimmedMean {
        /// Fraction removed from each tail before averaging.
        trim_fraction: f32,
    },
}

/// Marker spec in outer-normalized radius units.
///
/// NOTE: Defaults are derived from the synthetic renderer in `tools/gen_synth.py`
/// *and* the current edge sampler semantics in `ring::edge_sample::sample_edges`.
///
/// In `gen_synth.py` the marker uses:
/// - outer_radius = pitch_mm * 0.6
/// - inner_radius = pitch_mm * 0.4
/// - ring_width   = outer_radius * 0.12  (non-stress default)
///
/// The edge sampler finds the boundary of the merged dark band under blur, so
/// the expected (inner_edge / outer_edge) ratio in *outer-normalized* units is:
///   r_inner_expected = (inner_radius - ring_width) / (outer_radius + ring_width)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MarkerSpec {
    /// Expected inner radius as fraction of outer radius.
    pub r_inner_expected: f32,
    /// Allowed deviation in normalized radius around `r_inner_expected`.
    pub inner_search_halfwidth: f32,
    /// Expected sign of dI/dr at the inner edge.
    pub inner_grad_polarity: GradPolarity,
    /// Number of radii samples per theta.
    ///
    /// Same convention as [`OuterEstimationConfig::radial_samples`], calibrated
    /// independently for the inner estimation stage.
    pub radial_samples: usize,
    /// Number of theta samples (rays) for inner-scale estimation.
    ///
    /// Unlike the outer estimator (where ray count is set to `edge_sample.n_rays`
    /// at the call site), this value is used directly — the inner scan is not
    /// coupled to the edge-sampling resolution.
    pub theta_samples: usize,
    /// Aggregator across theta.
    ///
    /// Same convention as [`OuterEstimationConfig::aggregator`], applied to the
    /// inner radial profile.
    pub aggregator: AngularAggregator,
    /// Minimum fraction of theta samples required for a valid estimate.
    ///
    /// Same convention as [`OuterEstimationConfig::min_theta_coverage`], calibrated
    /// independently for the inner estimation stage.
    pub min_theta_coverage: f32,
    /// Minimum fraction of theta samples that must agree on the inner edge
    /// location (used as a quality gate).
    ///
    /// Same convention as [`OuterEstimationConfig::min_theta_consistency`]; the
    /// inner estimator uses a more permissive default (0.25) than the outer (0.35)
    /// because the inner edge is less anchored to a scale prior.
    ///
    /// Kept separate from `min_theta_coverage`: "coverage" is about in-bounds
    /// sampling, while "consistency" is about peak agreement.
    #[serde(default = "default_min_theta_consistency")]
    pub min_theta_consistency: f32,
}

impl Default for MarkerSpec {
    fn default() -> Self {
        // From tools/gen_synth.py (default, non-stress):
        //   outer_radius = pitch_mm * 0.6
        //   inner_radius = pitch_mm * 0.4
        //   ring_width   = outer_radius * 0.12 = pitch_mm * 0.072
        // and the edge sampler targets the *boundary* of the merged dark band:
        //   r_inner_edge = inner_radius - ring_width = pitch_mm * 0.328
        //   r_outer_edge = outer_radius + ring_width = pitch_mm * 0.672
        // so ratio is 0.328 / 0.672.
        let r_inner_expected = 0.328f32 / 0.672f32;

        Self {
            r_inner_expected,
            // Keep the window reasonably tight; polarity/consistency checks
            // should prevent snapping to code-band edges in difficult cases.
            inner_search_halfwidth: 0.08,
            // For the default synthetic marker, the inner edge is a light→dark transition.
            inner_grad_polarity: GradPolarity::LightToDark,
            radial_samples: 64,
            theta_samples: 96,
            aggregator: AngularAggregator::Median,
            min_theta_coverage: 0.6,
            min_theta_consistency: default_min_theta_consistency(),
        }
    }
}

impl MarkerSpec {
    /// Return normalized radial search window around `r_inner_expected`.
    pub fn search_window(&self) -> [f32; 2] {
        [
            self.r_inner_expected - self.inner_search_halfwidth,
            self.r_inner_expected + self.inner_search_halfwidth,
        ]
    }
}

fn default_min_theta_consistency() -> f32 {
    0.25
}
