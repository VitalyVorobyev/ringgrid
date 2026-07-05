//! JSON wire types shared with the Python and WASM bindings so all three
//! bindings stay isomorphic at the boundary.

use serde::{Deserialize, Serialize};

use crate::status::RinggridStatus;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleTierWire {
    diameter_min_px: f32,
    diameter_max_px: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleTiersWire {
    tiers: Vec<ScaleTierWire>,
}

/// Proposal payload (`{"image_size": [w, h], "proposals": [...]}`) — the heatmap
/// is delivered separately through the borrowed accessor.
#[derive(Serialize)]
pub(crate) struct ProposalPayload<'a> {
    pub image_size: [u32; 2],
    pub proposals: &'a [ringgrid::Proposal],
}

/// Combined payload for the diagnostics-returning entry points: the slim
/// [`ringgrid::DetectionResult`] under `result` and the opt-in
/// [`ringgrid::diagnostics::DetectionDiagnostics`] under `diagnostics`
/// (`diagnostics.markers` aligns 1:1 with `result.detected_markers`).
#[derive(Serialize)]
pub(crate) struct DetectionWithDiagnostics {
    pub result: ringgrid::DetectionResult,
    pub diagnostics: ringgrid::diagnostics::DetectionDiagnostics,
}

/// Parse a `{"tiers": [...]}` JSON string into [`ringgrid::ScaleTiers`].
pub(crate) fn parse_scale_tiers(tiers_json: &str) -> Result<ringgrid::ScaleTiers, RinggridStatus> {
    let wire: ScaleTiersWire =
        serde_json::from_str(tiers_json).map_err(|_| RinggridStatus::ErrBadJson)?;
    if wire.tiers.is_empty() {
        return Err(RinggridStatus::ErrBadJson);
    }
    Ok(ringgrid::ScaleTiers::new(
        wire.tiers
            .iter()
            .map(|t| ringgrid::ScaleTier::new(t.diameter_min_px, t.diameter_max_px))
            .collect(),
    ))
}

/// Serialize a [`ringgrid::ScaleTiers`] to the `{"tiers": [...]}` wire form.
pub(crate) fn scale_tiers_to_json(tiers: &ringgrid::ScaleTiers) -> Result<String, RinggridStatus> {
    let wire = ScaleTiersWire {
        tiers: tiers
            .tiers()
            .iter()
            .map(|t| ScaleTierWire {
                diameter_min_px: t.prior.diameter_min_px,
                diameter_max_px: t.prior.diameter_max_px,
            })
            .collect(),
    };
    serde_json::to_string(&wire).map_err(|_| RinggridStatus::ErrSerialize)
}
