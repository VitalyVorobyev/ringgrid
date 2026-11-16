//! Target JSON schema: canonical `ringgrid.target.v6` read/write.
//!
//! `ringgrid.target.v6` is the compositional schema (lattice / marker /
//! coding / fiducials) with **derived** origin-dot positions — `fiducials`
//! carries only `dot_radius_mm`. It is the only schema the loaders accept;
//! files written with the earlier v4 / v5 schemas are rejected with an
//! "unsupported schema" error that says how to convert them (load with
//! ringgrid 0.15 and save again).

#[cfg(feature = "std")]
use std::path::Path;

use super::error::{TargetLoadError, TargetValidationError};
use super::fiducials::OriginFiducials;
use super::lattice::LatticeGeometry;
use super::layout::TargetLayout;
use super::ring::{CodedRingSpec, MarkerCoding, RingGeometry};

/// Schema string every target spec is written with, and the only one the
/// loaders accept.
pub const TARGET_SCHEMA_VERSION: &str = "ringgrid.target.v6";

/// JSON Schema (draft 2020-12) of the canonical `ringgrid.target.v6` target
/// specification, as read and written by [`TargetLayout::from_json_str`] and
/// [`TargetLayout::to_json_string`].
///
/// The wire type itself is private; this is the supported way to obtain its
/// schema.
#[cfg(feature = "schemars")]
pub fn target_spec_schema() -> schemars::Schema {
    schemars::schema_for!(TargetSpecV6)
}

/// Minimal probe to dispatch on the schema tag before full deserialization.
#[derive(serde::Deserialize)]
struct SchemaProbe {
    schema: String,
}

/// The compositional `ringgrid.target.v6` schema.
#[derive(serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "schemars",
    schemars(
        title = "ringgrid target specification (v6)",
        description = "Physical description of a ringgrid calibration target: lattice, ring radii, marker coding and optional origin fiducials. All lengths are in millimeters. This is the only form the loaders accept and the writers emit."
    )
)]
#[serde(deny_unknown_fields)]
struct TargetSpecV6 {
    /// Schema identifier; always `"ringgrid.target.v6"`.
    #[cfg_attr(
        feature = "schemars",
        schemars(extend("const" = "ringgrid.target.v6"))
    )]
    schema: String,
    /// Human-readable target name (free text, e.g. for file naming).
    name: String,
    /// Cell arrangement: hexagonal or rectangular lattice, with its dimensions and pitch.
    lattice: LatticeGeometry,
    /// Ring radii shared by every marker on the target.
    marker: RingGeometry,
    /// How markers encode identity: 16-sector coded rings or plain annuli.
    coding: MarkerCoding,
    /// Optional origin fiducial dots that fix the board frame; omit for none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fiducials: Option<OriginFiducials>,
}

impl TargetSpecV6 {
    fn into_layout(self) -> Result<TargetLayout, TargetValidationError> {
        if self.schema != TARGET_SCHEMA_VERSION {
            return Err(TargetValidationError::UnsupportedSchema { found: self.schema });
        }
        TargetLayout::new(
            self.name,
            self.lattice,
            self.marker,
            self.coding,
            self.fiducials,
        )
    }
}

impl TargetLayout {
    /// Load a target layout from a `ringgrid.target.v6` JSON string.
    ///
    /// Any other schema tag, including the earlier `v4` / `v5`, fails with
    /// [`TargetValidationError::UnsupportedSchema`].
    pub fn from_json_str(data: &str) -> Result<Self, TargetLoadError> {
        // Dispatch on the tag first so an older file reports its schema, not
        // whichever field the v6 parser happens to trip over.
        let probe: SchemaProbe = serde_json::from_str(data)?;
        if probe.schema != TARGET_SCHEMA_VERSION {
            return Err(TargetValidationError::UnsupportedSchema {
                found: probe.schema,
            }
            .into());
        }
        let spec: TargetSpecV6 = serde_json::from_str(data)?;
        spec.into_layout().map_err(Into::into)
    }

    /// Load a target layout from a `ringgrid.target.v6` JSON file.
    #[cfg(feature = "std")]
    pub fn from_json_file(path: &Path) -> Result<Self, TargetLoadError> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Serialize the layout as canonical `ringgrid.target.v6` JSON.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.to_spec())
            .expect("target layout JSON serialization must succeed")
    }

    /// Write the canonical `ringgrid.target.v6` JSON representation to disk.
    #[cfg(feature = "std")]
    pub fn write_json_file(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, format!("{}\n", self.to_json_string()))
    }

    fn to_spec(&self) -> TargetSpecV6 {
        // Normalize a sequential id_assignment back to the implicit form.
        let coding = match self.coding() {
            MarkerCoding::Coded16(spec) => {
                let is_sequential = spec
                    .id_assignment
                    .as_ref()
                    .is_none_or(|ids| ids.iter().enumerate().all(|(i, &id)| id == i));
                MarkerCoding::Coded16(CodedRingSpec {
                    ring_width_mm: spec.ring_width_mm,
                    codebook_profile: spec.codebook_profile,
                    id_assignment: if is_sequential {
                        None
                    } else {
                        spec.id_assignment.clone()
                    },
                })
            }
            MarkerCoding::Plain => MarkerCoding::Plain,
        };
        TargetSpecV6 {
            schema: TARGET_SCHEMA_VERSION.to_string(),
            name: self.name().to_string(),
            lattice: *self.lattice(),
            marker: self.ring(),
            coding,
            fiducials: self.fiducials().copied(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marker::CodebookProfile;

    #[test]
    fn v6_round_trip_preserves_all_aspects() {
        let target = TargetLayout::rect_24x24();
        let json = target.to_json_string();
        let reloaded = TargetLayout::from_json_str(&json).expect("round-trip");

        assert_eq!(reloaded.name(), target.name());
        assert_eq!(reloaded.lattice(), target.lattice());
        assert_eq!(reloaded.ring(), target.ring());
        assert_eq!(reloaded.coding(), target.coding());
        assert_eq!(reloaded.fiducials(), target.fiducials());
        assert_eq!(reloaded.cells(), target.cells());
    }

    #[test]
    fn v6_json_shape_is_compositional() {
        let json = TargetLayout::rect_24x24().to_json_string();
        let val: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(val["schema"], "ringgrid.target.v6");
        assert_eq!(val["lattice"]["kind"], "rect");
        assert_eq!(val["lattice"]["rows"], 24);
        assert_eq!(val["coding"]["kind"], "plain");
        assert_eq!(val["marker"]["outer_radius_mm"], 5.6);
        // Fiducials carry size only — positions are derived from the lattice.
        assert_eq!(val["fiducials"]["dot_radius_mm"], 1.4);
        assert!(val["fiducials"].get("dots_mm").is_none());

        let hex_json = TargetLayout::default_hex().to_json_string();
        let hex: serde_json::Value = serde_json::from_str(&hex_json).expect("valid json");
        assert_eq!(hex["lattice"]["kind"], "hex");
        assert_eq!(hex["coding"]["kind"], "coded16");
        let ring_width = hex["coding"]["ring_width_mm"].as_f64().expect("number");
        assert!((ring_width - 1.152).abs() < 1e-6);
        assert!(hex.get("fiducials").is_none());
    }

    #[test]
    fn rejects_unknown_schema() {
        let raw = r#"{"schema":"ringgrid.target.v2","name":"x"}"#;
        let err = TargetLayout::from_json_str(raw).expect_err("unsupported");
        assert!(matches!(
            err,
            TargetLoadError::Validation(TargetValidationError::UnsupportedSchema { .. })
        ));
    }

    /// An earlier schema is reported by name, with the one supported version
    /// and how to convert the file, instead of a parse error about whichever
    /// v4 / v5 field the v6 parser meets first.
    #[test]
    fn earlier_schemas_fail_with_an_unsupported_schema_message() {
        let v4 = r#"{"schema":"ringgrid.target.v4","name":"legacy","pitch_mm":8.0,
            "rows":15,"long_row_cols":14,"marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2,"marker_ring_width_mm":1.152}"#;
        let v5 = r#"{"schema":"ringgrid.target.v5","name":"legacy",
            "lattice":{"kind":"rect","rows":8,"cols":8,"pitch_mm":14.0},
            "marker":{"outer_radius_mm":5.6,"inner_radius_mm":2.8},
            "coding":{"kind":"plain"},
            "fiducials":{"dot_radius_mm":1.4,"dots_mm":[[49.0,49.0]]}}"#;
        for (raw, tag) in [(v4, "ringgrid.target.v4"), (v5, "ringgrid.target.v5")] {
            let err = TargetLayout::from_json_str(raw).expect_err("v4 / v5 are not loadable");
            let message = err.to_string();
            assert!(
                matches!(
                    err,
                    TargetLoadError::Validation(TargetValidationError::UnsupportedSchema { .. })
                ),
                "{tag}: {message}"
            );
            assert!(message.contains(tag), "names the found schema: {message}");
            assert!(
                message.contains("'ringgrid.target.v6'"),
                "names the supported schema: {message}"
            );
            assert!(
                message.contains("ringgrid 0.15"),
                "says how to convert: {message}"
            );
        }
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let mut val: serde_json::Value =
            serde_json::from_str(&TargetLayout::default_hex().to_json_string()).expect("json");
        val["markers"] = serde_json::json!([]);
        let raw = serde_json::to_string(&val).expect("json");
        assert!(TargetLayout::from_json_str(&raw).is_err());
    }

    /// Every target JSON checked into the repository loads, validates and
    /// round-trips to the same layout.
    #[cfg(feature = "std")]
    #[test]
    fn checked_in_fixtures_load() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        // (path, cells, codebook profile)
        let fixtures = [
            ("tools/board/board_spec.json", 203, CodebookProfile::Base),
            (
                "tools/board/board_spec_optimized.json",
                203,
                CodebookProfile::Base,
            ),
            (
                "tools/board/board_spec_extended_opt.json",
                203,
                CodebookProfile::Extended,
            ),
            ("testdata/board_ringgrid.json", 203, CodebookProfile::Base),
            (
                "crates/ringgrid/examples/target.json",
                203,
                CodebookProfile::Base,
            ),
        ];
        for (path, n_cells, profile) in fixtures {
            let target = TargetLayout::from_json_file(&root.join(path))
                .unwrap_or_else(|e| panic!("{path} must load: {e}"));
            assert_eq!(target.n_cells(), n_cells, "{path}");
            assert!(
                matches!(target.coding(), MarkerCoding::Coded16(spec) if spec.codebook_profile == profile),
                "{path}: codebook profile"
            );
            let reloaded =
                TargetLayout::from_json_str(&target.to_json_string()).expect("round-trip");
            let ids: Vec<_> = target.cells().iter().map(|c| c.id).collect();
            let reloaded_ids: Vec<_> = reloaded.cells().iter().map(|c| c.id).collect();
            assert_eq!(ids, reloaded_ids, "{path}: ids survive a round-trip");
        }
    }
}
