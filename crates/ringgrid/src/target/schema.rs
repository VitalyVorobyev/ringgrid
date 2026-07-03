//! Target JSON schema: canonical v5 read/write plus v4 auto-migration.
//!
//! `ringgrid.target.v5` is the compositional schema (lattice / marker /
//! coding / fiducials). The legacy flat `ringgrid.target.v4` schema (hex
//! coded targets only) is still accepted by the loaders and migrated on the
//! fly; writers always emit v5.

#[cfg(feature = "std")]
use std::path::Path;

use super::error::{TargetLoadError, TargetValidationError};
use super::fiducials::OriginFiducials;
use super::lattice::{HexGeometry, LatticeGeometry};
use super::layout::TargetLayout;
use super::ring::{CodedRingSpec, MarkerCoding, RingGeometry};

pub(crate) const TARGET_SCHEMA_V5: &str = "ringgrid.target.v5";
pub(crate) const TARGET_SCHEMA_V4: &str = "ringgrid.target.v4";
const EXPECTED_SCHEMAS: &str = "'ringgrid.target.v5' or 'ringgrid.target.v4'";

/// Minimal probe to dispatch on the schema tag before full deserialization.
#[derive(serde::Deserialize)]
struct SchemaProbe {
    schema: String,
}

/// The compositional `ringgrid.target.v5` schema.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetSpecV5 {
    schema: String,
    name: String,
    lattice: LatticeGeometry,
    marker: RingGeometry,
    coding: MarkerCoding,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fiducials: Option<OriginFiducials>,
}

/// The legacy flat `ringgrid.target.v4` schema (hex coded targets only).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BoardSpecV4 {
    pub(crate) schema: String,
    pub(crate) name: String,
    pub(crate) pitch_mm: f32,
    pub(crate) rows: usize,
    pub(crate) long_row_cols: usize,
    pub(crate) marker_outer_radius_mm: f32,
    pub(crate) marker_inner_radius_mm: f32,
    pub(crate) marker_ring_width_mm: f32,
    /// Optional optimized ID assignment. When present, `id_assignment[i]` is the
    /// codebook ID for the i-th marker (in generation order). When absent, IDs
    /// are assigned sequentially (0, 1, 2, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) id_assignment: Option<Vec<usize>>,
}

impl BoardSpecV4 {
    pub(crate) fn into_layout(self) -> Result<TargetLayout, TargetValidationError> {
        if self.schema != TARGET_SCHEMA_V4 {
            return Err(TargetValidationError::UnsupportedSchema {
                found: self.schema,
                expected: EXPECTED_SCHEMAS,
            });
        }
        TargetLayout::new(
            self.name,
            LatticeGeometry::Hex(HexGeometry {
                rows: self.rows,
                long_row_cols: self.long_row_cols,
                pitch_mm: self.pitch_mm,
            }),
            RingGeometry {
                outer_radius_mm: self.marker_outer_radius_mm,
                inner_radius_mm: self.marker_inner_radius_mm,
            },
            MarkerCoding::Coded16(CodedRingSpec {
                ring_width_mm: self.marker_ring_width_mm,
                id_assignment: self.id_assignment,
            }),
            None,
        )
    }
}

impl TargetSpecV5 {
    fn into_layout(self) -> Result<TargetLayout, TargetValidationError> {
        if self.schema != TARGET_SCHEMA_V5 {
            return Err(TargetValidationError::UnsupportedSchema {
                found: self.schema,
                expected: EXPECTED_SCHEMAS,
            });
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
    /// Load a target layout from a JSON string (`v5`, or legacy `v4`
    /// auto-migrated to the compositional model).
    pub fn from_json_str(data: &str) -> Result<Self, TargetLoadError> {
        let probe: SchemaProbe = serde_json::from_str(data)?;
        match probe.schema.as_str() {
            TARGET_SCHEMA_V5 => {
                let spec: TargetSpecV5 = serde_json::from_str(data)?;
                spec.into_layout().map_err(Into::into)
            }
            TARGET_SCHEMA_V4 => {
                let spec: BoardSpecV4 = serde_json::from_str(data)?;
                spec.into_layout().map_err(Into::into)
            }
            other => Err(TargetValidationError::UnsupportedSchema {
                found: other.to_string(),
                expected: EXPECTED_SCHEMAS,
            }
            .into()),
        }
    }

    /// Load a target layout from a JSON file (`v5` or legacy `v4`).
    #[cfg(feature = "std")]
    pub fn from_json_file(path: &Path) -> Result<Self, TargetLoadError> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json_str(&data)
    }

    /// Serialize the layout as canonical `ringgrid.target.v5` JSON.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.to_spec())
            .expect("target layout JSON serialization must succeed")
    }

    /// Write the canonical `ringgrid.target.v5` JSON representation to disk.
    #[cfg(feature = "std")]
    pub fn write_json_file(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, format!("{}\n", self.to_json_string()))
    }

    fn to_spec(&self) -> TargetSpecV5 {
        // Normalize a sequential id_assignment back to the implicit form.
        let coding = match self.coding() {
            MarkerCoding::Coded16(spec) => {
                let is_sequential = spec
                    .id_assignment
                    .as_ref()
                    .is_none_or(|ids| ids.iter().enumerate().all(|(i, &id)| id == i));
                MarkerCoding::Coded16(CodedRingSpec {
                    ring_width_mm: spec.ring_width_mm,
                    id_assignment: if is_sequential {
                        None
                    } else {
                        spec.id_assignment.clone()
                    },
                })
            }
            MarkerCoding::Plain => MarkerCoding::Plain,
        };
        TargetSpecV5 {
            schema: TARGET_SCHEMA_V5.to_string(),
            name: self.name().to_string(),
            lattice: *self.lattice(),
            marker: self.ring(),
            coding,
            fiducials: self.fiducials().cloned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v5_round_trip_preserves_all_aspects() {
        let target = TargetLayout::isra_rect_24x24();
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
    fn v5_json_shape_is_compositional() {
        let json = TargetLayout::isra_rect_24x24().to_json_string();
        let val: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(val["schema"], "ringgrid.target.v5");
        assert_eq!(val["lattice"]["kind"], "rect");
        assert_eq!(val["lattice"]["rows"], 24);
        assert_eq!(val["coding"]["kind"], "plain");
        assert_eq!(val["marker"]["outer_radius_mm"], 5.6);
        assert_eq!(
            val["fiducials"]["dots_mm"].as_array().map(|a| a.len()),
            Some(3)
        );

        let hex_json = TargetLayout::default_hex().to_json_string();
        let hex: serde_json::Value = serde_json::from_str(&hex_json).expect("valid json");
        assert_eq!(hex["lattice"]["kind"], "hex");
        assert_eq!(hex["coding"]["kind"], "coded16");
        let ring_width = hex["coding"]["ring_width_mm"].as_f64().expect("number");
        assert!((ring_width - 1.152).abs() < 1e-6);
        assert!(hex.get("fiducials").is_none());
    }

    #[test]
    fn v4_auto_migrates_to_hex_coded() {
        let raw = r#"{
            "schema":"ringgrid.target.v4",
            "name":"legacy",
            "pitch_mm":8.0,
            "rows":3,
            "long_row_cols":4,
            "marker_outer_radius_mm":4.8,
            "marker_inner_radius_mm":3.2,
            "marker_ring_width_mm":1.152
        }"#;
        let target = TargetLayout::from_json_str(raw).expect("v4 accepted");
        assert_eq!(target.name(), "legacy");
        assert!(target.is_coded());
        assert!(matches!(target.lattice(), LatticeGeometry::Hex(_)));
        assert_eq!(target.fiducials(), None);

        // Migrated target re-serializes as v5.
        let json = target.to_json_string();
        assert!(json.contains("ringgrid.target.v5"));
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

    #[test]
    fn rejects_unknown_top_level_fields() {
        let mut val: serde_json::Value =
            serde_json::from_str(&TargetLayout::default_hex().to_json_string()).expect("json");
        val["markers"] = serde_json::json!([]);
        let raw = serde_json::to_string(&val).expect("json");
        assert!(TargetLayout::from_json_str(&raw).is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn checked_in_v4_fixtures_load() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let plain = TargetLayout::from_json_file(&root.join("tools/board/board_spec.json"))
            .expect("v4 board_spec fixture loads");
        assert_eq!(plain.n_cells(), 203);

        let optimized =
            TargetLayout::from_json_file(&root.join("tools/board/board_spec_optimized.json"))
                .expect("v4 optimized fixture loads");
        assert_eq!(optimized.n_cells(), 203);
        // Optimized assignment survives migration and round-trip.
        let reloaded = TargetLayout::from_json_str(&optimized.to_json_string()).expect("v5");
        let ids: Vec<_> = optimized.cells().iter().map(|c| c.id).collect();
        let reloaded_ids: Vec<_> = reloaded.cells().iter().map(|c| c.id).collect();
        assert_eq!(ids, reloaded_ids);
    }
}
