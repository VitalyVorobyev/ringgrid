//! The JSON Schemas generated from the `schemars` derives must accept exactly
//! the documents the library itself writes and reads.
//!
//! These tests validate real library output (defaults, presets, constructor
//! results) against the schemas, and hand-built invalid documents against the
//! same schemas, so a serde/schema mismatch shows up as a failing test rather
//! than a form that silently rejects valid configs.
#![cfg(feature = "schemars")]

use jsonschema::Validator;
use ringgrid::{
    CodebookProfile, CodedRingSpec, DetectConfig, HexGeometry, LatticeGeometry, MarkerCoding,
    MarkerScalePrior, OriginDots, PageOrientation, PageSize, PageSpec, RingGeometry, TargetLayout,
    TargetRenderOptions, target_spec_schema,
};
use schemars::schema_for;
use serde_json::{Value, json};

fn validator(schema: schemars::Schema) -> Validator {
    jsonschema::validator_for(&schema.to_value()).expect("generated schema must be valid")
}

fn assert_valid(validator: &Validator, instance: &Value, what: &str) {
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|e| format!("{} at {}", e, e.instance_path()))
        .collect();
    assert!(errors.is_empty(), "{what} must validate: {errors:#?}");
}

fn assert_invalid(validator: &Validator, instance: &Value, what: &str) {
    assert!(
        !validator.is_valid(instance),
        "{what} must be rejected by the schema"
    );
}

// ── DetectConfig ─────────────────────────────────────────────────────

fn detect_validator() -> Validator {
    validator(schema_for!(DetectConfig))
}

fn config_value(config: &DetectConfig) -> Value {
    serde_json::to_value(config).expect("serialize DetectConfig")
}

#[test]
fn default_detect_config_validates() {
    assert_valid(
        &detect_validator(),
        &config_value(&DetectConfig::default()),
        "DetectConfig::default()",
    );
}

#[test]
fn board_derived_detect_configs_validate() {
    // `default_config_json(board)` in the wasm package is exactly
    // `DetectConfig::from_target(parse(board))`.
    let v = detect_validator();
    let targets = [
        ("default_hex", TargetLayout::default_hex()),
        ("rect_24x24", TargetLayout::rect_24x24()),
        (
            "plain_hex",
            TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, OriginDots::None).unwrap(),
        ),
    ];
    for (name, target) in targets {
        let config = DetectConfig::from_target(target.clone());
        assert_valid(&v, &config_value(&config), &format!("from_target({name})"));

        let config = DetectConfig::from_target_and_marker_diameter(target.clone(), 32.0);
        assert_valid(
            &v,
            &config_value(&config),
            &format!("from_target_and_marker_diameter({name}, 32)"),
        );

        let config =
            DetectConfig::from_target_and_scale_prior(target, MarkerScalePrior::new(20.0, 60.0));
        assert_valid(
            &v,
            &config_value(&config),
            &format!("from_target_and_scale_prior({name})"),
        );
    }
}

#[test]
fn non_default_variants_and_overlays_validate() {
    let v = detect_validator();

    // Every enum representation the config can serialize to.
    for downscale in ["auto", "off"] {
        let doc = json!({ "advanced": { "proposal_downscale": downscale } });
        assert_valid(&v, &doc, &format!("proposal_downscale = {downscale}"));
        DetectConfig::default().with_json_overlay(doc).unwrap();
    }
    for doc in [
        json!({ "advanced": { "proposal_downscale": { "factor": 3 } } }),
        json!({ "circle_refinement": "None" }),
        json!({ "advanced": { "decode": { "codebook_profile": "extended" } } }),
        json!({ "advanced": { "outer_estimation": { "aggregator": { "trimmed_mean": { "trim_fraction": 0.2 } }, "grad_polarity": "auto" } } }),
        json!({ "advanced": { "marker_spec": { "inner_grad_polarity": "dark_to_light" } } }),
        // Option fields accept both a value and null.
        json!({ "advanced": { "completion": { "max_attempts": 5 }, "seed_proposals": { "max_seeds": null } } }),
        json!({ "advanced": { "projective_center": { "max_correction_shift_px": 7.5, "max_selected_residual": null } } }),
        // A partial overlay is a valid document.
        json!({}),
        json!({ "advanced": { "completion": { "enable": false } } }),
        json!({ "marker_scale": { "diameter_min_px": 20.0, "diameter_max_px": 80.0 }, "require_complete_board": true }),
        json!({ "self_undistort": { "enable": true, "lambda_range": [-1e-6, 1e-6] } }),
    ] {
        assert_valid(&v, &doc, "overlay");
        // The library must accept every document the schema accepts.
        DetectConfig::default()
            .with_json_overlay(doc.clone())
            .unwrap_or_else(|e| panic!("library rejected schema-valid overlay {doc}: {e}"));
    }
}

#[test]
fn detect_config_schema_rejects_invalid_documents() {
    let v = detect_validator();
    for (what, doc) in [
        (
            "string for a number",
            json!({ "marker_scale": { "diameter_min_px": "big" } }),
        ),
        ("number for a bool", json!({ "require_complete_board": 1 })),
        (
            "unknown enum value",
            json!({ "circle_refinement": "Sideways" }),
        ),
        (
            "unknown downscale",
            json!({ "advanced": { "proposal_downscale": "x9" } }),
        ),
        (
            "downscale factor above clamp",
            json!({ "advanced": { "proposal_downscale": { "factor": 9 } } }),
        ),
        (
            "zero radius_step",
            json!({ "advanced": { "proposal": { "radius_step": 0 } } }),
        ),
        (
            "negative count",
            json!({ "advanced": { "inner_fit": { "min_points": -1 } } }),
        ),
        (
            "trim fraction above clamp",
            json!({ "advanced": { "outer_estimation": { "aggregator": { "trimmed_mean": { "trim_fraction": 0.9 } } } } }),
        ),
        (
            "lambda_range of wrong length",
            json!({ "self_undistort": { "lambda_range": [0.0] } }),
        ),
        (
            "object for a section array",
            json!({ "advanced": { "id_correction": { "auto_search_radius_outer_muls": 2.4 } } }),
        ),
    ] {
        assert_invalid(&v, &doc, what);
    }
}

#[test]
fn detect_config_schema_documents_defaults_and_units() {
    let schema = schema_for!(DetectConfig).to_value();
    let description = schema["description"]
        .as_str()
        .expect("top-level description");
    assert!(
        description.contains("board-dependent"),
        "top-level description must warn that defaults depend on the board"
    );
    let defs = &schema["$defs"];
    assert_eq!(
        defs["ProposalConfig"]["properties"]["r_min"]["x-unit"],
        "px"
    );
    assert_eq!(
        defs["RansacConfig"]["properties"]["inlier_threshold"]["x-unit"],
        "px"
    );
    assert_eq!(
        defs["InnerFitConfig"]["properties"]["max_angular_gap_rad"]["x-unit"],
        "rad"
    );
    assert_eq!(
        defs["OuterFitConfig"]["properties"]["size_score_weight"]["maximum"],
        1.0
    );
}

// ── Target spec ──────────────────────────────────────────────────────

fn target_validator() -> Validator {
    validator(target_spec_schema())
}

fn target_value(target: &TargetLayout) -> Value {
    serde_json::from_str(&target.to_json_string()).expect("target JSON parses")
}

#[test]
fn target_presets_and_builders_validate() {
    let v = target_validator();
    let targets = [
        ("default_hex", TargetLayout::default_hex()),
        ("rect_24x24", TargetLayout::rect_24x24()),
        ("Default", TargetLayout::default()),
        (
            "coded_hex",
            TargetLayout::coded_hex(8.0, 15, 14, 4.8, 3.2, 1.152).unwrap(),
        ),
        (
            "coded_rect",
            TargetLayout::coded_rect(10.0, 6, 8, 4.0, 2.5, 0.8).unwrap(),
        ),
        (
            "plain_hex dots",
            TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, OriginDots::Auto).unwrap(),
        ),
        (
            "plain_hex no dots",
            TargetLayout::plain_hex(8.0, 15, 14, 4.8, 3.2, OriginDots::None).unwrap(),
        ),
        (
            "plain_rect dots",
            TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto).unwrap(),
        ),
        (
            "plain_rect no dots",
            TargetLayout::plain_rect(14.0, 12, 12, 5.6, 2.8, OriginDots::None).unwrap(),
        ),
        (
            "renamed",
            TargetLayout::default_hex().with_name("my board").unwrap(),
        ),
    ];
    for (name, target) in targets {
        let doc = target_value(&target);
        assert_valid(&v, &doc, name);
        // And the library reads back what the schema accepted.
        let reloaded = TargetLayout::from_json_str(&doc.to_string()).unwrap();
        assert_eq!(reloaded.cells(), target.cells(), "{name} round trip");
    }
}

#[test]
fn coded_target_with_profile_and_id_assignment_validates() {
    let v = target_validator();
    let target = TargetLayout::new(
        "assigned",
        LatticeGeometry::Hex(HexGeometry {
            rows: 3,
            long_row_cols: 4,
            pitch_mm: 8.0,
        }),
        RingGeometry {
            outer_radius_mm: 4.8,
            inner_radius_mm: 3.2,
        },
        MarkerCoding::Coded16(CodedRingSpec {
            ring_width_mm: 1.0,
            codebook_profile: CodebookProfile::Extended,
            id_assignment: Some(vec![9, 4, 7, 1, 0, 2, 3, 5, 6, 8, 10]),
        }),
        None,
    )
    .unwrap();
    let doc = target_value(&target);
    assert_eq!(doc["coding"]["codebook_profile"], "extended");
    assert!(doc["coding"]["id_assignment"].is_array());
    assert_valid(&v, &doc, "extended profile with id_assignment");

    let with_dots = TargetLayout::plain_rect(14.0, 24, 24, 5.6, 2.8, OriginDots::Auto)
        .unwrap()
        .with_name("dots")
        .unwrap();
    assert!(with_dots.fiducials().is_some());
    assert_valid(&v, &target_value(&with_dots), "plain rect with fiducials");
}

#[test]
fn target_schema_pins_the_schema_string() {
    let schema = target_spec_schema().to_value();
    assert_eq!(
        schema["properties"]["schema"]["const"],
        "ringgrid.target.v6"
    );
    assert_eq!(
        schema["properties"]["schema"]["const"],
        ringgrid::TARGET_SCHEMA_VERSION
    );
    assert_eq!(schema["additionalProperties"], false);
    let required: Vec<&str> = schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for field in ["schema", "name", "lattice", "marker", "coding"] {
        assert!(required.contains(&field), "{field} must be required");
    }
    assert!(!required.contains(&"fiducials"), "fiducials is optional");
}

#[test]
fn target_schema_rejects_invalid_documents() {
    let v = target_validator();
    let good = target_value(&TargetLayout::default_hex());
    assert_valid(&v, &good, "baseline");

    let mutate = |f: &dyn Fn(&mut Value)| {
        let mut doc = good.clone();
        f(&mut doc);
        doc
    };
    for (what, doc) in [
        (
            "legacy v5 schema string",
            mutate(&|d| d["schema"] = json!("ringgrid.target.v5")),
        ),
        (
            "missing schema",
            mutate(&|d| {
                d.as_object_mut().unwrap().remove("schema");
            }),
        ),
        (
            "missing lattice",
            mutate(&|d| {
                d.as_object_mut().unwrap().remove("lattice");
            }),
        ),
        (
            "unknown top-level field",
            mutate(&|d| d["extra"] = json!(1)),
        ),
        (
            "unknown lattice kind",
            mutate(&|d| d["lattice"]["kind"] = json!("triangular")),
        ),
        (
            "unknown coding kind",
            mutate(&|d| d["coding"]["kind"] = json!("coded32")),
        ),
        ("zero rows", mutate(&|d| d["lattice"]["rows"] = json!(0))),
        (
            "non-positive pitch",
            mutate(&|d| d["lattice"]["pitch_mm"] = json!(0.0)),
        ),
        (
            "non-positive radius",
            mutate(&|d| d["marker"]["outer_radius_mm"] = json!(-1.0)),
        ),
        (
            "string radius",
            mutate(&|d| d["marker"]["inner_radius_mm"] = json!("3")),
        ),
        (
            "hex field on rect lattice only",
            mutate(&|d| {
                d["lattice"] = json!({ "kind": "rect", "rows": 3, "pitch_mm": 8.0 });
            }),
        ),
        (
            "fiducials with stored dots",
            mutate(&|d| d["fiducials"] = json!({ "dot_radius_mm": 1.0, "dots_mm": [[0.0, 0.0]] })),
        ),
    ] {
        assert_invalid(&v, &doc, what);
    }
}

// ── Render options ───────────────────────────────────────────────────

fn render_validator() -> Validator {
    validator(schema_for!(TargetRenderOptions))
}

#[test]
fn render_options_validate() {
    let v = render_validator();
    let defaults = serde_json::to_value(TargetRenderOptions::default()).unwrap();
    assert_valid(&v, &defaults, "TargetRenderOptions::default()");

    let sizes = [
        PageSize::FitContent,
        PageSize::A4,
        PageSize::Letter,
        PageSize::Custom {
            width_mm: 300.0,
            height_mm: 420.0,
        },
    ];
    for size in sizes {
        for orientation in [PageOrientation::Portrait, PageOrientation::Landscape] {
            let options = TargetRenderOptions::default()
                .with_page(
                    PageSpec::new(size)
                        .with_orientation(orientation)
                        .with_margin_mm(7.5),
                )
                .with_scale_bar(false)
                .with_png_dpi(600.0);
            let doc = serde_json::to_value(options).unwrap();
            assert_valid(&v, &doc, &format!("{size:?} {orientation:?}"));
            let back: TargetRenderOptions = serde_json::from_value(doc).unwrap();
            assert_eq!(back, options);
        }
    }

    // The documented partial and empty forms.
    for doc in [
        json!({}),
        json!({ "png_dpi": 150 }),
        json!({ "page": { "size": { "kind": "a4" } } }),
        json!({ "page": { "orientation": "landscape", "margin_mm": 10.0 } }),
    ] {
        assert_valid(&v, &doc, "partial render options");
        serde_json::from_value::<TargetRenderOptions>(doc).expect("library accepts it too");
    }
}

#[test]
fn render_options_schema_rejects_invalid_documents() {
    let v = render_validator();
    for (what, doc) in [
        ("unknown field", json!({ "dpi": 300 })),
        ("zero dpi", json!({ "png_dpi": 0 })),
        ("negative dpi", json!({ "png_dpi": -300.0 })),
        ("negative margin", json!({ "page": { "margin_mm": -1.0 } })),
        (
            "unknown page size",
            json!({ "page": { "size": { "kind": "a0" } } }),
        ),
        (
            "custom size without height",
            json!({ "page": { "size": { "kind": "custom", "width_mm": 100.0 } } }),
        ),
        (
            "zero-width custom size",
            json!({ "page": { "size": { "kind": "custom", "width_mm": 0.0, "height_mm": 100.0 } } }),
        ),
        (
            "unknown orientation",
            json!({ "page": { "orientation": "diagonal" } }),
        ),
        ("string for bool", json!({ "include_scale_bar": "yes" })),
    ] {
        assert_invalid(&v, &doc, what);
    }
}

/// Fields the schema marks `readOnly` are re-derived from `marker_scale` and the
/// target whenever a config is loaded, so a form must not offer them as inputs.
/// This pins both halves: the annotation is present, and the claim is true.
#[test]
fn read_only_fields_are_overwritten_on_load() {
    let schema = schema_for!(DetectConfig).to_value();
    let defs = &schema["$defs"];
    let derived = [
        ("ProposalConfig", "proposal", "r_min", json!(123.0)),
        ("ProposalConfig", "proposal", "r_max", json!(123.0)),
        ("ProposalConfig", "proposal", "min_distance", json!(123.0)),
        ("EdgeSampleConfig", "edge_sample", "r_min", json!(123.0)),
        ("EdgeSampleConfig", "edge_sample", "r_max", json!(123.0)),
        (
            "CompletionConfig",
            "completion",
            "roi_radius_px",
            json!(123.0),
        ),
        (
            "OuterEstimationConfig",
            "outer_estimation",
            "search_halfwidth_px",
            json!(0.5),
        ),
        (
            "MarkerSpecConfig",
            "marker_spec",
            "r_inner_expected",
            json!(0.9),
        ),
        ("DecodeConfig", "decode", "code_band_ratio", json!(0.5)),
        (
            "DecodeConfig",
            "decode",
            "codebook_profile",
            json!("extended"),
        ),
    ];
    let base = DetectConfig::default();
    for (def, section, field, value) in derived {
        assert_eq!(
            defs[def]["properties"][field]["readOnly"], true,
            "{def}.{field} must be readOnly in the schema"
        );
        let overlay = json!({ "advanced": { section: { field: value.clone() } } });
        let loaded = base.with_json_overlay(overlay).unwrap();
        let after = serde_json::to_value(&loaded).unwrap();
        assert_ne!(
            after["advanced"][section][field], value,
            "advanced.{section}.{field} is documented as derived but kept the supplied value"
        );
    }
}
