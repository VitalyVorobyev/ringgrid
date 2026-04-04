//! Tests for the proposal module.

use super::*;
use image::{ImageReader, Luma};
use std::path::Path;

/// Create a small synthetic image with a dark ring on a bright background.
fn make_ring_image(w: u32, h: u32, cx: f32, cy: f32, radius: f32, ring_width: f32) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let val = if (dist - radius).abs() < ring_width {
                30u8 // dark ring
            } else {
                200u8 // bright background
            };
            img.put_pixel(x, y, Luma([val]));
        }
    }
    img
}

fn load_fixture_image() -> GrayImage {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/target_3_split_00.png");
    ImageReader::open(path)
        .expect("open fixture image")
        .decode()
        .expect("decode fixture image")
        .to_luma8()
}

#[test]
fn test_ring_proposal_finds_center() {
    let cx = 40.0f32;
    let cy = 40.0f32;
    let img = make_ring_image(80, 80, cx, cy, 10.0, 3.0);

    let config = ProposalConfig {
        r_min: 5.0,
        r_max: 15.0,
        grad_threshold: 0.03,
        min_distance: 5.0,
        min_vote_frac: 0.05,
        accum_sigma: 1.5,
        max_candidates: None,
        edge_thinning: false,
    };

    let proposals = find_ellipse_centers(&img, &config);
    assert!(!proposals.is_empty(), "should find at least one proposal");

    let best = &proposals[0];
    let err = ((best.x - cx).powi(2) + (best.y - cy).powi(2)).sqrt();
    assert!(
        err < 5.0,
        "best proposal ({}, {}) should be within 5 px of true center ({}, {}), error = {}",
        best.x,
        best.y,
        cx,
        cy,
        err
    );
}

#[test]
fn proposal_result_matches_proposals_and_heatmap_shape() {
    let cx = 40.0f32;
    let cy = 40.0f32;
    let img = make_ring_image(80, 80, cx, cy, 10.0, 3.0);
    let config = ProposalConfig {
        r_min: 5.0,
        r_max: 15.0,
        grad_threshold: 0.03,
        min_distance: 5.0,
        min_vote_frac: 0.05,
        accum_sigma: 1.5,
        max_candidates: None,
        edge_thinning: false,
    };

    let plain = find_ellipse_centers(&img, &config);
    let result = find_ellipse_centers_with_heatmap(&img, &config);
    assert_eq!(result.image_size, [80, 80]);
    assert_eq!(result.heatmap.len(), 80 * 80);
    assert_eq!(result.proposals, plain);

    let best = result.proposals[0];
    let err = ((best.x - cx).powi(2) + (best.y - cy).powi(2)).sqrt();
    assert!(err < 5.0);
}

#[test]
fn proposal_result_serde_roundtrip() {
    let img = make_ring_image(64, 64, 31.0, 29.0, 8.0, 2.0);
    let result = find_ellipse_centers_with_heatmap(&img, &ProposalConfig::default());
    let json = serde_json::to_string(&result).expect("serialize");
    let roundtrip: ProposalResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(roundtrip, result);
}

#[test]
fn blank_image_yields_empty_proposals_and_zero_heatmap() {
    let img = GrayImage::new(32, 24);
    let proposals = find_ellipse_centers(&img, &ProposalConfig::default());
    let result = find_ellipse_centers_with_heatmap(&img, &ProposalConfig::default());

    assert!(proposals.is_empty());
    assert!(result.proposals.is_empty());
    assert_eq!(result.image_size, [32, 24]);
    assert!(result.heatmap.iter().all(|&v| v == 0.0));
}

#[test]
fn min_distance_enforced_on_fixture() {
    let img = load_fixture_image();
    let config = ProposalConfig {
        r_min: 8.0,
        r_max: 28.0,
        grad_threshold: 0.05,
        min_distance: 18.0,
        min_vote_frac: 0.1,
        accum_sigma: 2.0,
        max_candidates: Some(64),
        edge_thinning: false,
    };

    let proposals = find_ellipse_centers(&img, &config);
    let result = find_ellipse_centers_with_heatmap(&img, &config);

    assert_eq!(proposals, result.proposals);
    for i in 0..proposals.len() {
        for j in (i + 1)..proposals.len() {
            let dx = proposals[i].x - proposals[j].x;
            let dy = proposals[i].y - proposals[j].y;
            let distance = (dx * dx + dy * dy).sqrt();
            assert!(
                distance + 1.0e-5 >= config.min_distance,
                "proposals too close: {distance} < {}",
                config.min_distance
            );
        }
    }
}

#[test]
fn fixture_produces_proposals() {
    let img = load_fixture_image();
    let config = ProposalConfig {
        edge_thinning: false,
        ..ProposalConfig::default()
    };

    let proposals = find_ellipse_centers(&img, &config);
    let result = find_ellipse_centers_with_heatmap(&img, &config);

    assert!(!proposals.is_empty(), "fixture should produce proposals");
    assert_eq!(proposals, result.proposals);
    assert_eq!(
        result.heatmap.len(),
        img.width() as usize * img.height() as usize
    );
}
