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

fn assert_voting_modes_match(img: &GrayImage, config: &ProposalConfig) {
    let scalar = find_proposals_with_mode(img, config, VotingMode::Scalar);
    let parallel = find_proposals_with_mode(img, config, VotingMode::Parallel);
    let scalar_result = find_proposals_result_with_mode(img, config, VotingMode::Scalar);
    let parallel_result = find_proposals_result_with_mode(img, config, VotingMode::Parallel);

    assert_eq!(parallel.len(), scalar.len());
    for (lhs, rhs) in parallel.iter().zip(&scalar) {
        assert_eq!(lhs.x, rhs.x);
        assert_eq!(lhs.y, rhs.y);
        assert!(
            (lhs.score - rhs.score).abs() <= 0.05,
            "proposal score drift too large at ({}, {}): {} vs {}",
            lhs.x,
            lhs.y,
            lhs.score,
            rhs.score
        );
    }

    assert_eq!(scalar_result.proposals.len(), scalar.len());
    assert_eq!(parallel_result.proposals.len(), parallel.len());
    for (lhs, rhs) in scalar_result.proposals.iter().zip(&scalar) {
        assert_eq!(lhs, rhs);
    }
    for (lhs, rhs) in parallel_result.proposals.iter().zip(&parallel) {
        assert_eq!(lhs, rhs);
    }
    let max_accum_diff = parallel_result
        .heatmap
        .iter()
        .zip(&scalar_result.heatmap)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_accum_diff <= 0.05,
        "parallel heatmap drift too large: max diff = {max_accum_diff}"
    );
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
    let best_idx = best.y as usize * result.image_size[0] as usize + best.x as usize;
    assert_eq!(result.heatmap[best_idx], best.score);

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
fn fused_scharr_matches_imageproc_on_image_interior() {
    let mut img = GrayImage::new(7, 6);
    for y in 0..img.height() {
        for x in 0..img.width() {
            let value = ((x * 17 + y * 29 + x * y * 3) % 251) as u8;
            img.put_pixel(x, y, Luma([value]));
        }
    }

    let (gx, gy, _) = gradient::build_scharr_gradients(&img);
    let gx_ref = imageproc::gradients::horizontal_scharr(&img);
    let gy_ref = imageproc::gradients::vertical_scharr(&img);
    let stride = img.width() as usize;
    let height = img.height() as usize;

    for y in 1..height - 1 {
        let row = y * stride;
        for x in 1..stride - 1 {
            let idx = row + x;
            assert_eq!(gx[idx], gx_ref.as_raw()[idx], "gx mismatch at ({x},{y})");
            assert_eq!(gy[idx], gy_ref.as_raw()[idx], "gy mismatch at ({x},{y})");
        }
    }
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
fn suppress_proposals_by_distance_drops_lower_scored_close_candidates() {
    let proposals = vec![
        Proposal {
            x: 10.0,
            y: 10.0,
            score: 9.0,
        },
        Proposal {
            x: 16.0,
            y: 10.0,
            score: 8.0,
        },
        Proposal {
            x: 40.0,
            y: 40.0,
            score: 7.0,
        },
        Proposal {
            x: 10.0,
            y: 29.0,
            score: 6.0,
        },
    ];

    let filtered = nms::suppress_proposals_by_distance(&proposals, 8.0);
    assert_eq!(
        filtered,
        vec![
            Proposal {
                x: 10.0,
                y: 10.0,
                score: 9.0,
            },
            Proposal {
                x: 40.0,
                y: 40.0,
                score: 7.0,
            },
            Proposal {
                x: 10.0,
                y: 29.0,
                score: 6.0,
            },
        ]
    );
}

#[test]
fn suppress_proposals_by_distance_keeps_candidates_at_threshold() {
    let proposals = vec![
        Proposal {
            x: 10.0,
            y: 10.0,
            score: 9.0,
        },
        Proposal {
            x: 20.0,
            y: 10.0,
            score: 8.0,
        },
    ];

    let filtered = nms::suppress_proposals_by_distance(&proposals, 10.0);
    assert_eq!(filtered, proposals);
}

#[test]
fn max_candidates_is_applied_after_distance_suppression() {
    let w = 80u32;
    let h = 40u32;
    let y = 16usize;
    let mut smoothed = vec![0.0f32; (w * h) as usize];
    for (x, score) in [(12usize, 10.0f32), (23, 9.0), (40, 8.0), (51, 7.0)] {
        smoothed[y * w as usize + x] = score;
    }

    let config = ProposalConfig {
        min_distance: 20.0,
        min_vote_frac: 0.05,
        max_candidates: Some(2),
        ..ProposalConfig::default()
    };

    let extracted = nms::extract_proposals_from_smoothed(&smoothed, w, h, &config);
    assert_eq!(
        extracted.len(),
        4,
        "peak extraction should not pre-truncate"
    );

    let mut final_proposals = nms::suppress_proposals_by_distance(&extracted, config.min_distance);
    nms::truncate_proposals(&mut final_proposals, config.max_candidates);

    assert_eq!(
        final_proposals,
        vec![
            Proposal {
                x: 12.0,
                y: y as f32,
                score: 10.0,
            },
            Proposal {
                x: 40.0,
                y: y as f32,
                score: 8.0,
            },
        ]
    );
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
fn fixture_proposals_match_result_in_scalar_mode() {
    let img = load_fixture_image();
    let config = ProposalConfig {
        edge_thinning: false,
        ..ProposalConfig::default()
    };

    let fast = find_ellipse_centers(&img, &config);
    let fast_forced = find_proposals_with_mode(&img, &config, VotingMode::Scalar);
    let result = find_proposals_result_with_mode(&img, &config, VotingMode::Scalar);

    assert!(!fast.is_empty(), "fixture should produce proposals");
    assert_eq!(fast, fast_forced);
    assert_eq!(result.proposals, fast);
    assert_eq!(
        result.heatmap.len(),
        img.width() as usize * img.height() as usize
    );

    let best = result.proposals[0];
    let best_idx = best.y as usize * img.width() as usize + best.x as usize;
    assert_eq!(result.heatmap[best_idx], best.score);
}

#[test]
fn ring_image_parallel_voting_matches_scalar() {
    let img = make_ring_image(160, 160, 79.0, 81.0, 20.0, 3.5);
    let config = ProposalConfig {
        r_min: 8.0,
        r_max: 28.0,
        grad_threshold: 0.03,
        min_distance: 6.0,
        min_vote_frac: 0.05,
        accum_sigma: 1.5,
        max_candidates: Some(32),
        edge_thinning: false,
    };

    assert_voting_modes_match(&img, &config);
}

#[test]
fn fixture_parallel_voting_matches_scalar() {
    let img = load_fixture_image();
    let config = ProposalConfig {
        edge_thinning: false,
        ..ProposalConfig::default()
    };
    assert_voting_modes_match(&img, &config);
}

// ── Edge thinning tests ──────────────────────────────────────────────

#[test]
fn edge_thinning_reduces_strong_edge_count() {
    let img = make_ring_image(160, 160, 79.0, 81.0, 20.0, 3.5);
    let (gx_orig, gy_orig, max_orig) = gradient::build_scharr_gradients(&img);

    let stride = img.width() as usize;
    let height = img.height() as usize;
    let threshold_sq = (0.05 * max_orig.sqrt()) * (0.05 * max_orig.sqrt());

    let edges_before =
        gradient::collect_strong_edges(&gx_orig, &gy_orig, stride, height, threshold_sq);

    let (mut gx, mut gy, _) = gradient::build_scharr_gradients(&img);
    let thinned_max = gradient::thin_edges_along_gradient(&mut gx, &mut gy, stride, height);
    let thinned_threshold_sq = (0.05 * thinned_max.sqrt()) * (0.05 * thinned_max.sqrt());
    let edges_after =
        gradient::collect_strong_edges(&gx, &gy, stride, height, thinned_threshold_sq);

    assert!(
        edges_after.len() < edges_before.len(),
        "thinning should reduce edges: {} >= {}",
        edges_after.len(),
        edges_before.len()
    );
    // Expect at least 40% reduction on a ring image
    let reduction = 1.0 - edges_after.len() as f64 / edges_before.len() as f64;
    assert!(
        reduction > 0.4,
        "expected >40% edge reduction, got {:.1}%",
        reduction * 100.0
    );
}

#[test]
fn edge_thinning_preserves_ring_center_detection() {
    let cx = 80.0f32;
    let cy = 80.0f32;
    let img = make_ring_image(160, 160, cx, cy, 20.0, 3.5);

    let config_thinned = ProposalConfig {
        r_min: 8.0,
        r_max: 28.0,
        min_distance: 6.0,
        grad_threshold: 0.03,
        min_vote_frac: 0.05,
        accum_sigma: 1.5,
        max_candidates: Some(32),
        edge_thinning: true,
    };
    let config_no_thin = ProposalConfig {
        edge_thinning: false,
        ..config_thinned.clone()
    };

    let proposals_thinned = find_ellipse_centers(&img, &config_thinned);
    let proposals_no_thin = find_ellipse_centers(&img, &config_no_thin);

    assert!(
        !proposals_thinned.is_empty(),
        "thinned should still find proposals"
    );
    assert!(
        !proposals_no_thin.is_empty(),
        "unthinned should find proposals"
    );

    // Both should find the center within tolerance
    let best_thin = &proposals_thinned[0];
    let err_thin = ((best_thin.x - cx).powi(2) + (best_thin.y - cy).powi(2)).sqrt();
    assert!(
        err_thin < 5.0,
        "thinned best at ({}, {}), err={err_thin}",
        best_thin.x,
        best_thin.y
    );

    let best_no = &proposals_no_thin[0];
    let err_no = ((best_no.x - cx).powi(2) + (best_no.y - cy).powi(2)).sqrt();
    assert!(
        err_no < 5.0,
        "unthinned best at ({}, {}), err={err_no}",
        best_no.x,
        best_no.y
    );
}

#[test]
fn edge_thinning_on_fixture_still_produces_proposals() {
    let img = load_fixture_image();
    let config = ProposalConfig {
        r_min: 8.0,
        r_max: 28.0,
        min_distance: 12.0,
        edge_thinning: true,
        ..ProposalConfig::default()
    };

    let proposals = find_ellipse_centers(&img, &config);
    assert!(
        !proposals.is_empty(),
        "fixture with thinning should still produce proposals"
    );
}
