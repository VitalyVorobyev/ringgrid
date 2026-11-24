import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_center_metrology import score


def row(capture, method, key, x):
    return dict(dataset="synthetic", camera="a", capture=capture, split="holdout",
                condition="gradient+", method=method, key=key, center=[x, 0],
                reference=[0, 0], cov=[[1, 0], [0, 1]], status="ok")


def test_capture_weighting_and_common_support():
    rows = [row("a", method, key, x) for key in range(10)
            for method, x in [("per_image", 1), ("moment", 0)]]
    rows += [row("b", "per_image", 1, -1), row("b", "moment", 1, 0),
             dict(row("b", "moment", 2, 0), status="wrong_id")]
    result = score(rows, "per_image")["results"][0]
    effect = result["paired"]["moment"]["paired_capture_effect"]
    assert effect["reduction_in_bias_norm_px"] == 0
    assert effect["n_captures"] == 2
    assert result["methods"]["moment"]["status_counts"]["wrong_id"] == 1
    assert result["paired"]["moment"]["common_candidate_error"]["n"] == 11
    assert result["methods"]["per_image"]["coverage_95"] == 1


def test_conflicting_split_or_duplicate_fails():
    observation = row("a", "per_image", 1, 0)
    with pytest.raises(ValueError, match="leakage"):
        score([observation, dict(observation, split="train")], "per_image")
    with pytest.raises(ValueError, match="duplicate"):
        score([observation, observation], "per_image")


def test_real_image_drift_does_not_validate_absolute_covariance():
    sample = dict(row("a", "per_image", 1, .1), dataset="ipgs")
    measured = score([sample], "per_image")["results"][0]["methods"]["per_image"]
    assert measured["covariance_n"] == 0
    assert measured["error"]["mean_xy_px"] == [.1, 0]


def test_planar_proxy_excludes_training_cells_and_uses_both_estimators():
    rows = []
    for i in range(6):
        for j in range(5):
            for method, offset in [("per_image", 0), ("nominal", .25)]:
                r = row("view", method, [i, j], 0)
                r.update(dataset="ipgs", board_xy_mm=[i, j],
                         center=[i * 2 + offset, j * 3], reference=None)
                rows.append(r)
    paired = score(rows, "per_image")["results"][0]["paired"]["nominal"]
    assert paired["intermethod_displacement"]["rms_px"] == .25
    proxy = paired["heldout_plane_proxy"]
    assert proxy["n_views"] == 1
    assert proxy["baseline"]["n"] == 10
    assert proxy["candidate"]["rms_px"] < 1e-10


def test_repeatability_excludes_between_feature_bias():
    rows = []
    for key, offset in enumerate([10, 20]):
        for rep, x in enumerate([offset - 1, offset + 1]):
            rows.append(dict(row("a", "per_image", key, x), repeat_group="fixed", replicate=rep))
    method = score(rows, "per_image")["results"][0]["methods"]["per_image"]
    assert method["repeatability_df"] == 2
    assert method["repeatability_sd_xy_px"] == pytest.approx([2 ** .5, 0])


def test_gradient_reversal_pairs_markers_and_clusters_shared_scene():
    rows = []
    for pose in range(3):
        for target in ("coded", "plain"):
            for cond, y in [("positive_gradient", -.1), ("negative_gradient", .1)]:
                r = row(f"{target}-{pose}", "per_image", 1, .04)
                r.update(scene_id=pose, target=target, condition=cond, center=[.04, y])
                rows.append(r)
    odd = score(rows, "per_image")["gradient_reversal"]["per_image"]
    assert odd["n"] == 6
    assert odd["n_scenes"] == 3
    assert odd["odd_xy_px"] == pytest.approx([0, -.1])
