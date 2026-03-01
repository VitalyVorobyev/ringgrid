from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import ringgrid
from ringgrid import viz


REPO_ROOT = Path(__file__).resolve().parents[3]
BOARD_JSON = REPO_ROOT / "tools" / "board" / "board_spec.json"
SAMPLE_IMAGE = REPO_ROOT / "testdata" / "target_3_split_00.png"


def test_board_layout_default_and_from_json() -> None:
    board_default = ringgrid.BoardLayout.default()
    assert board_default.schema == "ringgrid.target.v3"
    assert board_default.rows > 0
    assert board_default.long_row_cols > 0
    assert len(board_default.markers) > 0

    board_file = ringgrid.BoardLayout.from_json_file(BOARD_JSON)
    assert board_file.schema == "ringgrid.target.v3"
    assert board_file.rows == 15
    assert board_file.long_row_cols == 14
    assert len(board_file.markers) > 0


def test_detect_config_curated_properties_and_dict_roundtrip() -> None:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)

    baseline = cfg.to_dict()
    assert "decode" in baseline
    assert "completion" in baseline

    cfg.completion_enable = False
    cfg.use_global_filter = False
    cfg.decode_min_margin = 2
    cfg.decode_max_dist = 2
    cfg.decode_min_confidence = 0.4
    cfg.circle_refinement = ringgrid.CircleRefinementMethod.PROJECTIVE_CENTER
    cfg.inner_fit_required = True
    cfg.homography_inlier_threshold_px = 4.0

    d = cfg.to_dict()
    assert d["completion"]["enable"] is False
    assert d["use_global_filter"] is False
    assert d["decode"]["min_decode_margin"] == 2
    assert d["decode"]["max_decode_dist"] == 2
    assert d["decode"]["min_decode_confidence"] == pytest.approx(0.4)
    assert d["inner_fit"]["require_inner_fit"] is True
    assert d["ransac_homography"]["inlier_threshold"] == pytest.approx(4.0)

    cfg2 = ringgrid.DetectConfig.from_dict(board, {"decode": {"min_decode_margin": 1}})
    assert cfg2.decode_min_margin == 1

    cfg2.update_from_dict({"completion": {"enable": False}})
    assert cfg2.completion_enable is False


def _make_detector() -> ringgrid.Detector:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)
    cfg.update_from_dict({"completion": {"enable": False}, "use_global_filter": False})
    return ringgrid.Detector(board, cfg)


def test_detect_accepts_grayscale_rgb_rgba_uint8_arrays() -> None:
    detector = _make_detector()

    gray = np.zeros((96, 128), dtype=np.uint8)
    res_gray = detector.detect(gray)
    assert isinstance(res_gray, ringgrid.DetectionResult)

    rgb = np.zeros((96, 128, 3), dtype=np.uint8)
    res_rgb = detector.detect(rgb)
    assert isinstance(res_rgb, ringgrid.DetectionResult)

    rgba = np.zeros((96, 128, 4), dtype=np.uint8)
    res_rgba = detector.detect(rgba)
    assert isinstance(res_rgba, ringgrid.DetectionResult)


def test_detect_rejects_invalid_array_dtype_and_shape() -> None:
    detector = _make_detector()

    bad_dtype = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(TypeError):
        detector.detect(bad_dtype)

    bad_shape = np.zeros((64, 64, 2), dtype=np.uint8)
    with pytest.raises(TypeError):
        detector.detect(bad_shape)


def test_detect_path_smoke() -> None:
    detector = _make_detector()
    result = detector.detect(SAMPLE_IMAGE)
    assert isinstance(result, ringgrid.DetectionResult)
    assert result.image_size[0] > 0
    assert result.image_size[1] > 0


def test_detect_with_mapper_camera_and_division() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    cam = ringgrid.CameraModel(
        intrinsics=ringgrid.CameraIntrinsics(fx=900.0, fy=900.0, cx=64.0, cy=48.0),
        distortion=ringgrid.RadialTangentialDistortion(k1=-0.1, k2=0.02, p1=0.0, p2=0.0, k3=0.0),
    )
    res_cam = detector.detect_with_mapper(gray, cam)
    assert isinstance(res_cam, ringgrid.DetectionResult)

    division = ringgrid.DivisionModel(lambda_=-1e-7, cx=64.0, cy=48.0)
    res_div = detector.detect_with_mapper(gray, division)
    assert isinstance(res_div, ringgrid.DetectionResult)


def test_scale_tier_and_scale_tiers_roundtrip() -> None:
    tier = ringgrid.ScaleTier(diameter_min_px=12.0, diameter_max_px=48.0)
    tier_roundtrip = ringgrid.ScaleTier.from_dict(tier.to_dict())
    assert tier_roundtrip.to_dict() == tier.to_dict()

    tiers = ringgrid.ScaleTiers(
        tiers=[
            ringgrid.ScaleTier(diameter_min_px=10.0, diameter_max_px=30.0),
            ringgrid.ScaleTier(diameter_min_px=26.0, diameter_max_px=72.0),
        ]
    )
    tiers_roundtrip = ringgrid.ScaleTiers.from_dict(tiers.to_dict())
    assert tiers_roundtrip.to_dict() == tiers.to_dict()


def test_detect_adaptive_smoke_array_and_path() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    result_array = detector.detect_adaptive(gray)
    assert isinstance(result_array, ringgrid.DetectionResult)

    result_path = detector.detect_adaptive(SAMPLE_IMAGE)
    assert isinstance(result_path, ringgrid.DetectionResult)


def test_detect_adaptive_with_optional_hint() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    result_no_hint = detector.detect_adaptive(gray, None)
    assert isinstance(result_no_hint, ringgrid.DetectionResult)

    result_with_hint = detector.detect_adaptive(gray, 32.0)
    assert isinstance(result_with_hint, ringgrid.DetectionResult)


def test_detect_adaptive_with_hint_rejects_invalid_values() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    for value in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            detector.detect_adaptive(gray, value)


def test_detect_adaptive_with_hint_alias_warns_and_works() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    with pytest.warns(DeprecationWarning):
        result = detector.detect_adaptive_with_hint(gray, 32.0)
    assert isinstance(result, ringgrid.DetectionResult)


def test_adaptive_tiers_helper_smoke_and_validation() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    tiers_auto = detector.adaptive_tiers(gray)
    assert isinstance(tiers_auto, ringgrid.ScaleTiers)
    assert len(tiers_auto.tiers) > 0

    tiers_hint = detector.adaptive_tiers(gray, 32.0)
    assert isinstance(tiers_hint, ringgrid.ScaleTiers)
    assert len(tiers_hint.tiers) == 2

    for bad_value in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            detector.adaptive_tiers(gray, bad_value)


def test_detect_multiscale_presets_and_custom_tiers() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    wide = ringgrid.ScaleTiers.four_tier_wide()
    result_wide = detector.detect_multiscale(gray, wide)
    assert isinstance(result_wide, ringgrid.DetectionResult)

    standard = ringgrid.ScaleTiers.two_tier_standard()
    result_standard = detector.detect_multiscale(gray, standard)
    assert isinstance(result_standard, ringgrid.DetectionResult)

    custom = ringgrid.ScaleTiers(
        tiers=[
            ringgrid.ScaleTier(diameter_min_px=12.0, diameter_max_px=40.0),
            ringgrid.ScaleTier(diameter_min_px=36.0, diameter_max_px=90.0),
        ]
    )
    result_custom = detector.detect_multiscale(gray, custom)
    assert isinstance(result_custom, ringgrid.DetectionResult)


def test_detect_multiscale_rejects_invalid_tiers() -> None:
    detector = _make_detector()
    gray = np.zeros((96, 128), dtype=np.uint8)

    with pytest.raises(ValueError):
        ringgrid.ScaleTier(diameter_min_px=0.0, diameter_max_px=40.0)

    with pytest.raises(ValueError):
        ringgrid.ScaleTiers(tiers=[])

    with pytest.raises(TypeError):
        ringgrid.ScaleTiers(tiers=[{"diameter_min_px": 12.0, "diameter_max_px": 40.0}])

    with pytest.raises(TypeError):
        detector.detect_multiscale(gray, "bad_tiers")


def test_detection_result_roundtrips() -> None:
    detector = _make_detector()
    gray = np.zeros((80, 120), dtype=np.uint8)
    result = detector.detect(gray)

    data = result.to_dict()
    assert "detected_markers" in data
    assert "center_frame" in data
    assert "homography_frame" in data

    result2 = ringgrid.DetectionResult.from_dict(data)
    assert result2.to_dict() == data

    json_text = result.to_json()
    assert isinstance(json_text, str)
    parsed = json.loads(json_text)
    assert parsed["image_size"] == data["image_size"]

    result3 = ringgrid.DetectionResult.from_json(json_text)
    assert result3.to_dict() == data


def test_detection_result_to_json_path_and_plot(tmp_path: Path) -> None:
    detector = _make_detector()
    gray = np.zeros((100, 140), dtype=np.uint8)
    result = detector.detect(gray)

    out_json = tmp_path / "det.json"
    saved = result.to_json(out_json)
    assert saved is None
    assert out_json.exists()

    loaded = ringgrid.DetectionResult.from_json(out_json)
    assert loaded.to_dict() == result.to_dict()

    out_plot_1 = tmp_path / "plot_method.png"
    result.plot(image=gray, out=out_plot_1)
    assert out_plot_1.exists()
    assert out_plot_1.stat().st_size > 0

    out_plot_2 = tmp_path / "plot_func.png"
    viz.plot_detection(image=gray, detection=result, out=out_plot_2)
    assert out_plot_2.exists()
    assert out_plot_2.stat().st_size > 0
