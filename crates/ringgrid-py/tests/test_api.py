from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import ringgrid
from ringgrid import viz


REPO_ROOT = Path(__file__).resolve().parents[3]
BOARD_JSON = REPO_ROOT / "tools" / "board" / "board_spec.json"
SAMPLE_IMAGE = REPO_ROOT / "testdata" / "target_3_split_00.png"
TARGET_FIXTURE_DIR = (
    REPO_ROOT / "crates" / "ringgrid" / "tests" / "fixtures" / "target_generation"
)
TARGET_FIXTURE_JSON = TARGET_FIXTURE_DIR / "fixture_compact_hex.json"
TARGET_FIXTURE_SVG = TARGET_FIXTURE_DIR / "fixture_compact_hex.svg"
TARGET_FIXTURE_PNG = TARGET_FIXTURE_DIR / "fixture_compact_hex.png"
README_PATH = REPO_ROOT / "crates" / "ringgrid-py" / "README.md"

README_SECTION_TYPES: dict[str, type[object]] = {
    "marker_scale": ringgrid.MarkerScalePrior,
    "proposal": ringgrid.ProposalConfig,
    "edge_sample": ringgrid.EdgeSampleConfig,
    "outer_estimation": ringgrid.OuterEstimationConfig,
    "marker_spec": ringgrid.MarkerSpec,
    "outer_fit": ringgrid.OuterFitConfig,
    "inner_fit": ringgrid.InnerFitConfig,
    "decode": ringgrid.DecodeConfig,
    "seed_proposals": ringgrid.SeedProposalParams,
    "projective_center": ringgrid.ProjectiveCenterParams,
    "completion": ringgrid.CompletionParams,
    "ransac_homography": ringgrid.RansacHomographyConfig,
    "self_undistort": ringgrid.SelfUndistortConfig,
    "id_correction": ringgrid.IdCorrectionConfig,
    "inner_as_outer_recovery": ringgrid.InnerAsOuterRecoveryConfig,
}

README_ALIAS_NAMES = (
    "completion_enable",
    "self_undistort_enable",
    "inner_fit_required",
    "homography_inlier_threshold_px",
    "decode_min_margin",
    "decode_max_dist",
    "decode_min_confidence",
)


def _normalize_text_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def _markdown_section(text: str, heading: str) -> str:
    marker = f"### {heading}\n"
    start = text.find(marker)
    if start < 0:
        raise AssertionError(f"missing README heading: {heading}")
    start += len(marker)

    next_h3 = text.find("\n### ", start)
    next_h2 = text.find("\n## ", start)
    stops = [idx for idx in (next_h3, next_h2) if idx >= 0]
    end = min(stops) if stops else len(text)
    return text[start:end]


def _fixture_board() -> ringgrid.BoardLayout:
    return ringgrid.BoardLayout.from_geometry(
        8.0,
        3,
        4,
        4.8,
        3.2,
        1.152,
        name="fixture_compact_hex",
    )


def _png_phys(path: Path) -> tuple[int, int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")

    offset = 8
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + length
        chunk_data = data[chunk_data_start:chunk_data_end]
        if chunk_type == b"pHYs":
            return (
                int.from_bytes(chunk_data[0:4], "big"),
                int.from_bytes(chunk_data[4:8], "big"),
                int(chunk_data[8]),
            )
        offset = chunk_data_end + 4

    raise AssertionError("missing pHYs chunk")


def _png_pixels(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    pixels = np.asarray(mpimg.imread(path))
    if pixels.ndim == 3:
        rgb = pixels[..., :3]
        if rgb.dtype.kind == "f":
            rgb = np.rint(rgb * 255.0).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8, copy=False)
        assert np.array_equal(rgb[..., 0], rgb[..., 1])
        assert np.array_equal(rgb[..., 0], rgb[..., 2])
        return rgb[..., 0]

    if pixels.dtype.kind == "f":
        return np.rint(pixels * 255.0).astype(np.uint8)
    return pixels.astype(np.uint8, copy=False)


def test_board_layout_default_and_from_json() -> None:
    board_default = ringgrid.BoardLayout.default()
    assert board_default.schema == "ringgrid.target.v4"
    assert board_default.rows > 0
    assert board_default.long_row_cols > 0
    assert board_default.marker_ring_width_mm > 0.0
    assert len(board_default.markers) > 0

    board_file = ringgrid.BoardLayout.from_json_file(BOARD_JSON)
    assert board_file.schema == "ringgrid.target.v4"
    assert board_file.rows == 15
    assert board_file.long_row_cols == 14
    assert board_file.marker_ring_width_mm == pytest.approx(1.152)
    assert len(board_file.markers) > 0


def test_board_layout_from_geometry_matches_fixture_and_generated_name() -> None:
    board = _fixture_board()
    assert board.schema == "ringgrid.target.v4"
    assert board.name == "fixture_compact_hex"
    assert board.marker_ring_width_mm == pytest.approx(1.152)
    assert board.to_spec_dict() == json.loads(TARGET_FIXTURE_JSON.read_text())

    generated = ringgrid.BoardLayout.from_geometry(8.0, 3, 4, 4.8, 3.2, 1.152)
    assert generated.name == "ringgrid_hex_r3_c4_p8.000_o4.800_i3.200_w1.152"


def test_board_layout_spec_json_roundtrip_and_file_write(tmp_path: Path) -> None:
    board = _fixture_board()

    spec_json = board.to_spec_json()
    assert isinstance(spec_json, str)
    assert (
        _normalize_text_newlines(spec_json + "\n")
        == _normalize_text_newlines(TARGET_FIXTURE_JSON.read_text())
    )

    out_json = tmp_path / "nested" / "fixture.json"
    saved = board.to_spec_json(out_json)
    assert saved is None
    assert (
        _normalize_text_newlines(out_json.read_text())
        == _normalize_text_newlines(TARGET_FIXTURE_JSON.read_text())
    )

    loaded = ringgrid.BoardLayout.from_json_file(out_json)
    assert loaded.to_spec_dict() == board.to_spec_dict()


def test_board_layout_svg_and_png_generation_match_rust_fixtures(tmp_path: Path) -> None:
    board = _fixture_board()

    svg_path = tmp_path / "nested" / "fixture.svg"
    png_path = tmp_path / "nested" / "fixture.target"

    board.write_svg(svg_path)
    board.write_png(png_path, dpi=96.0)

    assert (
        _normalize_text_newlines(svg_path.read_text())
        == _normalize_text_newlines(TARGET_FIXTURE_SVG.read_text())
    )

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    expected_ppm = round(96.0 * 1000.0 / 25.4)
    assert _png_phys(png_path) == (expected_ppm, expected_ppm, 1)
    assert np.array_equal(_png_pixels(png_path), _png_pixels(TARGET_FIXTURE_PNG))


def test_board_layout_target_generation_tracks_mutated_spec_fields(
    tmp_path: Path,
) -> None:
    board = _fixture_board()
    original_marker_xy = list(board.markers[1].xy_mm)

    board.name = "mutated_fixture"
    board.pitch_mm = 10.0

    spec = json.loads(board.to_spec_json())
    assert spec["name"] == "mutated_fixture"
    assert spec["pitch_mm"] == pytest.approx(10.0)
    assert board.to_spec_dict() == spec
    assert board.markers[1].xy_mm != original_marker_xy

    svg_path = tmp_path / "mutated.svg"
    png_path = tmp_path / "mutated.png"
    board.write_svg(svg_path)
    board.write_png(png_path, dpi=96.0)

    assert _normalize_text_newlines(svg_path.read_text()) != _normalize_text_newlines(
        TARGET_FIXTURE_SVG.read_text()
    )
    assert _png_pixels(png_path).shape != _png_pixels(TARGET_FIXTURE_PNG).shape


def test_board_layout_target_generation_rejects_invalid_geometry_and_options(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        ringgrid.BoardLayout.from_geometry(8.0, 0, 4, 4.8, 3.2, 1.152)

    with pytest.raises(ValueError):
        ringgrid.BoardLayout.from_geometry(8.0, 2, 1, 4.8, 3.2, 1.152)

    with pytest.raises(ValueError):
        ringgrid.BoardLayout.from_geometry(8.0, 3, 4, 4.8, 4.8, 1.152)

    with pytest.raises(ValueError, match="no code band between rings"):
        ringgrid.BoardLayout.from_geometry(8.0, 3, 4, 4.8, 4.1, 1.152)

    with pytest.raises(ValueError, match="marker_ring_width_mm"):
        ringgrid.BoardLayout.from_geometry(8.0, 3, 4, 4.8, 3.2, 0.0)

    board = _fixture_board()

    with pytest.raises(ValueError):
        board.write_svg(tmp_path / "bad.svg", margin_mm=-1.0)

    for dpi in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            board.write_png(tmp_path / "bad.png", dpi=dpi)


def test_board_layout_target_generation_write_errors_raise_oserror(
    tmp_path: Path,
) -> None:
    board = _fixture_board()
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(OSError):
        board.to_spec_json(blocked_parent / "fixture.json")

    with pytest.raises(OSError):
        board.write_svg(blocked_parent / "fixture.svg")

    with pytest.raises(OSError):
        board.write_png(blocked_parent / "fixture.png")


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


def test_detect_config_typed_sections_are_settable_without_mapping_overlays() -> None:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)

    marker_scale = cfg.marker_scale
    marker_scale.diameter_min_px = 12.0
    marker_scale.diameter_max_px = 96.0
    cfg.marker_scale = marker_scale

    inner_fit = cfg.inner_fit
    inner_fit.require_inner_fit = True
    cfg.inner_fit = inner_fit

    outer_fit = cfg.outer_fit
    outer_fit.size_score_weight = 0.22
    cfg.outer_fit = outer_fit

    completion = cfg.completion
    completion.enable = False
    cfg.completion = completion

    projective_center = cfg.projective_center
    projective_center.ratio_penalty_weight = 0.45
    cfg.projective_center = projective_center

    seed_proposals = cfg.seed_proposals
    seed_proposals.max_seeds = 128
    cfg.seed_proposals = seed_proposals

    proposal = cfg.proposal
    proposal.max_candidates = 64
    cfg.proposal = proposal

    edge_sample = cfg.edge_sample
    edge_sample.n_rays = 56
    cfg.edge_sample = edge_sample

    decode = cfg.decode
    decode.codebook_profile = "extended"
    decode.min_decode_margin = 3
    cfg.decode = decode

    marker_spec = cfg.marker_spec
    marker_spec.theta_samples = 120
    cfg.marker_spec = marker_spec

    outer_estimation = cfg.outer_estimation
    outer_estimation.allow_two_hypotheses = not outer_estimation.allow_two_hypotheses
    cfg.outer_estimation = outer_estimation

    ransac_homography = cfg.ransac_homography
    ransac_homography.inlier_threshold = 5.5
    cfg.ransac_homography = ransac_homography

    self_undistort = cfg.self_undistort
    self_undistort.enable = True
    cfg.self_undistort = self_undistort

    id_correction = cfg.id_correction
    id_correction.enable = False
    cfg.id_correction = id_correction

    inner_as_outer_recovery = cfg.inner_as_outer_recovery
    inner_as_outer_recovery.enable = False
    cfg.inner_as_outer_recovery = inner_as_outer_recovery

    cfg.dedup_radius = 0.65
    cfg.max_aspect_ratio = 2.4
    cfg.use_global_filter = False

    d = cfg.to_dict()
    assert d["marker_scale"]["diameter_min_px"] == pytest.approx(12.0)
    assert d["marker_scale"]["diameter_max_px"] == pytest.approx(96.0)
    assert d["inner_fit"]["require_inner_fit"] is True
    assert d["outer_fit"]["size_score_weight"] == pytest.approx(0.22)
    assert d["completion"]["enable"] is False
    assert d["projective_center"]["ratio_penalty_weight"] == pytest.approx(0.45)
    assert d["seed_proposals"]["max_seeds"] == 128
    assert d["proposal"]["max_candidates"] == 64
    assert d["edge_sample"]["n_rays"] == 56
    assert d["decode"]["codebook_profile"] == "extended"
    assert d["decode"]["min_decode_margin"] == 3
    assert d["marker_spec"]["theta_samples"] == 120
    assert d["ransac_homography"]["inlier_threshold"] == pytest.approx(5.5)
    assert d["self_undistort"]["enable"] is True
    assert d["id_correction"]["enable"] is False
    assert d["inner_as_outer_recovery"]["enable"] is False
    assert d["dedup_radius"] == pytest.approx(0.65)
    assert d["max_aspect_ratio"] == pytest.approx(2.4)
    assert d["use_global_filter"] is False


def test_detect_config_uses_cached_snapshot_and_returns_copies() -> None:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)

    assert cfg._resolved_cache is None

    decode_first = cfg.decode
    cache = cfg._resolved_cache
    assert cache is not None

    decode_second = cfg.decode
    assert cfg._resolved_cache is cache
    assert decode_second.to_dict() == decode_first.to_dict()

    decode_first.min_decode_margin = 99
    assert cfg.decode.min_decode_margin != 99

    snapshot = cfg.to_dict()
    snapshot["decode"]["min_decode_margin"] = 123
    assert cfg.to_dict()["decode"]["min_decode_margin"] != 123
    assert cfg._resolved_cache is cache


def test_decode_config_matches_resolved_dump_surface_and_profile_override() -> None:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)

    assert cfg.decode.to_dict() == cfg.to_dict()["decode"]

    decode = cfg.decode
    decode.codebook_profile = "extended"
    decode.min_decode_margin = 2
    cfg.decode = decode

    assert cfg.decode.codebook_profile == "extended"
    assert cfg.decode.min_decode_margin == 2
    assert cfg.decode.to_dict() == cfg.to_dict()["decode"]

    roundtrip = ringgrid.DecodeConfig.from_dict(cfg.to_dict()["decode"])
    assert roundtrip.to_dict() == cfg.to_dict()["decode"]


def test_decode_config_from_dict_defaults_missing_profile_to_base() -> None:
    cfg = ringgrid.DetectConfig(ringgrid.BoardLayout.default())
    legacy_payload = dict(cfg.to_dict()["decode"])
    del legacy_payload["codebook_profile"]

    decoded = ringgrid.DecodeConfig.from_dict(legacy_payload)

    assert decoded.codebook_profile == "base"
    assert decoded.to_dict() == cfg.to_dict()["decode"]


def test_detect_config_to_dict_matches_native_dump_after_mixed_overlays() -> None:
    from ringgrid._ringgrid import DetectConfigCore as NativeDetectConfigCore

    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)
    native = NativeDetectConfigCore(board._spec_json)

    cfg.decode_min_confidence = 0.4
    native_decode = ringgrid.DecodeConfig.from_dict(
        json.loads(native.dump_json())["decode"]
    )
    native_decode.min_decode_confidence = 0.4
    native.apply_overlay_json(json.dumps({"decode": native_decode.to_dict()}))

    cfg.completion_enable = False
    native_completion = ringgrid.CompletionParams.from_dict(
        json.loads(native.dump_json())["completion"]
    )
    native_completion.enable = False
    native.apply_overlay_json(json.dumps({"completion": native_completion.to_dict()}))

    cfg.use_global_filter = False
    native.apply_overlay_json(json.dumps({"use_global_filter": False}))

    cfg.circle_refinement = ringgrid.CircleRefinementMethod.NONE
    native.apply_overlay_json(json.dumps({"circle_refinement": "None"}))

    assert cfg.to_dict() == json.loads(native.dump_json())


def test_detect_config_marker_scale_refreshes_cache_from_native() -> None:
    from ringgrid._ringgrid import DetectConfigCore as NativeDetectConfigCore

    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)
    native = NativeDetectConfigCore(board._spec_json)

    baseline = cfg.to_dict()
    prior = ringgrid.MarkerScalePrior(diameter_min_px=24.0, diameter_max_px=80.0)
    cfg.marker_scale = prior
    native.apply_overlay_json(json.dumps({"marker_scale": prior.to_dict()}))

    expected = json.loads(native.dump_json())
    assert cfg._resolved_cache == expected
    assert cfg.to_dict() == expected
    assert cfg.to_dict()["proposal"] != baseline["proposal"]


def test_readme_detect_config_field_guide_covers_python_surface() -> None:
    cfg = ringgrid.DetectConfig(ringgrid.BoardLayout.default())
    readme = README_PATH.read_text(encoding="utf-8")

    assert "## DetectConfig Field Guide" in readme
    assert "`cfg.board`" in readme

    resolved = cfg.to_dict()
    for key, value in resolved.items():
        if isinstance(value, dict):
            section = _markdown_section(readme, f"`{key}`")
            for field in dataclasses.fields(README_SECTION_TYPES[key]):
                assert f"`{field.name}`" in section
        else:
            assert f"`cfg.{key}`" in readme

    for alias in README_ALIAS_NAMES:
        assert f"`cfg.{alias}`" in readme


def test_detector_constructor_and_classmethods() -> None:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)

    detector = ringgrid.Detector(cfg)
    assert detector.config is cfg
    assert detector.board is cfg.board

    detector_from_board = ringgrid.Detector.from_board(board)
    assert isinstance(detector_from_board, ringgrid.Detector)
    assert detector_from_board.board is board
    assert detector_from_board.config.board is board

    detector_with_config = ringgrid.Detector.with_config(cfg)
    assert detector_with_config.config is cfg

    with pytest.raises(TypeError):
        ringgrid.Detector(board)
    with pytest.raises(TypeError):
        ringgrid.Detector(board, cfg)


def test_detector_refreshes_native_core_after_config_update() -> None:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)
    detector = ringgrid.Detector(cfg)

    before_version = detector._core_config_version
    cfg.decode_min_margin = cfg.decode_min_margin + 1
    assert cfg._version > before_version
    assert detector._core_config_version == before_version

    gray = np.zeros((64, 96), dtype=np.uint8)
    result = detector.detect(gray)
    assert isinstance(result, ringgrid.DetectionResult)
    assert detector._core_config_version == cfg._version


def _make_detector() -> ringgrid.Detector:
    board = ringgrid.BoardLayout.default()
    cfg = ringgrid.DetectConfig(board)
    cfg.completion_enable = False
    cfg.use_global_filter = False
    return ringgrid.Detector(cfg)


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
