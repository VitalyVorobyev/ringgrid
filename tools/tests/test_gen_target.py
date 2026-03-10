from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "tools" / "gen_target.py"
TARGET_FIXTURE_DIR = (
    REPO_ROOT / "crates" / "ringgrid" / "tests" / "fixtures" / "target_generation"
)
TARGET_FIXTURE_JSON = TARGET_FIXTURE_DIR / "fixture_compact_hex.json"
TARGET_FIXTURE_SVG = TARGET_FIXTURE_DIR / "fixture_compact_hex.svg"
TARGET_FIXTURE_PNG = TARGET_FIXTURE_DIR / "fixture_compact_hex.png"


def _normalize_text_newlines(text: str) -> str:
    return text.replace("\r\n", "\n")


def _run_gen_target(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
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


def test_gen_target_matches_committed_fixture_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "nested" / "fixture"
    result = _run_gen_target(
        "--pitch_mm",
        "8.0",
        "--rows",
        "3",
        "--long_row_cols",
        "4",
        "--marker_outer_radius_mm",
        "4.8",
        "--marker_inner_radius_mm",
        "3.2",
        "--name",
        "fixture_compact_hex",
        "--out_dir",
        str(out_dir),
        "--basename",
        "fixture_compact_hex",
        "--dpi",
        "96.0",
    )

    assert result.returncode == 0, result.stderr
    assert "Board spec JSON written to" in result.stdout
    assert "Print SVG written to" in result.stdout
    assert "Print PNG written to" in result.stdout

    json_path = out_dir / "board_spec.json"
    svg_path = out_dir / "fixture_compact_hex.svg"
    png_path = out_dir / "fixture_compact_hex.png"

    assert (
        _normalize_text_newlines(json_path.read_text())
        == _normalize_text_newlines(TARGET_FIXTURE_JSON.read_text())
    )
    assert (
        _normalize_text_newlines(svg_path.read_text())
        == _normalize_text_newlines(TARGET_FIXTURE_SVG.read_text())
    )

    expected_ppm = round(96.0 * 1000.0 / 25.4)
    assert _png_phys(png_path) == (expected_ppm, expected_ppm, 1)
    assert np.array_equal(_png_pixels(png_path), _png_pixels(TARGET_FIXTURE_PNG))


def test_gen_target_uses_generated_name_when_name_is_omitted(tmp_path: Path) -> None:
    out_dir = tmp_path / "generated_name"
    result = _run_gen_target(
        "--pitch_mm",
        "8.0",
        "--rows",
        "3",
        "--long_row_cols",
        "4",
        "--marker_outer_radius_mm",
        "4.8",
        "--marker_inner_radius_mm",
        "3.2",
        "--out_dir",
        str(out_dir),
    )

    assert result.returncode == 0, result.stderr
    spec = json.loads((out_dir / "board_spec.json").read_text())
    assert spec["name"] == "ringgrid_hex_r3_c4_p8.000_o4.800_i3.200"


def test_gen_target_rejects_invalid_geometry_and_options(tmp_path: Path) -> None:
    invalid_rows = _run_gen_target(
        "--pitch_mm",
        "8.0",
        "--rows",
        "0",
        "--long_row_cols",
        "4",
        "--marker_outer_radius_mm",
        "4.8",
        "--marker_inner_radius_mm",
        "3.2",
        "--out_dir",
        str(tmp_path / "bad_rows"),
    )
    assert invalid_rows.returncode != 0
    assert "rows" in invalid_rows.stderr

    invalid_margin = _run_gen_target(
        "--pitch_mm",
        "8.0",
        "--rows",
        "3",
        "--long_row_cols",
        "4",
        "--marker_outer_radius_mm",
        "4.8",
        "--marker_inner_radius_mm",
        "3.2",
        "--margin_mm",
        "-1.0",
        "--out_dir",
        str(tmp_path / "bad_margin"),
    )
    assert invalid_margin.returncode != 0
    assert "margin" in invalid_margin.stderr
