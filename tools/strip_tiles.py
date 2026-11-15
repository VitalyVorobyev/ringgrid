"""Shared helpers for the rtv3d horizontal-strip layout.

The rtv3d dataset stores N camera views side by side in one wide strip
(4320x540 = 6 tiles of 720x540). ``run_rtv3d_eval.py`` and the review tool
(``tools/review/server.py``) both split strips through this module, so a tile
means the same pixels in both. Pillow is imported lazily: modules that only
need the constants stay importable without it.
"""

from __future__ import annotations

from pathlib import Path

# Strips the evaluation skips (zero-detection, low-contrast frames).
SKIP_STRIPS = {"target_0", "target_7"}

# Number of cameras (tiles) per strip.
N_CAMERAS = 6


def tile_name(strip_stem: str, index: int) -> str:
    """Canonical tile name, also the stem of the tile PNG on disk."""
    return f"{strip_stem}_{index:02d}"


def _pil_image():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Pillow is required. Run: pip install Pillow") from exc
    return Image


def crop_tile(gray, index: int, n_parts: int):
    """Crop tile ``index`` out of a grayscale strip image."""
    w, h = gray.size
    assert w % n_parts == 0, f"Width {w} not divisible by {n_parts}"
    part_w = w // n_parts
    return gray.crop((index * part_w, 0, (index + 1) * part_w, h))


def split_strip(strip_path: Path, out_dir: Path, n_parts: int = 6) -> list[Path]:
    """Split a horizontal strip into n_parts tiles, save as grayscale PNGs."""
    Image = _pil_image()
    with Image.open(strip_path) as img:
        gray = img.convert("L")
        tiles = []
        for i in range(n_parts):
            out = out_dir / f"{tile_name(strip_path.stem, i)}.png"
            crop_tile(gray, i, n_parts).save(out)
            tiles.append(out)
    return tiles


def save_tile(strip_path: Path, index: int, n_parts: int, out_path: Path) -> None:
    """Write a single tile of a strip as a grayscale PNG."""
    Image = _pil_image()
    with Image.open(strip_path) as img:
        crop_tile(img.convert("L"), index, n_parts).save(out_path)
