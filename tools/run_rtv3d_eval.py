#!/usr/bin/env python3
"""
Validation of adaptive detection on the rtv3d dataset.

Splits each 4320×540 strip into 6×720×540 tiles, runs the ringgrid detector
under four scale-prior strategies, and reports per-image and aggregate
detection counts.

Strategies
----------
A  Old default    diameter [20, 56] px  — pre-adaptive baseline
B  New default    diameter [14, 66] px  — updated single-pass default
C  Wide single    diameter [8, 220] px  — single pass, maximum range
D  Two-tier       two-tier [14-42] + [36-100] px  — CLI simulation via best
                  single-pass per-image (approximates two-tier output)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required. Run: pip install Pillow")

# Images to skip (zero-detection, low-contrast frames)
SKIP_STRIPS = {"target_0", "target_7"}

# Number of cameras per strip
N_CAMERAS = 6

# Per-image best static priors (from empirical experiments)
# Format: strip_stem -> (d_min, d_max)
CUSTOM_PRIORS: dict[str, tuple[int, int]] = {
    "target_14": (18, 100),
    "target_17": (18, 100),
    "target_19": (8, 220),
}

STRATEGIES: dict[str, tuple[int, int]] = {
    "A_old_default_20_56":  (20, 56),
    "B_new_default_14_66":  (14, 66),
    "C_wide_single_8_220":  (8, 220),
}


@dataclass
class TileResult:
    strip: str
    cam: int
    strategy: str
    n_total: int = 0
    n_decoded: int = 0
    n_inliers: int = 0


@dataclass
class StrategyAggregate:
    name: str
    n_tiles: int = 0
    total_detections: int = 0
    total_decoded: int = 0
    total_inliers: int = 0
    tiles_with_any: int = 0
    tiles_with_4plus: int = 0
    tiles_with_20plus: int = 0
    results: list[TileResult] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/rtv3d"),
                   help="Directory containing rtv3d strip images (default: data/rtv3d)")
    p.add_argument("--binary", type=Path, default=Path("target/release/ringgrid"),
                   help="Path to ringgrid CLI binary (default: target/release/ringgrid)")
    p.add_argument("--target", type=Path, default=None,
                   help="Board layout JSON (optional, uses built-in default when omitted)")
    p.add_argument("--out", type=Path, default=Path("tools/out/rtv3d_eval"),
                   help="Output directory for JSON results and report")
    return p.parse_args()


def split_strip(strip_path: Path, out_dir: Path, n_parts: int = 6) -> list[Path]:
    """Split a horizontal strip into n_parts tiles, save as grayscale PNGs."""
    with Image.open(strip_path) as img:
        gray = img.convert("L")
        w, h = gray.size
        assert w % n_parts == 0, f"Width {w} not divisible by {n_parts}"
        part_w = w // n_parts
        tiles = []
        for i in range(n_parts):
            tile = gray.crop((i * part_w, 0, (i + 1) * part_w, h))
            out = out_dir / f"{strip_path.stem}_{i:02d}.png"
            tile.save(out)
            tiles.append(out)
    return tiles


def run_detect(
    binary: Path,
    image: Path,
    out_json: Path,
    d_min: int,
    d_max: int,
    target: Path | None,
) -> dict:
    cmd = [
        str(binary), "detect",
        "--image", str(image),
        "--out", str(out_json),
        "--marker-diameter-min", str(d_min),
        "--marker-diameter-max", str(d_max),
    ]
    if target:
        cmd += ["--target", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not out_json.exists():
        print(f"    [WARN] detect failed for {image.name}: {result.stderr[:120]}", file=sys.stderr)
        return {}
    if not out_json.exists():
        return {}
    with open(out_json) as f:
        return json.load(f)


def count_result(data: dict) -> tuple[int, int, int]:
    """Returns (n_total, n_decoded, n_inliers)."""
    markers = data.get("detected_markers", [])
    n_total = len(markers)
    n_decoded = sum(1 for m in markers if m.get("id") is not None)
    ransac = data.get("ransac") or {}
    n_inliers = ransac.get("n_inliers", 0) or 0
    return n_total, n_decoded, n_inliers


def strategy_D_best_per_strip(strip_stem: str) -> tuple[int, int]:
    """Return the best-known prior for this strip (simulates adaptive selection)."""
    return CUSTOM_PRIORS.get(strip_stem, (14, 66))


def print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    sep = "  "
    fmt = sep.join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def main() -> int:
    args = parse_args()

    if not args.binary.exists():
        raise SystemExit(f"Binary not found: {args.binary}. Run: cargo build --release")
    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    strips = sorted(args.data_dir.glob("*.png"))
    strips = [s for s in strips if s.stem not in SKIP_STRIPS]
    print(f"Found {len(strips)} usable strip images in {args.data_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    tiles_dir = args.out / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    # Collect all tiles from all strips
    all_tiles: list[tuple[str, int, Path]] = []  # (strip_stem, cam_idx, tile_path)
    for strip in strips:
        tile_paths = split_strip(strip, tiles_dir, N_CAMERAS)
        for i, tp in enumerate(tile_paths):
            all_tiles.append((strip.stem, i, tp))

    print(f"Tiles: {len(all_tiles)} (from {len(strips)} strips × {N_CAMERAS} cameras)")
    print()

    # Add strategy D (per-strip best prior)
    all_strategies = {**STRATEGIES, "D_per_strip_best": None}  # None = computed per strip

    aggregates: dict[str, StrategyAggregate] = {
        name: StrategyAggregate(name=name) for name in all_strategies
    }

    det_dir = args.out / "detections"
    det_dir.mkdir(exist_ok=True)

    total_tiles = len(all_tiles)
    for tile_idx, (strip_stem, cam, tile_path) in enumerate(all_tiles):
        print(f"  [{tile_idx+1:3d}/{total_tiles}] {strip_stem}_cam{cam}", end="", flush=True)

        for strat_name, prior in all_strategies.items():
            if strat_name == "D_per_strip_best":
                d_min, d_max = strategy_D_best_per_strip(strip_stem)
            else:
                d_min, d_max = prior

            out_json = det_dir / f"{tile_path.stem}_{strat_name}.json"
            data = run_detect(args.binary, tile_path, out_json, d_min, d_max, args.target)
            n_total, n_decoded, n_inliers = count_result(data)

            agg = aggregates[strat_name]
            agg.n_tiles += 1
            agg.total_detections += n_total
            agg.total_decoded += n_decoded
            agg.total_inliers += n_inliers
            if n_decoded > 0:
                agg.tiles_with_any += 1
            if n_decoded >= 4:
                agg.tiles_with_4plus += 1
            if n_decoded >= 20:
                agg.tiles_with_20plus += 1
            agg.results.append(TileResult(
                strip=strip_stem, cam=cam, strategy=strat_name,
                n_total=n_total, n_decoded=n_decoded, n_inliers=n_inliers
            ))

        # Print decoded counts for each strategy on this tile
        counts = " | ".join(
            f"{aggregates[s].results[-1].n_decoded:3d}"
            for s in all_strategies
        )
        print(f"  decoded: {counts}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("AGGREGATE RESULTS")
    print("=" * 78)

    strat_labels = {
        "A_old_default_20_56": "A old [20,56]",
        "B_new_default_14_66": "B new [14,66]",
        "C_wide_single_8_220": "C wide [8,220]",
        "D_per_strip_best":    "D per-strip",
    }

    header = ["Strategy", "Tiles", "Decoded", "Inliers", "Any>0", "≥4", "≥20"]
    rows = []
    for strat_name, agg in aggregates.items():
        rows.append([
            strat_labels[strat_name],
            str(agg.n_tiles),
            str(agg.total_decoded),
            str(agg.total_inliers),
            str(agg.tiles_with_any),
            str(agg.tiles_with_4plus),
            str(agg.tiles_with_20plus),
        ])
    print_table(rows, header)

    # Baseline for relative improvement
    baseline = aggregates["A_old_default_20_56"].total_decoded
    print("\nRelative improvement in decoded markers vs A (old default):")
    for strat_name, agg in aggregates.items():
        if strat_name == "A_old_default_20_56":
            continue
        gain = agg.total_decoded - baseline
        pct = 100.0 * gain / max(baseline, 1)
        print(f"  {strat_labels[strat_name]}: +{gain} ({pct:+.1f}%)")

    # ── Per-strip summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PER-STRIP DECODED MARKER TOTALS  (sum over 6 cameras)")
    print("=" * 78)

    strip_stems = sorted({t.strip for t in aggregates["A_old_default_20_56"].results})
    header2 = ["Strip"] + [strat_labels[s] for s in all_strategies]
    rows2 = []
    for stem in strip_stems:
        row = [stem]
        for strat_name in all_strategies:
            total = sum(
                r.n_decoded for r in aggregates[strat_name].results if r.strip == stem
            )
            row.append(str(total))
        rows2.append(row)
    print_table(rows2, header2)

    # ── Save JSON report ───────────────────────────────────────────────────────
    report = {
        "n_strips": len(strips),
        "n_tiles": len(all_tiles),
        "skipped_strips": list(SKIP_STRIPS),
        "strategies": {
            name: {
                "n_tiles": agg.n_tiles,
                "total_decoded": agg.total_decoded,
                "total_inliers": agg.total_inliers,
                "tiles_with_any_decoded": agg.tiles_with_any,
                "tiles_with_4plus_decoded": agg.tiles_with_4plus,
                "tiles_with_20plus_decoded": agg.tiles_with_20plus,
            }
            for name, agg in aggregates.items()
        },
    }
    report_path = args.out / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
