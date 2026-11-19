#!/usr/bin/env python3
"""
Evaluation of ringgrid on the private rtv3d dataset.

Splits each 4320x540 strip into 6 tiles of 720x540 (``strip_tiles.py``), runs
the detector under several strategies and reports, per strategy: decoded IDs
(total and per tile), tiles with 0 and with fewer than 10 IDs, IDs per
tile-row band (top / middle / bottom third: the sensor is a tilted
Scheimpflug one, sharp in the middle band and blurred at the top and bottom),
markers without an inner ellipse, and wall-clock runtime per tile (one CLI
process: start-up, PNG load, detection, JSON write; strategies run
sequentially so the timings are comparable).

Strategies
----------
adaptive              Primary. Published CLI ``ringgrid detect`` without
                      ``--marker-diameter`` = ``Detector::detect_adaptive``
                      (scale probe -> tiers -> candidate selection).
adaptive_photometric  The same with ``--refine photometric``.
A  old default        ``ringgrid-dev``, single pass, diameter [20, 56] px.
B  new default        ``ringgrid-dev``, single pass, diameter [14, 66] px.
D  per-strip best     ``ringgrid-dev``, single pass, a hand-picked prior per
                      strip (CUSTOM_PRIORS, [14, 66] otherwise).

A, B and D are kept for continuity with earlier reports and gates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from strip_tiles import N_CAMERAS, SKIP_STRIPS, split_strip  # noqa: E402  (tools/ is on sys.path)

# Per-strip best static priors (from empirical experiments), strategy D.
CUSTOM_PRIORS: dict[str, tuple[int, int]] = {
    "target_14": (18, 100),
    "target_17": (18, 100),
    "target_19": (8, 220),
}
DEFAULT_PRIOR = (14, 66)

BANDS = ("top", "middle", "bottom")
FEW_IDS = 10  # "fewer than this many decoded IDs" counts as a weak tile


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    tool: str  # "ringgrid" (published CLI) or "ringgrid-dev"
    args: tuple[str, ...] = ()
    per_strip_prior: bool = False

    def detect_args(self, strip_stem: str) -> list[str]:
        if not self.per_strip_prior:
            return list(self.args)
        d_min, d_max = CUSTOM_PRIORS.get(strip_stem, DEFAULT_PRIOR)
        return ["--marker-diameter-min", str(d_min), "--marker-diameter-max", str(d_max)]


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("adaptive", "adaptive", "ringgrid"),
    Strategy("adaptive_photometric", "adaptive+photo", "ringgrid", ("--refine", "photometric")),
    Strategy("A_old_default_20_56", "A [20,56]", "ringgrid-dev", ("--marker-diameter-min", "20", "--marker-diameter-max", "56")),
    Strategy("B_new_default_14_66", "B [14,66]", "ringgrid-dev", ("--marker-diameter-min", "14", "--marker-diameter-max", "66")),
    Strategy("D_per_strip_best", "D per-strip", "ringgrid-dev", per_strip_prior=True),
)


@dataclass
class TileResult:
    tile: str
    strip: str
    n_markers: int
    n_decoded: int
    bands: dict[str, int]
    n_no_inner: int
    runtime_s: float


@dataclass
class StrategyAggregate:
    strategy: Strategy
    results: list[TileResult] = field(default_factory=list)

    def summary(self) -> dict:
        decoded = [r.n_decoded for r in self.results]
        runtimes = [r.runtime_s for r in self.results]
        return {
            "n_tiles": len(self.results),
            "total_decoded": sum(decoded),
            "total_markers": sum(r.n_markers for r in self.results),
            "mean_ids_per_tile": round(statistics.fmean(decoded), 2) if decoded else 0.0,
            "median_ids_per_tile": statistics.median(decoded) if decoded else 0,
            "tiles_zero_ids": sum(1 for n in decoded if n == 0),
            f"tiles_lt{FEW_IDS}_ids": sum(1 for n in decoded if n < FEW_IDS),
            "decoded_by_band": {b: sum(r.bands[b] for r in self.results) for b in BANDS},
            "markers_without_inner": sum(r.n_no_inner for r in self.results),
            "runtime_total_s": round(sum(runtimes), 2),
            "runtime_median_ms": round(1000 * statistics.median(runtimes), 1) if runtimes else 0.0,
            "runtime_max_ms": round(1000 * max(runtimes), 1) if runtimes else 0.0,
            "per_tile_decoded": {r.tile: r.n_decoded for r in self.results},
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/rtv3d_0"),
                   help="Directory containing rtv3d strip images (default: data/rtv3d_0)")
    p.add_argument("--binary", type=Path, default=Path("target/release/ringgrid-dev"),
                   help="ringgrid-dev binary for A/B/D (default: target/release/ringgrid-dev)")
    p.add_argument("--ringgrid-binary", type=Path, default=Path("target/release/ringgrid"),
                   help="published ringgrid CLI for the adaptive strategies (default: target/release/ringgrid)")
    p.add_argument("--target", type=Path, default=Path("tools/board/board_spec.json"),
                   help="Target spec JSON (default: tools/board/board_spec.json, the built-in default_hex board)")
    p.add_argument("--strategies", nargs="+", default=None, metavar="NAME",
                   help=f"Subset of strategies to run (default: all of {', '.join(s.name for s in STRATEGIES)})")
    p.add_argument("--out", type=Path, default=Path("tools/out/rtv3d_eval"),
                   help="Output directory for JSON results and report")
    return p.parse_args()


def run_detect(binary: Path, image: Path, out_json: Path, args: list[str], target: Path) -> tuple[dict, float]:
    """Run one detection; returns (result JSON or {}, wall-clock seconds)."""
    cmd = [str(binary), "detect", "--image", str(image), "--out", str(out_json), "--target", str(target), *args]
    out_json.unlink(missing_ok=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0 or not out_json.exists():
        print(f"\n    [WARN] detect failed for {image.name}: {result.stderr.strip()[-200:]}", file=sys.stderr)
        return {}, elapsed
    with open(out_json) as f:
        return json.load(f), elapsed


def tile_counts(data: dict) -> tuple[int, int, dict[str, int], int]:
    """(n_markers, n_decoded, decoded per tile-row band, markers without inner ellipse)."""
    markers = data.get("detected_markers", [])
    height = (data.get("image_size") or [0, 0])[1]
    bands = dict.fromkeys(BANDS, 0)
    n_decoded = 0
    for m in markers:
        if m.get("id") is None:
            continue
        n_decoded += 1
        y = m["center"][1]
        band = min(int(3 * y / height), 2) if height > 0 else 1
        bands[BANDS[max(band, 0)]] += 1
    n_no_inner = sum(1 for m in markers if m.get("ellipse_inner") is None)
    return len(markers), n_decoded, bands, n_no_inner


def print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def select_strategies(names: list[str] | None) -> list[Strategy]:
    if names is None:
        return list(STRATEGIES)
    by_name = {s.name: s for s in STRATEGIES}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise SystemExit(f"unknown strategies {unknown}; choose from {list(by_name)}")
    return [by_name[n] for n in names]


def print_summary(aggregates: dict[str, StrategyAggregate]) -> None:
    print("\n" + "=" * 96)
    print("AGGREGATE RESULTS (decoded IDs; bands = top / middle / bottom third of each tile)")
    print("=" * 96)
    header = ["Strategy", "Tiles", "Decoded", "IDs/tile", "0 IDs", f"<{FEW_IDS} IDs",
              "Top", "Middle", "Bottom", "No inner", "Median ms", "Total s"]
    rows = []
    for agg in aggregates.values():
        s = agg.summary()
        b = s["decoded_by_band"]
        rows.append([
            agg.strategy.label, str(s["n_tiles"]), str(s["total_decoded"]), f"{s['mean_ids_per_tile']:.1f}",
            str(s["tiles_zero_ids"]), str(s[f"tiles_lt{FEW_IDS}_ids"]),
            str(b["top"]), str(b["middle"]), str(b["bottom"]), str(s["markers_without_inner"]),
            f"{s['runtime_median_ms']:.0f}", f"{s['runtime_total_s']:.1f}",
        ])
    print_table(rows, header)

    print("\n" + "=" * 96)
    print(f"PER-STRIP DECODED TOTALS  (sum over {N_CAMERAS} cameras)")
    print("=" * 96)
    first = next(iter(aggregates.values()))
    strip_stems = sorted({r.strip for r in first.results}, key=lambda s: int(s.rsplit("_", 1)[-1]))
    rows2 = []
    for stem in strip_stems:
        rows2.append([stem] + [
            str(sum(r.n_decoded for r in agg.results if r.strip == stem)) for agg in aggregates.values()
        ])
    print_table(rows2, ["Strip"] + [agg.strategy.label for agg in aggregates.values()])


def main() -> int:
    args = parse_args()
    strategies = select_strategies(args.strategies)
    tools = {"ringgrid-dev": args.binary, "ringgrid": args.ringgrid_binary}
    for s in strategies:
        if not tools[s.tool].exists():
            raise SystemExit(f"Binary not found: {tools[s.tool]}. Run: cargo build --release")
    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")
    if not args.target.exists():
        raise SystemExit(f"Target spec not found: {args.target}")

    strips = [s for s in sorted(args.data_dir.glob("*.png")) if s.stem not in SKIP_STRIPS]
    print(f"Found {len(strips)} usable strip images in {args.data_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    tiles_dir = args.out / "tiles"
    tiles_dir.mkdir(exist_ok=True)
    det_dir = args.out / "detections"
    det_dir.mkdir(exist_ok=True)

    all_tiles: list[tuple[str, Path]] = []  # (strip_stem, tile_path)
    for strip in strips:
        all_tiles += [(strip.stem, tp) for tp in split_strip(strip, tiles_dir, N_CAMERAS)]
    print(f"Tiles: {len(all_tiles)} (from {len(strips)} strips x {N_CAMERAS} cameras)")
    print("Strategies: " + ", ".join(s.name for s in strategies) + "\n")

    aggregates = {s.name: StrategyAggregate(s) for s in strategies}
    for i, (strip_stem, tile_path) in enumerate(all_tiles):
        print(f"  [{i + 1:3d}/{len(all_tiles)}] {tile_path.stem:<12}", end="", flush=True)
        for s in strategies:
            out_json = det_dir / f"{tile_path.stem}_{s.name}.json"
            data, elapsed = run_detect(tools[s.tool], tile_path, out_json, s.detect_args(strip_stem), args.target)
            n_markers, n_decoded, bands, n_no_inner = tile_counts(data)
            aggregates[s.name].results.append(
                TileResult(tile_path.stem, strip_stem, n_markers, n_decoded, bands, n_no_inner, elapsed)
            )
        print("  decoded: " + " | ".join(f"{aggregates[s.name].results[-1].n_decoded:3d}" for s in strategies))

    print_summary(aggregates)

    report = {
        "n_strips": len(strips),
        "n_tiles": len(all_tiles),
        "skipped_strips": sorted(SKIP_STRIPS),
        "target": str(args.target),
        "band_definition": "decoded IDs by marker centre y: top / middle / bottom third of the tile height",
        "runtime_definition": "wall clock per CLI process (start-up, PNG load, detection, JSON write), sequential",
        "strategies": {name: agg.summary() for name, agg in aggregates.items()},
    }
    report_path = args.out / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
