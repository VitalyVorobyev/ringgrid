#!/usr/bin/env python3
"""
Proposal backend benchmark: accuracy × performance across radii strategies.

Tests combinations of:
- Backend: rsd, frst (per-radius, via Python radsym bindings)
- Radii: all_integer, sparse_8, sparse_5
- Threshold: 0.4×max, 0.3×max, 0.2×max

Measures proposal count and timing on rtv3d tiles.
Run: .venv/bin/python tools/bench_proposal_variants.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import radsym
except ImportError:
    sys.exit("Requires: .venv/bin/uv pip install radsym")


@dataclass
class Variant:
    name: str
    backend: str  # "rsd" or "frst"
    n_radii: int | None  # None = all integer
    grad_threshold_scale: float


@dataclass
class Result:
    n_proposals: int
    elapsed_ms: float
    radii_used: int


def build_radii(r_min: float, r_max: float, n_radii: int | None) -> list[int]:
    lo = max(1, int(np.ceil(r_min)))
    hi = int(np.floor(r_max))
    if hi < lo:
        return []
    full = list(range(lo, hi + 1))
    if n_radii is None or len(full) <= n_radii:
        return full
    indices = np.round(np.linspace(0, len(full) - 1, n_radii)).astype(int)
    return sorted(set(full[i] for i in indices))


def run_variant(image_path: str, r_min: float, r_max: float, v: Variant) -> Result:
    radii = build_radii(r_min, r_max, v.n_radii)
    if not radii:
        return Result(0, 0.0, 0)

    image = radsym.load_grayscale(image_path)
    gradient = radsym.sobel_gradient(image)
    # Use a fixed absolute threshold since max_magnitude isn't available
    # in Python radsym 0.1.1. A threshold of 2.0 works well empirically.
    abs_thresh = 2.0 * v.grad_threshold_scale / 0.4

    t0 = time.perf_counter()

    if v.backend == "rsd":
        cfg = radsym.RsdConfig(
            radii=radii, gradient_threshold=abs_thresh,
            polarity="both", smoothing_factor=0.5,
        )
        response = radsym.rsd_response(gradient, cfg)
    else:
        cfg = radsym.FrstConfig(
            radii=radii, alpha=2.0, gradient_threshold=abs_thresh,
            polarity="both", smoothing_factor=0.5,
        )
        response = radsym.frst_response(gradient, cfg)

    nms = radsym.NmsConfig(radius=10, threshold=0.0, max_detections=512)
    proposals = radsym.extract_proposals(response, nms, polarity="both")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return Result(len(proposals), elapsed_ms, len(radii))


def main() -> int:
    tile_dir = Path("tools/out/rtv3d_eval/tiles")
    if not tile_dir.exists():
        sys.exit("Run rtv3d eval first: .venv/bin/python tools/run_rtv3d_eval.py")

    tiles = sorted(tile_dir.glob("target_5_*.png"))
    if not tiles:
        sys.exit("No target_5 tiles found")

    print(f"Testing on {len(tiles)} tiles from target_5 (720x540 each)")
    print()

    strategies = {
        "A [20,56]": (7.5, 37.8),
        "B [14,66]": (5.2, 44.6),
    }

    variants = [
        Variant("rsd_all_0.4", "rsd", None, 0.4),
        Variant("rsd_all_0.3", "rsd", None, 0.3),
        Variant("rsd_all_0.2", "rsd", None, 0.2),
        Variant("rsd_8_0.4", "rsd", 8, 0.4),
        Variant("rsd_8_0.3", "rsd", 8, 0.3),
        Variant("rsd_8_0.2", "rsd", 8, 0.2),
        Variant("rsd_5_0.4", "rsd", 5, 0.4),
        Variant("rsd_5_0.3", "rsd", 5, 0.3),
        Variant("rsd_5_0.2", "rsd", 5, 0.2),
        Variant("frst_all_0.4", "frst", None, 0.4),
        Variant("frst_5_0.4", "frst", 5, 0.4),
    ]

    header = f"{'Variant':<20s} {'Strategy':<12s} {'Radii':>5s} {'Proposals':>9s} {'Time ms':>8s}"
    print(header)
    print("-" * len(header))

    for strat_name, (r_min, r_max) in strategies.items():
        for v in variants:
            tot_prop = 0
            tot_ms = 0.0
            n_radii = 0
            for tp in tiles:
                r = run_variant(str(tp), r_min, r_max, v)
                tot_prop += r.n_proposals
                tot_ms += r.elapsed_ms
                n_radii = r.radii_used
            avg_p = tot_prop / len(tiles)
            avg_ms = tot_ms / len(tiles)
            print(f"{v.name:<20s} {strat_name:<12s} {n_radii:>5d} {avg_p:>9.0f} {avg_ms:>8.1f}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
