#!/usr/bin/env python3
"""Explicit multi-scale detection example.

Run from repository root after:
  maturin develop -m crates/ringgrid-py/Cargo.toml --release

Examples:
  python crates/ringgrid-py/examples/detect_multiscale.py \
    --image testdata/target_3_split_00.png \
    --tiers four_tier_wide

  python crates/ringgrid-py/examples/detect_multiscale.py \
    --image testdata/target_3_split_00.png \
    --tiers custom \
    --tier 12 40 \
    --tier 36 90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ringgrid


def build_tiers(args: argparse.Namespace) -> ringgrid.ScaleTiers:
    if args.tiers == "four_tier_wide":
        return ringgrid.ScaleTiers.four_tier_wide()
    if args.tiers == "two_tier_standard":
        return ringgrid.ScaleTiers.two_tier_standard()

    assert args.tiers == "custom"
    if not args.tier:
        raise ValueError("custom tier mode requires at least one --tier MIN MAX")
    return ringgrid.ScaleTiers(
        tiers=[
            ringgrid.ScaleTier(diameter_min_px=float(bounds[0]), diameter_max_px=float(bounds[1]))
            for bounds in args.tier
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit multi-scale ringgrid detection")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument(
        "--board",
        type=Path,
        default=None,
        help="Optional board spec JSON (defaults to built-in board)",
    )
    parser.add_argument(
        "--tiers",
        choices=["four_tier_wide", "two_tier_standard", "custom"],
        default="two_tier_standard",
        help="Tier strategy to use",
    )
    parser.add_argument(
        "--tier",
        action="append",
        nargs=2,
        metavar=("MIN_PX", "MAX_PX"),
        default=[],
        help="Custom tier bounds (repeat for multiple tiers)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output detection JSON path",
    )
    args = parser.parse_args()

    board = (
        ringgrid.BoardLayout.default()
        if args.board is None
        else ringgrid.BoardLayout.from_json_file(args.board)
    )
    detector = ringgrid.Detector(board, ringgrid.DetectConfig(board))
    tiers = build_tiers(args)

    result = detector.detect_multiscale(args.image, tiers)

    print(f"Detected markers: {len(result.detected_markers)}")
    with_id = sum(1 for marker in result.detected_markers if marker.id is not None)
    print(f"Decoded markers:  {with_id}")

    if args.out is not None:
        result.to_json(args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
