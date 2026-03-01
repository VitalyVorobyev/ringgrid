#!/usr/bin/env python3
"""Adaptive detection example.

Run from repository root after:
  maturin develop -m crates/ringgrid-py/Cargo.toml --release

Examples:
  python crates/ringgrid-py/examples/detect_adaptive.py \
    --image testdata/target_3_split_00.png

  python crates/ringgrid-py/examples/detect_adaptive.py \
    --image testdata/target_3_split_00.png \
    --hint 32.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ringgrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive ringgrid detection")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument(
        "--board",
        type=Path,
        default=None,
        help="Optional board spec JSON (defaults to built-in board)",
    )
    parser.add_argument(
        "--hint",
        type=float,
        default=None,
        help="Optional nominal marker diameter hint in pixels (> 0)",
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

    tiers = detector.adaptive_tiers(args.image, nominal_diameter_px=args.hint)
    tier_text = ", ".join(
        f"[{tier.diameter_min_px:.1f}, {tier.diameter_max_px:.1f}]"
        for tier in tiers.tiers
    )
    print(f"Adaptive tiers:   {tier_text}")

    result = detector.detect_adaptive(args.image, nominal_diameter_px=args.hint)

    print(f"Detected markers: {len(result.detected_markers)}")
    with_id = sum(1 for marker in result.detected_markers if marker.id is not None)
    print(f"Decoded markers:  {with_id}")

    if args.out is not None:
        result.to_json(args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
