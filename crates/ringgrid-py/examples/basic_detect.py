#!/usr/bin/env python3
"""Minimal high-level detection example.

Run from repository root after:
  maturin develop -m crates/ringgrid-py/Cargo.toml --release

Example:
  python crates/ringgrid-py/examples/basic_detect.py \
    --image data/target_3_split_00.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ringgrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ringgrid detection on one image")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument(
        "--board",
        type=Path,
        default=None,
        help="Optional board spec JSON (defaults to built-in board)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output detection JSON path",
    )
    args = parser.parse_args()

    board = ringgrid.BoardLayout.default() if args.board is None else ringgrid.BoardLayout.from_json_file(args.board)

    config = ringgrid.DetectConfig(board)
    detector = ringgrid.Detector(board, config)

    result = detector.detect(args.image)

    print(f"Detected markers: {len(result.detected_markers)}")
    with_id = sum(1 for m in result.detected_markers if m.id is not None)
    print(f"Decoded markers:  {with_id}")

    if args.out is not None:
        result.to_json(args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
