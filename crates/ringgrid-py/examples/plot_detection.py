#!/usr/bin/env python3
"""Run detection and render an overlay image.

Requires plotting extras:
  pip install -e crates/ringgrid-py[viz]

Example:
  python crates/ringgrid-py/examples/plot_detection.py \
    --image data/target_3_split_00.png \
    --out data/target_3_split_00_overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ringgrid
from ringgrid import viz


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and plot ringgrid markers")
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--out", required=True, type=Path, help="Output overlay PNG path")
    parser.add_argument(
        "--board",
        type=Path,
        default=None,
        help="Optional board spec JSON (defaults to built-in board)",
    )
    args = parser.parse_args()

    board = ringgrid.BoardLayout.default() if args.board is None else ringgrid.BoardLayout.from_json_file(args.board)
    detector = ringgrid.Detector(board, ringgrid.DetectConfig(board))
    result = detector.detect(args.image)

    viz.plot_detection(image=args.image, detection=result, out=args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
