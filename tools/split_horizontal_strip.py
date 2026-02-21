#!/usr/bin/env python3
"""Split a horizontally concatenated image strip into equal-width tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Pillow is required to run this script. "
        "Use the project venv or install it first."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a horizontal strip image into N equal-width 8-bit grayscale PNGs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/target_3.png"),
        help="Input strip image path (default: data/target_3.png).",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=6,
        help="Number of equal horizontal parts (default: 6).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for split images (default: data).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: input stem).",
    )
    parser.add_argument(
        "--one-based",
        action="store_true",
        help="Use 1-based file numbering instead of 0-based.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.parts <= 0:
        print("--parts must be > 0", file=sys.stderr)
        return 2
    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 2

    with Image.open(args.input) as image:
        gray = image.convert("L")
        width, height = gray.size
        if width % args.parts != 0:
            print(
                f"image width {width} is not divisible by parts={args.parts}",
                file=sys.stderr,
            )
            return 2

        part_width = width // args.parts
        args.out_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.prefix if args.prefix else args.input.stem

        for i in range(args.parts):
            x0 = i * part_width
            x1 = (i + 1) * part_width
            tile = gray.crop((x0, 0, x1, height))
            index = i + 1 if args.one_based else i
            out_path = args.out_dir / f"{prefix}_{index:02d}.png"
            tile.save(out_path)
            print(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
