#!/usr/bin/env python3
"""Minimal high-level detection example.

Run from repository root after:
  maturin develop -m crates/ringgrid-py/Cargo.toml --release

Example:
  python crates/ringgrid-py/examples/basic_detect.py \
    --image testdata/target_3_split_00.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ringgrid


def _compact_marker(marker: ringgrid.DetectedMarker) -> dict[str, object]:
    out: dict[str, object] = {
        "confidence": marker.confidence,
        "center": marker.center,
    }
    if marker.ellipse_outer is not None:
        out["ellipse_outer"] = marker.ellipse_outer.to_dict()
    if marker.ellipse_inner is not None:
        out["ellipse_inner"] = marker.ellipse_inner.to_dict()
    if marker.decode is not None:
        out["decode"] = marker.decode.to_dict()
    return out


def _compact_result(result: ringgrid.DetectionResult) -> dict[str, object]:
    return {
        "detected_markers": [_compact_marker(marker) for marker in result.detected_markers],
    }


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
        help="Optional output path for compact marker features JSON",
    )
    parser.add_argument(
        "--full-out",
        type=Path,
        default=None,
        help="Optional output path for the full DetectionResult JSON dump",
    )
    args = parser.parse_args()

    board = ringgrid.BoardLayout.default() if args.board is None else ringgrid.BoardLayout.from_json_file(args.board)

    config = ringgrid.DetectConfig(board)
    detector = ringgrid.Detector(config)

    result = detector.detect(args.image)

    print(f"Detected markers: {len(result.detected_markers)}")
    with_id = sum(1 for m in result.detected_markers if m.id is not None)
    print(f"Decoded markers:  {with_id}")

    if args.out is not None:
        args.out.write_text(json.dumps(_compact_result(result), indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")

    if args.full_out is not None:
        result.to_json(args.full_out)
        print(f"Wrote {args.full_out}")


if __name__ == "__main__":
    main()
