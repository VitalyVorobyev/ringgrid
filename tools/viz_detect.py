#!/usr/bin/env python3
"""Visualize DetectionResult overlays using the ringgrid Python package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ringgrid DetectionResult overlays")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--det_json", required=True, type=str)
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="Write PNG to this path (otherwise interactive window)",
    )
    parser.add_argument("--id", type=int, default=None, help="Focus on a single marker id")
    parser.add_argument("--zoom", type=float, default=None, help="Zoom factor when --id is set")
    parser.add_argument("--show-ellipses", dest="show_ellipses", action="store_true", default=True)
    parser.add_argument("--no-ellipses", dest="show_ellipses", action="store_false")
    parser.add_argument(
        "--show-confidence",
        dest="show_confidence",
        action="store_true",
        default=True,
        help="Show marker confidence in labels and center colors",
    )
    parser.add_argument(
        "--no-confidence",
        dest="show_confidence",
        action="store_false",
        help="Disable confidence indication",
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

    try:
        import ringgrid
        from ringgrid import viz
    except ImportError as exc:  # pragma: no cover - CLI env-dependent
        raise SystemExit(
            "ringgrid Python package is required for viz_detect.py. "
            "Install with: pip install -e crates/ringgrid-py[viz]"
        ) from exc

    det = ringgrid.DetectionResult.from_json(args.det_json)
    viz.plot_detection(
        image=args.image,
        detection=det,
        out=args.out,
        marker_id=args.id,
        zoom=args.zoom,
        show_ellipses=args.show_ellipses,
        show_confidence=args.show_confidence,
        alpha=args.alpha,
    )

    if args.out:
        out_path = Path(args.out)
        print(f"Wrote overlay to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        sys.exit(130)
