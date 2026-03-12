#!/usr/bin/env python3
"""Generate canonical ringgrid target JSON plus printable SVG/PNG."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


DEFAULT_OUT_DIR = Path("tools/out/target")
DEFAULT_BASENAME = "target_print"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pitch_mm", type=float, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--long_row_cols", type=int, required=True)
    parser.add_argument("--marker_outer_radius_mm", type=float, required=True)
    parser.add_argument("--marker_inner_radius_mm", type=float, required=True)
    parser.add_argument("--marker_ring_width_mm", type=float, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--basename", type=str, default=DEFAULT_BASENAME)
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument("--margin_mm", type=float, default=0.0)
    parser.add_argument(
        "--no-scale-bar",
        action="store_true",
        help="Omit the default scale bar from SVG/PNG outputs",
    )
    return parser


def load_ringgrid():
    try:
        import ringgrid
    except ModuleNotFoundError as exc:
        if exc.name != "ringgrid":
            raise
        print(
            "ringgrid is not installed in this Python environment.\n"
            "From the repository root, install the local binding first:\n"
            "  ./.venv/bin/python -m pip install -U pip maturin\n"
            "  ./.venv/bin/python -m maturin develop -m crates/ringgrid-py/Cargo.toml --release",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return ringgrid


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ringgrid = load_ringgrid()

    out_dir = args.out_dir
    json_path = out_dir / "board_spec.json"
    svg_path = out_dir / f"{args.basename}.svg"
    png_path = out_dir / f"{args.basename}.png"

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        board = ringgrid.BoardLayout.from_geometry(
            args.pitch_mm,
            args.rows,
            args.long_row_cols,
            args.marker_outer_radius_mm,
            args.marker_inner_radius_mm,
            args.marker_ring_width_mm,
            name=args.name,
        )
        include_scale_bar = not args.no_scale_bar
        board.to_spec_json(json_path)
        board.write_svg(
            svg_path,
            margin_mm=args.margin_mm,
            include_scale_bar=include_scale_bar,
        )
        board.write_png(
            png_path,
            dpi=args.dpi,
            margin_mm=args.margin_mm,
            include_scale_bar=include_scale_bar,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"gen_target.py: {exc}", file=sys.stderr)
        return 1

    print(f"Board spec JSON written to {json_path}")
    print(f"Print SVG written to {svg_path}")
    print(f"Print PNG written to {png_path} ({float(args.dpi):.1f} dpi)")
    print(
        f"Board: {board.name}, schema={board.schema}, rows={board.rows}, "
        f"long_row_cols={board.long_row_cols}, markers={len(board.markers)}, "
        f"pitch={board.pitch_mm}mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
