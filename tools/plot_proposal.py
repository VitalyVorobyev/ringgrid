#!/usr/bin/env python3
"""Plot proposal-stage diagnostics using the standalone proposal API.

Examples:
  python tools/plot_proposal.py \
    --image tools/out/synth_001/img_0000.png \
    --gt tools/out/synth_001/gt_0000.json \
    --out tools/out/synth_001/proposals_0000.png

  python tools/plot_proposal.py \
    --image testdata/target_3_split_00.png \
    --config tools/proposal_config.json

  python tools/plot_proposal.py \
    --image testdata/target_3_split_00.png \
    --r-min 4.0 --r-max 18.0 --min-distance 12.0 \
    --out proposals_overlay.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from rich import box
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

import ringgrid
from ringgrid import viz

CONSOLE = Console()


def normalize_rich_value(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return round(value, 6)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): normalize_rich_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_rich_value(v) for v in value]
    return value


def render_summary_table(rows: list[tuple[str, object]]) -> None:
    table = Table(box=box.ROUNDED, header_style="bold cyan", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for key, value in rows:
        table.add_row(key, str(value))
    CONSOLE.print(Panel(table, title="Run Summary", border_style="cyan"))


def render_json_panel(title: str, payload: dict) -> None:
    CONSOLE.print(
        Panel(
            JSON.from_data(normalize_rich_value(payload), indent=2, sort_keys=False),
            title=title,
            border_style="blue",
        )
    )


def proposal_config_dict(config: ringgrid.ProposalConfig) -> dict:
    return {
        "r_min": float(config.r_min),
        "r_max": float(config.r_max),
        "grad_threshold": float(config.grad_threshold),
        "min_distance": float(config.min_distance),
        "min_vote_frac": float(config.min_vote_frac),
        "accum_sigma": float(config.accum_sigma),
        "edge_thinning": bool(config.edge_thinning),
        "max_candidates": (
            None if getattr(config, "max_candidates", None) is None else int(config.max_candidates)
        ),
    }


def load_proposal_config(path: Path | None) -> ringgrid.ProposalConfig | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("proposal"), dict):
        payload = payload["proposal"]
    if not isinstance(payload, dict):
        raise SystemExit(f"proposal config must be a JSON object: {path}")
    return ringgrid.ProposalConfig.from_dict(payload)


def load_gt_points(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    markers = payload.get("markers")
    if not isinstance(markers, list):
        raise SystemExit(f"ground-truth JSON missing 'markers': {path}")

    points: list[list[float]] = []
    for marker in markers:
        if not isinstance(marker, dict) or not marker.get("visible", True):
            continue
        center = marker.get("true_image_center")
        if not isinstance(center, list) or len(center) < 2:
            continue
        points.append([float(center[0]), float(center[1])])
    return np.asarray(points, dtype=np.float32)


def proposal_hits(gt_points: np.ndarray | None, proposals: list[ringgrid.Proposal], gate_px: float) -> np.ndarray | None:
    if gt_points is None:
        return None
    if gt_points.size == 0:
        return np.zeros((0,), dtype=bool)
    if not proposals:
        return np.zeros((gt_points.shape[0],), dtype=bool)

    proposal_xy = np.asarray([[p.x, p.y] for p in proposals], dtype=np.float32)
    deltas = gt_points[:, None, :] - proposal_xy[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=2)
    return np.min(dist_sq, axis=1) <= float(gate_px) ** 2


def heatmap_out_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    suffix = path.suffix or ".png"
    return path.with_name(f"{path.stem}_heatmap{suffix}")


def build_config(args: argparse.Namespace) -> ringgrid.ProposalConfig:
    """Build ProposalConfig from --config JSON and/or CLI overrides."""
    config = load_proposal_config(args.config) or ringgrid.ProposalConfig()
    if args.r_min is not None:
        config.r_min = args.r_min
    if args.r_max is not None:
        config.r_max = args.r_max
    if args.min_distance is not None:
        config.min_distance = args.min_distance
    if args.grad_threshold is not None:
        config.grad_threshold = args.grad_threshold
    if args.min_vote_frac is not None:
        config.min_vote_frac = args.min_vote_frac
    if args.accum_sigma is not None:
        config.accum_sigma = args.accum_sigma
    if args.edge_thinning is not None:
        config.edge_thinning = args.edge_thinning
    if args.max_candidates is not None:
        config.max_candidates = args.max_candidates
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional ProposalConfig JSON (fields can be overridden by CLI flags)",
    )

    # ProposalConfig parameters — each overrides the config file / default
    parser.add_argument("--r-min", type=float, default=None, help="Minimum voting radius (px)")
    parser.add_argument("--r-max", type=float, default=None, help="Maximum voting radius (px)")
    parser.add_argument("--min-distance", type=float, default=None, help="Minimum distance between proposals (px)")
    parser.add_argument("--grad-threshold", type=float, default=None, help="Gradient magnitude threshold (fraction of max)")
    parser.add_argument("--min-vote-frac", type=float, default=None, help="Minimum accumulator peak (fraction of max)")
    parser.add_argument("--accum-sigma", type=float, default=None, help="Gaussian smoothing sigma")
    parser.add_argument("--edge-thinning", type=lambda s: s.lower() in ("true", "1", "yes"), default=None, help="Enable Canny-style gradient NMS (true/false)")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional hard cap on proposals")

    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Optional synth GT JSON for recall overlay using true_image_center",
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=8.0,
        help="GT-to-proposal hit gate in pixels when --gt is provided",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional proposal-overlay output path. "
            "When set, the script also writes a sibling '*_heatmap.png'."
        ),
    )
    args = parser.parse_args()

    proposal_config = build_config(args)

    config_payload = {
        "source": "default ProposalConfig()" if args.config is None else str(args.config),
        "proposal": proposal_config_dict(proposal_config),
    }

    # Run the standalone proposal API.
    t0 = time.perf_counter()
    result = ringgrid.propose_with_heatmap(args.image, config=proposal_config)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    gt_points = load_gt_points(args.gt)
    gt_hits = proposal_hits(gt_points, result.proposals, gate_px=args.gate)

    summary_rows: list[tuple[str, object]] = [
        ("image", args.image),
        ("propose_with_heatmap_ms", f"{elapsed_ms:.3f}"),
        ("proposals", len(result.proposals)),
        ("heatmap_shape", f"{result.heatmap.shape}"),
    ]
    if args.config is not None:
        summary_rows.append(("config", args.config))
    if gt_points is not None:
        n_gt = int(gt_points.shape[0])
        n_hit = int(np.count_nonzero(gt_hits))
        recall = 0.0 if n_gt == 0 else n_hit / n_gt
        summary_rows.extend(
            [
                ("gt_markers", n_gt),
                ("gt_hits", n_hit),
                ("recall", f"{recall:.3f}"),
                ("gate_px", f"{args.gate:.2f}"),
            ]
        )
    if args.out is not None:
        summary_rows.append(("overlay_out", args.out))
        heatmap_out = heatmap_out_path(args.out)
        if heatmap_out is not None:
            summary_rows.append(("heatmap_out", heatmap_out))
    else:
        heatmap_out = None

    render_summary_table(summary_rows)
    render_json_panel("Proposal Config", config_payload)

    viz.plot_proposal_diagnostics(
        image=args.image,
        diagnostics=result,
        out=args.out,
        heatmap_out=heatmap_out,
        gt_points=gt_points,
        gt_hits=gt_hits,
        show_proposals_on_heatmap=False,
    )

    if args.out is not None:
        CONSOLE.print(f"[green]wrote[/green] {args.out}")
        if heatmap_out is not None:
            CONSOLE.print(f"[green]wrote[/green] {heatmap_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
