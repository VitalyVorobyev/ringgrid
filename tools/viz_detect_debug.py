#!/usr/bin/env python3
"""Visualize ringgrid detection debug dumps (ringgrid.debug.v2).

Overlays detection candidates / final detections on top of the input image.

Examples:
  python3 tools/viz_detect_debug.py \
    --image tools/out/synth_001/img_0000.png \
    --debug_json tools/out/synth_001/debug_0000.json \
    --out tools/out/synth_001/det_overlay_0000.png

  # Focus on a single marker id
  python3 tools/viz_detect_debug.py \
    --image tools/out/synth_001/img_0000.png \
    --debug_json tools/out/synth_001/debug_0000.json \
    --id 42 --zoom 4.0 \
    --out tools/out/synth_001/det_overlay_id42.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable


def maybe_use_agg(out_path: str | None) -> None:
    # Use Agg for file output (headless-friendly).
    if out_path:
        import matplotlib

        matplotlib.use("Agg")


def load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


@dataclass(frozen=True)
class PlotStyle:
    # Candidates / points
    candidate_accepted: str = "lime"
    candidate_rejected: str = "red"
    candidate_ransac_outlier: str = "magenta"
    candidate_prior: str = "deepskyblue"

    # Geometry
    ellipse_outer: str = "lime"
    ellipse_inner: str = "cyan"

    # Completion (H-guided)
    completion_projected_added: str = "cyan"
    completion_projected_failed: str = "red"
    completion_projected_skipped: str = "gray"
    completion_added_marker: str = "cyan"
    completion_added_text: str = "cyan"

    # NL refine (board-plane circle fit)
    nl_before: str = "orange"
    nl_after_ok: str = "lime"
    nl_after_rejected: str = "gold"
    nl_after_failed: str = "red"

    # Labels (use outline for contrast)
    id_text: str = "white"
    id_text_outline: str = "black"

    # Misc
    warning_text: str = "white"


def add_id_label(
    ax,
    x: float,
    y: float,
    text: str,
    style: PlotStyle,
    *,
    text_color: str | None = None,
    outline_color: str | None = None,
) -> None:
    import matplotlib.patheffects as pe

    ax.text(
        x + 4,
        y - 4,
        text,
        fontsize=6,
        color=text_color or style.id_text,
        ha="left",
        va="bottom",
        path_effects=[
            pe.Stroke(linewidth=2.5, foreground=outline_color or style.id_text_outline),
            pe.Normal(),
        ],
    )


def sample_ellipse_xy(ell: dict[str, Any], n: int = 128) -> tuple[list[float], list[float]]:
    """Sample ellipse polyline from EllipseParams-like dict.

    Expects:
      center_xy: [cx, cy]
      semi_axes: [a, b]
      angle: radians
    """
    cx, cy = ell["center_xy"]
    a, b = ell["semi_axes"]
    ang = ell["angle"]

    xs: list[float] = []
    ys: list[float] = []
    ca = math.cos(ang)
    sa = math.sin(ang)
    for i in range(n + 1):
        t = (i / n) * 2.0 * math.pi
        x = a * math.cos(t)
        y = b * math.sin(t)
        xr = ca * x - sa * y + cx
        yr = sa * x + ca * y + cy
        xs.append(float(xr))
        ys.append(float(yr))
    return xs, ys


def iter_stage_candidates(debug: dict[str, Any], stage_name: str) -> Iterable[dict[str, Any]]:
    stages = debug.get("stages", {})
    stage = stages.get(stage_name, {})
    for c in stage.get("candidates", []) or []:
        yield c


def find_stage1_candidate_for_id(debug: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    for c in iter_stage_candidates(debug, "stage1_fit_decode"):
        did = (c.get("derived", {}) or {}).get("id")
        if did is not None and int(did) == int(marker_id):
            return c
    return None


def find_final_marker(debug: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    final_ = debug.get("stages", {}).get("final", {})
    for m in final_.get("detections", []) or []:
        if m.get("id") == marker_id:
            return m
    return None


def refine_entry_for_id(debug: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    refine = debug.get("stages", {}).get("stage4_refine")
    if not refine:
        return None
    for m in refine.get("refined_markers", []) or []:
        if m.get("id") == marker_id:
            return m
    return None


def completion_attempts(debug: dict[str, Any]) -> list[dict[str, Any]]:
    stage = debug.get("stages", {}).get("stage5_completion")
    if not stage:
        return []
    return stage.get("attempted", []) or []


def completion_added_ids(debug: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for a in completion_attempts(debug):
        if str(a.get("status", "")).lower() == "added":
            out.add(int(a.get("id")))
    return out


def ring_fit_debug_for_id(debug: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    """Best-effort lookup of RingFitDebug for an id.

    Prefer stage1 (proposal-based detection); fall back to completion attempt
    fit debug when the id is only present due to homography-guided completion.
    """
    c = find_stage1_candidate_for_id(debug, marker_id)
    if c:
        rf = c.get("ring_fit")
        if rf:
            return rf

    for a in completion_attempts(debug):
        mid = a.get("id")
        if mid is None or int(mid) != int(marker_id):
            continue
        fit = a.get("fit")
        if fit:
            return fit

    return None


def nl_refine_stage(debug: dict[str, Any]) -> dict[str, Any] | None:
    return debug.get("stages", {}).get("stage6_nl_refine")


def nl_refine_entries(debug: dict[str, Any]) -> list[dict[str, Any]]:
    stage = nl_refine_stage(debug)
    if not stage:
        return []
    return stage.get("refined_markers", []) or []


def nl_refine_entry_for_id(debug: dict[str, Any], marker_id: int) -> dict[str, Any] | None:
    for m in nl_refine_entries(debug):
        if m.get("id") == marker_id:
            return m
    return None


def plot_completion(
    ax,
    debug: dict[str, Any],
    *,
    marker_id: int | None,
    alpha: float,
    show_ellipses: bool,
    style: PlotStyle,
) -> None:
    for a in completion_attempts(debug):
        mid = a.get("id")
        if mid is None:
            continue
        mid = int(mid)
        if marker_id is not None and mid != int(marker_id):
            continue

        cx, cy = a.get("projected_center_xy", [None, None])
        if cx is None or cy is None:
            continue

        status = a.get("status")
        if status == "added":
            color = style.completion_projected_added
        elif status in ("failed_fit", "failed_gate"):
            color = style.completion_projected_failed
        else:
            color = style.completion_projected_skipped

        ax.plot(float(cx), float(cy), "x", color=color, markersize=6, alpha=alpha)
        if status == "added":
            add_id_label(
                ax,
                float(cx),
                float(cy),
                str(mid),
                style,
                text_color=style.completion_added_text,
            )

        if show_ellipses:
            fit = a.get("fit")
            if fit:
                for key, ecolor in (("ellipse_outer", style.ellipse_outer), ("ellipse_inner", style.ellipse_inner)):
                    ell = fit.get(key)
                    if ell:
                        xs, ys = sample_ellipse_xy(ell)
                        ax.plot(xs, ys, "-", color=ecolor, linewidth=1.0, alpha=alpha)


def plot_nl_refine(
    ax,
    debug: dict[str, Any],
    *,
    marker_id: int | None,
    only_inliers: bool,
    inlier_ids: set[int],
    alpha: float,
    show_after: bool,
    style: PlotStyle,
) -> None:
    stage = nl_refine_stage(debug) or {}
    if not stage.get("enabled", False):
        return

    for m in nl_refine_entries(debug):
        mid = m.get("id")
        if mid is None:
            continue
        mid = int(mid)
        if marker_id is not None and mid != int(marker_id):
            continue
        if only_inliers and mid not in inlier_ids:
            continue

        bx, by = m.get("center_img_before", [None, None])
        if bx is not None and by is not None:
            ax.plot(float(bx), float(by), "x", color=style.nl_before, markersize=6, alpha=alpha)

        if show_after:
            ax_, ay_ = (m.get("center_img_after") or [None, None])
            status = str(m.get("status", "failed")).lower()
            if status == "ok":
                color = style.nl_after_ok
            elif status == "rejected":
                color = style.nl_after_rejected
            else:
                color = style.nl_after_failed

            if ax_ is not None and ay_ is not None:
                ax.plot(float(ax_), float(ay_), "o", color=color, markersize=4, alpha=alpha)
                add_id_label(ax, float(ax_), float(ay_), str(mid), style)
            elif bx is not None and by is not None:
                add_id_label(ax, float(bx), float(by), f"{mid} ({status})", style)


def compute_zoom_window(center_xy: list[float], img_w: int, img_h: int, zoom: float) -> tuple[float, float, float, float]:
    span = min(img_w, img_h) / max(zoom, 1e-6)
    half = span / 2.0
    cx, cy = center_xy
    x0 = max(0.0, cx - half)
    x1 = min(float(img_w), cx + half)
    y0 = max(0.0, cy - half)
    y1 = min(float(img_h), cy + half)
    return x0, x1, y0, y1


def plot_candidates(
    ax,
    debug: dict[str, Any],
    stage: str,
    *,
    marker_id: int | None,
    only_inliers: bool,
    inlier_ids: set[int],
    outlier_ids: set[int],
    alpha: float,
    style: PlotStyle,
) -> None:
    stage_name = "stage0_proposals" if stage == "stage0_proposals" else "stage1_fit_decode"
    for c in iter_stage_candidates(debug, stage_name):
        prop = c.get("proposal", {})
        x = prop.get("x")
        y = prop.get("y")
        if x is None or y is None:
            x, y = prop.get("center_xy", [None, None])
        if x is None or y is None:
            continue

        status = str((c.get("decision", {}) or {}).get("status", "rejected")).lower()
        derived_id = (c.get("derived", {}) or {}).get("id")

        if marker_id is not None and derived_id is not None and int(derived_id) != int(marker_id):
            continue
        if only_inliers and derived_id is not None and int(derived_id) not in inlier_ids:
            continue

        color = style.candidate_accepted if status == "accepted" else style.candidate_rejected
        if stage == "stage3_ransac" and derived_id is not None:
            did = int(derived_id)
            if did in outlier_ids:
                color = style.candidate_ransac_outlier
            elif did in inlier_ids:
                color = style.candidate_accepted

        ax.plot(float(x), float(y), "o", color=color, markersize=3, alpha=alpha)


def plot_stage1_ellipses_for_id(
    ax,
    debug: dict[str, Any],
    marker_id: int,
    *,
    alpha: float,
    style: PlotStyle,
) -> None:
    for c in iter_stage_candidates(debug, "stage1_fit_decode"):
        derived_id = (c.get("derived", {}) or {}).get("id")
        if derived_id is None or int(derived_id) != int(marker_id):
            continue
        rf = c.get("ring_fit")
        if not rf:
            break
        for key, color in (("ellipse_outer", style.ellipse_outer), ("ellipse_inner", style.ellipse_inner)):
            ell = rf.get(key)
            if ell:
                xs, ys = sample_ellipse_xy(ell)
                ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=alpha)
        break


def plot_refine(
    ax,
    debug: dict[str, Any],
    *,
    marker_id: int | None,
    only_inliers: bool,
    inlier_ids: set[int],
    alpha: float,
    show_ellipses: bool,
    style: PlotStyle,
) -> None:
    refine = debug.get("stages", {}).get("stage4_refine")
    if not refine:
        return

    for m in refine.get("refined_markers", []) or []:
        mid = m.get("id")
        if marker_id is not None and mid != marker_id:
            continue
        if only_inliers and mid is not None and int(mid) not in inlier_ids:
            continue

        px, py = m.get("prior_center_xy", [None, None])
        refined_marker = m.get("refined_marker") or {}
        rx, ry = refined_marker.get("center", [None, None])
        if px is not None and py is not None:
            ax.plot(float(px), float(py), "x", color=style.candidate_prior, markersize=6, alpha=alpha)
        if rx is not None and ry is not None:
            ax.plot(float(rx), float(ry), "o", color=style.candidate_accepted, markersize=4, alpha=alpha)
            if mid is not None:
                add_id_label(ax, float(rx), float(ry), str(mid), style)

        if show_ellipses:
            for key, color in (("ellipse_outer", style.ellipse_outer), ("ellipse_inner", style.ellipse_inner)):
                ell = refined_marker.get(key)
                if ell:
                    xs, ys = sample_ellipse_xy(ell)
                    ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=alpha)


def plot_final(
    ax,
    debug: dict[str, Any],
    *,
    marker_id: int | None,
    only_inliers: bool,
    inlier_ids: set[int],
    completion_added: set[int],
    alpha: float,
    show_ellipses: bool,
    style: PlotStyle,
) -> None:
    final_ = debug.get("stages", {}).get("final", {})
    for m in final_.get("detections", []) or []:
        mid = m.get("id")
        if marker_id is not None and mid != marker_id:
            continue
        if only_inliers and mid is not None and int(mid) not in inlier_ids:
            continue

        cx, cy = m.get("center", [None, None])
        if cx is None or cy is None:
            continue

        is_completion = mid is not None and int(mid) in completion_added
        dot_color = style.completion_added_marker if is_completion else style.candidate_accepted
        ax.plot(float(cx), float(cy), "o", color=dot_color, markersize=4, alpha=alpha)
        if mid is not None:
            add_id_label(
                ax,
                float(cx),
                float(cy),
                str(mid),
                style,
                text_color=style.completion_added_text if is_completion else None,
            )

        if show_ellipses:
            for key, color in (("ellipse_outer", style.ellipse_outer), ("ellipse_inner", style.ellipse_inner)):
                ell = m.get(key)
                if ell:
                    xs, ys = sample_ellipse_xy(ell)
                    ax.plot(xs, ys, "-", color=color, linewidth=1.0, alpha=alpha)


def plot_edge_points_for_id(
    ax,
    debug: dict[str, Any],
    marker_id: int,
    *,
    alpha: float,
    style: PlotStyle,
) -> None:
    points_drawn = False
    for c in iter_stage_candidates(debug, "stage1_fit_decode"):
        derived_id = (c.get("derived", {}) or {}).get("id")
        if derived_id is None or int(derived_id) != int(marker_id):
            continue
        rf = c.get("ring_fit")
        if not rf:
            continue
        edge = rf.get("edge") or {}
        pts_outer = edge.get("outer_points") or []
        pts_inner = rf.get("inner_points_fit") or edge.get("inner_points") or []
        for pts, color in ((pts_outer, style.ellipse_outer), (pts_inner, style.ellipse_inner)):
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, ".", color=color, markersize=1.5, alpha=alpha)
                points_drawn = True
        break
    if not points_drawn:
        ax.text(
            10,
            20,
            "edge points not present (run with --debug-store-points)",
            color=style.warning_text,
            fontsize=10,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ringgrid debug overlays")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--debug_json", required=True, type=str)
    parser.add_argument("--out", default=None, type=str, help="Write PNG to this path (otherwise interactive window)")
    parser.add_argument(
        "--stage",
        type=str,
        default="final",
        choices=[
            "final",
            "stage0_proposals",
            "stage1_fit_decode",
            "stage3_ransac",
            "stage4_refine",
            "stage5_completion",
            "stage6_nl_refine",
        ],
    )
    parser.add_argument("--only-inliers", action="store_true")
    parser.add_argument("--id", type=int, default=None, help="Focus on a single decoded id")
    parser.add_argument("--zoom", type=float, default=None, help="Zoom factor when --id is provided (default 4.0)")
    parser.add_argument("--show-ellipses", dest="show_ellipses", action="store_true", default=True)
    parser.add_argument("--no-ellipses", dest="show_ellipses", action="store_false")
    parser.add_argument("--show-candidates", dest="show_candidates", action="store_true", default=True)
    parser.add_argument("--no-candidates", dest="show_candidates", action="store_false")
    parser.add_argument("--show-edge-points", action="store_true", default=False)
    parser.add_argument(
        "--show-completion",
        action="store_true",
        default=False,
        help="Overlay homography-guided completion projected centers (if present in debug dump).",
    )
    parser.add_argument(
        "--show-nl-refine",
        action="store_true",
        default=False,
        help="Overlay NL refinement centers (before/after) when present in debug dump.",
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    args = parser.parse_args()

    if args.id is not None and args.zoom is None:
        args.zoom = 4.0
    if args.stage == "stage5_completion":
        args.show_completion = True
    if args.stage == "stage6_nl_refine":
        args.show_nl_refine = True

    maybe_use_agg(args.out)
    import matplotlib.pyplot as plt

    style = PlotStyle()

    debug = load_json(args.debug_json)
    schema_version = debug.get("schema_version")
    if schema_version not in ("ringgrid.debug.v2", "ringgrid.debug.v1"):
        if "detected_markers" in debug and "image_size" in debug:
            raise SystemExit(
                "Input looks like DetectionResult (normal --out JSON), not a debug dump.\n"
                "Re-run detection with `ringgrid detect --debug-json <path>` and pass that file here."
            )
        raise SystemExit(f"Unsupported debug schema_version: {schema_version!r}")

    img = plt.imread(args.image)
    # image dims from debug (authoritative for plotting extents)
    img_w = int(debug.get("image", {}).get("width", img.shape[1]))
    img_h = int(debug.get("image", {}).get("height", img.shape[0]))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img, cmap="gray")

    title = f"ringgrid debug overlay ({args.stage})"
    if args.id is not None:
        title += f" id={args.id}"

        # Add inner/outer estimation metrics (if available) to the title.
        rf = ring_fit_debug_for_id(debug, args.id) or {}
        inner = rf.get("inner_estimation")
        outer = rf.get("outer_estimation")
        if outer:
            status = outer.get("status")
            chosen = rf.get("chosen_outer_hypothesis")
            hypotheses = outer.get("hypotheses") or []
            r_found = None
            if chosen is not None and 0 <= int(chosen) < len(hypotheses):
                h = hypotheses[int(chosen)]
                r_found = h.get("r_outer_px")
            tc = outer.get("theta_consistency")
            ps = outer.get("peak_strength")
            reason = outer.get("reason")
            title += (
                f"\nouter: status={status}"
                f" r={r_found if r_found is not None else 'NA'}"
                f" tc={tc if tc is not None else 'NA'}"
                f" ps={ps if ps is not None else 'NA'}"
                f" hyp={chosen if chosen is not None else 'NA'}"
            )
            if reason:
                title += f" ({reason})"
        if inner:
            status = inner.get("status")
            r_found = inner.get("r_inner_found")
            tc = inner.get("theta_consistency")
            ps = inner.get("peak_strength")
            reason = inner.get("reason")
            title += (
                f"\ninner: status={status}"
                f" r={r_found if r_found is not None else 'NA'}"
                f" tc={tc if tc is not None else 'NA'}"
                f" ps={ps if ps is not None else 'NA'}"
            )
            if reason:
                title += f" ({reason})"

        nl = nl_refine_entry_for_id(debug, args.id)
        if nl:
            status = nl.get("status")
            br = nl.get("before_rms_mm")
            ar = nl.get("after_rms_mm")
            dc = nl.get("delta_center_mm")
            reason = nl.get("reason")
            title += (
                f"\nnl: status={status}"
                f" rms_mm={br if br is not None else 'NA'}→{ar if ar is not None else 'NA'}"
                f" Δc_mm={dc if dc is not None else 'NA'}"
            )
            if reason:
                title += f" ({reason})"
    ax.set_title(title)

    # If focusing on a marker id and the debug contains the aggregated radial
    # response curve, plot it as an inset.
    if args.id is not None:
        rf = ring_fit_debug_for_id(debug, args.id) or {}
        inner = rf.get("inner_estimation")
        outer = rf.get("outer_estimation")
        if outer and outer.get("radial_response_agg") is not None and outer.get("r_samples") is not None:
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                r_samples = outer["r_samples"]
                resp = outer["radial_response_agg"]
                inset = inset_axes(ax, width="35%", height="25%", loc="upper left", borderpad=1.0)
                inset.plot(r_samples, resp, color="cyan", linewidth=1.0)
                inset.set_title("outer dI/dr (agg)", fontsize=8, color="white")
                inset.tick_params(axis="both", labelsize=7, colors="white")
                inset.grid(True, alpha=0.2)
                inset.set_facecolor((0, 0, 0, 0.35))
            except Exception:
                pass

        if inner and inner.get("radial_response_agg") is not None and inner.get("r_samples") is not None:
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                r_samples = inner["r_samples"]
                resp = inner["radial_response_agg"]
                inset = inset_axes(ax, width="35%", height="25%", loc="lower left", borderpad=1.0)
                inset.plot(r_samples, resp, color="white", linewidth=1.0)
                inset.set_title("inner dI/dr (agg)", fontsize=8, color="white")
                inset.tick_params(axis="both", labelsize=7, colors="white")
                inset.grid(True, alpha=0.2)
                inset.set_facecolor((0, 0, 0, 0.35))
            except Exception:
                # Inset plotting is best-effort (backend/toolkit availability).
                pass

    # Helpers for ransac inlier/outlier sets (ids)
    ransac = debug.get("stages", {}).get("stage3_ransac", {})
    inlier_ids = set(int(x) for x in (ransac.get("inlier_ids", []) or []))
    outlier_ids = set(int(x) for x in (ransac.get("outlier_ids", []) or []))
    comp_added = completion_added_ids(debug)

    # Candidate scatter
    if args.show_candidates and args.stage in ("stage0_proposals", "stage1_fit_decode", "stage3_ransac"):
        plot_candidates(
            ax,
            debug,
            args.stage,
            marker_id=args.id,
            only_inliers=args.only_inliers,
            inlier_ids=inlier_ids,
            outlier_ids=outlier_ids,
            alpha=args.alpha,
            style=style,
        )

    # Refine stage overlay
    if args.stage == "stage4_refine":
        plot_refine(
            ax,
            debug,
            marker_id=args.id,
            only_inliers=args.only_inliers,
            inlier_ids=inlier_ids,
            alpha=args.alpha,
            show_ellipses=args.show_ellipses,
            style=style,
        )

    # Final detections overlay
    if args.stage == "final":
        plot_final(
            ax,
            debug,
            marker_id=args.id,
            only_inliers=args.only_inliers,
            inlier_ids=inlier_ids,
            completion_added=comp_added,
            alpha=args.alpha,
            show_ellipses=args.show_ellipses,
            style=style,
        )

    # Completion overlay (projected centers + optional fit ellipses)
    if args.show_completion and args.stage in ("final", "stage5_completion"):
        plot_completion(
            ax,
            debug,
            marker_id=args.id,
            alpha=args.alpha,
            show_ellipses=args.show_ellipses,
            style=style,
        )

    # NL refine overlay (board-plane circle fit)
    if args.show_nl_refine and args.stage in ("final", "stage6_nl_refine"):
        plot_nl_refine(
            ax,
            debug,
            marker_id=args.id,
            only_inliers=args.only_inliers,
            inlier_ids=inlier_ids,
            alpha=args.alpha,
            show_after=args.stage == "stage6_nl_refine",
            style=style,
        )

    # If focusing on an id and not in final/refine stage, draw stage1 ellipses if available.
    if args.id is not None and args.show_ellipses and args.stage in ("stage0_proposals", "stage1_fit_decode", "stage3_ransac"):
        plot_stage1_ellipses_for_id(ax, debug, args.id, alpha=args.alpha, style=style)

    # Optional edge points when focusing on an id
    if args.show_edge_points and args.id is not None:
        plot_edge_points_for_id(ax, debug, args.id, alpha=args.alpha, style=style)

    # Default axes extents in image coordinates
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)

    # Zoom if requested
    if args.id is not None and args.zoom is not None:
        # Choose zoom center: refine refined center, else final center, else stage1 derived center.
        center = None
        refine_m = refine_entry_for_id(debug, args.id)
        if refine_m:
            center = (refine_m.get("refined_marker") or {}).get("center")
        if center is None:
            fm = find_final_marker(debug, args.id)
            if fm:
                center = fm.get("center")
        if center is None:
            for c in iter_stage_candidates(debug, "stage1_fit_decode"):
                if (c.get("derived", {}) or {}).get("id") == args.id:
                    center = (c.get("derived", {}) or {}).get("center_xy")
                    break

        if center:
            x0, x1, y0, y1 = compute_zoom_window(center, img_w, img_h, args.zoom)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)  # invert y for image coords
    ax.set_aspect("equal")
    fig.tight_layout()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote overlay to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
