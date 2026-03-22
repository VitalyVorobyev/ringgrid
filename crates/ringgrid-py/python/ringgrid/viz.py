"""Visualization helpers for ringgrid detections.

This module is imported lazily from :meth:`ringgrid.DetectionResult.plot`.
It requires `matplotlib` (install with `ringgrid[viz]`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load_matplotlib(out: str | Path | None):
    import matplotlib

    if out is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def _sample_ellipse_xy(ellipse: Mapping[str, Any], n: int = 128) -> tuple[list[float], list[float]]:
    cx = float(ellipse["cx"])
    cy = float(ellipse["cy"])
    a = float(ellipse["a"])
    b = float(ellipse["b"])
    angle = float(ellipse["angle"])

    ca = math.cos(angle)
    sa = math.sin(angle)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n + 1):
        t = (i / n) * 2.0 * math.pi
        x = a * math.cos(t)
        y = b * math.sin(t)
        xr = ca * x - sa * y + cx
        yr = sa * x + ca * y + cy
        xs.append(float(xr))
        ys.append(float(yr))
    return xs, ys


def _marker_confidence(marker: Mapping[str, Any]) -> float | None:
    value = marker.get("confidence")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(0.0, min(1.0, value))


def _confidence_color(conf: float | None) -> tuple[float, float, float]:
    if conf is None:
        return (0.0, 1.0, 0.0)
    return (1.0 - 0.9 * conf, 0.1 + 0.9 * conf, 0.1)


def _compute_zoom_window(
    center_xy: list[float], img_w: int, img_h: int, zoom: float
) -> tuple[float, float, float, float]:
    span = min(img_w, img_h) / max(zoom, 1e-6)
    half = span / 2.0
    cx, cy = center_xy
    x0 = max(0.0, cx - half)
    x1 = min(float(img_w), cx + half)
    y0 = max(0.0, cy - half)
    y1 = min(float(img_h), cy + half)
    return x0, x1, y0, y1


def _to_detection_dict(detection: Any) -> dict[str, Any]:
    if hasattr(detection, "to_dict"):
        return detection.to_dict()
    if isinstance(detection, Mapping):
        return dict(detection)
    if isinstance(detection, Path):
        return json.loads(detection.read_text(encoding="utf-8"))
    if isinstance(detection, str):
        text = detection
        if text.lstrip().startswith("{"):
            return json.loads(text)
        path = Path(text)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return json.loads(text)
    raise TypeError("detection must be DetectionResult, mapping, JSON text, or JSON path")


def _to_image_array(image: np.ndarray | str | Path, plt) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return plt.imread(str(Path(image)))


def plot_detection(
    *,
    image: np.ndarray | str | Path,
    detection: Any,
    out: str | Path | None = None,
    marker_id: int | None = None,
    zoom: float | None = None,
    show_ellipses: bool = True,
    show_confidence: bool = True,
    alpha: float = 0.8,
) -> None:
    """Render a detection overlay over an image.

    Parameters
    ----------
    image:
        Path to image file or image array.
    detection:
        `DetectionResult`, dictionary, JSON text, or JSON file path.
    out:
        Optional output file path. If omitted, opens an interactive window.
    marker_id:
        Optional marker id filter.
    zoom:
        Optional zoom factor (used when `marker_id` is set).
    show_ellipses:
        Draw fitted outer/inner ellipses when available.
    show_confidence:
        Color marker centers by confidence and include confidence in labels.
    alpha:
        Overlay alpha for points/ellipses.
    """

    plt = _load_matplotlib(out)
    image_arr = _to_image_array(image, plt)
    detection_dict = _to_detection_dict(detection)

    markers = detection_dict.get("detected_markers")
    if not isinstance(markers, list):
        raise ValueError("detection payload missing 'detected_markers'")

    image_size = detection_dict.get("image_size")
    if isinstance(image_size, list) and len(image_size) == 2:
        img_w, img_h = int(image_size[0]), int(image_size[1])
    else:
        img_h, img_w = int(image_arr.shape[0]), int(image_arr.shape[1])

    if marker_id is not None and zoom is None:
        zoom = 4.0

    render_dpi = 100
    fig = plt.figure(
        figsize=(img_w / render_dpi, img_h / render_dpi),
        dpi=render_dpi,
        frameon=False,
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    if image_arr.ndim == 2:
        ax.imshow(image_arr, cmap="gray")
    else:
        ax.imshow(image_arr)
    ax.set_axis_off()

    import matplotlib.patheffects as pe

    for marker in markers:
        mid = marker.get("id")
        if marker_id is not None and mid != marker_id:
            continue

        center = marker.get("center")
        if not isinstance(center, list) or len(center) != 2:
            continue

        cx, cy = float(center[0]), float(center[1])
        conf = _marker_confidence(marker) if show_confidence else None
        color = _confidence_color(conf)

        ax.plot(cx, cy, "o", color=color, markersize=4, alpha=alpha)

        label = None
        if mid is not None and conf is not None:
            label = f"{mid} ({conf:.2f})"
        elif mid is not None:
            label = str(mid)
        elif conf is not None:
            label = f"{conf:.2f}"

        if label is not None:
            ax.text(
                cx + 4.0,
                cy - 4.0,
                label,
                fontsize=6,
                color="white",
                ha="left",
                va="bottom",
                path_effects=[pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()],
            )

        if not show_ellipses:
            continue

        for key, color_name in (("ellipse_outer", "lime"), ("ellipse_inner", "cyan")):
            ellipse = marker.get(key)
            if not isinstance(ellipse, Mapping):
                continue
            xs, ys = _sample_ellipse_xy(ellipse)
            ax.plot(xs, ys, "-", color=color_name, linewidth=1.0, alpha=alpha)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")

    if marker_id is not None and zoom is not None:
        center = None
        for marker in markers:
            if marker.get("id") == marker_id:
                center = marker.get("center")
                break
        if isinstance(center, list) and len(center) == 2:
            x0, x1, y0, y1 = _compute_zoom_window(
                [float(center[0]), float(center[1])],
                img_w,
                img_h,
                float(zoom),
            )
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)

    if out is None:
        plt.show()
        return

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=render_dpi, pad_inches=0)
    plt.close(fig)


def _to_proposal_result(diagnostics: Any):
    from ._api import ProposalResult

    if isinstance(diagnostics, ProposalResult):
        return diagnostics
    if isinstance(diagnostics, Mapping):
        return ProposalResult.from_dict(diagnostics)
    raise TypeError("diagnostics must be ProposalResult or a mapping")


def _proposal_xy_sizes(proposals: list[Any]) -> tuple[list[float], list[float], list[float]]:
    scores: list[float] = []
    for proposal in proposals:
        score = proposal.get("score") if isinstance(proposal, Mapping) else getattr(proposal, "score", 0.0)
        try:
            parsed = float(score)
        except (TypeError, ValueError):
            parsed = 0.0
        if math.isfinite(parsed):
            scores.append(parsed)
    max_score = max(scores) if scores else 1.0
    if max_score <= 0.0:
        max_score = 1.0

    xs: list[float] = []
    ys: list[float] = []
    sizes: list[float] = []
    for proposal in proposals:
        x = proposal.get("x") if isinstance(proposal, Mapping) else getattr(proposal, "x", None)
        y = proposal.get("y") if isinstance(proposal, Mapping) else getattr(proposal, "y", None)
        score = proposal.get("score") if isinstance(proposal, Mapping) else getattr(proposal, "score", 0.0)
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
            sf = float(score)
        except (TypeError, ValueError):
            continue
        xs.append(xf)
        ys.append(yf)
        sizes.append(6.0 + max(0.0, sf) / max_score * 18.0)
    return xs, ys, sizes


def _plot_proposals(ax, proposals: list[Any], *, alpha: float) -> None:
    xs, ys, sizes = _proposal_xy_sizes(proposals)
    if xs:
        ax.scatter(xs, ys, s=sizes, c="magenta", linewidths=0, alpha=alpha, zorder=3)


def _plot_gt_points(
    ax,
    gt_points: np.ndarray | list[list[float]] | None,
    gt_hits: np.ndarray | list[bool] | None,
) -> None:
    if gt_points is None:
        return

    points = np.asarray(gt_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("gt_points must have shape (N, 2)")

    if gt_hits is None:
        ax.scatter(points[:, 0], points[:, 1], s=26.0, c="cyan", marker="+", alpha=0.9, zorder=4)
        return

    hits = np.asarray(gt_hits, dtype=bool)
    if hits.shape != (points.shape[0],):
        raise ValueError("gt_hits must have shape (N,) and match gt_points length")

    if np.any(hits):
        matched = points[hits]
        ax.scatter(
            matched[:, 0],
            matched[:, 1],
            s=26.0,
            c="lime",
            marker="+",
            alpha=0.9,
            zorder=4,
        )
    if np.any(~hits):
        missed = points[~hits]
        ax.scatter(
            missed[:, 0],
            missed[:, 1],
            s=22.0,
            c="red",
            marker="x",
            alpha=0.9,
            zorder=4,
        )


def plot_proposal_diagnostics(
    *,
    image: np.ndarray | str | Path,
    diagnostics: Any,
    out: str | Path | None = None,
    heatmap_out: str | Path | None = None,
    gt_points: np.ndarray | list[list[float]] | None = None,
    gt_hits: np.ndarray | list[bool] | None = None,
    alpha: float = 0.8,
    heatmap_cmap: str = "magma",
    show_proposals_on_heatmap: bool = True,
) -> None:
    """Render proposal candidates and the post-NMS accumulator heatmap."""

    plt = _load_matplotlib(out)
    image_arr = _to_image_array(image, plt)
    diagnostics_obj = _to_proposal_result(diagnostics)

    proposals = diagnostics_obj.proposals
    accumulator = np.asarray(diagnostics_obj.heatmap, dtype=np.float32)
    if accumulator.ndim != 2:
        raise ValueError("diagnostics.heatmap must be a 2D array")

    if diagnostics_obj.image_size and len(diagnostics_obj.image_size) == 2:
        img_w, img_h = int(diagnostics_obj.image_size[0]), int(diagnostics_obj.image_size[1])
    else:
        img_h, img_w = int(accumulator.shape[0]), int(accumulator.shape[1])

    def _finalize_axes(ax) -> None:
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_aspect("equal")

    def _save_or_close(fig, output_path: str | Path | None) -> None:
        if output_path is None:
            return
        resolved = Path(output_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved, dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig_image, ax_image = plt.subplots(figsize=(8, 6), dpi=140)
    if image_arr.ndim == 2:
        ax_image.imshow(image_arr, cmap="gray")
    else:
        ax_image.imshow(image_arr)
    _plot_proposals(ax_image, proposals, alpha=alpha)
    _plot_gt_points(ax_image, gt_points, gt_hits)
    _finalize_axes(ax_image)
    fig_image.tight_layout()

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6), dpi=140)
    heat = ax_heatmap.imshow(accumulator, cmap=heatmap_cmap)
    if show_proposals_on_heatmap:
        _plot_proposals(ax_heatmap, proposals, alpha=alpha)
    _plot_gt_points(ax_heatmap, gt_points, gt_hits)
    _finalize_axes(ax_heatmap)
    fig_heatmap.colorbar(heat, ax=ax_heatmap, fraction=0.046, pad=0.04)
    fig_heatmap.tight_layout()

    if out is None and heatmap_out is None:
        plt.show()
        return

    _save_or_close(fig_image, out)
    _save_or_close(fig_heatmap, heatmap_out)
