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
