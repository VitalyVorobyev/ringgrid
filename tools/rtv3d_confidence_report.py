#!/usr/bin/env python3
"""Photometric confidence against centre-error proxies on strip captures (ADR-029).

Input: the JSON lines of ``cargo run --release -p ringgrid-cli --example
photometric_confidence_dump`` (one per reported marker, ``min_confidence = 0``).
There is no ground truth on real captures, so two leave-one-out proxies are
reported per tile-row band (top / middle / bottom third):

- ``loo8``: residual of the marker's centre against the homography of its 8
  nearest labelled neighbours (board mm -> image px). It also measures the
  displayed board's geometry and smooth lens curvature, both fixed in board
  mm, so it grows with the marker's pixel size.
- ``cubic``: residual against a full cubic warp (10 terms per coordinate)
  fitted to the 20 nearest labelled neighbours within 40 mm; it absorbs the
  smooth part. Markers without a full neighbourhood have no value.

``--display`` also tests whether the per-ID ``loo8`` offset (board mm,
averaged over views) is a fixed property of the displayed target: split-half
repeatability across strips and the share explained by the marker's board
column (x) / row (y) alone.

Prints aggregates only (no per-tile output), so it is safe on private data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

BANDS = (("top", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("bottom", 2 / 3, 1.01), ("all", 0.0, 1.01))


def ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y) -> float:
    return float(np.corrcoef(ranks(np.asarray(x, float)), ranks(np.asarray(y, float)))[0, 1])


def roc_auc(score, bad) -> float:
    """P(score of a bad marker > score of a good one)."""
    bad = np.asarray(bad, bool)
    nb, ng = bad.sum(), (~bad).sum()
    if nb == 0 or ng == 0:
        return float("nan")
    r = ranks(np.asarray(score, float))
    return float(((r[bad] + 1).sum() - nb * (nb + 1) / 2) / (nb * ng))


def dlt(src, dst) -> np.ndarray:
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    return np.linalg.svd(np.asarray(a))[2][-1].reshape(3, 3)


def jacobian(h: np.ndarray, p) -> np.ndarray:
    q = h @ np.array([p[0], p[1], 1.0])
    u, v, w = q[0] / q[2], q[1] / q[2], q[2]
    return np.array([[(h[0, 0] - u * h[2, 0]) / w, (h[0, 1] - u * h[2, 1]) / w],
                     [(h[1, 0] - v * h[2, 0]) / w, (h[1, 1] - v * h[2, 1]) / w]])


def cubic_terms(p: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    return np.column_stack([np.ones(len(p)), x, y, x * x, x * y, y * y, x ** 3, x * x * y, x * y * y, y ** 3])


def tile_rows(ms: list[dict]) -> list[dict]:
    xy = np.array([m["board_xy_mm"] for m in ms])
    c = np.array([m["center"] for m in ms])
    rows = []
    for a, m in enumerate(ms):
        d = np.hypot(*(xy - xy[a]).T)
        d[a] = np.inf
        order = np.argsort(d)
        row = {"y": m["center"][1] / m["height"], "conf": m["confidence"], "id": m["id"],
               "board": m["board_xy_mm"], "tile": m["tile"], "loo8": np.nan, "loo8_mm": None,
               "cubic": np.nan}
        nb = order[:8]
        if len(ms) > 8 and d[nb].max() <= 28.0:
            h = dlt(xy[nb], c[nb])
            q = h @ np.array([*xy[a], 1.0])
            r = c[a] - q[:2] / q[2]
            row["loo8"] = float(np.hypot(*r))
            row["loo8_mm"] = np.linalg.solve(jacobian(h, xy[a]), r)
        nb = order[:20]
        if len(ms) > 20 and d[nb].max() <= 40.0:
            coef = np.linalg.lstsq(cubic_terms((xy[nb] - xy[a]) / 10.0), c[nb], rcond=None)[0]
            row["cubic"] = float(np.hypot(*(c[a] - coef[0])))
        rows.append(row)
    return rows


def load(path: str) -> list[dict]:
    tiles = defaultdict(list)
    with open(path) as f:
        for line in f:
            m = json.loads(line)
            if m.get("board_xy_mm") is not None and m["center_source"] == "photometric":
                tiles[m["tile"]].append(m)
    return [r for ms in tiles.values() if len(ms) > 8 for r in tile_rows(ms)]


def band_table(rows: list[dict]) -> None:
    print("band     centres  conf med  conf p10  <0.03   | cubic: n  med px  AUC>0.1  rho | loo8: med px  AUC>0.5  rho")
    for name, lo, hi in BANDS:
        s = [r for r in rows if lo <= r["y"] < hi]
        conf = np.array([r["conf"] for r in s])
        cub = np.array([r["cubic"] for r in s])
        loo = np.array([r["loo8"] for r in s])
        ok, okl = np.isfinite(cub), np.isfinite(loo)
        print(f"{name:7} {len(s):8d} {np.median(conf):9.3f} {np.percentile(conf, 10):9.3f} {np.mean(conf < 0.03):7.4f} | "
              f"{ok.sum():8d} {np.median(cub[ok]):7.3f} {roc_auc(-conf[ok], cub[ok] > 0.1):8.3f} "
              f"{spearman(conf[ok], cub[ok]):+5.2f} | {np.median(loo[okl]):12.3f} "
              f"{roc_auc(-conf[okl], loo[okl] > 0.5):8.3f} {spearman(conf[okl], loo[okl]):+5.2f}")


def display_test(rows: list[dict]) -> None:
    by_id, half, pos = defaultdict(list), defaultdict(lambda: ([], [])), {}
    strips = sorted({r["tile"].rsplit("_", 1)[0] for r in rows})
    even = set(strips[0::2])
    for r in rows:
        v = r["loo8_mm"]
        if v is None or np.hypot(*v) >= 0.5:
            continue
        by_id[r["id"]].append(v)
        half[r["id"]][0 if r["tile"].rsplit("_", 1)[0] in even else 1].append(v)
        pos[r["id"]] = r["board"]
    ids = [i for i in half if len(half[i][0]) >= 2 and len(half[i][1]) >= 2]
    a = np.array([np.mean(half[i][0], 0) for i in ids])
    b = np.array([np.mean(half[i][1], 0) for i in ids])
    print(f"per-ID loo8 offset (board mm), {len(ids)} IDs: split-half correlation across strips "
          f"x {np.corrcoef(a[:, 0], b[:, 0])[0, 1]:.2f} y {np.corrcoef(a[:, 1], b[:, 1])[0, 1]:.2f}")
    ids = [i for i, v in by_id.items() if len(v) >= 6]
    m = np.array([np.mean(by_id[i], 0) for i in ids])
    p = np.array([pos[i] for i in ids])
    for axis, name in ((0, "x by column"), (1, "y by row")):
        groups = defaultdict(list)
        for v, k in zip(m[:, axis], p[:, axis]):
            groups[round(float(k), 3)].append(v)
        within = sum(np.var(g) * len(g) for g in groups.values())
        print(f"  offset {name}: between-group share {1 - within / (np.var(m[:, axis]) * len(m)):.2f} "
              f"({len(groups)} groups, {len(ids)} IDs)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump", help="JSON lines from the photometric_confidence_dump example")
    p.add_argument("--display", action="store_true", help="also test the displayed board's per-marker offsets")
    args = p.parse_args()
    rows = load(args.dump)
    band_table(rows)
    if args.display:
        display_test(rows)


if __name__ == "__main__":
    main()
