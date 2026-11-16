#!/usr/bin/env python3
"""Score a ringgrid detection against gen_synth_target.py ground truth.

Works for every target kind (hex | rect lattice, coded | plain, with or without
origin dots). Predictions are matched to ground-truth cells by center distance
(greedy nearest within --gate), and label quality is measured on lattice
coordinates:

- board_frame == "absolute": every matched prediction's grid_coord must equal
  the GT (u, v) — IDs or origin dots anchored the board, and a wrong anchor is a
  hard failure (origin_correct = false).
- board_frame == "relative_canonical": coordinates are only defined up to the
  board's rotational symmetry plus a lattice translation, so the score finds
  the best (lattice rotation x translation) map from predicted to GT
  coordinates and reports the agreement fraction under it. Only rotations
  count: the board images through an orientation-preserving homography, so a
  mirrored labeling is wrong. ``coord_accuracy_up_to_mirror`` repeats the
  search with reflections allowed, which separates "internally consistent but
  mirrored" (``labels_mirrored``) from "inconsistent" labels.

Coded targets are also scored on identity: ``id_accuracy`` is the fraction of
matched predictions carrying the GT ID, ``decoded_fraction`` the fraction of
matched predictions whose ID came from decoding their own code band
(``source == "fit_decoded"``) rather than from homography completion. On an
Extended-profile target the ``ext_*`` metrics restrict this to cells whose ID
lies beyond the base table, i.e. the extension codewords.

Usage:
    python tools/score_detect_target.py --gt gt_0000.json --pred det_0000.json \
        --gate 8.0 --out score_0000.json
"""

import argparse
import json
import math

# Orientation-preserving lattice rotations acting on integer grid coordinates
# (rows of a 2x2 matrix). Rect: C4. Hex: C6 on axial (q, r), where a 60° turn
# maps (q, r) -> (-r, q + r).
IDENTITY = ((1, 0), (0, 1))
ROTATIONS = {
    "rect": [
        IDENTITY,
        ((0, -1), (1, 0)),
        ((-1, 0), (0, -1)),
        ((0, 1), (-1, 0)),
    ],
    "hex": [
        IDENTITY,
        ((0, -1), (1, 1)),
        ((-1, -1), (1, 0)),
        ((-1, 0), (0, -1)),
        ((0, 1), (-1, -1)),
        ((1, 1), (-1, 0)),
    ],
}
# A lattice reflection (y -> -y); composed with ROTATIONS it spans the
# orientation-reversing lattice maps. Hex: (q, r) -> (q + r, -r).
REFLECTIONS = {
    "rect": ((1, 0), (0, -1)),
    "hex": ((1, 1), (0, -1)),
}


def compose(m, n):
    """Matrix product m·n of two 2x2 integer matrices."""
    (a, b), (c, d) = m
    (e, f), (g, h) = n
    return ((a * e + b * g, a * f + b * h), (c * e + d * g, c * f + d * h))


def match_by_center(gt_cells, preds, gate_px):
    """Greedy one-to-one nearest matching of predictions to visible GT cells."""
    pairs = []
    for pi, pred in enumerate(preds):
        px, py = pred["center"]
        for gi, cell in enumerate(gt_cells):
            if not cell["visible"]:
                continue
            gx, gy = cell["true_image_center"]
            d = math.hypot(px - gx, py - gy)
            if d <= gate_px:
                pairs.append((d, pi, gi))
    pairs.sort(key=lambda t: t[0])
    used_pred, used_gt, matches = set(), set(), []
    for d, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append({"pred_idx": pi, "gt_idx": gi, "center_err_px": d})
    return matches


def coord_agreement(matched_coords, rotation, translation):
    """Fraction of (pred, gt) coordinate pairs consistent with R·pred + t."""
    (a, b), (c, d) = rotation
    n_ok = 0
    for (pu, pv), (gu, gv) in matched_coords:
        mu = a * pu + b * pv + translation[0]
        mv = c * pu + d * pv + translation[1]
        if (mu, mv) == (gu, gv):
            n_ok += 1
    return n_ok / max(1, len(matched_coords))


def best_symmetry_agreement(matched_coords, rotations):
    """Best coordinate agreement over lattice rotations x solved translations."""
    best = 0.0
    for rotation in rotations:
        (a, b), (c, d) = rotation
        # Solve the translation from the most common offset.
        offsets = {}
        for (pu, pv), (gu, gv) in matched_coords:
            t = (gu - (a * pu + b * pv), gv - (c * pu + d * pv))
            offsets[t] = offsets.get(t, 0) + 1
        if not offsets:
            continue
        translation = max(offsets, key=offsets.get)
        best = max(best, coord_agreement(matched_coords, rotation, translation))
    return best


def project(h, x, y):
    w = h[0][0] * x + h[0][1] * y + h[0][2]
    v = h[1][0] * x + h[1][1] * y + h[1][2]
    z = h[2][0] * x + h[2][1] * y + h[2][2]
    return w / z, v / z


def marker_sources(pred):
    """Per-marker ``source`` from the diagnostics block, when it is aligned."""
    diag = (pred.get("diagnostics") or {}).get("markers")
    if diag is None or len(diag) != len(pred.get("detected_markers", [])):
        return None
    return [m.get("source") for m in diag]


def identity_metrics(gt, preds, sources, matches):
    """ID accuracy / decode provenance for coded targets."""
    gt_cells = gt["cells"]
    n_matched = len(matches)
    correct = [
        m for m in matches if preds[m["pred_idx"]].get("id") == gt_cells[m["gt_idx"]]["id"]
    ]
    decoded = [m for m in correct if sources and sources[m["pred_idx"]] == "fit_decoded"]
    result = {
        "id_accuracy": len(correct) / n_matched if n_matched else 0.0,
        "n_id_wrong": n_matched - len(correct),
        "decoded_fraction": (len(decoded) / n_matched if n_matched else 0.0)
        if sources is not None
        else None,
    }

    if gt.get("codebook_profile") == "extended":
        base_n = gt["codebook_base_n"]
        n_ext = sum(1 for c in gt_cells if c["visible"] and c["id"] >= base_n)

        def ext(ms):
            return sum(1 for m in ms if gt_cells[m["gt_idx"]]["id"] >= base_n)

        result["n_gt_visible_ext"] = n_ext
        result["ext_recall"] = ext(correct) / n_ext if n_ext else None
        result["ext_decoded_rate"] = (
            ext(decoded) / n_ext if n_ext and sources is not None else None
        )
    return result


def score(gt, pred, gate_px):
    gt_cells = gt["cells"]
    preds = pred.get("detected_markers", [])
    board_frame = pred.get("board_frame")

    matches = match_by_center(gt_cells, preds, gate_px)
    n_visible = sum(1 for c in gt_cells if c["visible"])
    errs = sorted(m["center_err_px"] for m in matches)
    center_mean = sum(errs) / len(errs) if errs else float("nan")
    center_p95 = errs[min(len(errs) - 1, int(len(errs) * 0.95))] if errs else float("nan")

    matched_coords = []
    for m in matches:
        coord = preds[m["pred_idx"]].get("grid_coord")
        if coord is not None:
            cell = gt_cells[m["gt_idx"]]
            matched_coords.append((tuple(coord), (cell["u"], cell["v"])))

    origin_resolved = board_frame == "absolute"
    mirror_metrics = {}
    if origin_resolved:
        coord_accuracy = coord_agreement(matched_coords, IDENTITY, (0, 0))
    else:
        kind = gt["lattice"]["kind"]
        rotations = ROTATIONS[kind]
        coord_accuracy = best_symmetry_agreement(matched_coords, rotations)
        mirrored = [compose(r, REFLECTIONS[kind]) for r in rotations]
        up_to_mirror = best_symmetry_agreement(matched_coords, rotations + mirrored)
        mirror_metrics = {
            "coord_accuracy_up_to_mirror": up_to_mirror,
            "labels_mirrored": up_to_mirror > coord_accuracy,
        }
    origin_correct = origin_resolved and coord_accuracy >= 0.99

    # Homography quality is only meaningful in the absolute frame.
    h_vs_gt_mean = None
    if origin_resolved and pred.get("homography"):
        h = pred["homography"]
        errs_h = []
        for cell in gt_cells:
            if not cell["visible"]:
                continue
            px, py = project(h, *cell["board_xy_mm"])
            gx, gy = cell["true_image_center"]
            errs_h.append(math.hypot(px - gx, py - gy))
        if errs_h:
            h_vs_gt_mean = sum(errs_h) / len(errs_h)

    result = {
        "n_gt_visible": n_visible,
        "n_pred": len(preds),
        "n_matched": len(matches),
        "precision": len(matches) / len(preds) if preds else 0.0,
        "recall": len(matches) / n_visible if n_visible else 0.0,
        "center_err_mean_px": center_mean,
        "center_err_p95_px": center_p95,
        "board_frame": board_frame,
        "origin_resolved": origin_resolved,
        "origin_correct": origin_correct,
        "coord_accuracy": coord_accuracy,
        "n_coord_pairs": len(matched_coords),
        "h_vs_gt_mean_px": h_vs_gt_mean,
        **mirror_metrics,
    }
    if "codebook_profile" in gt:
        result.update(identity_metrics(gt, preds, marker_sources(pred), matches))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gate", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    with open(args.pred) as f:
        pred = json.load(f)

    result = score(gt, pred, args.gate)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    line = (
        f"precision {result['precision']:.3f}  recall {result['recall']:.3f}  "
        f"center {result['center_err_mean_px']:.3f}px  "
        f"frame {result['board_frame']}  coord_acc {result['coord_accuracy']:.3f}"
    )
    if "id_accuracy" in result:
        line += f"  id_acc {result['id_accuracy']:.3f}"
    print(line)


if __name__ == "__main__":
    main()
