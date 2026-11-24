"""Study a camera-fixed focus-band direction using spatially withheld blur measurements."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def unit(degrees):
    a = np.deg2rad(degrees)
    return np.array([np.cos(a), np.sin(a)])


def profile(x, y):
    """BIC selects a flat profile, a slope, or two flanks of a focus band."""
    # Knot only in observed interior, both slopes supported by >=3 samples.
    best = (np.inf, None)
    choices = [(0, 0.0)]
    if np.ptp(x) > 1e-6:
        choices.append((1, 0.0))
        choices.extend(
            (2, k)
            for k in np.quantile(x, [0.2, 0.35, 0.5, 0.65, 0.8])
            if (x < k).sum() >= 3 and (x > k).sum() >= 3
        )
    for degree, k in choices:
        design = np.column_stack(
            [np.ones(len(x)), x, np.maximum(x - k, 0)][: degree + 1]
        )
        c, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank != len(c) or (degree == 2 and not (c[1] < 0 < c[1] + c[2])):
            continue
        mse = np.mean((design @ c - y) ** 2)
        loss = max(mse, 1e-15) * np.exp(
            (len(c) + int(degree == 2)) * np.log(len(x)) / len(x)
        )
        if loss < best[0]:
            best = loss, (np.pad(c, (0, 3 - len(c))), k)
    return best


def direction(views):
    angles = np.arange(0, 180, 2)
    losses = []
    for data in views:
        xy, sigma = data[:, :2], data[:, 2]
        if len(xy) < 8 or np.var(sigma) < 1e-10:
            continue
        losses.append([profile(xy @ unit(a), sigma)[0] / np.var(sigma) for a in angles])
    if not losses:
        raise ValueError("camera lacks directional training measurements")
    return int(angles[np.argmin(np.mean(losses, axis=0))]), angles[
        np.argmin(losses, axis=1)
    ]


def local(support, xy):
    # Production rule: nearest nine points, centered LS and its spread gate.
    near = support[
        np.argsort(np.sum((support[:, :2] - xy) ** 2, axis=1), kind="stable")[:9]
    ]
    x = near[:, :2] - near[:, :2].mean(axis=0)
    gram = x.T @ x
    if len(x) < 4 or np.linalg.det(gram) <= 1e-3 * np.trace(gram) ** 2:
        return near[:, 2].mean(), np.zeros(2), False
    g = np.linalg.solve(gram, x.T @ (near[:, 2] - near[:, 2].mean()))
    return near[:, 2].mean() + (xy - near[:, :2].mean(axis=0)) @ g, g, True


def profile_prediction(model, normal, xy):
    c, knot = model
    t = xy @ normal
    return c[0] + c[1] * t + c[2] * max(0, t - knot), (
        c[1] + c[2] * (t > knot)
    ) * normal


def predict(data, normal, size, ablate=False):
    xy = data[:, :2]
    # Withhold groups of nearby pixels; leave every observed marker out once.
    block = np.floor(4 * xy / size).astype(int).clip(0, 3)
    fold = block[:, 0] % 2 + 2 * (block[:, 1] % 2)
    for i in np.unique(fold):
        support = data[fold != i]
        if len(support) < 4:
            continue
        model = profile(support[:, :2] @ normal, support[:, 2])[1]
        alternatives = {}
        if ablate and len(support) >= 8 and np.var(support[:, 2]) > 1e-10:
            free = unit(direction([support])[0])
            for name, axis in (("free", free), ("vertical", unit(90))):
                alternatives[name] = (
                    profile(support[:, :2] @ axis, support[:, 2])[1],
                    axis,
                )
        for idx in np.flatnonzero(fold == i):
            sigma, g = profile_prediction(model, normal, xy[idx])
            alt = {
                name: profile_prediction(m, axis, xy[idx])[0]
                for name, (m, axis) in alternatives.items()
            }
            near_sigma, near_g, ok = local(support, xy[idx])
            yield idx, sigma, g, near_sigma, near_g, ok, np.mean(support[:, 2]), alt


def real_views(path):
    files = sorted(path.glob("*_adaptive_photometric.json"))
    views = defaultdict(dict)
    for file in files:
        m = re.fullmatch(r"target_(\d+)_(\d+)_adaptive_photometric.json", file.name)
        capture, camera = map(int, m.groups())
        result = json.loads(file.read_text())
        rows = [
            [*r["center"], r["blur"]["sigma_px"]]
            for r in result["detected_markers"]
            if r.get("blur") and r["center_source"] == "photometric"
        ]
        views[camera][capture] = np.array(rows).reshape(-1, 3)
    if not views:
        raise ValueError(f"no adaptive photometric files at {path}")
    return views


def synthetic(path, split, conditions=None):
    views = defaultdict(list)
    statuses = defaultdict(lambda: defaultdict(int))
    for line in path.open():
        r = json.loads(line)
        cond = r["condition"]
        if (
            r["split"] != split
            or r["method"] != "per_image"
            or (conditions and cond not in conditions)
        ):
            continue
        statuses[cond][r["status"]] += 1
        if r["status"] != "ok" or r["blur_sigma_px"] is None:
            continue
        key = (cond, r["capture"], r["replicate"], r["scene_id"])
        views[key].append(
            [
                *r["center"],
                r["blur_sigma_px"],
                *r["reference"],
                r["reference_sigma_px"],
                *r.get("true_blur_gradient_px_per_px", [0, r.get("focus_gradient", 0)]),
            ]
        )
    return {k: np.array(v) for k, v in views.items()}, statuses


def rms(arr):
    a = np.array(arr)
    return float(np.sqrt(np.mean(np.sum(a * a, axis=-1))))


def geometric(rows):
    result = {}
    for method in ("raw", "field", "neighbor", "oracle", "field_full", "neighbor_full"):
        err = np.stack([r[method] for r in rows])
        signed = [
            r[method] @ r["gradient"] / np.linalg.norm(r["gradient"])
            for r in rows
            if np.linalg.norm(r["gradient"]) > 0
        ]
        result[method] = {
            "mean_error_along_increasing_blur_px": float(np.mean(signed))
            if signed
            else None,
            "rms_px": rms(err),
            "p95_px": np.quantile(np.linalg.norm(err, axis=1), 0.95),
            "signed_bias_xy_px": err.mean(axis=0).tolist(),
        }
    return result


def capture_ci(rows):
    # Keep transformations of the same pose together. Paired gain in RMS on
    # the tested marker population, resampling whole scenes with replacement.
    groups = defaultdict(list)
    for r in rows:
        groups[r["scene"]].append(r)
    stats = np.array(
        [
            [
                len(rs),
                *[
                    sum(v[m] @ v[m] for v in rs)
                    for m in ("raw", "field", "neighbor", "oracle")
                ],
            ]
            for rs in groups.values()
        ]
    )
    rng = np.random.default_rng(93)
    sampled = stats[rng.integers(len(stats), size=(4000, len(stats)))].sum(axis=1)
    errs = np.sqrt(sampled[:, 1:] / sampled[:, :1])
    return {
        "n_scenes": len(stats),
        "raw_minus_field_px": np.quantile(
            errs[:, 0] - errs[:, 1], [0.025, 0.975]
        ).tolist(),
        "neighbor_minus_field_px": np.quantile(
            errs[:, 2] - errs[:, 1], [0.025, 0.975]
        ).tolist(),
    }


def synthetic_study(train_path, test_path, conditions=None):
    train, _ = synthetic(train_path, "train")
    if conditions is None:
        conditions = [
            "positive_gradient",
            "negative_gradient",
            "zero_gradient",
            "anisotropic",
        ]
    test, counts = synthetic(test_path, "holdout", conditions)
    if not train or not test or {k[3] for k in train} & {k[3] for k in test}:
        raise ValueError("expected nonempty, disjoint training and test scenes")
    angle, _ = direction(train.values())
    rows = defaultdict(list)
    for (cond, _, _, scene), data in test.items():
        axis = unit(angle)
        complete = profile(data[:, :2] @ axis, data[:, 2])[1]
        for i, _, gf, _, gn, ok, _, _ in predict(
            data, unit(angle), np.array([236, 188])
        ):
            value = data[i]
            raw = value[:2] - value[3:5]
            # The local sigma is an observed measurement available at inference;
            # its own value is excluded from gradient estimation.
            gt = value[6:8]
            # Also evaluate operational density. Production includes the
            # marker itself in the nine-point fit; these are geometric scores
            # against independent truth, never blur-prediction scores.
            _, gn_full, _ = local(data, value[:2])
            _, gf_full = profile_prediction(complete, axis, value[:2])
            rows[cond].append(
                {
                    "scene": scene,
                    "raw": raw,
                    "field": raw + 1.24 * value[2] * gf,
                    "neighbor": raw + 1.24 * value[2] * gn,
                    "oracle": raw + 1.24 * value[5] * gt,
                    "field_full": raw + 1.24 * value[2] * gf_full,
                    "neighbor_full": raw + 1.24 * value[2] * gn_full,
                    "gradient": gt,
                    "field_g": gf - gt,
                    "neighbor_g": gn - gt,
                    "neighbor_ok": ok,
                }
            )
    report = {}
    for cond, rs in rows.items():
        scores = geometric(rs)
        gain = scores["raw"]["rms_px"] - scores["oracle"]["rms_px"]
        report[cond] = {
            "n": len(rs),
            "cohort": dict(counts[cond]),
            "metrics": scores,
            "ci95": capture_ci(rs),
            "field_oracle_gain_fraction": (
                scores["raw"]["rms_px"] - scores["field"]["rms_px"]
            )
            / gain
            if gain > 1e-9
            else None,
            "field_gradient_rmse": rms([r["field_g"] for r in rs]),
            "neighbor_gradient_rmse": rms([r["neighbor_g"] for r in rs]),
            "neighbor_fallbacks": sum(not r["neighbor_ok"] for r in rs),
        }
    return {"normal_deg": angle, "training_views": len(train), "conditions": report}


def real_study(views, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), layout="constrained", sharex=True, sharey=True
    )
    details = {}
    for (camera, captures), ax in zip(sorted(views.items()), axes.flat, strict=True):
        train, test = sorted(captures)[::2], sorted(captures)[1::2]
        a, indiv = direction([captures[k] for k in train])
        records = []
        for k in test:
            data = captures[k]
            for i, f, gf, n, _, ok, flat, alt in predict(
                data, unit(a), np.array([720, 540]), ablate=True
            ):
                records.append(
                    [
                        k,
                        data[i, 2],
                        f,
                        n,
                        flat,
                        ok,
                        alt.get("free", flat),
                        alt.get("vertical", flat),
                        1.24 * data[i, 2] * np.linalg.norm(gf),
                    ]
                )
        arr = np.array(records)
        sq = (arr[:, 2:5] - arr[:, 1:2]) ** 2
        by_capture = [
            (int(k), sq[arr[:, 0] == k].sum(axis=0), int(np.sum(arr[:, 0] == k)))
            for k in np.unique(arr[:, 0])
        ]
        sample = np.array([[n, *s] for _, s, n in by_capture])
        rng = np.random.default_rng(55)
        boot = sample[rng.integers(len(sample), size=(4000, len(sample)))].sum(axis=1)
        b = np.sqrt(boot[:, 1:] / boot[:, :1])
        data = np.vstack([captures[k] for k in test])
        points = ax.scatter(
            data[:, 0], data[:, 1], c=data[:, 2], s=7, vmin=0.5, vmax=4, cmap="viridis"
        )
        # The white line denotes orientation only; its intercept is arbitrary.
        v = np.array([-unit(a)[1], unit(a)[0]])
        pts = np.array([360, 270]) + np.array([[-125], [125]]) * v
        ax.plot(*pts.T, color="white", linewidth=3)
        ax.plot(*pts.T, color="black", linewidth=1)
        ax.set(title=f"Camera {camera} · normal {a}°", xlim=(0, 720), ylim=(540, 0))
        details[camera] = {
            "train": train,
            "test": test,
            "normal_deg": a,
            "individual_train_deg": indiv.tolist(),
            "free_angle_rmse": float(np.mean((arr[:, 6] - arr[:, 1]) ** 2) ** 0.5),
            "vertical_rmse": float(np.mean((arr[:, 7] - arr[:, 1]) ** 2) ** 0.5),
            "n": len(arr),
            "available": len(data),
            "predicted_center_shift_median_p95_px": np.quantile(
                arr[:, 8], [0.5, 0.95]
            ).tolist(),
            "neighbor_fallbacks": int((arr[:, 5] == 0).sum()),
            "sigma_rmse_field_neighbor_flat": np.sqrt(sq.mean(axis=0)).tolist(),
            "neighbor_minus_field_ci95": np.quantile(
                b[:, 1] - b[:, 0], [0.025, 0.975]
            ).tolist(),
            "n_captures_field_wins": sum(s[0] < s[1] for _, s, _ in by_capture).item(),
        }
    fig.colorbar(points, ax=axes, label="Fitted blur σ (px)", extend="both")
    fig.supxlabel("Tile x (px)")
    fig.supylabel("Tile y (px)")
    fig.savefig(out / "focus_fields.png", dpi=150)
    plt.close(fig)
    return details


def main():
    root = Path("tools/out/center_metrology")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", type=Path, default=root / "rtv3d_final/detections")
    p.add_argument(
        "--train", type=Path, default=root / "synthetic_directional_train.jsonl"
    )
    p.add_argument(
        "--test", type=Path, default=root / "synthetic_independent_final.jsonl"
    )
    p.add_argument(
        "--band", type=Path, help="Optional independent moving-band challenge JSONL"
    )
    p.add_argument("--out", type=Path, default=root / "focus_field")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        "real": real_study(real_views(args.real), args.out),
        "synthetic": synthetic_study(args.train, args.test),
    }
    if args.band:
        results["moving_band"] = synthetic_study(args.band, args.band, conditions=[])
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
