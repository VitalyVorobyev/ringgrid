"""Score held-out center observations in image pixels. See docs/studies/center-metrology.md."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from score_coverage import fit_homography_dlt


def point(row, field):
    value = row.get(field)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"invalid {field}: {value!r}")
    return array


def identity(row):
    return row["capture"], row["camera"], row.get("replicate", 0), json.dumps(row["key"], sort_keys=True)


def summary(vectors):
    if not vectors:
        return {"n": 0}
    data = np.asarray(vectors)
    lengths = np.linalg.norm(data, axis=1)
    return dict(n=len(data), mean_xy_px=data.mean(axis=0).tolist(),
                mean_error_px=float(np.mean(lengths)),
                rms_px=float(np.sqrt(np.mean(lengths**2))),
                p95_px=float(np.quantile(lengths, .95)))


def usable(row):
    return row["status"] == "ok" and point(row, "center") is not None


def error(row):
    reference = point(row, "reference")
    return None if reference is None else point(row, "center") - reference


def condition_summary(rows):
    errors, within, runtimes = [], [], {}
    repeats = defaultdict(list)
    blur_errors = []
    statuses, sources = Counter(), Counter()
    for row in rows:
        statuses[row["status"]] += 1
        source = row.get("source") or row.get("center_source")
        if source is not None:
            sources[source] += 1
        if row.get("runtime_ms") is not None:
            runtimes[(row["camera"], row["capture"], row.get("replicate", 0))] = row["runtime_ms"]
        if not usable(row):
            continue
        if row.get("reference_sigma_px") is not None and row.get("blur_sigma_px") is not None:
            blur_errors.append(row["blur_sigma_px"] - row["reference_sigma_px"])
        if row.get("repeat_group") is not None:
            repeats[(row["camera"], row["capture"], row["repeat_group"],
                     json.dumps(row["key"], sort_keys=True))].append(point(row, "center"))
        delta = error(row)
        if delta is None:
            continue
        errors.append(delta)
        # A real sharp/blurred pair has an uncertain and correlated reference.
        # Its displacement cannot validate a one-observation center covariance.
        if row.get("cov") is not None and row["dataset"] == "synthetic":
            cov = np.asarray(row["cov"], dtype=float)
            if (cov.shape == (2, 2) and np.isfinite(cov).all()
                    and np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov)[0] > 0):
                within.append(float(delta @ np.linalg.solve(cov, delta)) <= 5.991465)
    # Pool only within-marker, same-pose noise variance (Bessel corrected).
    squared = np.zeros(2)
    df = 0
    for values in repeats.values():
        if len(values) > 1:
            squared += np.sum((values - np.mean(values, axis=0))**2, axis=0)
            df += len(values) - 1
    blur = dict(n=len(blur_errors), mean_px=float(np.mean(blur_errors)),
                rms_px=float(np.sqrt(np.mean(np.square(blur_errors))))) if blur_errors else None
    return dict(blur_error=blur, repeatability_sd_xy_px=np.sqrt(squared / df).tolist() if df else None,
                repeatability_df=df, status_counts=dict(statuses), sources=dict(sources), error=summary(errors),
                covariance_n=len(within), coverage_95=float(np.mean(within)) if within else None,
                median_runtime_ms=float(np.median(list(runtimes.values()))) if runtimes else None)


def paired_interval(clusters):
    """Equal capture weight; resample whole captures and keep methods paired."""
    if not clusters:
        return None
    means = np.array([np.mean(vectors, axis=0) for _, vectors in sorted(clusters.items())])
    # Columns are baseline dx,dy and candidate dx,dy for common observations.
    def effects(matrix):
        change = matrix[..., 2:] - matrix[..., :2]
        bias_gain = np.linalg.norm(matrix[..., :2], axis=-1) - np.linalg.norm(matrix[..., 2:], axis=-1)
        return np.concatenate((change, bias_gain[..., None]), axis=-1)
    mean = effects(means.mean(axis=0))
    interval = None
    if len(means) >= 2:
        rng = np.random.default_rng(1729)
        draws = rng.integers(len(means), size=(2000, len(means)))
        interval = np.quantile(effects(means[draws].mean(axis=1)), [.025, .975], axis=0).T.tolist()
    return dict(n_captures=len(means), delta_bias_xy_px=mean[:2].tolist(),
                reduction_in_bias_norm_px=float(mean[2]), ci95_xy_and_reduction_px=interval)


def heldout_planar_residual(views):
    """Relative geometry proxy on real images. Use identical cells for both methods."""
    errors = [[], []]
    n_views = 0
    for cells in views.values():
        # Stable ordering independent of the estimator and of table row order.
        cells.sort(key=lambda pair: hashlib.sha256(
            json.dumps(pair[0]["key"], sort_keys=True).encode()).digest())
        test = cells[::3]
        fit = [pair for i, pair in enumerate(cells) if i % 3][:40]
        if len(fit) < 8 or len(test) < 4:
            continue
        src = np.array([a["board_xy_mm"] for a, _ in fit])
        probes = np.array([a["board_xy_mm"] for a, _ in test])
        projections = []
        for method in range(2):
            dst = np.array([pair[method]["center"] for pair in fit])
            h = fit_homography_dlt(src, dst)
            if h is None:
                break
            xyw = np.c_[probes, np.ones(len(probes))] @ h.T
            if np.any(abs(xyw[:, 2]) < 1e-10):
                break
            expected = xyw[:, :2] / xyw[:, 2:]
            projections.append(np.array([pair[method]["center"] for pair in test]) - expected)
        if len(projections) == 2:
            n_views += 1
            for out, residuals in zip(errors, projections):
                out.extend(residuals)
    return dict(n_views=n_views, baseline=summary(errors[0]), candidate=summary(errors[1]))


def scene_id(row):
    return row.get("scene_id", row["capture"])


def gradient_reversals(groups):
    # Odd component isolates the bias that follows gradient direction; a
    # constant per-cell offset or display phase cancels in the paired contrast.
    plus = groups.get(("synthetic", "holdout", "positive_gradient"), {})
    minus = groups.get(("synthetic", "holdout", "negative_gradient"), {})
    result = {}
    for method in plus.keys() & minus.keys():
        captures = defaultdict(list)
        n = 0
        for key in plus[method].keys() & minus[method].keys():
            p, m = plus[method][key], minus[method][key]
            if not (usable(p) and usable(m)):
                continue
            ep, em = error(p), error(m)
            if ep is None or em is None:
                continue
            captures[scene_id(p)].append((ep - em) / 2)
            n += 1
        means = np.array([np.mean(v, axis=0) for _, v in sorted(captures.items())])
        if len(means):
            ci = None
            if len(means) > 1:
                draws = np.random.default_rng(1730).integers(len(means), size=(2000, len(means)))
                ci = np.quantile(means[draws].mean(axis=1), [.025, .975], axis=0).T.tolist()
            result[method] = dict(n=n, n_scenes=len(means), odd_xy_px=means.mean(axis=0).tolist(),
                                  ci95_xy_px=ci)
    return result


def compare(base, candidate, plane=True):
    clusters = defaultdict(list)
    views = defaultdict(list)
    displacement, errors_a, errors_b = [], [], []
    for key in sorted(base.keys() & candidate.keys()):
        a, b = base[key], candidate[key]
        if not (usable(a) and usable(b)):
            continue
        displacement.append(point(b, "center") - point(a, "center"))
        if plane and a["dataset"] != "synthetic" and a.get("board_xy_mm") is not None:
            views[(a["camera"], a["capture"])].append((a, b))
        ea, eb = error(a), error(b)
        if ea is not None and eb is not None:
            errors_a.append(ea)
            errors_b.append(eb)
            clusters[scene_id(a)].append(np.r_[ea, eb])
    sharp_change = None
    if errors_a:
        sharp_change = float(np.mean(np.linalg.norm(errors_b, axis=1)
                                     - np.linalg.norm(errors_a, axis=1)))
    return dict(intermethod_displacement=summary(displacement),
                common_mean_error_change_px=sharp_change,
                common_baseline_error=summary(errors_a), common_candidate_error=summary(errors_b),
                paired_capture_effect=paired_interval(clusters),
                heldout_plane_proxy=heldout_planar_residual(views) if views else None)


def score(rows, baseline):
    groups = defaultdict(lambda: defaultdict(dict))
    splits = {}
    strata = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        physical_capture = row["dataset"], row["capture"]
        if splits.setdefault(physical_capture, row["split"]) != row["split"]:
            raise ValueError(f"capture leakage: {physical_capture}")
        group = row["dataset"], row["split"], row["condition"]
        index = groups[group][row["method"]]
        key = identity(row)
        if key in index:
            raise ValueError(f"duplicate observation: {group} {key}")
        index[key] = row
        if row["dataset"] == "synthetic":
            strata[(*group, "target_and_print", f"{row.get('target')} {row.get('revision')}")][row["method"]][key] = row
        else:
            for field in ("camera", "focus_band", "scale_bin", "contrast_bin"):
                if row.get(field) not in (None, "unknown"):
                    strata[(*group, field, str(row[field]))][row["method"]][key] = row
    results = []
    for (dataset, split, condition), methods in sorted(groups.items()):
        base = methods.get(baseline, {})
        results.append(dict(dataset=dataset, split=split, condition=condition,
                            methods={name: condition_summary(list(index.values()))
                                     for name, index in methods.items()},
                            paired={name: compare(base, idx) for name, idx in methods.items()
                                    if name != baseline and base}))
    breakdowns = []
    for (dataset, split, condition, field, value), methods in sorted(strata.items()):
        base = methods.get(baseline, {})
        for name, index in methods.items():
            paired = compare(base, index, plane=False) if name != baseline and base else None
            breakdowns.append(dict(dataset=dataset, split=split, condition=condition,
                                   field=field, value=value, method=name,
                                   paired=paired, **condition_summary(index.values())))
    return dict(baseline=baseline, results=results, strata=breakdowns,
                gradient_reversal=gradient_reversals(groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--baseline", default="per_image")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    def records():
        for path in args.input:
            with path.open() as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
    result = score(records(), args.baseline)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
