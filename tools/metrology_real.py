#!/usr/bin/env python3
"""Run paired center metrology on private real captures. Outputs stay local.

Build the runner with:
    cargo build --release -p ringgrid --features cli,study --example metrology_frame
Run with:
    .venv/bin/python tools/metrology_real.py --jobs 4

Centers are in the input image's integer-pixel-center frame. Strip images use
tile-local coordinates. Same-method references measure repeatability under
added blur. Archived labels and sharp-position anchors establish correspondence
only. Native rtv3d observations have no center reference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from math import dist
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

from run_ipgs_eval import DEFAULT_DATA as IPGS_DATA, PLATE
from strip_tiles import N_CAMERAS, SKIP_STRIPS, crop_tile

REPO = Path(__file__).resolve().parents[1]
METHODS = ("projective", "per_image", "nominal", "pooled")
SCALES = {"ipgs": [50, 170], "rtv.t": [32, 32]}
TARGET_SCHEMA = "ringgrid.target.v6"


@dataclass(frozen=True)
class Frame:
    dataset: str
    capture: str
    camera: str
    source: Path
    tile: int | None = None
    sha256: str | None = None


def capture_split(index: int) -> str:
    return "holdout" if index % 2 else "train"


def select_frames(frames: list[Frame], limit: int) -> list[tuple[Frame, str]]:
    captures = list(dict.fromkeys(f.capture for f in frames))
    kept = {}
    for offset in (0, 1):
        group = captures[offset::2]
        if limit and len(group) > limit:
            group = [group[i] for i in np.linspace(0, len(group) - 1, limit, dtype=int)]
        kept.update((c, capture_split(offset)) for c in group)
    return [(frame, kept[frame.capture]) for frame in frames if frame.capture in kept]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def select_rtv_manifest(path):
    candidates = [path] if path.is_file() else sorted(path.glob("*/dataset.json"))
    matches = []
    for candidate in candidates:
        manifest = read_json(candidate)
        spec = manifest.get("target", {}).get("spec", {})
        if spec.get("kind") == "ring_grid":
            matches.append((candidate, manifest))
    if len(matches) != 1:
        raise ValueError(f"expected one embedded ring_grid manifest at {path}, got {len(matches)}")
    return matches[0]


def archive_labels(manifest_path, manifest_id, spec):
    result = {}
    for run_path in sorted(manifest_path.parent.glob("detections/*/run.json")):
        run = read_json(run_path)
        if (run["status"] != "succeeded" or run["dataset"] != manifest_id
                or run["config"]["target"]["spec"] != spec):
            continue
        labels = read_json(run_path.with_name("features.json"))
        if labels["detection_run"] != run["id"]:
            raise ValueError(f"inconsistent observation run ID at {run_path}")
        result = {}
        for row in labels["images"]:
            if row["observation"] is None:
                continue
            features = row["observation"]["features"]
            result[str(row["capture"]), str(row["camera"])] = {
                key: (point, None, xyz[:2])
                for key, point, xyz in zip(
                    features["ids"], features["image_px"], features["object_mm"], strict=True
                )
            }
    return result


def prepare_datasets(args):
    ipgs = {
        "schema": TARGET_SCHEMA, "name": "ipgs_58x58_2mm",
        "lattice": {"kind": "rect", **{k: PLATE[k] for k in ("rows", "cols", "pitch_mm")}},
        "marker": {k: PLATE[k] for k in ("outer_radius_mm", "inner_radius_mm")},
        "coding": {"kind": "plain"}, "fiducials": {"dot_radius_mm": 0.2},
    }
    args.rtv_archive = {}
    args.rtv_manifest = None
    targets = {"ipgs": ipgs}
    if "rtv3d" in args.datasets:
        targets["rtv3d"] = read_json(args.rtv3d_target)
    rtv_frames = []
    if "rtv.t" in args.datasets:
        manifest_path, manifest = select_rtv_manifest(args.rtvt)
        spec = manifest["target"]["spec"]
        args.rtv_archive = archive_labels(manifest_path, manifest["id"], spec)
        args.rtv_manifest = str(manifest_path)
        targets["rtv.t"] = {
            "schema": TARGET_SCHEMA, **{k: v for k, v in spec.items() if k != "kind"}
        }
        rtv_frames = [
            Frame("rtv.t", str(record["index"]), str(im["camera"]),
                  manifest_path.parent / im["path"], sha256=im.get("sha256"))
            for record in sorted(manifest["captures"], key=lambda row: row["index"])
            for im in record["images"]
        ]
    ipgs_frames = [
        Frame("ipgs", capture, cam, path)
        for path in sorted(args.ipgs.glob("Cam*.png"))
        for cam, capture in [path.stem.split("_", 1)]
    ]
    frames = {
        "ipgs": sorted(ipgs_frames, key=lambda f: (f.capture, f.camera)),
        "rtv.t": rtv_frames,
        "rtv3d": [
            Frame("rtv3d", path.stem, str(tile), path, tile)
            for path in sorted(args.rtv3d.glob("target_*.png"),
                               key=lambda p: int(p.stem.split("_")[-1]))
            if path.stem not in SKIP_STRIPS
            for tile in range(N_CAMERAS)
        ],
    }
    return targets, frames


def run_frame(args, frame, image_path, method, radii=None):
    cmd = [str(args.bin), str(image_path), str(args.out / f"{frame.dataset}-target.json"),
           method.replace("_", "-")]
    if frame.dataset in SCALES:
        cmd += ["--scale", *map(str, SCALES[frame.dataset])]
    if radii is not None:
        cmd += ["--radii", *map(str, radii)]
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(f"{frame.dataset}/{frame.capture}/{frame.camera}/{method}: {process.stderr}")
    return json.loads(process.stdout)


def centered_gaussian(image, sigma):
    """Separable normalized discrete Gaussian with symmetric kernel and edge padding."""
    radius = int(np.ceil(4.0 * sigma))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    pixels = np.asarray(image, dtype=np.float64)
    pixels = np.apply_along_axis(
        lambda a: np.convolve(np.pad(a, radius, mode="edge"), kernel, mode="valid"), 1, pixels
    )
    pixels = np.apply_along_axis(
        lambda a: np.convolve(np.pad(a, radius, mode="edge"), kernel, mode="valid"), 0, pixels
    )
    return Image.fromarray(np.rint(pixels).astype(np.uint8))


def load_frame(frame):
    if frame.sha256 and hashlib.sha256(frame.source.read_bytes()).hexdigest() != frame.sha256:
        raise ValueError(f"manifest checksum mismatch for {frame.source}")
    with Image.open(frame.source) as source:
        image = source.convert("L")
    return image if frame.tile is None else crop_tile(image, frame.tile, N_CAMERAS)


def training_frame(args, frame):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.png"
        load_frame(frame).save(path)
        return run_frame(args, frame, path, "per_image")


def fit_pools(args, selections, executor):
    training = [frame for frame, split in selections if split == "train"]
    fits = {}
    by_capture = defaultdict(list)
    for frame, run in zip(training, executor.map(lambda f: training_frame(args, f), training),
                          strict=True):
        fits[frame] = run
        radii = run["estimated_radii_mm"]
        if radii is not None and any(m["source"] == "photometric" for m in run["markers"]):
            by_capture[frame.dataset, frame.capture].append(radii)
    by_dataset = defaultdict(dict)
    for (dataset, capture), radii in by_capture.items():
        by_dataset[dataset][capture] = np.median(radii, axis=0).tolist()
    pools = {dataset: np.median(list(group.values()), axis=0).tolist()
             for dataset, group in by_dataset.items()}
    return pools, by_dataset, fits


def valid_center(center):
    return center is not None and len(center) == 2 and np.isfinite(center).all()


def make_anchors(args, frame, sharp):
    """Use a frozen projective sharp cohort with archive locations when available."""
    archived = args.rtv_archive.get((frame.capture, frame.camera)) if frame.dataset == "rtv.t" else None
    if archived:
        return archived, "archived_projective"
    output = {}
    for marker in sharp["markers"]:
        if not valid_center(marker["center"]):
            continue
        if frame.dataset == "ipgs":
            # A stable within-image identity even if the plate triad was not visible.
            key = "ring_%05d" % len(output)
        else:
            key = marker["id"]
        if key is not None:
            board_xy = marker["board_xy_mm"]
            if frame.dataset == "ipgs" and board_xy is None and marker["coord"] is not None:
                # Rectangular assignment-frame coordinates are scaled by pitch.
                board_xy = [PLATE["pitch_mm"] * value for value in marker["coord"]]
            output[key] = marker["center"], marker["coord"], board_xy
    return output, "sharp_projective"


def match_markers(anchors, render, gate=5.0):
    """Match uniquely to frozen image positions before checking a claimed label."""
    buckets = defaultdict(list)
    for key, (center, _, _) in anchors.items():
        buckets[int(center[0] // gate), int(center[1] // gate)].append((key, center))
    matches = defaultdict(list)
    unmatched = ambiguous = 0
    for marker in render["markers"]:
        if not valid_center(marker["center"]):
            unmatched += 1
            continue
        x, y = (int(value // gate) for value in marker["center"])
        close = [
            key
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for key, center in buckets[x + dx, y + dy]
            if dist(center, marker["center"]) <= gate
        ]
        if len(close) == 1:
            matches[close[0]].append(marker)
        elif close:
            ambiguous += 1
            for key in close:
                matches[key].append(None)
        else:
            unmatched += 1
    ambiguous += sum(
        sum(marker is not None for marker in candidates)
        for candidates in matches.values() if len(candidates) > 1
    )
    return matches, unmatched, ambiguous


def observation(frame, key, anchor, markers, run, method, cohort, sharp_frame):
    status = "missing"
    if len(markers) > 1 or markers == [None]:
        status = "ambiguous"
    if status == "ambiguous" or not markers:
        return {"status": status, "center": None}
    marker = markers[0]
    if frame.dataset != "ipgs":
        correct_id = marker["id"] == key
        identity = "archive_agreement" if cohort == "archived_projective" else "sharp_decode_agreement"
    elif sharp_frame == run["board_frame"] == "absolute" and anchor[1] is not None:
        correct_id = marker["coord"] == anchor[1]
        identity = "absolute_triad_and_position"
    else:
        correct_id = True
        identity = "within_image_position_only"
    center = marker["moment_center"] if method == "moment" else marker["center"]
    if not correct_id:
        status = "wrong_id" if marker["id"] is not None or frame.dataset == "ipgs" else "unverified"
    elif not valid_center(center):
        status = "fit_failed"
    elif method in ("nominal", "per_image", "pooled") and marker["source"] != "photometric":
        status = "fit_failed"
    elif method == "projective" and marker["source"] != "projective_center":
        status = "fit_failed"
    else:
        status = "ok"
    diameter, contrast = marker["diameter_px"], marker["contrast"]
    return {
        "center": center if status == "ok" else None, "status": status, "identity": identity,
        "source": marker["source"] if method != "moment" else "moment", "fit_status": marker["status"],
        "cov": marker["cov"] if status == "ok" and method != "moment" else None,
        "radii_mm": marker["radii_mm"], "blur_sigma_px": marker["blur_sigma_px"],
        "contrast": contrast, "diameter_px": diameter,
    }


def classify(value, lo, hi, names):
    return "unknown" if value is None else names[int(value >= lo) + int(value >= hi)]


def evaluate_frame(args, entry, pools, fits):
    frame, split = entry
    methods = METHODS if frame.dataset in pools else METHODS[:-1]
    if frame.dataset == "rtv3d":
        conditions = [("native", 0.0)]
    else:
        conditions = [("sharp", 0.0), *[(f"blur_{s:g}", s) for s in args.blur_sigma]]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        image = load_frame(frame)
        image_path = Path(tmp) / "sample.png"
        for condition, sigma in conditions:
            (centered_gaussian(image, sigma) if sigma else image).save(image_path)
            runs = {
                method: (
                    fits[frame] if sigma == 0 and method == "per_image" and frame in fits
                    else run_frame(args, frame, image_path, method,
                                   pools[frame.dataset] if method == "pooled" else None)
                )
                for method in methods
            }
            results[condition] = runs
    sharp_condition = conditions[0][0]
    sharp = results[sharp_condition]["projective"]
    anchors, cohort = make_anchors(args, frame, sharp)
    data = {}
    full_output = []
    for condition, runs in results.items():
        for method in (*methods, "moment"):
            run = runs["per_image"] if method == "moment" else runs[method]
            matched, unmatched, ambiguous = match_markers(anchors, run)
            detected = run["markers"]
            full_output.append({
                "condition": condition, "method": method,
                "n_detected": len(detected),
                "n_unlabelled": sum(m["id"] is None and m["coord"] is None for m in detected),
                "n_unmatched": unmatched, "n_ambiguous": ambiguous,
                "source_counts": dict(Counter(m["source"] for m in detected)),
                "fit_status_counts": dict(Counter(m["status"] or "not_run" for m in detected)),
                "n_moment_centers": sum(m["moment_center"] is not None for m in detected)
                if method == "moment" else None,
                "runtime_ms": run["runtime_ms"] if method != "moment" else None,
            })
            for key, anchor in anchors.items():
                data[condition, method, key] = observation(
                    frame, key, anchor, matched[key], run, method, cohort, sharp["board_frame"]
                ) | {"runtime_ms": run["runtime_ms"] if method != "moment" else None}
    for (condition, method, key), row in data.items():
        reference = data[sharp_condition, method, key]["center"] if condition != sharp_condition else None
        sharp_pi = data[sharp_condition, "per_image", key]
        sharp_projective = data[sharp_condition, "projective", key]
        diameter = sharp_pi.get("diameter_px") or sharp_projective.get("diameter_px")
        contrast = sharp_pi.get("contrast")
        y = anchors[key][0][1]
        band = ("top", "middle", "bottom")[min(2, max(0, int(3 * y / image.height)))]
        row.update(
            dataset=frame.dataset, camera=frame.camera, capture=frame.capture, split=split,
            condition=condition, method=method, key=str(key), reference=reference,
            focus_band=band if frame.dataset == "rtv3d" else "unknown",
            scale_bin=classify(diameter, 24, 48, ("<24", "24–48", ">=48")),
            contrast_bin=classify(contrast, 25, 75, ("<25", "25–75", ">=75")),
            cohort=cohort, sharp_board_frame=sharp["board_frame"], board_xy_mm=anchors[key][2],
        )
    print(f"{frame.dataset} {frame.capture} camera {frame.camera} {split}: {len(anchors)} anchors",
          flush=True)
    return list(data.values()), {
        "dataset": frame.dataset, "capture": frame.capture, "camera": frame.camera,
        "split": split, "anchors": len(anchors), "cohort": cohort,
        "sharp_board_frame": sharp["board_frame"], "full_output": full_output,
    }


def verify_ipgs_origin(args, spec, selection, executor):
    """Check the grid frame using pixels near the three origin dots."""
    from run_ipgs_eval import anchor_by_dots
    from score_coverage import Lattice, fit_models

    lattice = Lattice(spec)

    def check(entry):
        frame, split = entry
        image = load_frame(frame)
        projective = run_frame(args, frame, frame.source, "projective")
        board_frame = projective["board_frame"]
        det = {
            "detected_markers": [
                {"center": m["center"], "grid_coord": m["coord"]}
                for m in projective["markers"]
            ],
        }
        row = {
            "capture": frame.capture, "camera": frame.camera, "split": split,
            "board_frame": board_frame, "confirmed_absolute": False,
        }
        models, reason = fit_models(det, lattice, image.size, ransac_px=4.0, min_fit=8)
        if models is None:
            return row | {"reason": reason}
        rotation, translation, evidence = anchor_by_dots(
            np.asarray(image, dtype=np.float32), models["hk"], lattice, spec
        )
        return row | {
            "rotation": rotation, "translation": translation, "intensity_check": evidence,
            "confirmed_absolute": (
                board_frame == "absolute"
                and rotation is not None
                and np.array_equal(rotation, np.eye(2))
                and tuple(translation) == (0, 0)
            ),
        }

    checks = list(executor.map(check, selection))
    write_json(args.out / "ipgs-triad-check.json", checks)
    n_confirmed = sum(row["confirmed_absolute"] for row in checks)
    print(f"IPGS dark-dot check confirms the absolute frame in {n_confirmed}/{len(checks)} views")



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipgs", type=Path, default=IPGS_DATA)
    parser.add_argument("--rtvt", type=Path, default=IPGS_DATA.parent / "datasets",
                        help="dataset.json or directory of dataset directories")
    parser.add_argument("--rtv3d", type=Path, default=REPO / "data/rtv3d_0")
    parser.add_argument("--rtv3d-target", type=Path, default=REPO / "tools/board/board_spec.json")
    parser.add_argument("--bin", type=Path, default=REPO / "target/release/examples/metrology_frame")
    parser.add_argument("--out", type=Path, default=REPO / "tools/out/center_metrology/real")
    parser.add_argument("--datasets", nargs="+", choices=["ipgs", "rtv.t", "rtv3d"],
                        default=["ipgs", "rtv.t", "rtv3d"])
    parser.add_argument("--captures-per-split", type=int, default=3,
                        help="evenly selected per dataset and split (0 for all)")
    parser.add_argument("--blur-sigma", nargs="+", type=float, default=[1.5])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--verify-ipgs-origin", action="store_true",
                        help="run the separate native IPGS origin-dot intensity check")
    args = parser.parse_args()
    if not args.bin.is_file():
        parser.error(f"missing study runner {args.bin}. Build the example before running.")
    if (args.jobs < 1 or args.captures_per_split < 0
            or any(not np.isfinite(sigma) or sigma <= 0 for sigma in args.blur_sigma)
            or len({f"{sigma:g}" for sigma in args.blur_sigma}) != len(args.blur_sigma)):
        parser.error("need positive jobs, unique positive blur sigmas and nonnegative capture limit")
    return args


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.verify_ipgs_origin:
        args.datasets = ["ipgs"]
        targets, frames = prepare_datasets(args)
        write_json(args.out / "ipgs-target.json", targets["ipgs"])
        with ThreadPoolExecutor(args.jobs) as executor:
            verify_ipgs_origin(args, targets["ipgs"],
                               select_frames(frames["ipgs"], args.captures_per_split), executor)
        return
    targets, frames = prepare_datasets(args)
    selections = []
    for dataset in args.datasets:
        if not frames[dataset]:
            raise ValueError(f"no frames found for {dataset}")
        write_json(args.out / f"{dataset}-target.json", targets[dataset])
        selections.extend(select_frames(frames[dataset], args.captures_per_split))
    with ThreadPoolExecutor(args.jobs) as executor:
        pools, training, fits = fit_pools(args, selections, executor)
        write_json(args.out / "pools.json", {
            "weighting": "coordinatewise median across capture medians; training native only",
            "radii_mm": pools, "training_radii_mm": training,
        })
        print(f"training-only pools: {pools}", flush=True)
        counts = Counter()
        evaluated = []
        tmp = args.out / "observations.jsonl.partial"
        with tmp.open("w") as out:
            for rows, frame_info in executor.map(
                lambda e: evaluate_frame(args, e, pools, fits), selections
            ):
                evaluated.append(frame_info)
                for row in rows:
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    counts[row["dataset"], row["split"], row["condition"], row["method"], row["status"]] += 1
        tmp.replace(args.out / "observations.jsonl")
        write_json(args.out / "manifest.json", {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "runner_sha256": hashlib.sha256(args.bin.read_bytes()).hexdigest(),
            "targets": {d: targets[d]["name"] for d in args.datasets},
            "rtv_manifest": args.rtv_manifest,
            "blur_sigmas_px": args.blur_sigma, "jobs": args.jobs,
            "screening": {
                "position_gate_px": 5.0,
                "ipgs": "within-image position; also check coordinate when both frames are triad-anchored",
                "rtv.t": "prior detector archive when available; otherwise native projective sharp decode",
                "rtv3d": "native projective decode and position agreement; absolute identity unverified",
            },
            "frames": evaluated,
            "counts": [{"dataset": d, "split": s, "condition": c, "method": m, "status": q, "n": n}
                       for (d, s, c, m, q), n in sorted(counts.items())],
        })


if __name__ == "__main__":
    main()
