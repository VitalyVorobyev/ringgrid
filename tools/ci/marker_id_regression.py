#!/usr/bin/env python3
"""Regression check: marker ID correctness on target_3_split_00..05.

Runs the detector on the six split images with the standard pass-1 config and
compares produced marker IDs against committed expected outputs in `data/`.
Marker matching uses nearest-center pairing to remain stable if ordering changes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CONFIG = ROOT / "crates" / "ringgrid-cli" / "config_sample.json"
SPLITS = [f"{i:02d}" for i in range(6)]
MATCH_GATE_PX = 1.0


@dataclass(frozen=True)
class Marker:
    center: tuple[float, float]
    marker_id: int | None


def load_markers(path: Path) -> list[Marker]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[Marker] = []
    for m in payload["detected_markers"]:
        cx, cy = m["center"]
        out.append(Marker(center=(float(cx), float(cy)), marker_id=m.get("id")))
    return out


def run_detect(image_path: Path, out_path: Path) -> None:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "ringgrid-cli",
        "--",
        "detect",
        "--image",
        str(image_path),
        "--out",
        str(out_path),
        "--circle-refine-method",
        "projective-center",
        "--complete-require-perfect-decode",
        "--no-global-filter",
        "--include-proposals",
        "--config",
        str(CONFIG),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"detect failed for {image_path.name}:\n{proc.stdout}")


def pair_markers(expected: list[Marker], actual: list[Marker]) -> list[tuple[Marker, Marker, float]]:
    if len(expected) != len(actual):
        raise RuntimeError(
            f"marker count mismatch: expected {len(expected)}, got {len(actual)}"
        )
    used: set[int] = set()
    pairs: list[tuple[Marker, Marker, float]] = []
    for e in expected:
        best_idx = -1
        best_dist = float("inf")
        for j, a in enumerate(actual):
            if j in used:
                continue
            d = math.hypot(e.center[0] - a.center[0], e.center[1] - a.center[1])
            if d < best_dist:
                best_dist = d
                best_idx = j
        if best_idx < 0 or best_dist > MATCH_GATE_PX:
            raise RuntimeError(
                "failed center pairing within gate "
                f"({MATCH_GATE_PX}px) for expected center={e.center}"
            )
        used.add(best_idx)
        pairs.append((e, actual[best_idx], best_dist))
    return pairs


def compare_split(split: str, workdir: Path) -> list[str]:
    image = DATA / f"target_3_split_{split}.png"
    expected = DATA / f"target_3_split_{split}_det.json"
    actual = workdir / f"target_3_split_{split}_det_actual.json"
    run_detect(image, actual)

    expected_markers = load_markers(expected)
    actual_markers = load_markers(actual)
    pairs = pair_markers(expected_markers, actual_markers)

    errors: list[str] = []
    for e, a, dist in pairs:
        if e.marker_id != a.marker_id:
            errors.append(
                f"split {split}: center={e.center} (pair_dist={dist:.3f}px) "
                f"expected id={e.marker_id}, got id={a.marker_id}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary detection outputs for debugging.",
    )
    args = parser.parse_args()

    if not CONFIG.exists():
        print(f"ERROR: config not found: {CONFIG}")
        return 1

    if args.keep_temp:
        workdir = Path(tempfile.mkdtemp(prefix="ringgrid_marker_reg_"))
        all_errors: list[str] = []
        for split in SPLITS:
            all_errors.extend(compare_split(split, workdir))
        if all_errors:
            print("ERROR: marker ID regression detected:")
            for err in all_errors:
                print(f"  - {err}")
            print(f"Temporary outputs kept in: {workdir}")
            return 1
        print("OK: marker ID regression passed for target_3_split_00..05.")
        print(f"Temporary outputs kept in: {workdir}")
        return 0

    with tempfile.TemporaryDirectory(prefix="ringgrid_marker_reg_") as td:
        workdir = Path(td)
        all_errors: list[str] = []
        for split in SPLITS:
            all_errors.extend(compare_split(split, workdir))
        if all_errors:
            print("ERROR: marker ID regression detected:")
            for err in all_errors:
                print(f"  - {err}")
            return 1
        print("OK: marker ID regression passed for target_3_split_00..05.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
