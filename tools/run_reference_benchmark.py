#!/usr/bin/env python3
"""Reference benchmark for clean synthetic samples.

Pipeline:
  1) Generate a 3-image synthetic set with low blur (<1 px) and no noise.
  2) Run detection in four refinement variants.
  3) Score each output and write a compact summary.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ALL_MODES = {
    "none": ["--circle-refine-method", "none"],
    "projective_center": ["--circle-refine-method", "projective-center"],
    "nl_board_lm": ["--circle-refine-method", "nl-board", "--nl-solver", "lm"],
    "nl_board_irls": ["--circle-refine-method", "nl-board", "--nl-solver", "irls"],
}

ALL_CORRECTIONS = ("none", "external", "self_undistort")


def find_ringgrid_binary() -> str | None:
    for candidate in ("target/release/ringgrid", "target/debug/ringgrid"):
        if os.path.isfile(candidate):
            return candidate
    return None


def binary_supports_camera_cli(binary: str) -> bool:
    try:
        result = subprocess.run(
            [binary, "detect", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    return "--cam-fx" in text


def binary_supports_self_undistort_cli(binary: str) -> bool:
    try:
        result = subprocess.run(
            [binary, "detect", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    return "--self-undistort" in text


def camera_cli_args_from_gt(gt_data: dict) -> list[str]:
    camera = gt_data.get("camera")
    if not isinstance(camera, dict):
        return []
    intr = camera.get("intrinsics")
    dist = camera.get("distortion")
    if not isinstance(intr, dict) or not isinstance(dist, dict):
        return []
    required_intr = ("fx", "fy", "cx", "cy")
    required_dist = ("k1", "k2", "p1", "p2", "k3")
    if any(k not in intr for k in required_intr) or any(k not in dist for k in required_dist):
        return []
    return [
        f"--cam-fx={intr['fx']}",
        f"--cam-fy={intr['fy']}",
        f"--cam-cx={intr['cx']}",
        f"--cam-cy={intr['cy']}",
        f"--cam-k1={dist['k1']}",
        f"--cam-k2={dist['k2']}",
        f"--cam-p1={dist['p1']}",
        f"--cam-p2={dist['p2']}",
        f"--cam-k3={dist['k3']}",
    ]


def run_checked(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}"
        )
    return res.stdout


def score_to_row(score: dict) -> dict:
    ce = score.get("center_error") or {}
    hs = score.get("homography_self_error") or {}
    hg = score.get("homography_error_vs_gt") or {}
    return {
        "n_tp": score.get("n_tp"),
        "n_fp": score.get("n_fp"),
        "n_miss": score.get("n_miss"),
        "precision": score.get("precision"),
        "recall": score.get("recall"),
        "center_mean_px": ce.get("mean"),
        "h_self_mean_px": hs.get("mean"),
        "h_self_p95_px": hs.get("p95"),
        "h_vs_gt_mean_px": hg.get("mean"),
        "h_vs_gt_p95_px": hg.get("p95"),
    }


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def resolve_pred_frame_for_detection(correction: str, det_data: dict) -> str:
    if correction == "external":
        return "working"
    if correction == "self_undistort":
        su = det_data.get("self_undistort")
        if isinstance(su, dict) and bool(su.get("applied", False)):
            return "working"
        return "image"
    return "image"


def parse_correction_names(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.corrections is None:
        return ["external"] if args.pass_camera_to_detector else ["none"]
    if args.pass_camera_to_detector:
        parser.error("--pass-camera-to-detector is deprecated; use --corrections external")
    return list(ALL_CORRECTIONS) if "all" in args.corrections else args.corrections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean reference benchmark")
    parser.add_argument("--out_dir", type=str, default="tools/out/reference_benchmark")
    parser.add_argument("--n_images", type=int, default=3)
    parser.add_argument("--blur_px", type=float, default=0.8)
    parser.add_argument("--marker_diameter", type=float, default=32.0)
    parser.add_argument("--gate", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tilt_strength", type=float, default=0.3)
    parser.add_argument("--illum_strength", type=float, default=0.15)
    parser.add_argument("--noise_sigma", type=float, default=0.0)
    parser.add_argument("--cam-fx", type=float, default=None)
    parser.add_argument("--cam-fy", type=float, default=None)
    parser.add_argument("--cam-cx", type=float, default=None)
    parser.add_argument("--cam-cy", type=float, default=None)
    parser.add_argument("--cam-k1", type=float, default=0.0)
    parser.add_argument("--cam-k2", type=float, default=0.0)
    parser.add_argument("--cam-p1", type=float, default=0.0)
    parser.add_argument("--cam-p2", type=float, default=0.0)
    parser.add_argument("--cam-k3", type=float, default=0.0)
    parser.add_argument(
        "--pass-camera-to-detector",
        action="store_true",
        help="Deprecated alias for --corrections external.",
    )
    parser.add_argument(
        "--corrections",
        nargs="+",
        default=None,
        choices=["all", *ALL_CORRECTIONS],
        help=(
            "Distortion correction variants to benchmark. "
            "Use 'none', 'external', 'self_undistort', or 'all'. "
            "Default: 'none' unless --pass-camera-to-detector is set."
        ),
    )
    parser.add_argument("--self-undistort-lambda-min", type=float, default=-8e-7)
    parser.add_argument("--self-undistort-lambda-max", type=float, default=8e-7)
    parser.add_argument("--self-undistort-min-markers", type=int, default=6)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["all"],
        choices=["all", *ALL_MODES.keys()],
        help=(
            "Detection refinement modes to benchmark. "
            "Use 'all' (default) or one/more explicit mode names."
        ),
    )
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument(
        "--use-prebuilt-binary",
        action="store_true",
        help=(
            "Use target/(release|debug)/ringgrid if present. "
            "Default uses `cargo run` to avoid stale-binary drift."
        ),
    )
    args = parser.parse_args()

    intr_values = [args.cam_fx, args.cam_fy, args.cam_cx, args.cam_cy]
    has_any_intr = any(v is not None for v in intr_values)
    has_all_intr = all(v is not None for v in intr_values)
    has_dist_coeff = any(
        abs(float(v)) > 1e-15 for v in (args.cam_k1, args.cam_k2, args.cam_p1, args.cam_p2, args.cam_k3)
    )
    if has_any_intr and not has_all_intr:
        parser.error("camera intrinsics are partial; provide all of --cam-fx --cam-fy --cam-cx --cam-cy")
    if not has_any_intr and has_dist_coeff:
        parser.error("non-zero distortion requires camera intrinsics")

    mode_names = list(ALL_MODES.keys()) if "all" in args.modes else args.modes
    correction_names = parse_correction_names(args, parser)
    if "external" in correction_names and not has_all_intr:
        parser.error("correction 'external' requires camera intrinsics in generated GT")

    out_dir = Path(args.out_dir)
    synth_dir = out_dir / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_gen:
        print(
            f"[1/3] Generate synth set n={args.n_images}, blur={args.blur_px}, "
            f"noise_sigma={args.noise_sigma}"
        )
        run_checked(
            [
                sys.executable,
                "tools/gen_synth.py",
                "--out_dir",
                str(synth_dir),
                "--n_images",
                str(args.n_images),
                "--seed",
                str(args.seed),
                "--blur_px",
                str(args.blur_px),
                "--tilt_strength",
                str(args.tilt_strength),
                "--illum_strength",
                str(args.illum_strength),
                "--noise_sigma",
                str(args.noise_sigma),
                *(
                    [
                        "--cam-fx",
                        str(args.cam_fx),
                        "--cam-fy",
                        str(args.cam_fy),
                        "--cam-cx",
                        str(args.cam_cx),
                        "--cam-cy",
                        str(args.cam_cy),
                        "--cam-k1",
                        str(args.cam_k1),
                        "--cam-k2",
                        str(args.cam_k2),
                        "--cam-p1",
                        str(args.cam_p1),
                        "--cam-p2",
                        str(args.cam_p2),
                        "--cam-k3",
                        str(args.cam_k3),
                    ]
                    if has_all_intr
                    else []
                ),
            ]
        )
    else:
        print(f"[1/3] Reuse existing synth set at {synth_dir}")

    ringgrid_bin = None
    use_cargo_run = True
    if args.use_prebuilt_binary:
        ringgrid_bin = find_ringgrid_binary()
        use_cargo_run = ringgrid_bin is None
        if (
            not use_cargo_run
            and "external" in correction_names
            and not binary_supports_camera_cli(ringgrid_bin)
        ):
            print(
                "  NOTE: selected ringgrid binary does not support camera CLI flags; "
                "falling back to cargo run"
            )
            use_cargo_run = True
        if (
            not use_cargo_run
            and "self_undistort" in correction_names
            and not binary_supports_self_undistort_cli(ringgrid_bin)
        ):
            print(
                "  NOTE: selected ringgrid binary does not support self-undistort CLI flags; "
                "falling back to cargo run"
            )
            use_cargo_run = True
    if use_cargo_run:
        print("  Runner: cargo run")
    else:
        print(f"  Runner: prebuilt binary ({ringgrid_bin})")

    print("[2/3] Run detection variants")
    summary: dict[str, dict] = {
        "config": {
            "n_images": args.n_images,
            "blur_px": args.blur_px,
            "noise_sigma": args.noise_sigma,
            "illum_strength": args.illum_strength,
            "marker_diameter": args.marker_diameter,
            "gate": args.gate,
            "seed": args.seed,
            "tilt_strength": args.tilt_strength,
            "metric_frame": "image",
            "pass_camera_to_detector": bool(args.pass_camera_to_detector),
            "modes": mode_names,
            "corrections": correction_names,
            "camera": (
                {
                    "fx": args.cam_fx,
                    "fy": args.cam_fy,
                    "cx": args.cam_cx,
                    "cy": args.cam_cy,
                    "k1": args.cam_k1,
                    "k2": args.cam_k2,
                    "p1": args.cam_p1,
                    "p2": args.cam_p2,
                    "k3": args.cam_k3,
                }
                if has_all_intr
                else None
            ),
            "runner": "cargo_run" if use_cargo_run else f"binary:{ringgrid_bin}",
        },
        "modes": {},
    }

    for correction in correction_names:
        for mode in mode_names:
            mode_args = ALL_MODES[mode]
            run_name = f"{mode}__{correction}"
            mode_dir = out_dir / correction / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            scores: list[dict] = []
            for idx in range(args.n_images):
                img_path = synth_dir / f"img_{idx:04d}.png"
                gt_path = synth_dir / f"gt_{idx:04d}.json"
                det_path = mode_dir / f"det_{idx:04d}.json"
                score_path = mode_dir / f"score_{idx:04d}.json"

                if use_cargo_run:
                    detect_cmd = [
                        "cargo",
                        "run",
                        "--quiet",
                        "--",
                        "detect",
                        "--image",
                        str(img_path),
                        "--out",
                        str(det_path),
                        "--marker-diameter",
                        str(args.marker_diameter),
                        *mode_args,
                    ]
                else:
                    detect_cmd = [
                        ringgrid_bin,
                        "detect",
                        "--image",
                        str(img_path),
                        "--out",
                        str(det_path),
                        "--marker-diameter",
                        str(args.marker_diameter),
                        *mode_args,
                    ]
                if correction == "external":
                    with open(gt_path) as f:
                        gt_data = json.load(f)
                    detect_cmd.extend(camera_cli_args_from_gt(gt_data))
                elif correction == "self_undistort":
                    detect_cmd.extend(
                        [
                            "--self-undistort",
                            f"--self-undistort-lambda-min={args.self_undistort_lambda_min}",
                            f"--self-undistort-lambda-max={args.self_undistort_lambda_max}",
                            f"--self-undistort-min-markers={args.self_undistort_min_markers}",
                        ]
                    )
                run_checked(detect_cmd)
                with open(det_path) as f:
                    det_data = json.load(f)
                pred_frame = resolve_pred_frame_for_detection(correction, det_data)
                run_checked(
                    [
                        sys.executable,
                        "tools/score_detect.py",
                        "--gt",
                        str(gt_path),
                        "--pred",
                        str(det_path),
                        "--gate",
                        str(args.gate),
                        "--center-gt-key",
                        "image",
                        "--homography-gt-key",
                        "image",
                        "--pred-center-frame",
                        pred_frame,
                        "--pred-homography-frame",
                        pred_frame,
                        "--out",
                        str(score_path),
                    ]
                )
                with open(score_path) as f:
                    scores.append(json.load(f))

            rows = [score_to_row(s) for s in scores]
            summary["modes"][run_name] = {
                "mode": mode,
                "correction": correction,
                "per_image": rows,
                "avg_precision": mean_of(rows, "precision"),
                "avg_recall": mean_of(rows, "recall"),
                "avg_center_mean_px": mean_of(rows, "center_mean_px"),
                "avg_h_self_mean_px": mean_of(rows, "h_self_mean_px"),
                "avg_h_self_p95_px": mean_of(rows, "h_self_p95_px"),
                "avg_h_vs_gt_mean_px": mean_of(rows, "h_vs_gt_mean_px"),
                "avg_h_vs_gt_p95_px": mean_of(rows, "h_vs_gt_p95_px"),
                "avg_tp": mean_of(rows, "n_tp"),
                "avg_fp": mean_of(rows, "n_fp"),
                "avg_miss": mean_of(rows, "n_miss"),
            }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[3/3] Summary")
    print("correction | mode | precision | recall | center_mean | H_self_mean/p95 | H_vs_GT_mean/p95")
    for _, vals in summary["modes"].items():
        print(
            f"{vals['correction']} | {vals['mode']} | "
            f"{vals['avg_precision']:.3f} | {vals['avg_recall']:.3f} | "
            f"{vals['avg_center_mean_px']:.3f} | "
            f"{vals['avg_h_self_mean_px']:.3f}/{vals['avg_h_self_p95_px']:.3f} | "
            f"{vals['avg_h_vs_gt_mean_px']:.3f}/{vals['avg_h_vs_gt_p95_px']:.3f}"
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
