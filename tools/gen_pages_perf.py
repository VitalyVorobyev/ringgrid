#!/usr/bin/env python3
"""Generate the documentation performance-page data (``data.json``).

This is the *performance test* behind the GitHub Pages performance dashboard
(``.github/pages/performance/index.html``). It drives two reusable measurement
paths and assembles them into the schema the page reads:

  * **Timing + counts** — the ``ringgrid bench`` subcommand: single-pass
    detection with warmup + repeats, median per stage (proposal / fit+decode /
    finalize) and end-to-end wall-clock, plus proposal / decoded marker counts.
  * **Accuracy** — the synthetic reference benchmark
    (``tools/run_reference_benchmark.sh``): precision, recall and subpixel
    centre error scored against generated ground truth, for both the
    ``none`` and ``projective_center`` refinement modes.

The result is a stable, committed snapshot (``--out``). CI only deploys it; it
is regenerated locally on a quiet machine when the algorithm changes:

    .venv/bin/python tools/gen_pages_perf.py

Run from the repository root.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / ".github/pages/performance/data.json"
DEFAULT_IMG_DIR = REPO_ROOT / ".github/pages/performance/img"
RELEASE_BIN = REPO_ROOT / "target/release/ringgrid"
REFERENCE_OUT_DIR = REPO_ROOT / "tools/out/reference_benchmark_post_pipeline"


# ── shell helpers ────────────────────────────────────────────────────────────
def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kwargs)


def capture(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_sha() -> str:
    return capture(["git", "rev-parse", "--short", "HEAD"]) or "unknown"


def rustc_version() -> str:
    raw = capture(["rustc", "--version"])  # "rustc 1.88.0 (abc 2025-...)"
    return raw.split()[1] if raw else "unknown"


def cpu_name() -> str:
    if sys.platform == "darwin":
        name = capture(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown CPU"


def today() -> str:
    return datetime.date.today().isoformat()


# ── timing (ringgrid bench) ──────────────────────────────────────────────────
def bench_group(binary: Path, images: list[str], repeats: int,
                marker_diameter: float | None) -> dict[str, dict]:
    """Run ``ringgrid bench`` on a group of images sharing one scale prior."""
    out_path = REPO_ROOT / "tools/out/_pages_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "bench", "--out", str(out_path), "--repeats", str(repeats)]
    for img in images:
        cmd += ["--image", img]
    if marker_diameter is not None:
        cmd += ["--marker-diameter", str(marker_diameter)]
    run(cmd, env={"RUST_LOG": "warn", **_env()})
    report = json.loads(out_path.read_text())
    return {entry["file"]: entry for entry in report["images"]}


def _env() -> dict:
    import os
    return dict(os.environ)


# ── accuracy (reference benchmark) ───────────────────────────────────────────
def run_reference_benchmark(reuse: bool) -> dict | None:
    summary_path = REFERENCE_OUT_DIR / "summary.json"
    if not reuse:
        script = REPO_ROOT / "tools/run_reference_benchmark.sh"
        try:
            run(["bash", str(script)])
        except subprocess.CalledProcessError as exc:
            print(f"  ! reference benchmark failed ({exc}); accuracy omitted")
            return None
    if not summary_path.exists():
        print(f"  ! {summary_path} not found; accuracy omitted")
        return None
    return json.loads(summary_path.read_text())


def accuracy_sets(summary: dict) -> list[dict]:
    """Extract precision / recall / centre-error rows for the page."""
    modes = summary.get("modes", {})
    labels = [
        ("none__none", "No refinement"),
        ("projective_center__none", "Projective centre"),
    ]
    sets = []
    for key, label in labels:
        m = modes.get(key)
        if not m:
            continue
        sets.append({
            "label": label,
            "precision": m.get("avg_precision"),
            "recall": m.get("avg_recall"),
            "center_mean_px": m.get("avg_center_mean_px"),
            "h_vs_gt_mean_px": m.get("avg_h_vs_gt_mean_px"),
            "h_vs_gt_p95_px": m.get("avg_h_vs_gt_p95_px"),
        })
    return sets


# ── assembly ─────────────────────────────────────────────────────────────────
def bench_items() -> list[dict]:
    """The image set shown on the page (real capture + synthetic boards)."""
    synth = REFERENCE_OUT_DIR / "synth"
    items = [{
        "path": "testdata/target_3_split_00.png",
        "display": "testdata/target_3_split_00.png",
        "marker_diameter": None,
        "label": "Coded ring board — real capture",
        "kind": "ringgrid",
        "note": "Dense hex lattice, decoded to absolute IDs with homography.",
        "preview": "target_3_split_00.png",
    }]
    for i in range(2):
        p = synth / f"img_{i:04d}.png"
        if p.exists():
            items.append({
                "path": str(p.relative_to(REPO_ROOT)),
                "display": f"synth/img_{i:04d}.png",
                "marker_diameter": 32.0,
                "label": f"Synthetic board #{i + 1}",
                "kind": "synthetic",
                "note": "Generated target, <1 px blur — also scored for accuracy.",
                "preview": f"synth_{i:04d}.png",
            })
    return items


def copy_preview(src_rel: str, preview_name: str, img_dir: Path) -> bool:
    src = REPO_ROOT / src_rel
    if not src.exists():
        return False
    img_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, img_dir / preview_name)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--img-out-dir", type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument("--repeats", type=int, default=15,
                    help="Timed repeats per image (median reported).")
    ap.add_argument("--reuse-summary", action="store_true",
                    help="Reuse an existing reference-benchmark summary.json "
                         "instead of re-running the synthetic accuracy benchmark.")
    ap.add_argument("--skip-accuracy", action="store_true",
                    help="Skip the accuracy panel entirely (timing only).")
    args = ap.parse_args()

    print("[1/4] Building release CLI")
    run(["cargo", "build", "--release", "-p", "ringgrid-cli"])
    if not RELEASE_BIN.exists():
        print(f"error: {RELEASE_BIN} not found after build", file=sys.stderr)
        return 1

    print("[2/4] Accuracy — synthetic reference benchmark")
    summary = None if args.skip_accuracy else run_reference_benchmark(args.reuse_summary)

    print("[3/4] Timing — ringgrid bench")
    items = bench_items()
    measured: dict[str, dict] = {}
    # Group by scale prior so each bench call uses a single --marker-diameter.
    groups: dict[float | None, list[str]] = {}
    for it in items:
        groups.setdefault(it["marker_diameter"], []).append(it["path"])
    for diameter, paths in groups.items():
        measured.update(bench_group(RELEASE_BIN, paths, args.repeats, diameter))

    print("[4/4] Assembling data.json")
    images, frames = [], []
    for it in items:
        m = measured.get(it["path"])
        if not m:
            print(f"  ! no measurement for {it['path']}; skipping")
            continue
        has_preview = copy_preview(it["path"], it["preview"], args.img_out_dir)
        display = it.get("display", it["path"])
        entry = {
            "label": it["label"],
            "kind": it["kind"],
            "file": display,
            "width": m["width"],
            "height": m["height"],
            "raw_corners": m["raw_corners"],
            "labelled": m["labelled"],
            "markers": m["markers"],
            "proposal_ms": m["proposal_ms"],
            "fit_decode_ms": m["fit_decode_ms"],
            "finalize_ms": m["finalize_ms"],
        }
        if has_preview:
            entry["img"] = f"./img/{it['preview']}"
        if it.get("note"):
            entry["note"] = it["note"]
        images.append(entry)
        frames.append({"file": display, "note": it.get("note", ""),
                       "ms": m["total_ms"]})

    data = {
        "meta": {
            "cpu": cpu_name(),
            "rustc": rustc_version(),
            "git_sha": git_sha(),
            "generated": today(),
            "repeats": args.repeats,
            "source": "tools/gen_pages_perf.py",
        },
        "images": images,
        "end_to_end": {
            "desc": "End-to-end single-pass latency (median of "
                    f"{args.repeats} repeats) on public sample images.",
            "frames": frames,
        },
    }

    if summary is not None:
        sets = accuracy_sets(summary)
        if sets:
            data["accuracy"] = {
                "desc": "Synthetic reference set (3 boards, <1 px blur) scored "
                        "against generated ground truth. Centre error is the "
                        "mean per-marker distance to the true centre; lower is "
                        "sharper. Projective-centre refinement removes the "
                        "perspective bias of the raw ellipse centre.",
                "sets": sets,
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)} "
          f"({len(images)} images, accuracy={'yes' if 'accuracy' in data else 'no'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
