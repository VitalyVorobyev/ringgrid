#!/usr/bin/env python3
"""Maintainability guardrails for ringgrid CI.

Checks:
1) No new `allow(dead_code)` in configured hot modules.
2) No new `allow(clippy::too_many_arguments)` in configured hot modules.
3) No new oversized functions in hot modules (baseline-locked).
4) Rustdoc missing-docs warning count does not increase.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "tools" / "ci" / "maintainability_baseline.json"

FUNCTION_START_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
ALLOW_DEAD_CODE_RE = re.compile(r"allow\s*\(\s*dead_code\s*\)")
ALLOW_TOO_MANY_ARGS_RE = re.compile(r"allow\s*\(\s*clippy::too_many_arguments\s*\)")
MISSING_DOCS_WARNING_RE = re.compile(r"warning: missing documentation")
STRING_LITERAL_RE = re.compile(r'"(?:\\.|[^"\\])*"')
CHAR_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])'")


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    start_line: int
    end_line: int
    lines: int


def _normalize_ws(text: str) -> str:
    return " ".join(text.strip().split())


def _strip_strings_and_line_comments(line: str) -> str:
    cleaned = STRING_LITERAL_RE.sub('""', line)
    cleaned = CHAR_LITERAL_RE.sub("''", cleaned)
    comment_idx = cleaned.find("//")
    if comment_idx != -1:
        cleaned = cleaned[:comment_idx]
    return cleaned


def _brace_delta(line: str) -> int:
    cleaned = _strip_strings_and_line_comments(line)
    return cleaned.count("{") - cleaned.count("}")


def _iter_hot_rust_files(prefixes: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for rel_prefix in prefixes:
        prefix_dir = ROOT / rel_prefix
        if not prefix_dir.exists():
            continue
        for path in sorted(prefix_dir.rglob("*.rs")):
            rel = path.relative_to(ROOT).as_posix()
            yield rel, path


def _read_production_lines(path: Path) -> list[str]:
    """Read file and drop unit-test module tail (`#[cfg(test)]`)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "#[cfg(test)]":
            return lines[:idx]
    return lines


def _extract_function_spans(lines: list[str]) -> list[FunctionSpan]:
    out: list[FunctionSpan] = []
    i = 0
    n = len(lines)
    while i < n:
        match = FUNCTION_START_RE.match(lines[i])
        if not match:
            i += 1
            continue

        name = match.group(1)
        sig_end = i
        has_body = False
        while sig_end < n:
            line = _strip_strings_and_line_comments(lines[sig_end])
            open_idx = line.find("{")
            semi_idx = line.find(";")
            # Treat as declaration-without-body only when the signature line
            # itself terminates with `;` (avoid false positives like `[T; N]`).
            if (
                semi_idx != -1
                and line.rstrip().endswith(";")
                and (open_idx == -1 or semi_idx < open_idx)
            ):
                # Trait-style declaration (no body).
                break
            if open_idx != -1:
                has_body = True
                break
            sig_end += 1

        if not has_body:
            i += 1
            continue

        depth = 0
        end = sig_end
        k = i
        while k < n:
            depth += _brace_delta(lines[k])
            if depth == 0 and k >= sig_end:
                end = k
                break
            k += 1

        out.append(
            FunctionSpan(
                name=name,
                start_line=i + 1,
                end_line=end + 1,
                lines=(end - i + 1),
            )
        )
        i = end + 1

    return out


def collect_oversized_functions(
    prefixes: Iterable[str], threshold: int
) -> dict[str, int]:
    oversized: dict[str, int] = {}
    key_counts: Counter[str] = Counter()
    for rel_path, abs_path in _iter_hot_rust_files(prefixes):
        lines = _read_production_lines(abs_path)
        for span in _extract_function_spans(lines):
            if span.lines <= threshold:
                continue
            key_base = f"{rel_path}::{span.name}"
            key_counts[key_base] += 1
            key = key_base
            if key_counts[key_base] > 1:
                key = f"{key_base}#{key_counts[key_base]}"
            oversized[key] = span.lines
    return oversized


def collect_dead_code_allow_occurrences(prefixes: Iterable[str]) -> Counter[str]:
    out: Counter[str] = Counter()
    for rel_path, abs_path in _iter_hot_rust_files(prefixes):
        for line in abs_path.read_text(encoding="utf-8").splitlines():
            if not ALLOW_DEAD_CODE_RE.search(line):
                continue
            key = f"{rel_path}::{_normalize_ws(line)}"
            out[key] += 1
    return out


def collect_too_many_args_allow_occurrences(prefixes: Iterable[str]) -> Counter[str]:
    out: Counter[str] = Counter()
    for rel_path, abs_path in _iter_hot_rust_files(prefixes):
        for line in abs_path.read_text(encoding="utf-8").splitlines():
            if not ALLOW_TOO_MANY_ARGS_RE.search(line):
                continue
            key = f"{rel_path}::{_normalize_ws(line)}"
            out[key] += 1
    return out


def run_static_guardrails(baseline: dict) -> bool:
    ok = True
    hot_prefixes = baseline["hot_module_prefixes"]

    fn_cfg = baseline["function_line_guard"]
    threshold = int(fn_cfg["line_threshold"])
    allowed_oversized: dict[str, int] = {
        str(k): int(v) for k, v in fn_cfg["allowed_oversized_functions"].items()
    }
    current_oversized = collect_oversized_functions(hot_prefixes, threshold)

    new_oversized: list[tuple[str, int]] = []
    grown_oversized: list[tuple[str, int, int]] = []
    for key, lines in sorted(current_oversized.items()):
        if key not in allowed_oversized:
            new_oversized.append((key, lines))
            continue
        if lines > allowed_oversized[key]:
            grown_oversized.append((key, lines, allowed_oversized[key]))

    if new_oversized:
        ok = False
        print("ERROR: new oversized functions detected:")
        for key, lines in new_oversized:
            print(f"  - {key} ({lines} lines)")
    if grown_oversized:
        ok = False
        print("ERROR: oversized baseline functions grew:")
        for key, lines, max_allowed in grown_oversized:
            print(f"  - {key} grew to {lines} lines (baseline max {max_allowed})")

    dead_cfg = baseline["dead_code_allow_guard"]
    allowed_occurrences = Counter(
        {str(k): int(v) for k, v in dead_cfg["allowed_occurrences"].items()}
    )
    current_occurrences = collect_dead_code_allow_occurrences(hot_prefixes)
    dead_code_regressions: list[tuple[str, int, int]] = []
    for key, current_count in sorted(current_occurrences.items()):
        allowed_count = allowed_occurrences.get(key, 0)
        if current_count > allowed_count:
            dead_code_regressions.append((key, current_count, allowed_count))

    if dead_code_regressions:
        ok = False
        print("ERROR: new `allow(dead_code)` occurrences in hot modules:")
        for key, current_count, allowed_count in dead_code_regressions:
            extra = current_count - allowed_count
            print(f"  - {key} (count {current_count}, baseline {allowed_count}, +{extra})")

    too_many_cfg = baseline["too_many_arguments_allow_guard"]
    allowed_occurrences = Counter(
        {str(k): int(v) for k, v in too_many_cfg["allowed_occurrences"].items()}
    )
    current_occurrences = collect_too_many_args_allow_occurrences(hot_prefixes)
    too_many_regressions: list[tuple[str, int, int]] = []
    for key, current_count in sorted(current_occurrences.items()):
        allowed_count = allowed_occurrences.get(key, 0)
        if current_count > allowed_count:
            too_many_regressions.append((key, current_count, allowed_count))

    if too_many_regressions:
        ok = False
        print("ERROR: new `allow(clippy::too_many_arguments)` occurrences in hot modules:")
        for key, current_count, allowed_count in too_many_regressions:
            extra = current_count - allowed_count
            print(f"  - {key} (count {current_count}, baseline {allowed_count}, +{extra})")

    if ok:
        print(
            "OK: static maintainability guardrails passed "
            f"(oversized threshold={threshold}, allow baselines locked)."
        )
    return ok


def run_rustdoc_guardrail(baseline: dict) -> bool:
    rustdoc_cfg = baseline["rustdoc_missing_docs_guard"]
    crate_name = str(rustdoc_cfg["crate"])
    max_warnings = int(rustdoc_cfg["max_warnings"])
    cmd = [
        "cargo",
        "rustdoc",
        "-p",
        crate_name,
        "--all-features",
        "--",
        "-W",
        "missing-docs",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout
    if proc.returncode != 0:
        print("ERROR: rustdoc invocation failed.")
        print(output)
        return False

    warning_count = len(MISSING_DOCS_WARNING_RE.findall(output))
    if warning_count > max_warnings:
        print(
            "ERROR: rustdoc missing-docs warnings regressed: "
            f"{warning_count} > baseline {max_warnings}"
        )
        return False

    print(
        "OK: rustdoc coverage guardrail passed "
        f"(missing-docs warnings {warning_count} <= baseline {max_warnings})."
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="Path to baseline JSON (default: tools/ci/maintainability_baseline.json).",
    )
    parser.add_argument(
        "--check",
        choices=("all", "static", "rustdoc"),
        default="all",
        help="Which guardrails to run.",
    )
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = (ROOT / baseline_path).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    ok = True
    if args.check in ("all", "static"):
        ok &= run_static_guardrails(baseline)
    if args.check in ("all", "rustdoc"):
        ok &= run_rustdoc_guardrail(baseline)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
