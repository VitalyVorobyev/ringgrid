#!/usr/bin/env python3
"""Generate/check committed typing artifacts for ringgrid Python API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "tools" / "typing_artifacts.pyi.template"
OUTPUT = ROOT / "python" / "ringgrid" / "__init__.pyi"


def read_template() -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if artifacts are out of date")
    args = parser.parse_args()

    expected = read_template()

    if args.check:
        if not OUTPUT.exists():
            print(f"missing typing artifact: {OUTPUT}", file=sys.stderr)
            return 1
        current = OUTPUT.read_text(encoding="utf-8")
        if current != expected:
            print("typing artifacts are out of date", file=sys.stderr)
            print(
                f"run: python {ROOT / 'tools' / 'generate_typing_artifacts.py'}",
                file=sys.stderr,
            )
            return 1
        print("typing artifacts are up to date")
        return 0

    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
