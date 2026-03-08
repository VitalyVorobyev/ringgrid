#!/usr/bin/env python3
"""Microbenchmark hot DetectConfig getter/setter paths."""

from __future__ import annotations

import argparse
import statistics
import timeit

import ringgrid


def build_config() -> ringgrid.DetectConfig:
    cfg = ringgrid.DetectConfig(ringgrid.BoardLayout.default())
    cfg.to_dict()
    return cfg


def benchmark_scenarios(
    *,
    repeat: int,
    getter_number: int,
    setter_number: int,
) -> list[tuple[str, int, float, list[float]]]:
    cfg = build_config()
    state_conf = [False]
    state_completion = [False]
    state_global = [False]

    def get_decode_section() -> ringgrid.DecodeConfig:
        return cfg.decode

    def get_decode_min_confidence() -> float:
        return cfg.decode_min_confidence

    def get_completion_enable() -> bool:
        return cfg.completion_enable

    def set_decode_min_confidence() -> None:
        cfg.decode_min_confidence = 0.31 if state_conf[0] else 0.41
        state_conf[0] = not state_conf[0]

    def set_completion_enable() -> None:
        cfg.completion_enable = state_completion[0]
        state_completion[0] = not state_completion[0]

    def set_use_global_filter() -> None:
        cfg.use_global_filter = state_global[0]
        state_global[0] = not state_global[0]

    scenarios = [
        ("get_decode_section", get_decode_section, getter_number),
        ("get_decode_min_confidence", get_decode_min_confidence, getter_number),
        ("get_completion_enable", get_completion_enable, getter_number),
        ("set_decode_min_confidence", set_decode_min_confidence, setter_number),
        ("set_completion_enable", set_completion_enable, setter_number),
        ("set_use_global_filter", set_use_global_filter, setter_number),
    ]

    results: list[tuple[str, int, float, list[float]]] = []
    for name, fn, number in scenarios:
        samples = timeit.Timer(fn).repeat(repeat=repeat, number=number)
        samples_us = [(sample / number) * 1_000_000.0 for sample in samples]
        median_us = statistics.median(samples_us)
        results.append((name, number, median_us, samples_us))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=7, help="Timeit repeat count")
    parser.add_argument(
        "--getter-number",
        type=int,
        default=5000,
        help="Iterations per getter repeat",
    )
    parser.add_argument(
        "--setter-number",
        type=int,
        default=1000,
        help="Iterations per setter repeat",
    )
    args = parser.parse_args()

    results = benchmark_scenarios(
        repeat=args.repeat,
        getter_number=args.getter_number,
        setter_number=args.setter_number,
    )

    for name, number, median_us, samples_us in results:
        rounded = ", ".join(f"{sample:.2f}" for sample in samples_us)
        print(
            f"{name}: number={number} median={median_us:.2f} us/op "
            f"samples_us=[{rounded}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
