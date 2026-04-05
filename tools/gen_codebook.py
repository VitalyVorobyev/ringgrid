#!/usr/bin/env python3
"""Generate 16-bit codebook profiles for ring markers.

Each codeword is a 16-bit value encoding a marker ID via 16 angular sectors.
Codewords are selected to be:
  - rotationally unique (no nontrivial cyclic rotation equals itself)
  - pairwise separated under cyclic Hamming distance

Algorithm:
  - baseline profile: greedy selection with multiple random restarts,
    relaxing the minimum distance until the target count is reached
  - extended profile: append the remaining rotationally unique 16-bit
    codewords that do not introduce new polarity-complement collisions
    beyond the fixed baseline, preserving the baseline prefix exactly

Usage:
    python tools/gen_codebook.py --n 900 --seed 1 \\
        --out_json tools/codebook.json \\
        --out_rs crates/ringgrid/src/marker/codebook.rs
"""

import argparse
import json
import random
import sys
from pathlib import Path


BITS = 16
MASK = (1 << BITS) - 1


def popcount(x: int) -> int:
    return bin(x).count("1")


def rotate_left(word: int, k: int) -> int:
    k = k % BITS
    return ((word << k) | (word >> (BITS - k))) & MASK


def all_rotations(word: int) -> list[int]:
    return [rotate_left(word, k) for k in range(BITS)]


def canonical(word: int) -> int:
    return min(all_rotations(word))


def complement_canonical(word: int) -> int:
    return canonical((~word) & MASK)


def is_rotationally_symmetric(word: int) -> bool:
    for k in range(1, BITS):
        if rotate_left(word, k) == word:
            return True
    return False


def cyclic_hamming(a: int, b: int) -> int:
    """Min Hamming distance between a and any rotation of b."""
    return min(popcount(a ^ rotate_left(b, k)) for k in range(BITS))


def build_candidate_pool() -> list[int]:
    """Build pool: one canonical representative per rotation equivalence class,
    excluding rotationally symmetric words and 0x0000 / 0xFFFF."""
    seen: set[int] = set()
    pool: list[int] = []
    for w in range(1, MASK):
        if is_rotationally_symmetric(w):
            continue
        c = canonical(w)
        if c not in seen:
            seen.add(c)
            pool.append(c)
    return pool


def greedy_select(
    pool: list[int],
    n: int,
    min_dist: int,
    rng: random.Random,
) -> list[int]:
    """Greedy selection of codewords from shuffled pool."""
    shuffled = pool.copy()
    rng.shuffle(shuffled)

    chosen: list[int] = []

    for cand in shuffled:
        if len(chosen) >= n:
            break
        ok = True
        for w in chosen:
            if cyclic_hamming(cand, w) < min_dist:
                ok = False
                break
        if ok:
            chosen.append(cand)

    return chosen


def generate_codebook(
    n: int,
    min_dist: int,
    seed: int,
    restarts: int = 30,
) -> tuple[list[int], int]:
    """Generate up to n codewords, relaxing distance as needed.

    Returns (codewords, achieved_min_dist).
    """
    pool = build_candidate_pool()
    print(f"Candidate pool size (rotation equivalence classes): {len(pool)}")

    current_dist = min_dist
    best_chosen: list[int] = []

    while len(best_chosen) < n and current_dist >= 2:
        print(f"Trying min_cyclic_dist={current_dist}, {restarts} restarts...")

        for restart in range(restarts):
            rng = random.Random(seed + restart * 7919)
            chosen = greedy_select(pool, n, current_dist, rng)

            if len(chosen) > len(best_chosen):
                best_chosen = chosen
                print(f"  restart {restart}: {len(chosen)} codewords")

            if len(best_chosen) >= n:
                break

        if len(best_chosen) >= n:
            break

        print(
            f"  Best at dist={current_dist}: {len(best_chosen)}/{n}, "
            f"relaxing to {current_dist - 1}"
        )
        current_dist -= 1

    best_chosen = best_chosen[:n]

    # Compute verified minimum pairwise cyclic distance (sampling for speed)
    achieved = verified_min_dist(best_chosen)

    return best_chosen, achieved


def compute_distance_histogram(codewords: list[int]) -> dict:
    """Compute full pairwise cyclic distance histogram, worst pairs, and per-word badness."""
    nc = len(codewords)
    histogram = [0] * (BITS + 1)
    min_d = BITS
    worst_pairs: list[tuple[int, int]] = []

    for i in range(nc):
        for j in range(i + 1, nc):
            d = cyclic_hamming(codewords[i], codewords[j])
            histogram[d] += 1
            if d < min_d:
                min_d = d
                worst_pairs = [(i, j)]
            elif d == min_d:
                worst_pairs.append((i, j))

    # Per-codeword count of how many d_min pairs it participates in
    badness: dict[int, int] = {}
    for i, j in worst_pairs:
        badness[i] = badness.get(i, 0) + 1
        badness[j] = badness.get(j, 0) + 1

    return {
        "histogram": histogram,
        "min_dist": min_d,
        "worst_pairs": worst_pairs,
        "badness": badness,
    }


def estimate_removal_for_target(
    codewords: list[int], target_dist: int
) -> tuple[list[int], list[int]]:
    """Greedy removal of codewords until min cyclic distance reaches target_dist.

    Returns (remaining_codewords, removed_indices_in_original_order).
    """
    # Work with (original_index, word) pairs
    active: list[tuple[int, int]] = list(enumerate(codewords))
    removed: list[int] = []

    while True:
        # Find current min distance and worst pairs among active words
        min_d = BITS
        pairs_at_min: list[tuple[int, int]] = []  # indices into `active`

        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                d = cyclic_hamming(active[ai][1], active[aj][1])
                if d < min_d:
                    min_d = d
                    pairs_at_min = [(ai, aj)]
                elif d == min_d:
                    pairs_at_min.append((ai, aj))

        if min_d >= target_dist:
            break

        # Count how many min-dist pairs each active index participates in
        counts: dict[int, int] = {}
        for ai, aj in pairs_at_min:
            counts[ai] = counts.get(ai, 0) + 1
            counts[aj] = counts.get(aj, 0) + 1

        # Remove the word with the most min-dist pairs
        worst_active_idx = max(counts, key=lambda k: counts[k])
        removed.append(active[worst_active_idx][0])
        active.pop(worst_active_idx)

    remaining = [w for _, w in active]
    return remaining, sorted(removed)


def print_analysis(codewords: list[int]) -> None:
    """Print full distance distribution analysis."""
    nc = len(codewords)
    total_pairs = nc * (nc - 1) // 2
    print(f"\nDistance histogram ({nc} codewords, {total_pairs} pairs):")

    result = compute_distance_histogram(codewords)
    histogram = result["histogram"]
    min_d = result["min_dist"]
    worst_pairs = result["worst_pairs"]
    badness = result["badness"]

    for d in range(BITS + 1):
        if histogram[d] > 0:
            pct = 100.0 * histogram[d] / total_pairs
            print(f"  d={d:2d}: {histogram[d]:7d} pairs ({pct:.3f}%)")

    print(f"\nWorst pairs (d={min_d}): {len(worst_pairs)} pairs")
    for i, j in worst_pairs[:20]:
        print(
            f"  ID {i:4d} (0x{codewords[i]:04X}) <-> "
            f"ID {j:4d} (0x{codewords[j]:04X})"
        )
    if len(worst_pairs) > 20:
        print(f"  ... and {len(worst_pairs) - 20} more")

    if badness:
        print(f"\nCodewords with most d={min_d} neighbors:")
        ranked = sorted(badness.items(), key=lambda kv: -kv[1])
        for idx, count in ranked[:15]:
            print(f"  ID {idx:4d} (0x{codewords[idx]:04X}): {count} pairs at d={min_d}")
        if len(ranked) > 15:
            print(f"  ... and {len(ranked) - 15} more")

    # Estimate removals needed for d_min=3 and d_min=4
    for target in (3, 4):
        if min_d >= target:
            print(f"\nAlready at d_min >= {target}, no removals needed.")
            continue
        print(f"\nGreedy removal to reach d_min={target}...")
        remaining, removed_ids = estimate_removal_for_target(codewords, target)
        print(
            f"  Removed {len(removed_ids)} codewords -> "
            f"{len(remaining)} remaining, d_min={target}"
        )
        if len(removed_ids) <= 30:
            print(f"  Removed IDs: {removed_ids}")
        else:
            print(f"  Removed IDs (first 30): {removed_ids[:30]} ...")


def verified_min_dist(codewords: list[int], max_pairs: int = 500_000) -> int:
    """Compute min pairwise cyclic Hamming distance.
    For large codebooks, sample pairs to keep runtime bounded."""
    nc = len(codewords)
    if nc < 2:
        return BITS

    total_pairs = nc * (nc - 1) // 2

    if total_pairs <= max_pairs:
        # Exhaustive
        min_d = BITS
        for i in range(nc):
            for j in range(i + 1, nc):
                d = cyclic_hamming(codewords[i], codewords[j])
                if d < min_d:
                    min_d = d
                    if min_d <= 1:
                        return min_d
        return min_d
    else:
        # Sample
        rng = random.Random(42)
        min_d = BITS
        for _ in range(max_pairs):
            i = rng.randrange(nc)
            j = rng.randrange(nc)
            if i == j:
                continue
            d = cyclic_hamming(codewords[i], codewords[j])
            if d < min_d:
                min_d = d
                if min_d <= 1:
                    return min_d
        return min_d


def profile_record(
    codewords: list[int],
    achieved_dist: int,
    seed: int,
    *,
    base_n: int | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {
        "bits": BITS,
        "n": len(codewords),
        "min_cyclic_dist": achieved_dist,
        "seed": seed,
        "codewords": [f"0x{w:04X}" for w in codewords],
    }
    if base_n is not None:
        data["base_n"] = base_n
        data["extension_n"] = len(codewords) - base_n
    return data


def load_profile(path: Path) -> tuple[list[int], int, int]:
    with open(path) as f:
        data = json.load(f)
    return (
        [int(s, 16) for s in data["codewords"]],
        int(data["min_cyclic_dist"]),
        int(data["seed"]),
    )


def build_extended_profile(base_codewords: list[int], pool: list[int]) -> tuple[list[int], int]:
    base_set = set(base_codewords)
    blocked_complements = {complement_canonical(w) for w in base_codewords}
    extension: list[int] = []

    for w in pool:
        if w in base_set or w in blocked_complements:
            continue
        extension.append(w)
        base_set.add(w)
        blocked_complements.add(complement_canonical(w))

    extended = base_codewords + extension
    achieved_dist = 1 if len(extended) > 1 else BITS
    return extended, achieved_dist


def write_json(
    base_codewords: list[int],
    base_achieved_dist: int,
    extended_codewords: list[int],
    extended_achieved_dist: int,
    seed: int,
    path: Path,
) -> None:
    base_profile = profile_record(base_codewords, base_achieved_dist, seed)
    extended_profile = profile_record(
        extended_codewords,
        extended_achieved_dist,
        seed,
        base_n=len(base_codewords),
    )
    data = {
        **base_profile,
        "profiles": {
            "base": base_profile,
            "extended": extended_profile,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(
        "JSON written to "
        f"{path} (base={len(base_codewords)} codewords, "
        f"extended={len(extended_codewords)} codewords)"
    )


def write_rust(
    base_codewords: list[int],
    base_achieved_dist: int,
    extended_codewords: list[int],
    extended_achieved_dist: int,
    seed: int,
    path: Path,
) -> None:
    lines = [
        "//! Generated codebook profiles for 16-sector ring markers.",
        "//!",
        f"//! Generated by `tools/gen_codebook.py --n {len(base_codewords)} --seed {seed}`.",
        f"//! Base profile achieved minimum cyclic Hamming distance: {base_achieved_dist}.",
        f"//! Extended profile achieved minimum cyclic Hamming distance: {extended_achieved_dist}.",
        "//! Do not edit manually.",
        "",
        f"pub const CODEBOOK_BITS: usize = {BITS};",
        f"pub const CODEBOOK_N: usize = {len(base_codewords)};",
        f"pub const CODEBOOK_MIN_CYCLIC_DIST: usize = {base_achieved_dist};",
        f"pub const CODEBOOK_SEED: u64 = {seed};",
        "",
        f"pub const CODEBOOK_EXTENDED_N: usize = {len(extended_codewords)};",
        f"pub const CODEBOOK_EXTENDED_MIN_CYCLIC_DIST: usize = {extended_achieved_dist};",
        f"pub const CODEBOOK_EXTENDED_SEED: u64 = {seed};",
        f"pub const CODEBOOK_EXTENDED_EXTENSION_N: usize = {len(extended_codewords) - len(base_codewords)};",
        "",
        "/// Shipped baseline profile. This remains the default decode/codegen surface.",
        f"pub const CODEBOOK: [u16; {len(base_codewords)}] = [",
    ]
    for i in range(0, len(base_codewords), 8):
        chunk = base_codewords[i : i + 8]
        entries = ", ".join(f"0x{w:04X}" for w in chunk)
        lines.append(f"    {entries},")
    lines.append("];")
    lines.append("")
    lines.append("/// Additive opt-in profile: baseline prefix plus the remaining valid 16-bit")
    lines.append("/// rotationally unique codewords that do not introduce new")
    lines.append("/// polarity-complement collisions beyond the shipped baseline.")
    lines.append("/// This expands ID capacity but lowers the minimum cyclic Hamming distance to 1.")
    lines.append(
        f"pub const CODEBOOK_EXTENDED: [u16; {len(extended_codewords)}] = ["
    )
    for i in range(0, len(extended_codewords), 8):
        chunk = extended_codewords[i : i + 8]
        entries = ", ".join(f"0x{w:04X}" for w in chunk)
        lines.append(f"    {entries},")
    lines.append("];")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(
        "Rust written to "
        f"{path} (base={len(base_codewords)} codewords, "
        f"extended={len(extended_codewords)} codewords)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ring-marker codebook")
    parser.add_argument("--n", type=int, default=900, help="Number of codewords")
    parser.add_argument("--bits", type=int, default=16, help="Bits per codeword (fixed at 16)")
    parser.add_argument("--min_cyclic_dist", type=int, default=6, help="Target min cyclic Hamming dist")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--restarts", type=int, default=30, help="Random restarts per distance level")
    parser.add_argument(
        "--base_json",
        type=str,
        default="tools/codebook.json",
        help="Existing baseline codebook JSON to preserve as the profile prefix",
    )
    parser.add_argument("--out_json", type=str, default="tools/codebook.json")
    parser.add_argument(
        "--out_rs",
        type=str,
        default="crates/ringgrid/src/marker/codebook.rs",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Print distance distribution analysis of the codebook and exit",
    )
    args = parser.parse_args()

    if args.bits != 16:
        print("WARNING: only 16 bits supported; forcing bits=16", file=sys.stderr)

    base_source = Path(args.base_json)
    effective_seed = args.seed
    if base_source.exists():
        base_codewords, base_achieved, base_seed = load_profile(base_source)
        effective_seed = base_seed
        print(
            "Loaded baseline profile from "
            f"{base_source}: n={len(base_codewords)}, "
            f"min_cyclic_dist={base_achieved}, seed={base_seed}"
        )
        if base_seed != args.seed:
            print(
                "WARNING: baseline seed from source JSON does not match "
                f"--seed ({base_seed} != {args.seed}); using source JSON",
                file=sys.stderr,
            )
    else:
        print(
            "Generating baseline profile: "
            f"n={args.n}, target min_cyclic_dist={args.min_cyclic_dist}, seed={args.seed}"
        )
        base_codewords, base_achieved = generate_codebook(
            n=args.n,
            min_dist=args.min_cyclic_dist,
            seed=args.seed,
            restarts=args.restarts,
        )

    if args.analyze:
        print_analysis(base_codewords)
        return

    pool = build_candidate_pool()
    extended_codewords, extended_achieved = build_extended_profile(base_codewords, pool)

    print(
        f"\nBase profile: {len(base_codewords)} codewords, "
        f"achieved min cyclic dist = {base_achieved}"
    )
    print(
        f"Extended profile: {len(extended_codewords)} codewords "
        f"({len(extended_codewords) - len(base_codewords)} appended), "
        f"achieved min cyclic dist = {extended_achieved}"
    )
    write_json(
        base_codewords,
        base_achieved,
        extended_codewords,
        extended_achieved,
        effective_seed,
        Path(args.out_json),
    )
    write_rust(
        base_codewords,
        base_achieved,
        extended_codewords,
        extended_achieved,
        effective_seed,
        Path(args.out_rs),
    )


if __name__ == "__main__":
    main()
