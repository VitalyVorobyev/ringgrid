#!/usr/bin/env python3
"""Generate a 16-bit cyclic-distance-separated codebook for ring markers.

Each codeword is a 16-bit value encoding a marker ID via 16 angular sectors.
Codewords are selected to be:
  - rotationally unique (no nontrivial cyclic rotation equals itself)
  - pairwise separated under cyclic Hamming distance

Algorithm: greedy selection with multiple random restarts, relaxing the
minimum distance until the target count is reached.

Usage:
    python tools/gen_codebook.py --n 900 --seed 1 \\
        --out_json tools/codebook.json \\
        --out_rs crates/ringgrid-core/src/codebook.rs
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


def write_json(codewords: list[int], achieved_dist: int, seed: int, path: Path) -> None:
    data = {
        "bits": BITS,
        "n": len(codewords),
        "min_cyclic_dist": achieved_dist,
        "seed": seed,
        "codewords": [f"0x{w:04X}" for w in codewords],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON written to {path} ({len(codewords)} codewords)")


def write_rust(codewords: list[int], achieved_dist: int, seed: int, path: Path) -> None:
    lines = [
        "//! Generated codebook for 16-sector ring markers.",
        "//!",
        f"//! Generated by `tools/gen_codebook.py --n {len(codewords)} --seed {seed}`.",
        f"//! Achieved minimum cyclic Hamming distance: {achieved_dist}.",
        "//! Do not edit manually.",
        "",
        f"pub const CODEBOOK_BITS: usize = {BITS};",
        f"pub const CODEBOOK_N: usize = {len(codewords)};",
        f"pub const CODEBOOK_MIN_CYCLIC_DIST: usize = {achieved_dist};",
        f"pub const CODEBOOK_SEED: u64 = {seed};",
        "",
        f"pub const CODEBOOK: [u16; {len(codewords)}] = [",
    ]
    for i in range(0, len(codewords), 8):
        chunk = codewords[i : i + 8]
        entries = ", ".join(f"0x{w:04X}" for w in chunk)
        lines.append(f"    {entries},")
    lines.append("];")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Rust written to {path} ({len(codewords)} codewords)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ring-marker codebook")
    parser.add_argument("--n", type=int, default=900, help="Number of codewords")
    parser.add_argument("--bits", type=int, default=16, help="Bits per codeword (fixed at 16)")
    parser.add_argument("--min_cyclic_dist", type=int, default=6, help="Target min cyclic Hamming dist")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--restarts", type=int, default=30, help="Random restarts per distance level")
    parser.add_argument("--out_json", type=str, default="tools/codebook.json")
    parser.add_argument("--out_rs", type=str, default="crates/ringgrid-core/src/codebook.rs")
    args = parser.parse_args()

    if args.bits != 16:
        print("WARNING: only 16 bits supported; forcing bits=16", file=sys.stderr)

    print(f"Generating codebook: n={args.n}, target min_cyclic_dist={args.min_cyclic_dist}, seed={args.seed}")
    codewords, achieved = generate_codebook(
        n=args.n,
        min_dist=args.min_cyclic_dist,
        seed=args.seed,
        restarts=args.restarts,
    )

    print(f"\nFinal: {len(codewords)} codewords, achieved min cyclic dist = {achieved}")
    write_json(codewords, achieved, args.seed, Path(args.out_json))
    write_rust(codewords, achieved, args.seed, Path(args.out_rs))


if __name__ == "__main__":
    main()
