from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "tools" / "gen_codebook.py"
COMMITTED_CODEBOOK_JSON = REPO_ROOT / "tools" / "codebook.json"
BITS = 16
MASK = (1 << BITS) - 1


def _rotate_left(word: int, k: int) -> int:
    k %= BITS
    return ((word << k) | (word >> (BITS - k))) & MASK


def _canonical(word: int) -> int:
    return min(_rotate_left(word, k) for k in range(BITS))


def _run_gen_codebook(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_gen_codebook_preserves_loaded_seed_metadata(tmp_path: Path) -> None:
    base_json = tmp_path / "base.json"
    out_json = tmp_path / "out.json"
    out_rs = tmp_path / "out.rs"

    data = json.loads(COMMITTED_CODEBOOK_JSON.read_text())
    data["seed"] = 99
    data["profiles"]["base"]["seed"] = 99
    data["profiles"]["extended"]["seed"] = 99
    base_json.write_text(json.dumps(data))

    result = _run_gen_codebook(
        "--seed",
        "1",
        "--base_json",
        str(base_json),
        "--out_json",
        str(out_json),
        "--out_rs",
        str(out_rs),
    )

    assert result.returncode == 0, result.stderr
    assert "using source JSON" in result.stderr

    regenerated = json.loads(out_json.read_text())
    assert regenerated["seed"] == 99
    assert regenerated["profiles"]["base"]["seed"] == 99
    assert regenerated["profiles"]["extended"]["seed"] == 99


def test_gen_codebook_excludes_complement_collisions_in_extended_profile(
    tmp_path: Path,
) -> None:
    out_json = tmp_path / "out.json"
    out_rs = tmp_path / "out.rs"

    result = _run_gen_codebook(
        "--seed",
        "1",
        "--base_json",
        str(COMMITTED_CODEBOOK_JSON),
        "--out_json",
        str(out_json),
        "--out_rs",
        str(out_rs),
    )

    assert result.returncode == 0, result.stderr

    regenerated = json.loads(out_json.read_text())
    base_n = int(regenerated["profiles"]["base"]["n"])
    extended_words = [int(word, 16) for word in regenerated["profiles"]["extended"]["codewords"]]
    canonical_words = {_canonical(word) for word in extended_words}

    for word in extended_words[base_n:]:
        canonical = _canonical(word)
        complement = _canonical((~word) & MASK)
        if complement == canonical:
            continue
        assert complement not in canonical_words
