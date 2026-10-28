#!/usr/bin/env python3
"""Extract the preregistered pilot subsets from the official PCam
HDF5 files (Zenodo record 2546921) into compact .npz caches.

Usage:
    python scripts/extract_subset.py --data-dir /path/to/h5 --out-dir data_cache

Expects in --data-dir:
    camelyonpatch_level_2_split_valid_x.h5.gz / _valid_y.h5.gz
    camelyonpatch_level_2_split_test_x.h5.gz  / _test_y.h5.gz

Writes pool.npz (6,000 class-stratified patches from the official
validation split, seed 42) and test.npz (2,000 class-stratified
patches from the official test split, seed 43) at native 96x96.
The pilot trains on the validation split subset and evaluates on
the test split subset; see PREREG.md.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpath_bench.data import stratified_subset  # noqa: E402

POOL_N = 6000
POOL_SEED = 42
TEST_N = 2000
TEST_SEED = 43


def load_h5_gz(path: Path, dataset: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with gzip.open(path, "rb") as src:
            tmp.write(src.read())
        tmp_path = tmp.name
    try:
        with h5py.File(tmp_path, "r") as f:
            return f[dataset][:]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, n, seed, out_name in (
        ("valid", POOL_N, POOL_SEED, "pool.npz"),
        ("test", TEST_N, TEST_SEED, "test.npz"),
    ):
        x = load_h5_gz(
            data_dir / f"camelyonpatch_level_2_split_{split}_x.h5.gz", "x"
        )
        y = load_h5_gz(
            data_dir / f"camelyonpatch_level_2_split_{split}_y.h5.gz", "y"
        ).reshape(-1).astype(np.int64)
        assert x.shape[0] == y.shape[0], (x.shape, y.shape)
        idx = stratified_subset(y, n, seed)
        np.savez_compressed(
            out_dir / out_name,
            x=x[idx],
            y=y[idx],
            source_indices=idx,
        )
        print(
            f"{out_name}: {len(idx)} patches from official {split} split "
            f"(positives={int(y[idx].sum())}, negatives={int((y[idx] == 0).sum())})"
        )


if __name__ == "__main__":
    main()
