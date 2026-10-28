#!/usr/bin/env python3
"""Run the fedpath-bench pilot exactly as preregistered (PREREG.md).

Usage:
    python scripts/run_pilot.py --cache-dir data_cache --out-dir results

Defaults ARE the preregistered protocol: pool 6,000 (official
validation split subset), test 2,000 (official test split subset),
K=4 clients, Dirichlet alpha=0.5, 15 centralized epochs, 15 FedAvg
rounds x 1 local epoch, 15 local-only epochs, seed 42, Adam lr 1e-3,
batch 64. The --epochs / --rounds / --pool-limit / --test-limit
overrides exist only for smoke tests and are recorded in the
summary when used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpath_bench.data import (  # noqa: E402
    apply_client_stains,
    client_class_counts,
    dirichlet_label_partition,
    partition_digest,
)
from fedpath_bench.federated import evaluate, run_fedavg, set_seed, train_epochs  # noqa: E402
from fedpath_bench.model import SmallCNN, count_parameters  # noqa: E402
from fedpath_bench.stain import PROFILES, apply_profile  # noqa: E402

SEED = 42
N_CLIENTS = 4
ALPHA = 0.5
IMG_SIZE = 48


def resize_batch(x: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    out = []
    for i in range(0, len(t), 1000):
        chunk = F.interpolate(t[i : i + 1000], size=size, mode="bilinear", antialias=True)
        out.append(chunk)
    t = torch.cat(out).permute(0, 2, 3, 1)
    return np.clip(np.round(t.numpy()), 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--pool-limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(2)
    t_start = time.time()
    cache = Path(args.cache_dir)
    pool = np.load(cache / "pool.npz")
    test = np.load(cache / "test.npz")
    pool_x, pool_y = pool["x"], pool["y"].astype(np.int64)
    test_x, test_y = test["x"], test["y"].astype(np.int64)
    if args.pool_limit:
        pool_x, pool_y = pool_x[: args.pool_limit], pool_y[: args.pool_limit]
    if args.test_limit:
        test_x, test_y = test_x[: args.test_limit], test_y[: args.test_limit]

    pool_x = resize_batch(pool_x, IMG_SIZE)
    test_x = resize_batch(test_x, IMG_SIZE)

    mean = (pool_x.astype(np.float32) / 255.0).mean(axis=(0, 1, 2))
    std = (pool_x.astype(np.float32) / 255.0).std(axis=(0, 1, 2))

    partition = dirichlet_label_partition(pool_y, N_CLIENTS, ALPHA, SEED)
    counts = client_class_counts(pool_y, partition)
    stained_pool = apply_client_stains(pool_x, partition)
    digest = partition_digest(partition)
    print("partition:", counts, flush=True)
    print("partition sha256:", digest, flush=True)

    # Incremental checkpointing: the rows file is rewritten after
    # every arm (and every FedAvg round) so a machine restart loses
    # at most the arm in flight, never a finished measurement.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def flush_rows() -> None:
        with open(out_dir / "pilot_rows.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    # --- Arm 1: centralized -------------------------------------------
    print("arm: centralized", flush=True)
    set_seed(SEED)
    central = SmallCNN()
    train_epochs(central, stained_pool, pool_y, mean, std, args.epochs, SEED)
    central_metrics = evaluate(central, test_x, test_y, mean, std)
    rows.append({"arm": "centralized", **central_metrics})
    flush_rows()
    print("centralized:", central_metrics, flush=True)

    # --- Arm 2: FedAvg -------------------------------------------------
    print("arm: fedavg", flush=True)

    def _on_round(t: dict) -> None:
        rows.append({"arm": "fedavg_round", **t})
        flush_rows()
        print("fedavg round:", t, flush=True)

    fed_model, trace = run_fedavg(
        stained_pool, pool_y, partition, mean, std,
        rounds=args.rounds, local_epochs=1, seed=SEED,
        test_images=test_x, test_labels=test_y,
        on_round=_on_round,
    )
    fed_metrics = evaluate(fed_model, test_x, test_y, mean, std)
    rows.append({"arm": "fedavg", **fed_metrics})
    flush_rows()
    print("fedavg:", fed_metrics, flush=True)

    # --- Arm 3: local-only --------------------------------------------
    print("arm: local-only", flush=True)
    local_rows = []
    for client_id, idx in enumerate(partition):
        set_seed(SEED + client_id)
        local = SmallCNN()
        train_epochs(local, stained_pool[idx], pool_y[idx], mean, std, args.epochs, SEED + client_id)
        m = evaluate(local, test_x, test_y, mean, std)
        local_rows.append({"client": client_id, "train_n": int(len(idx)), **m})
        rows.append({"arm": "local_only", "client": client_id, **m})
        flush_rows()
        print(f"local client {client_id}:", m, flush=True)
    local_mean_acc = float(np.average([r["accuracy"] for r in local_rows], weights=[r["train_n"] for r in local_rows]))

    # --- Secondary: stained-test robustness ----------------------------
    stained_test: dict[str, dict] = {}
    for profile in PROFILES:
        sx = apply_profile(test_x, profile)
        stained_test[profile.name] = {
            "centralized": evaluate(central, sx, test_y, mean, std),
            "fedavg": evaluate(fed_model, sx, test_y, mean, std),
        }
        print("stained test", profile.name, stained_test[profile.name], flush=True)

    r1 = fed_metrics["accuracy"] >= local_mean_acc + 0.03
    r2 = fed_metrics["accuracy"] >= central_metrics["accuracy"] - 0.10
    summary = {
        "config": {
            "pool_n": int(len(pool_y)),
            "pool_source": "official PCam validation split, class-stratified subset, seed 42",
            "test_n": int(len(test_y)),
            "test_source": "official PCam test split, class-stratified subset, seed 43",
            "n_clients": N_CLIENTS,
            "dirichlet_alpha": ALPHA,
            "image_size": IMG_SIZE,
            "epochs": args.epochs,
            "rounds": args.rounds,
            "seed": SEED,
            "model_params": count_parameters(SmallCNN()),
        },
        "client_counts": counts,
        "partition_sha256": digest,
        "centralized": central_metrics,
        "fedavg": fed_metrics,
        "fedavg_rounds_trace": trace,
        "local_only": {"per_client": local_rows, "weighted_mean_accuracy": local_mean_acc},
        "stained_test_secondary": stained_test,
        "readout": {
            "R1_fedavg_beats_local_mean_by_3pp": bool(r1),
            "R2_fedavg_within_10pp_of_centralized": bool(r2),
            "fedavg_minus_local_mean_pp": round(100 * (fed_metrics["accuracy"] - local_mean_acc), 2),
            "centralized_minus_fedavg_pp": round(100 * (central_metrics["accuracy"] - fed_metrics["accuracy"]), 2),
        },
        "wall_time_seconds": round(time.time() - t_start, 1),
    }

    summary_bytes = json.dumps(summary, indent=2).encode()
    (out_dir / "pilot_summary.json").write_bytes(summary_bytes)
    print("summary sha256:", hashlib.sha256(summary_bytes).hexdigest())
    print("DONE in", summary["wall_time_seconds"], "s")


if __name__ == "__main__":
    main()
