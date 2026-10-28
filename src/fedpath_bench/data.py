"""Data handling: PCam subsets, Dirichlet label-skew partitioning,
and per-client stain application.

The pilot trains on a class-stratified subset of the official PCam
VALIDATION split and evaluates on a class-stratified subset of the
official TEST split (see PREREG.md for why, and for the exact sizes
and seeds). Subsets are extracted once from the official HDF5 files
by scripts/extract_subset.py into compact .npz caches.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .stain import apply_profile, profile_for_client


def stratified_subset(labels: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    """Class-stratified subset of indices, as equal per class as
    possible, drawn with numpy RandomState(seed). Deterministic."""
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    per_class = n_total // len(classes)
    chosen: list[np.ndarray] = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(len(idx))]
        chosen.append(idx[:per_class])
    out = np.concatenate(chosen)
    return out[rng.permutation(len(out))]


def dirichlet_label_partition(
    labels: np.ndarray, n_clients: int, alpha: float, seed: int
) -> list[np.ndarray]:
    """Split indices across clients with per-class Dirichlet label
    skew. Every index is assigned to exactly one client
    (disjoint + covering; verified by tests). Deterministic in seed.
    """
    rng = np.random.RandomState(seed)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(len(idx))]
        proportions = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(idx, cuts)):
            client_indices[client_id].extend(chunk.tolist())
    return [np.array(sorted(ix), dtype=np.int64) for ix in client_indices]


def partition_digest(partition: list[np.ndarray]) -> str:
    """SHA-256 over the sorted per-client index lists."""
    h = hashlib.sha256()
    for ix in partition:
        h.update(json.dumps([int(i) for i in ix]).encode())
    return h.hexdigest()


def apply_client_stains(
    images: np.ndarray, partition: list[np.ndarray]
) -> np.ndarray:
    """Return a copy of the pool where each client's patches carry
    that client's stain profile. Client k uses profile k."""
    out = images.copy()
    for client_id, idx in enumerate(partition):
        if len(idx) == 0:
            continue
        out[idx] = apply_profile(images[idx], profile_for_client(client_id))
    return out


def client_class_counts(
    labels: np.ndarray, partition: list[np.ndarray]
) -> list[dict[str, int]]:
    rows = []
    for client_id, idx in enumerate(partition):
        ys = labels[idx]
        rows.append(
            {
                "client": client_id,
                "n": int(len(idx)),
                "n_negative": int((ys == 0).sum()),
                "n_positive": int((ys == 1).sum()),
            }
        )
    return rows
