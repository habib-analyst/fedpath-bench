"""Tests for fedpath-bench: partition integrity, stain determinism,
exact FedAvg averaging, and a model forward pass. CPU, no data
downloads (synthetic arrays only).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedpath_bench.data import (  # noqa: E402
    dirichlet_label_partition,
    partition_digest,
    stratified_subset,
)
from fedpath_bench.federated import fedavg_aggregate  # noqa: E402
from fedpath_bench.model import SmallCNN  # noqa: E402
from fedpath_bench.stain import PROFILES, apply_profile, profile_for_client  # noqa: E402


def test_partition_disjoint_and_covering():
    labels = np.array([0] * 500 + [1] * 500)
    part = dirichlet_label_partition(labels, n_clients=4, alpha=0.5, seed=42)
    all_idx = np.concatenate(part)
    assert len(all_idx) == 1000
    assert len(np.unique(all_idx)) == 1000  # disjoint
    assert set(all_idx.tolist()) == set(range(1000))  # covering


def test_partition_deterministic():
    labels = np.array([0] * 200 + [1] * 200)
    a = dirichlet_label_partition(labels, 4, 0.5, 42)
    b = dirichlet_label_partition(labels, 4, 0.5, 42)
    assert partition_digest(a) == partition_digest(b)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_partition_is_actually_skewed():
    # With alpha=0.5 over 4 clients, per-client positive counts must
    # not all be equal (label skew is the point of the partition).
    labels = np.array([0] * 1000 + [1] * 1000)
    part = dirichlet_label_partition(labels, 4, 0.5, 42)
    pos = [int((labels[ix] == 1).sum()) for ix in part]
    assert len(set(pos)) > 1


def test_stratified_subset_balanced_and_deterministic():
    labels = np.array([0] * 1000 + [1] * 1000)
    a = stratified_subset(labels, 200, seed=42)
    b = stratified_subset(labels, 200, seed=42)
    assert np.array_equal(np.sort(a), np.sort(b))
    assert int((labels[a] == 0).sum()) == 100
    assert int((labels[a] == 1).sum()) == 100


def test_stain_identity_profile_is_noop():
    rng = np.random.RandomState(0)
    imgs = rng.randint(0, 256, size=(8, 12, 12, 3), dtype=np.uint8)
    out = apply_profile(imgs, PROFILES[0])
    assert np.array_equal(out, imgs)


def test_stain_deterministic_and_changes_pixels():
    rng = np.random.RandomState(0)
    imgs = rng.randint(0, 256, size=(8, 12, 12, 3), dtype=np.uint8)
    warm = PROFILES[1]
    a = apply_profile(imgs, warm)
    b = apply_profile(imgs, warm)
    assert np.array_equal(a, b)  # deterministic
    assert not np.array_equal(a, imgs)  # non-identity profile shifts pixels
    assert a.dtype == np.uint8 and a.shape == imgs.shape
    assert profile_for_client(1).name == "warm_scanner"
    assert profile_for_client(0).name == "reference"


def test_fedavg_single_client_is_identity():
    torch.manual_seed(0)
    model = SmallCNN()
    sd = model.state_dict()
    agg = fedavg_aggregate([sd], [10.0])
    for k in sd:
        assert torch.allclose(agg[k], sd[k])


def test_fedavg_weighted_mean_exact():
    torch.manual_seed(0)
    m1, m2 = SmallCNN(), SmallCNN()
    agg = fedavg_aggregate([m1.state_dict(), m2.state_dict()], [1.0, 3.0])
    for k in agg:
        expected = 0.25 * m1.state_dict()[k].float() + 0.75 * m2.state_dict()[k].float()
        assert torch.allclose(agg[k].float(), expected, atol=1e-6)


def test_model_forward_shape():
    model = SmallCNN()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(5, 3, 48, 48))
    assert out.shape == (5, 2)
