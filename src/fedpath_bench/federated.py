"""Training loops for the three preregistered arms: centralized,
FedAvg, and local-only. All arms share the model, optimizer family,
batch size, and seed (PREREG.md). CPU only.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .model import SmallCNN

LR = 1e-3
BATCH_SIZE = 64


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    shuffle: bool,
    seed: int,
) -> torch.utils.data.DataLoader:
    """images: uint8 (N, H, W, 3). Normalization applied on the fly."""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    y = torch.from_numpy(labels.astype(np.int64))
    ds = torch.utils.data.TensorDataset(x, y)
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=gen
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float | int]:
    model.eval()
    loader = make_loader(images, labels, mean, std, shuffle=False, seed=0)
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    for xb, yb in loader:
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum())
        total += int(len(yb))
        tp += int(((pred == 1) & (yb == 1)).sum())
        fp += int(((pred == 1) & (yb == 0)).sum())
        tn += int(((pred == 0) & (yb == 0)).sum())
        fn += int(((pred == 0) & (yb == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def train_epochs(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    epochs: int,
    seed: int,
) -> nn.Module:
    """Train in place for `epochs` passes and return the model."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loader = make_loader(images, labels, mean, std, shuffle=True, seed=seed + epoch)
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def fedavg_aggregate(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float]
) -> dict[str, torch.Tensor]:
    """Sample-count-weighted average of model state dicts. Floating
    tensors are averaged; integer buffers (none expected with
    GroupNorm) are copied from the first client. Exact: aggregating
    a single model returns that model (verified by tests)."""
    total = float(sum(weights))
    normed = [w / total for w in weights]
    out: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        first = state_dicts[0][key]
        if first.dtype.is_floating_point:
            acc = sum(sd[key].float() * w for sd, w in zip(state_dicts, normed))
            out[key] = acc.to(first.dtype)
        else:
            out[key] = first.clone()
    return out


def run_fedavg(
    images: np.ndarray,
    labels: np.ndarray,
    partition: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    rounds: int,
    local_epochs: int,
    seed: int,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    on_round=None,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    """Full-participation FedAvg. Returns (global model, per-round
    clean-test metrics trace). `on_round`, if given, is called with
    each round's metrics dict as soon as it is computed."""
    set_seed(seed)
    global_model = SmallCNN()
    trace: list[dict[str, float | int]] = []
    weights = [float(len(ix)) for ix in partition]
    for rnd in range(rounds):
        client_states = []
        for client_id, idx in enumerate(partition):
            local = SmallCNN()
            local.load_state_dict(copy.deepcopy(global_model.state_dict()))
            train_epochs(
                local,
                images[idx],
                labels[idx],
                mean,
                std,
                epochs=local_epochs,
                seed=seed + 1000 * rnd + client_id,
            )
            client_states.append(local.state_dict())
        global_model.load_state_dict(fedavg_aggregate(client_states, weights))
        metrics = evaluate(global_model, test_images, test_labels, mean, std)
        row = {"round": rnd + 1, **metrics}
        trace.append(row)
        if on_round is not None:
            on_round(row)
    return global_model, trace
