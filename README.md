# fedpath-bench — a federated computational-pathology benchmark pilot on PatchCamelyon

Federated learning papers in medical imaging mostly borrow generic
benchmarks. In computational pathology, the cross-site problem is
specific: hospitals differ in label mix AND in acquisition — scanner
white balance and H&E stain intensity shift every pixel. This pilot
builds a small, fully reproducible benchmark on **PatchCamelyon
(PCam)** where simulated hospitals differ in both ways at once, and
measures what that does to the three training regimes everyone
compares: centralized, FedAvg, and local-only.

## Pilot results (measured, this repo, CPU, seed 42)

Primary endpoint — accuracy on 2,000 clean patches from the
official PCam **test** split (1,000 positive / 1,000 negative):

| Arm | Accuracy | Correct | Sensitivity | Specificity |
|---|---|---|---|---|
| Centralized (pooled) | 0.7845 | 1569/2000 | 0.783 | 0.786 |
| FedAvg (15 rounds) | 0.7445 | 1489/2000 | 0.677 | 0.812 |
| Local-only, client 0 (n=386) | 0.6650 | 1330/2000 | 0.440 | 0.890 |
| Local-only, client 1 (n=1298) | 0.6690 | 1338/2000 | 0.549 | 0.789 |
| Local-only, client 2 (n=2177) | 0.6110 | 1222/2000 | 0.527 | 0.695 |
| Local-only, client 3 (n=2139) | 0.5890 | 1178/2000 | 0.227 | 0.951 |
| Local-only, sample-weighted mean | 0.6192 | — | — | — |

Preregistered readout (PREREG.md, locked before the run):
**R1 PASS** — FedAvg beats the local-only mean by **+12.53pp**
(bar: +3pp). **R2 PASS** — FedAvg sits **4.0pp** behind centralized
(bar: within 10pp).

Secondary readout — the same two global models evaluated on the
test set re-rendered under each client's acquisition profile:

| Test profile | Centralized | FedAvg |
|---|---|---|
| reference (clean) | 0.7845 | 0.7445 |
| warm_scanner | 0.7980 | 0.7305 |
| cool_scanner | 0.7870 | **0.5895** |
| overstained_lab | 0.7245 | 0.7065 |

The pooled model barely moves under profile shift. The FedAvg
model collapses under the cool-scanner profile (sensitivity falls
to 0.265): averaging weights across differently-shifted clients
produced a model whose decision balance is fragile to one
acquisition direction. Single seed, pilot scale — an observation
to test at full scale, not a law.

FedAvg's round trace oscillates (0.634–0.7445) before ending at
its best round, 0.7445 — the sensitivity/specificity pair swings
hard between rounds under this label skew. The full trace is in
`results/pilot_summary.json`.

## Protocol (preregistered, PREREG.md)

- **Data**: PCam (Veeling et al. 2018), Zenodo record 2546921.
  The pilot trains on a class-stratified **6,000-patch subset of
  the official validation split** (seed 42) and evaluates on a
  class-stratified **2,000-patch subset of the official test
  split** (seed 43). The official train split is not used in the
  pilot (its x-file is 6.4 GB; subset honesty over silent
  shortcuts). Patches resized 96→48 px for CPU feasibility.
- **Clients**: 4 simulated hospitals. Label skew via per-class
  Dirichlet(α=0.5): client sizes 386 / 1,298 / 2,177 / 2,139, with
  positive rates 43% / 80% / 59% / 24%. Partition SHA-256:
  `e58678e535f5e9a1c68311e8fe259327b2cb74162d0882d3fc1362cd6453b799`.
- **Acquisition shift**: each client applies one fixed,
  deterministic stain/scanner profile (white-balance gains,
  saturation, brightness, contrast) to all its patches — client 0
  is the identity reference. Profiles are **simulations**: PCam
  ships no per-patch scanner metadata, and no real scanner claim
  is made. See `src/fedpath_bench/stain.py`.
- **Model**: SmallCNN (102,082 params) with GroupNorm — chosen so
  FedAvg weight averaging is exact, with no BatchNorm statistics
  aggregation artifact. Adam lr 1e-3, batch 64, cross-entropy, no
  augmentation, seed 42 for every arm.
- **Budgets matched**: centralized 15 epochs; FedAvg 15 rounds ×
  1 local epoch (same total sample passes); local-only 15 epochs
  per client on its own shard.
- **Determinism**: fixed seeds everywhere; the centralized arm was
  run twice (an earlier attempt was killed by a machine restart
  after the centralized arm finished) and reproduced **exactly**:
  1,569/2,000 correct, accuracy 0.7845, identical
  sensitivity/specificity.

## Honest limits

- Pilot scale: 6,000 training patches, one seed, one α, one model.
  Absolute accuracies (~0.78) are far below full-data PCam results
  (~0.95 with pretraining on 262k patches) — the pilot measures
  *relative* arm behaviour under heterogeneity, not SOTA.
- Training pool comes from the official validation split, not the
  train split (download size). Splits remain disjoint and the test
  set is the official test split, but full-scale runs should use
  the official train split.
- Stain/scanner profiles are parametric color simulations, not
  measured scanner characterizations and not stain-separation
  (Macenko/Reinhard) models.
- Simulation-only federation: all clients run in one process on
  one CPU machine. No networking, no secure aggregation, no
  differential privacy.

## Prior art (kill-gate, KILLGATE.md)

FLamby's **Fed-Camelyon16** is the nearest benchmark: Camelyon16
slides split by hospital (2 clients), FedAvg/FedProx/FedOpt
baselines. It is not PCam, has no simulated acquisition shift, and
is a heavyweight suite. One-off papers run FL on PCam as single
experiments. No maintained federated PCam benchmark with
pathology-specific (label + acquisition) splits was found —
details and sources in `KILLGATE.md`.

## Run it

```bash
pip install torch numpy h5py pytest   # CPU wheels are fine
# Download the 4 PCam HDF5 files (valid_x/y, test_x/y) from
# https://zenodo.org/records/2546921 into a folder, then:
python scripts/extract_subset.py --data-dir /path/to/h5 --out-dir data_cache
bash run_all.sh   # pytest (9 tests) + the preregistered pilot (~30 min on 2 CPU cores)
```

Data credit: PatchCamelyon, Veeling et al. 2018 (CC0). This repo
redistributes no PCam data — only the extraction script, the
protocol, and measured results.
