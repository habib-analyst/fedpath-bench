# PREREGISTRATION — fedpath-bench PILOT (locked BEFORE the run)

Locked: 2026-10-28, after the kill-gate passed and before any pilot
number exists. No training run, exploratory or otherwise, has touched
these subsets at lock time.

## Data (frozen)

- Source: PatchCamelyon (PCam), Veeling et al. 2018, Zenodo record
  2546921, official HDF5 files.
- The official TRAIN split x-file is 6.4 GB (gz); the pilot instead
  trains on the official VALIDATION split and evaluates on the
  official TEST split. This is stated everywhere results appear.
- Training pool: 6,000 patches drawn class-stratified from the
  official validation split (32,768), numpy RandomState seed 42.
- Test set: 2,000 patches drawn class-stratified from the official
  test split (32,768), seed 43.
- No patch appears in both pool and test (disjoint official splits
  by construction: validation vs test).
- Images resized 96x96 -> 48x48 (bilinear) for CPU feasibility.
  Normalization: pool mean/std computed once from the 6,000-patch
  pool, applied to all arms identically.

## Clients and heterogeneity (frozen)

- K = 4 simulated hospitals.
- Label skew: per class, client proportions drawn from
  Dirichlet(alpha = 0.5) with RandomState(42); indices shuffled per
  class with the same RNG before assignment. Per-client n and class
  counts are reported, not adjusted.
- Acquisition shift: each client applies ONE fixed, deterministic
  stain/scanner profile to all its training patches (white-balance
  channel gains, saturation scale, brightness shift, contrast
  scale; client 0 = identity reference). Profiles are constants in
  src/fedpath_bench/stain.py, documented as SIMULATION — PCam does
  not ship per-patch scanner metadata, so no real scanner claim is
  made.
- The test set is evaluated CLEAN (no profile) as the primary
  endpoint for every arm, so all arms face the same target. A
  secondary table reports centralized and FedAvg accuracy on the
  test set under each client's profile.

## Arms (frozen)

All arms use the SAME model (SmallCNN with GroupNorm — chosen so
FedAvg weight averaging is exact, with no BatchNorm statistics
aggregation artifact), SAME optimizer family (Adam, lr 1e-3),
batch size 64, cross-entropy, no augmentation, seed 42.

- **centralized**: pooled 6,000 (each patch carrying its client's
  stain profile), 15 epochs.
- **fedavg**: 15 communication rounds x 1 local epoch per client,
  sample-count-weighted averaging of full weights, all 4 clients
  participate every round. Total sample passes match centralized
  (15 x 6,000).
- **local-only**: one model per client, 15 epochs on that client's
  shard only. Reported per client and as the sample-weighted mean.

## Endpoint and readout criteria (frozen)

- Primary endpoint: accuracy (correct / 2,000) on the clean test
  subset for centralized, fedavg, and local-only (mean + per
  client).
- Pilot readout criteria (judged after the run, bars never moved):
  - R1: fedavg accuracy >= local-only mean accuracy + 3pp
    (collaboration helps under this heterogeneity).
  - R2: fedavg accuracy >= centralized accuracy - 10pp
    (federation cost is bounded at pilot scale).
- Whether R1/R2 pass or fail, the numbers are reported as measured.
  A fail is a finding about the pilot configuration, published
  as-is — it does not retroactively change the protocol.

## Determinism

- Seeds: 42 everywhere (torch, numpy). torch threads fixed at 2.
- SHA-256 of the client partition (sorted index lists) and of the
  results summary JSON are recorded in the results.
- Unit tests cover partition coverage/disjointness, stain
  determinism and identity profile, and exact FedAvg averaging.

## Compute

- CPU only (2 cores), $0. Dataset download via curl from Zenodo.
- Expected wall time: tens of minutes. Actual wall time reported.
