# KILL-GATE — fedpath-bench (checked BEFORE any build, 2026-10-28)

Question: does a maintained federated PatchCamelyon (PCam) benchmark
with client splits + baselines already exist? If yes, STOP.

## What was checked

1. **FLamby** (Ogier du Terrail et al., NeurIPS 2022 Datasets &
   Benchmarks; maintained, github.com/owkin/FLamby). Its histology
   dataset is **Fed-Camelyon16** — built from Camelyon16 whole-slide
   images, split naturally by hospital (2 clients), with FedAvg /
   FedProx / FedOpt / Cyclic baselines. It is NOT PatchCamelyon:
   different patch extraction, no PCam official train/val/test
   splits, only 2 clients, no simulated acquisition heterogeneity.
2. **Fed-ISIC2019** (also FLamby): dermoscopy, not histopathology
   patches; different modality entirely.
3. **LEAF** (Caldas et al.): FEMNIST / CelebA / Reddit / Sent140 /
   Shakespeare / synthetic. No PCam, no histopathology.
4. **Flower / flwr-datasets and TFF examples**: generic partitioners
   (Dirichlet etc.) and tutorial examples; no maintained,
   PCam-specific benchmark with pathology splits and a baseline
   table. Generic partitioners do not model stain/scanner shift.
5. **One-off paper experiments**: e.g. a 2026 Research Square
   blockchain-FL framework runs FedAvg/FedProx on PCam with a
   Dirichlet (alpha=0.8) split, K=5 — a single paper's experiment
   section, not a reusable benchmark: no released benchmark harness,
   no stain/scanner simulation, no per-client acquisition profiles,
   EfficientNet-B0 at 224px (GPU-scale), numbers not a maintained
   baseline table.
6. **GitHub search** ("patchcamelyon federated", "fedpcam",
   PCam + FedAvg): only single-model PCam classifiers and the
   one-off experiments above. No benchmark repo found.

## Verdict: PASS (build), on this differentiation

- **Dataset**: PCam proper (Veeling et al. 2018), official splits —
  the patch benchmark the field actually reports on — not
  Camelyon16 slides (FLamby) and not ISIC (Fed-ISIC).
- **Pathology-specific non-IID**: each simulated hospital gets BOTH
  label skew (Dirichlet) AND a deterministic H&E stain / scanner
  acquisition profile (white-balance gains, saturation, brightness,
  contrast). FLamby's Camelyon16 split is natural-hospital only;
  generic Dirichlet suites have no acquisition shift at all. In real
  computational pathology the dominant cross-site shift is stain and
  scanner, not just label mix — that is the gap this benchmark
  measures.
- **Form factor**: CPU-runnable in minutes, one command, fixed
  seeds, preregistered arms (centralized / FedAvg / local-only),
  per-client reporting. FLamby is a heavyweight multi-dataset suite;
  nothing maintained offers a small, reproducible PCam FL pilot.
- **Honest scope**: this is a pilot on a subset of PCam (see
  PREREG.md). It does not claim FLamby-scale coverage, and Fed-
  Camelyon16 is cited as the nearest prior art in the README.

If any of the above is later shown wrong (a maintained federated
PCam benchmark with acquisition-shift splits surfaces), this
project's claim narrows to the stain/scanner simulation harness.
