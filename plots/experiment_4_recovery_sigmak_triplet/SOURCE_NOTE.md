# Experiment 4 Source Note

This directory now contains both the rendered Experiment 4 PDFs and a local
regenerated NPZ payload:

- `experiment_4_recovery_sigmak_triplet.py`
- `coverage_sigmak_triplet_data.npz`

The payload was rebuilt from the current fixed-beta helpers at:

- `beta = 0.9`
- `seed = 45`
- `n = 12`
- `train_m = 200`
- `sigma in {0.5, 1, 2, 3}`
- `K in {128, 256, 512}`

Selection rule:

- `best spectral` is defined explicitly as the spectral setting with maximal
  recovery `R(Q)` at `Q = 1000`
