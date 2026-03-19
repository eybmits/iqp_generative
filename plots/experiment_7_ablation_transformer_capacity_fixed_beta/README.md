# Transformer Capacity Ablation

This directory contains the fixed-beta Transformer capacity sweep against the IQP parity baseline.

Protocol:

- fixed beta: `0.9`
- n: `12`
- seeds: `111,112,113,114,115,116,117,118,119,120`
- train sample size: `m=200`
- validation sample size: `m_val=1000` sampled independently from `p_train`
- holdout mode: `global`
- same `D_train`, parity band, and validation sample are reused across Transformer sizes within each seed
- reported metrics: validation NLL and exact forward KL
- training protocol file: `TRAINING_PROTOCOL.md`

Transformer sizes:

- `tiny`: `d_model=8`, `layers=1`, `heads=2`, `dim_ff=16`
- `small`: `d_model=16`, `layers=1`, `heads=2`, `dim_ff=32`
- `medium`: `d_model=32`, `layers=1`, `heads=4`, `dim_ff=64`
- `large`: `d_model=64`, `layers=2`, `heads=4`, `dim_ff=128`

Headline results:

- `IQP parity` is best overall on both reported means: `val NLL = 7.495` and `test KL = 0.980` at `24` parameters
- best Transformer by mean test KL: `tiny` with `729` parameters (`d_model=8`, `layers=1`, `heads=2`, `dim_ff=16`), `test KL = 2.167`
- best Transformer by mean validation NLL: `tiny` with `729` parameters, `val NLL = 7.698`
- the KL curve shows no interior minimum in this pilot: the lowest mean Transformer KL is already the smallest tested model, and the `67,969`-parameter model is worst
- seedwise comparison: `IQP parity` beats the best Transformer on `10/10` seeds

Artifacts are stored under `plots/experiment_7_ablation_transformer_capacity_fixed_beta`.
