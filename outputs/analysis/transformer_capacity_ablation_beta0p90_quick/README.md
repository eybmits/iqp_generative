# Transformer Capacity Ablation

This directory contains the fixed-beta Transformer capacity sweep against the IQP parity baseline.

Protocol:

- fixed beta: `0.9`
- n: `12`
- seeds: `101,102,103,104,105`
- train sample size: `m=200`
- validation sample size: `m_val=1000` sampled independently from `p_train`
- holdout mode: `global`
- same `D_train`, parity band, and validation sample are reused across Transformer sizes within each seed
- reported metrics: validation NLL and exact forward KL

Transformer sizes:

- `tiny`: `d_model=8`, `layers=1`, `heads=2`, `dim_ff=16`
- `small`: `d_model=16`, `layers=1`, `heads=2`, `dim_ff=32`
- `medium`: `d_model=32`, `layers=1`, `heads=4`, `dim_ff=64`
- `large`: `d_model=64`, `layers=2`, `heads=4`, `dim_ff=128`

Headline results:

- `IQP parity` is best overall on both reported means: `val NLL = 6.669` and `test KL = 0.741` at `24` parameters
- best Transformer by mean test KL: `tiny` with `729` parameters (`d_model=8`, `layers=1`, `heads=2`, `dim_ff=16`), `test KL = 0.983`
- best Transformer by mean validation NLL: `medium` with `9,057` parameters, `val NLL = 6.937`
- the KL curve shows no interior minimum in this pilot: the lowest mean Transformer KL is already the smallest tested model, and the `67,969`-parameter model is worst
- seedwise comparison: `IQP parity` beats the best Transformer on `4/5` seeds

Artifacts are stored under `outputs/analysis/transformer_capacity_ablation_beta0p90_quick`.
