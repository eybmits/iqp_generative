# Transformer Capacity Ablation at `beta = 0.9`

This note records the fixed-`beta` capacity ablation for the autoregressive Transformer baseline against `IQP parity`.

## Protocol

- target slice: `beta = 0.9`
- system size: `n = 12`
- seeds: `101, 102, 103, 104, 105`
- train set size: `m = 200`
- validation set size: `m_val = 1000`, sampled independently from `p_train`
- holdout mode: `global`
- shared randomness per seed: same `D_train`, same parity band `Omega`, same validation sample across all Transformer sizes
- budgets: `IQP steps = 300`, `Transformer epochs = 300`

## Transformer Sizes

| Variant | Parameters | Configuration |
| --- | ---: | --- |
| `tiny` | `729` | `d_model=8`, `layers=1`, `heads=2`, `dim_ff=16` |
| `small` | `2,481` | `d_model=16`, `layers=1`, `heads=2`, `dim_ff=32` |
| `medium` | `9,057` | `d_model=32`, `layers=1`, `heads=4`, `dim_ff=64` |
| `large` | `67,969` | `d_model=64`, `layers=2`, `heads=4`, `dim_ff=128` |

The `IQP parity` reference uses `24` trainable parameters.

## Results

| Model | Mean val NLL | Mean test KL |
| --- | ---: | ---: |
| `IQP parity` | `6.669` | `0.741` |
| Transformer `tiny` | `6.996` | `0.983` |
| Transformer `small` | `6.950` | `1.055` |
| Transformer `medium` | `6.937` | `1.120` |
| Transformer `large` | `7.094` | `1.416` |

Key observations:

- `IQP parity` is best overall on both reported means.
- The best Transformer by mean test KL is the smallest tested model, `tiny` (`729` parameters).
- The best Transformer by mean validation NLL is `medium` (`9,057` parameters), but it still has worse test KL than `tiny`.
- The Transformer KL curve shows no interior minimum in this pilot. Increasing capacity does not recover the baseline; the largest model is worst.
- On a seedwise comparison, `IQP parity` beats the best Transformer on `4/5` seeds.

## Artifact Paths

- plots: `outputs/analysis/transformer_capacity_ablation_beta0p90_quick/transformer_capacity_ablation_fixed_beta_beta0p90.pdf`
- per-seed points: `outputs/analysis/transformer_capacity_ablation_beta0p90_quick/transformer_capacity_ablation_fixed_beta_points_beta0p90.csv`
- summaries: `outputs/analysis/transformer_capacity_ablation_beta0p90_quick/transformer_capacity_ablation_fixed_beta_summary_beta0p90.csv`
- run config: `outputs/analysis/transformer_capacity_ablation_beta0p90_quick/RUN_CONFIG.json`
