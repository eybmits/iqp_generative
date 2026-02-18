# Evaluation Manifest v1 (Pre-Registered Protocol)

This document fixes the Nature-impact evaluation protocol for the `paper_even` family.
All main claims and figures under the `nature_comms_v1` profile must use this protocol.

## 1) Scope

- Dataset family: `paper_even` only.
- Task type: in-distribution unseen-state discovery under fixed target-family support.
- Primary objective: discovery quality, not generic global fit.

## 2) Fixed Evaluation Matrix

- `holdout_mode`: `global`, `high_value`
- `train_m`: `200`, `1000`, `5000`
- `beta`: `0.6` to `1.4` in `0.1` increments
- `seed`: `42`, `43`, `44`, `45`, `46`
- IQP parity settings:
  - global: `sigma=1`, `K=512`
  - high_value: `sigma=2`, `K=256`

## 3) Fixed Metrics

Primary:
- `Q80` (lower is better)

Secondary:
- `AUC_R_0_10000` (area under recovery curve on `Q in [0,10000]`)
- `q(H)`
- `q(H)/q_unif(H)`
- `R_Q1000`, `R_Q10000`
- `fit_tv_to_pstar` (diagnostic, not primary discovery metric)

## 4) Statistical Procedure

- Paired permutation test, `n_perm=10000`.
- Bootstrap CI (95%), `n_boot=10000`.
- Multiple-testing correction: Holm.
- Effect size: Cliff's delta.

All p-values must be reported as:
- raw p-value
- Holm-corrected p-value

## 5) Fairness Rules

- Same target distribution, same holdout, same train sample count per seed.
- Same training budget class for compared models (steps/epochs disclosed).
- Any hyperparameter tuning must use inner train/validation only.
- Holdout remains untouched until final evaluation.

## 6) Loss-Ablation Rule

For IQP-only loss ablation:
- Compare `parity_mse` vs `mmd` vs `xent` on identical model/circuit setup.
- `mmd` uses Hamming-RBF kernel with `tau in {1,2,4}` selected by inner validation.
- For published main figures, report selected tau and selection criterion.

## 7) Reporting Minimum

Every claim in the main text must include:
- metric summary (mean, std, median, IQR)
- CI
- statistical significance result
- effect size
- explicit statement of scope (in-distribution, paper_even)

## 8) Artifact Expectations

Main artifact root:
- `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/`

Statistical tables:
- `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/99_stats_tables/main_table.csv`
- `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/99_stats_tables/supp_table.csv`
- `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/99_stats_tables/significance_report.md`

