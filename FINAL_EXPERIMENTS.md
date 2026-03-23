# Final Experiments

This repository has been reduced to a minimal final plotting package.

Final reporting convention:

- all statistical summary figures in the final package are intended to use `10` matched seeds
- the canonical seed protocol is documented in `STANDARD_TRAINING_PROTOCOL.md`
- fixed-case illustration panels are the only explicit exception

Kept code in the repository root:

- `experiment_1_kl_diagnostics.py`
- `experiment_2_beta_kl_summary.py`
- `experiment_3_beta_quality_coverage.py`
- `experiment_4_recovery_sigmak_triplet.py`
- `experiment_5_kl_coverage_scatter.py`
- `experiment_6_ablation_n_sweep_all_baselines.py`
- `experiment_7_ablation_transformer_capacity_fixed_beta.py`
- `final_plot_style.py`

Kept plot folders:

- `plots/experiment_1_kl_diagnostics`
- `plots/experiment_2_beta_kl_summary`
- `plots/experiment_3_beta_quality_coverage`
- `plots/experiment_4_recovery_sigmak_triplet`
- `plots/experiment_5_kl_coverage_scatter`

## Experiment 1

- File: `experiment_1_kl_diagnostics.py`
- Purpose: fixed-`beta` KL diagnostics with
  - `(a)` sigma-K heatmap
  - `(b)` sigma-K rank ordering
  - `(c)` seedwise-best parity-in-grid vs `IQP MSE`
- Protocol:
  - panels `(a)` and `(b)`: `beta=0.9`, one fixed seed, 12 parity configurations
  - panel `(c)`: 10 matched seeds, seedwise best parity KL over the full sigma-K grid vs `IQP MSE`
- Visual standard:
  - the approved Experiment 1 panel aesthetics are locked in `experiment_1_kl_diagnostics.py`
  - the concrete rendering notes are documented in `plots/experiment_1_kl_diagnostics/README.md`
- Saved artifacts in the output folder include PDFs plus `npz/csv/json` for rerendering.

## Experiment 2

- File: `experiment_2_beta_kl_summary.py`
- Purpose: beta-sweep summary of forward KL across the five main models
- Final reporting protocol:
  - use `10` matched seeds
  - report seed-aggregated spread consistently with the standard training protocol
  - fix the Transformer baseline to the `medium` capacity selected in Experiment 7
  - plot `median` with an `IQR` band; store `mean` and `95% CI` in the saved artifacts
- Default source:
  - `plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary_series.csv`
- Output:
  - one PDF panel
  - one normalized series CSV
  - `RUN_CONFIG.json`
  - `RERENDER_CONFIG.json`

## Experiment 3

- File: `experiment_3_beta_quality_coverage.py`
- Purpose: beta-sweep quality-coverage plots for three budgets
- Final reporting protocol:
  - use `10` matched seeds
  - keep the same matched-seed pool across all reported budgets
  - fix the Transformer baseline to the `medium` capacity selected in Experiment 7
  - plot `mean` with an `IQR` band; store `median` and `95% CI` in the saved artifacts
- Budgets:
  - `Q=1000`
  - `Q=2000`
  - `Q=5000`
- Output:
  - three PDF panels
  - one series CSV per budget
  - `RUN_CONFIG.json`
  - `RERENDER_CONFIG.json`

## Experiment 4

- File: `experiment_4_recovery_sigmak_triplet.py`
- Purpose: three recovery panels from the saved sigma-K single-beta NPZ
- Status:
  - fixed-case illustration
  - not a 10-seed statistical summary figure
- Panels:
  - best IQP vs matched spectral
  - parity family vs `IQP MSE`
  - spectral family only
- Output:
  - three PDF panels
  - `RUN_CONFIG.json`
  - `RERENDER_CONFIG.json`

## Experiment 5

- File: `experiment_5_kl_coverage_scatter.py`
- Purpose: KL-vs-coverage scatter over fixed-`beta` slices
- Final reporting protocol:
  - use `10` matched seeds
  - aggregate each fixed-`beta` slice over the same matched-seed pool before plotting the model means
- Default view:
  - x-axis: `D_KL(p* || q)`
  - y-axis: `C_q(Q=1000)`
- Output:
  - one PDF scatter
  - one normalized scatter CSV
  - `RUN_CONFIG.json`
  - `RERENDER_CONFIG.json`

## Experiment 6

- File: `experiment_6_ablation_n_sweep_all_baselines.py`
- Purpose: fixed-`beta` ablation study over system size `n` against all classical baselines
- Final reporting protocol:
  - use `10` matched seeds at each reported `n`
  - keep the matched-seed comparison across all baselines at fixed `beta`
- Intended role:
  - scaling-style ablation / size sweep
  - IQP parity versus classical baselines on a matched fixed-`beta` slice

## Experiment 7

- File: `experiment_7_ablation_transformer_capacity_fixed_beta.py`
- Purpose: fixed-`beta` Transformer capacity ablation against IQP parity
- Final reporting protocol:
  - use `10` matched seeds for each reported Transformer capacity point
  - compare against IQP parity on the same matched data instances
- Intended role:
  - capacity ablation
  - validation NLL and forward KL as a function of Transformer size

## Common Plot Style

- File: `final_plot_style.py`
- Purpose: common dimensions and styling conventions for the final figure family
- All final plots are intended to follow this common look.
