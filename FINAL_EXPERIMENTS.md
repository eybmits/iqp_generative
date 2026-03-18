# Final Experiments

This repository has been reduced to a minimal final plotting package.

Kept code in the repository root:

- `experiment_1_kl_diagnostics.py`
- `experiment_2_beta_kl_summary.py`
- `experiment_3_beta_quality_coverage.py`
- `experiment_4_recovery_sigmak_triplet.py`
- `experiment_5_kl_coverage_scatter.py`
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
- Saved artifacts in the output folder include PDFs plus `npz/csv/json` for rerendering.

## Experiment 2

- File: `experiment_2_beta_kl_summary.py`
- Purpose: beta-sweep summary of forward KL across the five main models
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
- Default view:
  - x-axis: `D_KL(p* || q)`
  - y-axis: `C_q(Q=1000)`
- Output:
  - one PDF scatter
  - one normalized scatter CSV
  - `RUN_CONFIG.json`
  - `RERENDER_CONFIG.json`

## Common Plot Style

- File: `final_plot_style.py`
- Purpose: common dimensions and styling conventions for the final figure family
- All final plots are intended to follow this common look.
