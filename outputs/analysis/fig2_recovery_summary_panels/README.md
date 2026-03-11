# Fig2 Recovery Summary Panels

This directory contains a companion summary view for the frozen Fig2 sigma-K ablation.

Inputs:

- frozen snapshot: `outputs/final_plots/fig2_iqp_sigmak_ablation_recovery/fig2_data_default.npz`
- no retraining; all metrics are derived from the stored recovery curves

Primary summary metric:

- `Q80`: first interpolated `Q` where `R(Q) >= 0.8`
- `AUC_norm`: normalized area under the recovery curve on `[0, 10000]`
- `R10000`: terminal recovery at `Q=10000`

Main visual:

- left panel: heatmap over the 12 parity settings in the frozen sigma-K grid
- heatmap color: `Delta Q80 vs IQP MSE = Q80(MSE) - Q80(Parity)`
- positive values mean parity reaches 80% recovery earlier than IQP MSE
- `IQP MSE` is the zero-reference baseline for `Delta Q80`
- cell text: signed `Delta Q80 vs IQP MSE`, not absolute parity `Q80`

Right benchmark panel:

- `Target p*`, `Best IQP Parity`, `IQP MSE`, `Uniform`
- x-axis is `Q80` in samples; lower is better
- inline labels report `AUC` and `R10000`

Headline result:

- best parity setting: `sigma=1, K=512`
- best parity `Q80 = 1375.88`
- IQP MSE `Q80 = 2103.78`
- best parity advantage vs IQP MSE: `727.89` fewer samples to reach 80% recovery
- target `Q80 = 1114.06`
- uniform `Q80 = 6591.52`

Kept files:

- `fig2_recovery_summary_panels.pdf`
- `fig2_recovery_summary_panels.png`
- `fig2_recovery_summary_panels_heatmap_only.pdf`
- `fig2_recovery_summary_panels_heatmap_only.png`
- `fig2_recovery_summary_panels_benchmark_only.pdf`
- `fig2_recovery_summary_panels_benchmark_only.png`
- `fig2_recovery_summary_panels_metrics.csv`
- `RUN_CONFIG.json`
