# Fig6 Beta Q80 Summary

This directory contains a compact companion summary for the documented Fig6 multiseed rerun.

Inputs:

- per-seed metrics: `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed_metrics.csv`
- multiseed recovery curves: `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed_data.npz`
- style snapshot: `outputs/final_plots/fig6_beta_sweep_recovery_grid/fig6_data_default.npz`
- no retraining; all values are derived from the stored multiseed Fig6 artifacts

Primary summary metric:

- `Q80`: first interpolated `Q` where `R(Q) >= 0.8`
- x-axis: target sharpness parameter `beta`
- y-axis: absolute `Q80` on a log scale; lower is better

Visual encoding:

- one line per model using the same labels, colors, and line styles as Fig6
- thin background lines show individual seed-specific `Q80(beta)` traces
- the thick foreground line shows the seed-median and the shaded band shows the IQR
- the companion metrics CSV contains `q25/q50/q75/mean/std` for the same runs
- `Uniform` is shown as a deterministic reference line without a seed band
- `Target` is intentionally omitted to keep the plot compact

Kept files:

- `fig6_beta_q80_summary.pdf`
- `fig6_beta_q80_summary.png`
- `fig6_beta_q80_summary_metrics.csv`
- `README.md`
- `RUN_CONFIG.json`
