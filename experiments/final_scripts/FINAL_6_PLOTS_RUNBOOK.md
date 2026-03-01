# Final 6 Plots Runbook (Locked)

This runbook defines the exact commands for the final six figures in this minimal package.

## Global policy

- Exactly 6 plot scripts are used.
- No legacy training scripts are required for rerendering.
- Frozen `.npz`/`.csv` data snapshots are part of the package.

## Style baseline (single-panel plots)

Implemented directly inside each script:
- figure size: `243.12/72 x 185.52/72` inches (3.3767 x 2.5767)
- `font.size=12`
- `axes.labelsize=12`
- `xtick.labelsize=10`
- `ytick.labelsize=10`
- `legend.fontsize=7.2` (script-local override for some legends)
- `lines.linewidth=2.0`
- `axes.linewidth=1.2`
- `pdf.fonttype=42`, `ps.fonttype=42`

## Fig1

Script:
- `experiments/final_scripts/plot_target_sharpness_beta_sweep.py`

Command:
```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
```

Output:
- `outputs/final_plots/fig1_target_sharpness/fig1_target_sharpness_beta_sweep.pdf`

## Fig2

Script:
- `experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py`

Command:
```bash
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
```

Input snapshot:
- `outputs/final_plots/fig2_iqp_sigmak_ablation_recovery/fig2_data_default.npz`

Output:
- `outputs/final_plots/fig2_iqp_sigmak_ablation_recovery/fig2_iqp_sigmak_ablation_recovery.pdf`

## Fig3

Script:
- `experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py`

Default command (locked manuscript panel):
```bash
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
```

Equivalent explicit command:
```bash
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py \
  --points-csv outputs/final_plots/fig3_tv_bshs_seedmean_scatter/tv_bshs_points_multiseed_beta_q1000_no_iqp_mse_beta0p9_newseeds12.csv \
  --beta-fixed 0.90 \
  --beta-fixed-dual-axis-boxplot 1
```

Output:
- `outputs/final_plots/fig3_tv_bshs_seedmean_scatter/fig3_tv_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`

## Fig4

Script:
- `experiments/final_scripts/plot_visibility_mechanistic_recovery.py`

Command:
```bash
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
```

Input snapshot:
- `outputs/final_plots/fig4_visibility_mechanistic_recovery/fig4_data_default.npz`

Output:
- `outputs/final_plots/fig4_visibility_mechanistic_recovery/fig4_visibility_mechanistic_recovery.pdf`

## Fig5

Script:
- `experiments/final_scripts/plot_visibility_visible_invisible_recovery.py`

Command:
```bash
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
```

Input snapshot:
- `outputs/final_plots/fig5_visibility_visible_invisible_recovery/fig5_data_default.npz`

Output:
- `outputs/final_plots/fig5_visibility_visible_invisible_recovery/fig5_visibility_visible_invisible_recovery.pdf`

## Fig6

Script:
- `experiments/final_scripts/plot_beta_sweep_recovery_grid.py`

Command:
```bash
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
```

Input snapshot:
- `outputs/final_plots/fig6_beta_sweep_recovery_grid/fig6_data_default.npz`

Output:
- `outputs/final_plots/fig6_beta_sweep_recovery_grid/fig6_beta_sweep_recovery_grid.pdf`

## Kept files

- 6 scripts above
- outputs in `outputs/final_plots/`
- root docs: `README.md`, `REPRODUCIBILITY.md`, `PUBLISHING_CHECKLIST.md`
- style/settings lock docs in this folder
