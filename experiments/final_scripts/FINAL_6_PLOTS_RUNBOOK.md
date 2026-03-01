# Final 7 Plots Runbook (Locked)

This runbook defines the exact commands for the final seven figures in this minimal package.

## Global policy

- Exactly 7 plot scripts are used.
- No legacy training scripts are required for rerendering.
- Frozen `.npz`/`.csv` data snapshots are part of the package.

## Style baseline

Implemented directly inside each script:
- single-panel base size: `243.12/72 x 185.52/72` inches (3.3767 x 2.5767)
- `font.size=12`
- `axes.labelsize=12`
- `xtick.labelsize=10`
- `ytick.labelsize=10`
- `legend.fontsize=7.2` (script-local overrides where needed)
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

Default command:
```bash
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
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

## Fig7 (Appendix Ablation)

Script:
- `experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py`

Command:
```bash
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

Input snapshots:
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_data_default.npz`
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_seed_table.csv`

Output:
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_appendix_ablation_beta0p8_nsweep.pdf`
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_appendix_ablation_beta0p8_nsweep_qholdout_vs_n.pdf`
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_appendix_ablation_beta0p8_nsweep_rq10000_vs_n.pdf`

Frozen ablation configuration:
- `beta=0.8`
- `n in {12,14,16,18,20}`
- seeds `42..46` (5 seeds)
- models: IQP parity vs IQP MSE
- exact evaluation for `n<=14`
- shot-based evaluation for `n>=16` with `100000` shots

## Kept files

- 7 scripts above
- outputs in `outputs/final_plots/`
- root docs: `README.md`, `REPRODUCIBILITY.md`, `PUBLISHING_CHECKLIST.md`, `RESEARCH_EFFORT.md`
- style/settings lock docs in this folder
