# Analysis Scripts

Kept analysis drivers:

- `plot_fig2_recovery_summary_panels.py`
- `plot_fig6_beta_sweep_recovery_grid_multiseed.py`
- `plot_fig3_kl_bshs_dual_axis_boxplot.py`

Both scripts contain their training logic locally and only reuse frozen final plotting assets where that is appropriate for style consistency.

Prerequisites for rerunning these analyses:

- install `requirements-analysis.txt`

Selected analysis runs:

1. Fig2 recovery-summary companion figure

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig2_recovery_summary_panels.py \
  --outdir outputs/analysis/fig2_recovery_summary_panels
```

2. Fig6 multiseed recovery rerun

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py \
  --recompute 1 \
  --seeds 42,43,44,45,46 \
  --holdout-seed 46 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600 \
  --outdir outputs/analysis/fig6_multiseed_all600_seeds42_46
```

3. Fig3 KL-BSHS dual-axis boxplot

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py \
  --recompute 1 \
  --outdir outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600 \
  --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600
```

Archived outputs live in:

- `outputs/analysis/fig2_recovery_summary_panels/`
- `outputs/analysis/fig6_multiseed_all600_seeds42_46/`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/`
