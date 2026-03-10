# Fig3 KL-BSHS 20 Seeds All600

This directory contains the selected Fig3-style dual-axis boxplot with:

- left axis: `Support BSHS(Q)`
- right axis: forward KL `D_KL(p* || q)`

Selected rerun command:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py \
  --recompute 1 \
  --outdir outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600 \
  --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600
```

Fixed protocol:

- `beta = 0.9`
- `n = 12`
- seeds `101..120` (20 seeds)
- `holdout_mode = global`
- `holdout_k = 20`
- `holdout_pool = 400`
- `holdout_m_train = 5000`
- `train_m = 200`
- `sigma = 1.0`
- `K = 512`
- `q_eval = 1000`
- `iqp_steps = 600`
- `artr_epochs = 600`
- `maxent_steps = 600`
- KL variant: forward KL `D_KL(p* || q)` in natural-log units

Interpretation notes:

- The holdout is regenerated per seed using the same global smart-holdout protocol.
- The selected output is the dual-axis boxplot `fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`.
- The point CSV stores the per-seed model values that feed the boxplot.

Kept files:

- `fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`
- `fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.png`
- `kl_bshs_points_multiseed_beta_q1000_beta0p90_newseeds20.csv`
- `kl_bshs_summary_multiseed_beta_q1000_beta0p90_newseeds20.json`
- `RUN_CONFIG.json`
