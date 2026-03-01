# Final Scripts Settings Lock

This document freezes the current defaults and visual settings for the seven final scripts.

## Common single-panel style (Fig1-Fig5)

Embedded in each respective script:
- figure size: `243.12/72 x 185.52/72` inch
- `font.size=12`
- `axes.labelsize=12`
- `xtick.labelsize=10`
- `ytick.labelsize=10`
- `legend.fontsize=7.2` (some script-local overrides)
- `lines.linewidth=2.0`
- `lines.markersize=6`
- `axes.linewidth=1.2`
- `xtick.major.width=1.0`, `ytick.major.width=1.0`
- `xtick.major.size=4`, `ytick.major.size=4`
- `pdf.fonttype=42`, `ps.fonttype=42`
- `savefig.pad_inches=0.03`

## Script locks

1. `plot_target_sharpness_beta_sweep.py`
- defaults:
  - `n=12`
  - `highlight-betas=0.6,0.8,1.0,1.2,1.4`
  - `bg-beta-min=0.1`, `bg-beta-max=2.0`, `bg-beta-step=0.1`
- output:
  - `fig1_target_sharpness_beta_sweep.pdf/.png`

2. `plot_iqp_sigmak_ablation_recovery.py`
- default input:
  - `fig2_data_default.npz`
- legend lock:
  - lower-right with inward anchor `bbox_to_anchor=(0.975, 0.055)`
  - compact legend fontsize `6.6`
- output:
  - `fig2_iqp_sigmak_ablation_recovery.pdf/.png`

3. `plot_tv_bshs_seedmean_scatter.py`
- default input:
  - `tv_bshs_points_multiseed_beta_q1000_no_iqp_mse_beta0p9_newseeds12.csv`
- default mode lock:
  - `beta-fixed=0.90`
  - `beta-fixed-dual-axis-boxplot=1`
- dual-axis legend lock:
  - upper-left, white background, labels only: `Support`, `TVscore`
- output:
  - `fig3_tv_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf/.png`

4. `plot_visibility_mechanistic_recovery.py`
- default input:
  - `fig4_data_default.npz`
- legend lock:
  - upper-right with `bbox_to_anchor=(1.0, 0.78)`
- output:
  - `fig4_visibility_mechanistic_recovery.pdf/.png`

5. `plot_visibility_visible_invisible_recovery.py`
- default input:
  - `fig5_data_default.npz`
- legend lock:
  - upper-right with `bbox_to_anchor=(1.0, 0.93)`
- x-axis label lock:
  - `Q samples from model`
- output:
  - `fig5_visibility_visible_invisible_recovery.pdf/.png`

6. `plot_beta_sweep_recovery_grid.py`
- default input:
  - `fig6_data_default.npz`
- panel layout lock:
  - `grid-cols=4` (2x4 panels)
- default range lock:
  - `qmax=10000`, `q80-thr=0.8`
- figure style lock (script-specific):
  - serif layout with compact panel typography
  - dynamic canvas from panel geometry (`panel_w=3.0`, `panel_h=2.18`)
- output:
  - `fig6_beta_sweep_recovery_grid.pdf/.png`

7. `plot_appendix_ablation_beta0p8_nsweep.py`
- default input:
  - `fig7_data_default.npz`
- data lock:
  - `beta=0.8`
  - `n={12,14,16,18,20}`
  - seeds `42..46` (5 seeds)
  - models: `iqp_parity`, `iqp_mse`
  - exact evaluation for `n<=14`
  - shot-based eval for `n>=16` with `100000` shots
- layout lock:
  - two-panel horizontal figure (`6.95 x 2.60` inch)
  - left panel `q(H)` vs `n`
  - right panel `R(10000)` vs `n`
  - summary lines shown as mean±std over seeds
- output:
  - `fig7_appendix_ablation_beta0p8_nsweep.pdf/.png`
  - `fig7_appendix_ablation_beta0p8_nsweep_qholdout_vs_n.pdf/.png`
  - `fig7_appendix_ablation_beta0p8_nsweep_rq10000_vs_n.pdf/.png`
