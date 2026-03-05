# Final Scripts

This folder contains exactly 7 standalone final plotting scripts.

Each script uses:
- local Matplotlib style (no shared helper module)
- fixed default output folder in `outputs/final_plots/...`
- deterministic rendering from frozen final data (where applicable)

## Scripts and outputs

1. `plot_target_sharpness_beta_sweep.py`
- Produces:
  - `fig1_target_sharpness_beta_sweep.pdf`
  - `fig1_target_sharpness_beta_sweep.png`

2. `plot_iqp_sigmak_ablation_recovery.py`
- Input snapshot:
  - `fig2_data_default.npz`
- Produces:
  - `fig2_iqp_sigmak_ablation_recovery.pdf`
  - `fig2_iqp_sigmak_ablation_recovery.png`

3. `plot_tv_bshs_seedmean_scatter.py`
- Default input snapshot:
  - `tv_bshs_points_multiseed_beta_q1000_no_iqp_mse_beta0p9_newseeds12.csv`
- Default output mode:
  - dual-axis boxplot for `beta=0.90`
- Produces:
  - `fig3_tv_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`
  - `fig3_tv_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.png`

4. `plot_visibility_mechanistic_recovery.py`
- Input snapshot:
  - `fig4_data_default.npz`
- Produces:
  - `fig4_visibility_mechanistic_recovery.pdf`
  - `fig4_visibility_mechanistic_recovery.png`

5. `plot_visibility_visible_invisible_recovery.py`
- Input snapshot:
  - `fig5_data_default.npz`
- Produces:
  - `fig5_visibility_visible_invisible_recovery.pdf`
  - `fig5_visibility_visible_invisible_recovery.png`

6. `plot_beta_sweep_recovery_grid.py`
- Input snapshot:
  - `fig6_data_default.npz`
- Produces:
  - `fig6_beta_sweep_recovery_grid.pdf`
  - `fig6_beta_sweep_recovery_grid.png`

7. `plot_appendix_ablation_beta0p8_nsweep.py`
- Input snapshot:
  - `fig7_data_default.npz`
- Frozen setup:
  - `beta=0.8`
  - `n in {12,14,16,18}`
  - seeds `42..46` (5 seeds)
  - `iqp_steps=300` for parity and mse
- Optional classical baseline overlay:
  - pass `--include-classical-baselines 1 --baseline-csv <path>`
  - CSV schema: `n,model_key,q_holdout,R_Q10000[,seed]`
- Produces:
  - `fig7_appendix_ablation_beta0p8_nsweep.pdf`
  - `fig7_appendix_ablation_beta0p8_nsweep.png`
  - `fig7_appendix_ablation_beta0p8_nsweep_qholdout_vs_n.pdf`
  - `fig7_appendix_ablation_beta0p8_nsweep_qholdout_vs_n.png`
  - `fig7_appendix_ablation_beta0p8_nsweep_rq10000_vs_n.pdf`
  - `fig7_appendix_ablation_beta0p8_nsweep_rq10000_vs_n.png`
