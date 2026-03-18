# Final Scripts Settings Lock

This document records the publish-time policy for the standalone scripts in `experiments/final_scripts/`.

## Lock Policy

- the seven scripts in this folder are the canonical frozen figure entry points
- each script owns its default output directory under `outputs/final_plots/`
- figure appearance, default file names, and committed input snapshots are treated as frozen publication artifacts
- if a committed final output changes intentionally, rebuild `outputs/final_plots/ARTIFACT_MANIFEST.csv` and `outputs/final_plots/ARTIFACT_MANIFEST.md`

## Script Inventory

### 1. `plot_target_sharpness_beta_sweep.py`

- produces `outputs/final_plots/fig1_target_sharpness/`
- deterministic script-owned render

### 2. `plot_iqp_sigmak_ablation_recovery.py`

- reads `outputs/final_plots/fig2_iqp_sigmak_ablation_recovery/fig2_data_default.npz`
- produces `outputs/final_plots/fig2_iqp_sigmak_ablation_recovery/`

### 3. `plot_tv_bshs_seedmean_scatter.py`

- reads the committed Fig3 CSV snapshot in `outputs/final_plots/fig3_tv_bshs_seedmean_scatter/`
- produces the frozen dual-axis boxplot in the same directory
- the committed input snapshot is a historical `12`-seed artifact and is intentionally not rewritten to the benchmark-standard 20-seed schedule inside the frozen package

### 4. `plot_visibility_mechanistic_recovery.py`

- reads `outputs/final_plots/fig4_visibility_mechanistic_recovery/fig4_data_default.npz`
- produces `outputs/final_plots/fig4_visibility_mechanistic_recovery/`

### 5. `plot_visibility_visible_invisible_recovery.py`

- reads `outputs/final_plots/fig5_visibility_visible_invisible_recovery/fig5_data_default.npz`
- produces `outputs/final_plots/fig5_visibility_visible_invisible_recovery/`

### 6. `plot_beta_sweep_recovery_grid.py`

- reads `outputs/final_plots/fig6_beta_sweep_recovery_grid/fig6_data_default.npz`
- produces `outputs/final_plots/fig6_beta_sweep_recovery_grid/`

### 7. `plot_appendix_ablation_beta0p8_nsweep.py`

- reads `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_data_default.npz`
- produces `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/`
- the committed appendix snapshot remains a historical `5`-seed artifact; benchmark-standard 20-seed reruns belong in `experiments/analysis/`, not in the frozen final package

## Change Rules

- do not add shared helper code that changes the visual contract of the frozen scripts without updating this lock document
- do not replace committed default input snapshots casually
- do not commit local scratch renders into `outputs/final_plots/`
