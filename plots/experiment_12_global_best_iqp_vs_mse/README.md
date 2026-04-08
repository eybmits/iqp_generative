# Experiment 12 Summary

Global-best IQP parity grid selection versus IQP MSE over the active 10 matched seeds.

Key settings:

- beta: `0.9`
- n: `12`
- train sample size: `200`
- layers: `1`
- IQP budget: `steps=600`, `lr=0.05`
- selection metric: `mean`
- matched seeds: `111,112,113,114,115,116,117,118,119,120`

Selected global-best parity grid point:

- best `(sigma, K)`: `(1, 512)`
- parity mean KL: `0.451550`
- parity median KL: `0.446872`
- parity 95% CI: `0.031920`

Parity vs IQP MSE at the selected grid point:

- parity mean KL: `0.451550`
- MSE mean KL: `0.481647`
- parity median KL: `0.446872`
- MSE median KL: `0.495066`
- mean delta `(parity - mse)`: `-0.030097`
- seed wins `(parity < mse)`: `6/10`

Artifacts:

- summary plot PDF: `plots/experiment_12_global_best_iqp_vs_mse/experiment_12_global_best_iqp_vs_mse.pdf`
- summary plot PNG: `plots/experiment_12_global_best_iqp_vs_mse/experiment_12_global_best_iqp_vs_mse.png`
- grid metrics per seed CSV: `plots/experiment_12_global_best_iqp_vs_mse/experiment_12_global_best_iqp_vs_mse_grid_metrics_per_seed.csv`
- grid summary CSV: `plots/experiment_12_global_best_iqp_vs_mse/experiment_12_global_best_iqp_vs_mse_grid_summary.csv`
- global-best vs MSE CSV: `plots/experiment_12_global_best_iqp_vs_mse/experiment_12_global_best_iqp_vs_mse_global_best_vs_mse_per_seed.csv`
- data NPZ: `plots/experiment_12_global_best_iqp_vs_mse/experiment_12_global_best_iqp_vs_mse_data.npz`
