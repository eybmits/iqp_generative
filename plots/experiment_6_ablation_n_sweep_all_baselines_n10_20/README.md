# Experiment 6: Fixed-Beta n-Sweep

This directory contains the final fixed-beta n-sweep KL summary over the full baseline set.

Protocol:

- fixed beta: `0.9`
- n values: `10,11,12,13,14,15,16,17,18,19,20`
- matched seeds: `111,112,113,114,115,116,117,118,119,120`
- seed count: `10` under the active standard protocol
- target distribution: even-parity score-tilted family
- train sample count: `m=200`
- parity band: `sigma=1`, `K=512`
- IQP parity budget: `steps=600`, `lr=0.05`
- Ising+fields budgets: `steps=600`, `lr=0.05`
- MaxEnt parity budget: `steps=600`, `lr=0.05`
- Transformer baseline: `variant=medium`, `d_model=32`, `layers=1`, `heads=4`, `dim_ff=64`, `epochs=600`, `lr=0.001`, `batch_size=256`

Plot semantics:

- line: seedwise median KL over the matched-seed pool
- band: interquartile range (Q1 to Q3)
- saved artifacts additionally include mean, standard deviation, and 95% CI for each n/model pair
- rerun the script with a new `--n-values` slice (for example `22`) and keep `--append-existing 1` to merge the new n-values into the existing metrics/series/npz artifacts before rerendering

Saved artifacts:

- per-seed KL metrics: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines_metrics_per_seed.csv`
- aggregated n-series: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines_series.csv`
- saved data cube: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines_data.npz`
- final PDF: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines.pdf`
- final PNG: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines.png`
- pointcloud PDF: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines_pointcloud.pdf`
- pointcloud PNG: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines_pointcloud.png`
- compact LaTeX table: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20/experiment_6_ablation_n_sweep_all_baselines_compact_table.tex`
- local protocol doc: `TRAINING_PROTOCOL.md`

- source driver: `experiment_6_ablation_n_sweep_all_baselines.py`
- outdir: `plots/experiment_6_ablation_n_sweep_all_baselines_n10_20`
