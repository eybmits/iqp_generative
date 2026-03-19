# Experiment 2: Beta-KL Summary

This directory contains the final beta-sweep KL summary used for the main reporting protocol.

Protocol:

- betas: `0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0`
- matched seeds: `111,112,113,114,115,116,117,118,119,120`
- seed count: `10` by default under the active standard protocol
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
- saved artifacts additionally include mean, standard deviation, and 95% CI for each beta/model pair

Saved artifacts:

- per-seed KL metrics: `plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary_metrics_per_seed.csv`
- aggregated beta series: `plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary_series.csv`
- saved data cube: `plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary_data.npz`
- final PDF: `plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary.pdf`
- local protocol doc: `TRAINING_PROTOCOL.md`

- source driver: `experiment_2_beta_kl_summary.py`
- outdir: `plots/experiment_2_beta_kl_summary`
