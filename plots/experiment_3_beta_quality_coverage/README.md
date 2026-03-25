# Experiment 3: Beta-Quality-Coverage Summary

This directory contains the final beta-sweep quality-coverage summaries under the active 10-seed protocol.

Protocol:

- betas: `0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0`
- matched seeds: `111,112,113,114,115,116,117,118,119,120`
- seed count: `10` by default under the active standard protocol
- target distribution: even-parity score-tilted family
- train sample count: `m=200`
- parity band: `sigma=1`, `K=512`
- elite set: top `10.0%` of valid states by score
- unseen criterion: states not observed in the matched training sample multiset
- IQP parity budget: `steps=600`, `lr=0.05`
- Ising+fields budgets: `steps=600`, `lr=0.05`
- MaxEnt parity budget: `steps=600`, `lr=0.05`
- Transformer baseline: `variant=medium`, `d_model=32`, `layers=1`, `heads=4`, `dim_ff=64`, `epochs=600`, `lr=0.001`, `batch_size=256`

Coverage metric:

- for each budget $Q$, we report $C_q(Q)=Q^{-1}\sum_{x\in A_{\mathrm{elite}}\setminus D_{\mathrm{train}}}\left(1-(1-q(x))^Q\right)$
- line: seedwise mean quality coverage over the matched-seed pool
- band: interquartile range (Q1 to Q3)
- saved artifacts additionally include median, standard deviation, and 95% CI for each beta/model pair

Saved artifacts:

- per-seed metrics: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_metrics_per_seed.csv`
- saved data cube: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_data.npz`
- series Q=1000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q1000_series.csv`
- PDF Q=1000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q1000.pdf`
- pointcloud PDF Q=1000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q1000_pointcloud.pdf`
- series Q=2000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q2000_series.csv`
- PDF Q=2000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q2000.pdf`
- pointcloud PDF Q=2000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q2000_pointcloud.pdf`
- series Q=5000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q5000_series.csv`
- PDF Q=5000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q5000.pdf`
- pointcloud PDF Q=5000: `plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q5000_pointcloud.pdf`
- paper export alias (Q=1000 pointcloud): `plots/experiment_3_beta_quality_coverage/fig7_beta_quality_coverage_q1000.pdf`
- manifest: `experiment_3_beta_quality_coverage_manifest.csv`
- local protocol doc: `TRAINING_PROTOCOL.md`

- source driver: `experiment_3_beta_quality_coverage.py`
- outdir: `plots/experiment_3_beta_quality_coverage`
