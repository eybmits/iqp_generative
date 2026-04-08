# Experiment 13 Summary

IBM hardware sampling for the Experiment 12 seedwise-best IQP parity setting versus IQP MSE.

Key settings:

- backend: `ibm_marrakesh`
- account: `seedwise-open-plan`
- beta: `0.9`
- n: `12`
- train sample size: `200`
- layers: `1`
- parity selection: `seedwise-best oracle over (sigma, K) per seed`
- shots per circuit: `10000`
- seeds: `111,112,113,114,115,116,117,118,119,120`

Coverage definition:

- high-value states: top-10% of valid states by score
- unseen subset: elite states not present in the matched `D_train` for that seed
- metric: `C_q(Q) = Q^{-1} sum_x (1 - (1-q(x))^Q)` over elite unseen states

Artifacts:

- summary plot PDF: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage.pdf`
- summary plot PNG: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage.png`
- per-seed metrics CSV: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage_per_seed_metrics.csv`
- model summary CSV: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage_summary.csv`
- pairwise parity-vs-mse summary CSV: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage_pairwise_summary.csv`
- job rows CSV: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage_job_rows.csv`
- jobs JSON: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage_jobs.json`
- data NPZ: `plots/experiment_15_ibm_hardware_seedwise_best_coverage/experiment_15_ibm_hardware_seedwise_best_coverage_data.npz`
