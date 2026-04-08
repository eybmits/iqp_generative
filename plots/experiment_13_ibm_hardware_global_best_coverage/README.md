# Experiment 13 Summary

IBM hardware sampling for the Experiment 12 global-best IQP parity setting versus IQP MSE.

Key settings:

- backend: `ibm_fez`
- account: `seedwise-open-plan`
- beta: `0.9`
- n: `12`
- train sample size: `200`
- layers: `1`
- global-best `(sigma, K)`: `(1, 512)`
- shots per circuit: `10000`
- seeds: `111,112,113,114,115,116,117,118,119,120`

Coverage definition:

- high-value states: top-10% of valid states by score
- unseen subset: elite states not present in the matched `D_train` for that seed
- metric: `C_q(Q) = Q^{-1} sum_x (1 - (1-q(x))^Q)` over elite unseen states

Artifacts:

- summary plot PDF: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage.pdf`
- summary plot PNG: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage.png`
- per-seed metrics CSV: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage_per_seed_metrics.csv`
- model summary CSV: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage_summary.csv`
- pairwise parity-vs-mse summary CSV: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage_pairwise_summary.csv`
- job rows CSV: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage_job_rows.csv`
- jobs JSON: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage_jobs.json`
- data NPZ: `plots/experiment_13_ibm_hardware_global_best_coverage/experiment_13_ibm_hardware_global_best_coverage_data.npz`
