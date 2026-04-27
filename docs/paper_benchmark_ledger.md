# Paper Benchmark Ledger

_Auto-generated from `paper_benchmark_ledger.py`. Last updated: 2026-04-27T17:07:31Z_

This file is the central paper-side benchmark note for the current repository state.
It combines the static disclosure needed by the manuscript with the live registry of active benchmark-standard experiment runs.

## Draft Sync

- Current repository benchmark standard: `10` matched seeds `111..120`.
- Current matched-instance count for the wide beta sweep: `20 betas x 10 seeds = 200`.
- Historical 5-seed, 12-seed, and 20-seed snapshots are legacy artifacts and do not override the active final-reporting standard.
- The source of truth is the script plus each experiment-local `RUN_CONFIG.json` under `plots/`.

## Current Active Experiment Status

| Experiment | Paper Target | Status | Artifact |
| --- | --- | --- | --- |
| `experiment_1_kl_diagnostics` | KL diagnostic panels and fixed-beta comparison | available | `plots/experiment_1_kl_diagnostics` |
| `experiment_2_beta_kl_summary` | Full beta-sweep KL summary | available | `plots/experiment_2_beta_kl_summary` |
| `experiment_3_beta_quality_coverage` | Full beta-sweep quality coverage summary | available | `plots/experiment_3_beta_quality_coverage` |
| `experiment_4_recovery_sigmak_triplet` | Recovery curves over parity/spectral settings | available | `plots/experiment_4_recovery_sigmak_triplet` |
| `experiment_12_global_best_iqp_vs_mse` | Seedwise IQP-parity vs IQP-MSE global-best comparison | available | `plots/experiment_12_global_best_iqp_vs_mse` |
| `experiment_15_ibm_hardware_seedwise_best_coverage` | Matched real-hardware validation | available | `plots/experiment_15_ibm_hardware_seedwise_best_coverage` |
| `aligned_publication_figures` | Composite LaTeX-sized publication figures | available | `plots/aligned_kl_triptych` |

## Static Benchmark Disclosure

### Matched Instances

- Index: `(beta, s)`
- Beta values: `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2`
- Seed IDs: `111..120 (10 seeds)`
- Total matched instances in the wide sweep: `200`
- Shared train data within instance: `True`
- Shared parity band within instance: `True`
- Definition text: A matched instance is indexed by (beta, s), with beta in {0.1, 0.2, ..., 2.0} and s in {111, ..., 120}, yielding 200 matched instances in total.

### Randomness Stack

- sample D_train
- sample parity band Omega
- initialize each model once
- train without restarts unless explicitly disclosed

Derived seed formulas used in the current analysis drivers:
- `train_dataset_seed`: `s + 7`
- `parity_band_seed`: `s + 222`
- `fig3_holdout_seed`: `s + 111`
- `fig6_holdout_seed`: `holdout_seed + 111`
- `iqp_init_seed`: `s + 10000 + 7*K`
- `classical_nnn_fields_parity_init_seed`: `s + 30001`
- `classical_dense_fields_xent_init_seed`: `s + 30004`
- `transformer_init_seed`: `s + 35501`
- `transformer_dataloader_seed`: `s + 35512`
- `maxent_init_seed`: `s + 36001`
- `restart_policy`: `no restarts in the current analysis drivers`

### Training Budgets

| Model | Optimizer | LR | Budget | Early stopping | Batch size | Max objective evaluations |
| --- | --- | --- | --- | --- | --- | --- |
| IQP parity | PennyLane AdamOptimizer | 0.05 | 600 | none | full-distribution objective | 600 |
| Ising+fields (NN+NNN) | PennyLane AdamOptimizer | 0.05 | 600 | none | full-distribution objective | 600 |
| Dense Ising+fields | PennyLane AdamOptimizer | 0.05 | 600 | none | full-distribution objective | 600 |
| AR Transformer | torch.optim.Adam | 0.001 | 600 | none | 256 | 600 |
| MaxEnt parity | torch.optim.Adam | 0.05 | 600 | none | full-distribution objective | 600 |

### Model Hyperparameters

| Model | Architecture / features | Capacity | Key settings |
| --- | --- | --- | --- |
| IQP parity / IQP-MSE / IQP-MLE | 1-layer IQP ZZ circuit on cyclic NN+NNN pairs | 24 parameters | lr=0.05, steps=600, restart=none |
| Ising+fields (NN+NNN) | NN+NNN Ising pair products plus local fields | 36 features | solver=first-order gradient descent via PennyLane AdamOptimizer, search_budget=600, regularization=none |
| Dense Ising+fields | dense Ising pair products plus local fields | 78 features | solver=first-order gradient descent via PennyLane AdamOptimizer, search_budget=600, regularization=none |
| MaxEnt parity | parity-band sufficient statistics | 512 parameters | solver=first-order gradient descent via torch.optim.Adam, search_budget=600 |
| AR Transformer | autoregressive MLE on big-endian bit ordering | 9057 parameters | d_model=32, layers=1, heads=4, dropout=0.0, weight_decay=0.0, lr=0.001, epochs=600 |

### Transformer Transparency

- The AR Transformer is intentionally over-documented because it is the easiest baseline to challenge as under-tuned.
- Bit ordering: `most-significant to least-significant bit`
- d_model: `32`
- layers: `1`
- heads: `4`
- dropout: `0.0`
- weight decay: `0.0`
- learning rate: `0.001`
- epochs: `600`
- batch size: `256`
- early stopping: `none`
- final selected config: `fixed single config in script defaults`

### Restarts, Statistics, and Package Contents

- `iqp_parity_mse`: `single initialization, no restart sweep`
- `classical_nnn_fields_parity`: `single initialization, no restart sweep`
- `classical_dense_fields_xent`: `single initialization, no restart sweep`
- `classical_transformer_mle`: `single initialization, no restart sweep`
- `classical_maxent_parity`: `single initialization, no restart sweep`
- `selection_rule`: `single trained run per matched instance`
- Paired tests: `paired Wilcoxon signed-rank, Sign test`
- Paired sample size: `200`
- Sweep summaries: `mean ± 95% CI, median + IQR`
- Capacity note: Capacities are disclosed rather than force-matched; all models share the same n, matched-instance data, and fixed training-budget protocol.
- Robustness axes: `sample_complexity_m=50, 100, 200, 500, 1000`, `elite_thresholds=top-5%, top-10%, top-20%`, `larger_n_pilots=14, 16, 18`
- Package contents: `raw per-instance metrics, seed lists, scripts to regenerate manuscript tables and figures, configs`

## Latest Registered Benchmark-Standard Runs

No active benchmark-standard runs have been registered yet by the auto-logger.

## Run History

No runs recorded yet.
