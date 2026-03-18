# Paper Benchmark Ledger

_Auto-generated from `experiments/analysis/paper_benchmark_ledger.py`. Last updated: 2026-03-18T11:39:29Z_

This file is the central paper-side benchmark note for the current repository state.
It combines the static disclosure needed by the manuscript with the live registry of benchmark-standard 20-seed experiment runs.

## Draft Sync

- Current repository benchmark standard: `20` matched seeds `101..120`.
- Current matched-instance count for the wide beta sweep: `20 betas x 20 seeds = 400`.
- Any paper draft that still says `10` seeds or `200` matched instances is out of sync with the current repo standard.
- Frozen final artifacts under `outputs/final_plots/` remain historical snapshots and are not the benchmark-standard source of truth.

## Current 20-Seed Experiment Status

| Experiment | Paper Target | Status | Artifact |
| --- | --- | --- | --- |
| `fig2_fixed_beta_sigmak_kl_20seed` | Fig. 3 / Table III fixed-beta sigma-K KL study | planned | `No dedicated 20-seed analysis driver exists yet.` |
| `fig3_fixed_beta_kl_bshs_20seed` | Fig. 4 fixed-beta KL-BSHS boxplot at beta = 0.9 | available | `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600` |
| `fig6_base_multiseed_20seed` | Base beta sweep recovery grid (beta = 0.5..1.2) | missing | `outputs/analysis/fig6_multiseed_all600_seeds101_120` |
| `fig6_base_q80_summary_20seed` | Base beta-vs-Q80 summary | legacy/out-of-sync | `outputs/analysis/fig6_beta_q80_summary/RUN_CONFIG.json` |
| `fig6_wide_multiseed_20seed` | Wide beta sweep recovery grid (beta = 0.1..2.0) | missing | `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_120` |
| `fig6_wide_q80_summary_iqr_20seed` | Wide beta-vs-Q80 summary (median + IQR) | legacy/out-of-sync | `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/RUN_CONFIG.json` |
| `fig6_wide_q80_summary_iqr_seed_traces_20seed` | Wide beta-vs-Q80 summary with seed traces | legacy/out-of-sync | `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/RUN_CONFIG.json` |
| `fig6_wide_q80_summary_mean_std_20seed` | Wide beta-vs-Q80 summary (mean +/- std) | legacy/out-of-sync | `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/RUN_CONFIG.json` |

## Static Benchmark Disclosure

### Matched Instances

- Index: `(beta, s)`
- Beta values: `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2`
- Seed IDs: `101..120 (20 seeds)`
- Total matched instances in the wide sweep: `400`
- Shared train data within instance: `True`
- Shared parity band within instance: `True`
- Definition text: A matched instance is indexed by (beta, s), with beta in {0.1, 0.2, ..., 2.0} and s in {1, ..., 20}, yielding 400 matched instances in total.

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
| AR Transformer | autoregressive MLE on big-endian bit ordering | 67969 parameters | d_model=64, layers=2, heads=4, dropout=0.0, weight_decay=0.0, lr=0.001, epochs=600 |

### Transformer Transparency

- The AR Transformer is intentionally over-documented because it is the easiest baseline to challenge as under-tuned.
- Bit ordering: `most-significant to least-significant bit`
- d_model: `64`
- layers: `2`
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
- Paired sample size: `400`
- Sweep summaries: `mean ± 95% CI, median + IQR`
- Capacity note: Capacities are disclosed rather than force-matched; all models share the same n, matched-instance data, and fixed training-budget protocol.
- Robustness axes: `sample_complexity_m=50, 100, 200, 500, 1000`, `elite_thresholds=top-5%, top-10%, top-20%`, `larger_n_pilots=14, 16, 18`
- Package contents: `raw per-instance metrics, seed lists, scripts to regenerate Table II and all figures, configs`

## Latest Registered Benchmark-Standard Runs

### `fig3_fixed_beta_kl_bshs_20seed`

- Title: Fig4 fixed-beta KL-BSHS benchmark at beta = 0.9 (registered from existing artifact)
- Recorded at: `2026-03-18T11:39:29Z`
- Run config: `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/RUN_CONFIG.json`
- Outdir: `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600`
- Seeds: `101..120 (20 seeds)`
- Betas: `0.9`
- Key config:
  - `K`: `512`
  - `artr_epochs`: `600`
  - `beta`: `0.9`
  - `holdout_k`: `20`
  - `holdout_m_train`: `5000`
  - `holdout_mode`: `global`
  - `holdout_pool`: `400`
  - `iqp_steps`: `600`
  - `maxent_steps`: `600`
  - `n`: `12`
  - `q_eval`: `1000`
  - `sigma`: `1`
  - `train_m`: `200`
- Outputs:
  - `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`
  - `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.png`
- Metrics:
  - `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/kl_bshs_points_multiseed_beta_q1000_beta0p90_newseeds20.csv`
  - `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/kl_bshs_summary_multiseed_beta_q1000_beta0p90_newseeds20.json`
- Notes:
  - Backfilled from the already committed 20-seed artifact directory.
  - This establishes the initial ledger entry before the next fresh rerun.

## Run History

1. `2026-03-18T11:39:29Z` | `fig3_fixed_beta_kl_bshs_20seed` | `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600` | seeds `101..120 (20 seeds)`
