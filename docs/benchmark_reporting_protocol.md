# Benchmark Reporting Protocol

This document defines the benchmark-side disclosure standard for the curated analysis reruns in this repository.
It is separate from the frozen figure rerendering contract in `docs/methodology_section_full.tex`.
The live auto-updated run ledger for benchmark-standard 20-seed experiments is written to `docs/paper_benchmark_ledger.md`.

## 1. Named Disclosure Blocks

The benchmark protocol is organized into these named disclosure blocks:

1. `Matched-instance definition`
2. `Seed schedule and randomness stack`
3. `Training budget fairness`
4. `Restart policy`
5. `Model hyperparameters`
6. `Transformer transparency`
7. `Capacity fairness`
8. `Aggregation and statistics`
9. `Benchmark constants`
10. `Robustness axes`
11. `Reproducibility package contents`

## 2. Matched-Instance Definition

The benchmark standard uses `20` matched seeds:

- matched seed IDs: `101, 102, ..., 120`
- beta grid: `0.1, 0.2, ..., 2.0`
- matched-instance index: `(beta, s)` with `s in {1, ..., 20}`
- total matched instances for the wide beta sweep: `20 betas x 20 seeds = 400`

Within each matched instance, all models receive the same `D_train`.
Parity-based models additionally receive the same parity band `Omega`.

The canonical seed schedule is committed in `docs/benchmark_seed_schedule_20seeds.csv`.

## 3. Seed Schedule and Randomness Stack

### Base seed list

- matched seed IDs: `101..120`
- canonical seed schedule file: `docs/benchmark_seed_schedule_20seeds.csv`

### Exact derived seed rules

- train-data sampling seed: `matched_seed + 7`
- parity-band sampling seed: `matched_seed + 222`
- IQP initialization seed: `matched_seed + 10000 + 7*K`
- Ising parity initialization seed: `matched_seed + 30001`
- dense Ising xent initialization seed: `matched_seed + 30004`
- Transformer initialization seed: `matched_seed + 35501`
- Transformer dataloader shuffle seed: `matched_seed + 35512`
- MaxEnt initialization seed: `matched_seed + 36001`
- restarts: none

### Holdout policy by rerun family

- `Fig6` multiseed reruns use a fixed holdout seed `46`, so the holdout mask is derived from `46 + 111 = 157` and is shared across matched seeds.
- `Fig3` KL-BSHS reruns derive the holdout mask from `matched_seed + 111`, so the holdout randomness changes with the matched seed but remains shared across models within that matched instance.

### Randomness order per run

For each matched instance `(beta, s)`:

1. Build `p*` on the even-parity support.
2. Construct the holdout mask according to the rerun-family holdout policy.
3. Remove the holdout mass to obtain `p_train`.
4. Sample `D_train` from `p_train`.
5. Sample parity band `Omega`.
6. Initialize each model with its own model-specific seed.
7. Train exactly one run per model with no restart selection stage.

## 4. Training Budget Fairness

The scripts disclose fixed optimizer budgets rather than wall-clock matching.
The comparison is update-budget based and not wall-clock normalized.

| Model | Optimizer | LR | Budget | Batch size | Early stopping | Max objective evaluations |
| --- | --- | --- | --- | --- | --- | --- |
| IQP parity QCBM | `qml.AdamOptimizer` | `0.05` | `600` steps | full objective | none | `600` |
| Ising+fields parity | `qml.AdamOptimizer` | `0.05` | `600` steps | full objective | none | `600` |
| Dense Ising+fields xent | `qml.AdamOptimizer` | `0.05` | `600` steps | full objective | none | `600` |
| AR Transformer MLE | `torch.optim.Adam` | `1e-3` | `600` epochs | `256` | none | `600` optimizer steps because `m=200 < batch_size` |
| MaxEnt parity | `torch.optim.Adam` | `5e-2` | `600` steps | full objective | none | `600` |

## 5. Restart Policy

The benchmark scripts currently use a single predetermined run per model per matched instance.

- number of restarts: `1`
- varied across restarts: `none`
- selection rule: `single_run_only`
- validation-based restart selection: `not used`

## 6. Model Hyperparameters

| Model | Parameterization / feature set | Parameter count | Other fixed settings |
| --- | --- | --- | --- |
| IQP parity QCBM | `ZZ`-only IQP circuit on ring `NN + NNN` pairs, `layers=1` | `24` trainable angles for `n=12` | parity-MSE objective, `eval_every=20` in Fig6 and `50` in Fig3 |
| Ising+fields parity | `NN + NNN` pairwise Ising features plus local fields | `24 + 12 = 36` | parity-MSE objective, no regularization |
| Dense Ising+fields xent | dense pairwise Ising features plus local fields | `66 + 12 = 78` | empirical cross-entropy objective, no regularization |
| AR Transformer MLE | big-endian bit ordering from `int2bits`, `d_model=64`, `layers=2`, `heads=4`, `dim_ff=128` | `67,969` | `dropout=0.0`, `weight_decay=0.0`, `batch_size=256` |
| MaxEnt parity | parity feature vector `theta in R^K` with `K=512` | `512` | log-partition objective, no regularization |

## 7. Transformer Transparency

The Transformer baseline is intentionally over-documented because it is the easiest target for an under-tuned-baseline criticism.

- bit ordering: big-endian integer-to-bit expansion from `int2bits`
- embedding vocabulary: `{0, 1, BOS}`
- `d_model = 64`
- heads: `4`
- layers: `2`
- feed-forward width: `128`
- activation: `GELU`
- dropout: `0.0`
- weight decay: `0.0`
- optimizer: `Adam`
- learning rate: `1e-3`
- epochs: `600`
- batch size: `256`
- early stopping: `none`
- restart policy: `single_run_only`
- final selected config: the script defaults above; there is no hidden post-hoc config swap inside the rerun drivers

## 8. Capacity Fairness

Capacities are not matched exactly across model families.
The comparison instead fixes:

- the same matched-instance data for all models
- the same parity band `Omega` for parity-based models
- the same nominal optimizer-update budget of `600`

Capacity disclosure:

- IQP parity QCBM: `24` trainable parameters
- Ising+fields parity: `36` trainable parameters
- Dense Ising+fields xent: `78` trainable parameters
- AR Transformer MLE: `67,969` trainable parameters
- MaxEnt parity: `512` trainable parameters

This is acceptable because the benchmark compares qualitatively different model classes; fairness is enforced through shared data, shared randomness, and fixed training budgets rather than equal parameter counts.

## 9. Aggregation and Statistics

### Per-instance-first aggregation rule

Metrics are computed per matched instance first and only then aggregated across instances.
There is no pooling over samples before metric evaluation.

Per-instance metrics exposed by the current rerun scripts include:

- `D_KL(p* || q)` for the Fig3 fixed-beta analysis
- `TV_score`
- `BSHS`
- `Composite = BSHS * (1 - TV_score)`
- `qH`
- `R_Q10000`
- `Q80`
- the full recovery curve `R_q(Q)` via `expected_unique_fraction`

### Summary conventions

- use `SD` when the goal is to show the actual spread across matched seeds / instances
- use `95% CI` when the goal is to quantify uncertainty of the estimated mean
- do not use `SE` as the default error bar, because it can visually understate seed-to-seed variability
- wide beta-sweep summaries: prefer `median + IQR` when reporting `Q80`
- mean-based summaries remain allowed when explicitly labeled `mean +/- std`
- Fig3 fixed-beta summaries report per-seed distributions and seed means

### Paired statistical protocol

For manuscript reporting, paired tests should be run on per-instance metrics:

- wide beta sweep comparisons: `N = 400`
- fixed-beta Fig3 comparisons: `N = 20`
- recommended tests: paired Wilcoxon signed-rank and Sign Test
- always report effect direction and `p`-value

### KL wins

`KL wins` counts matched `(beta, seed)` instances on which a model achieves the lowest forward KL among all compared models.
The count is over matched instances, not over beta-aggregated means.

## 10. Benchmark Constants

| Quantity | Value |
| --- | --- |
| `n` | `12` |
| support | even parity only |
| support size `|S|` | `2048` |
| wide beta sweep | `0.1..2.0` in steps of `0.1` |
| base multiseed beta sweep | `0.5..1.2` |
| matched seeds | `101..120` |
| train sample count `m` | `200` |
| holdout training mass proxy | `5000` |
| holdout size `k` | `20` |
| holdout candidate pool | `400` |
| parity-band parameters | `sigma = 1.0`, `K = 512` |
| IQP layers | `1` |
| elite threshold / good fraction | `0.05` (`top-5%`) |
| `Q_eval` for Fig3 | `1000` |
| `R_Q10000` evaluation budget | `10000` |
| `Q80` threshold | `0.8` |
| legacy frozen appendix pilot | `beta = 0.8`, `n in {12, 14, 16, 18}` |

## 11. Robustness Axes

Pre-specified robustness axes for benchmark reporting:

- sample-complexity sweep: `m in {50, 100, 200, 500, 1000}`
- elite-threshold sweep: `top-5%`, `top-10%`, `top-20`
- parity-band robustness over `beta`
- larger-`n` pilots

Current repository coverage:

- beta robustness is covered by the Fig6 rerun family
- larger-`n` pilot coverage appears in the frozen Fig7 appendix snapshot
- the sample-complexity and elite-threshold sweeps are protocol-level robustness axes that should be disclosed when added

## 12. Reproducibility Package Contents

The benchmark-facing reproducibility package should expose:

- raw per-instance metrics in machine-readable CSV form
- the canonical seed schedule in `docs/benchmark_seed_schedule_20seeds.csv`
- rerun drivers under `experiments/analysis/`
- frozen final figure rerender scripts under `experiments/final_scripts/`
- run configs in `RUN_CONFIG.json`
- rendered figures and kept `.npz`/`.csv`/`.json` intermediates

Current canonical paths:

- Fig3 per-instance metrics: `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/kl_bshs_points_multiseed_beta_q1000_beta0p90_newseeds20.csv`
- Fig6 per-instance metrics: `fig6_beta_sweep_recovery_grid_multiseed_metrics.csv` produced by `plot_fig6_beta_sweep_recovery_grid_multiseed.py`
- benchmark seed schedule: `docs/benchmark_seed_schedule_20seeds.csv`
- rerun commands: `REPRODUCIBILITY.md` and `experiments/analysis/README.md`

## 13. Legacy Frozen Artifacts

Some committed frozen artifacts predate the 20-seed benchmark standard and are kept as historical snapshots:

- frozen Fig3 final snapshot uses `12` seeds (`101..112`)
- frozen Fig7 appendix snapshot uses `5` seeds (`42..46`)
- committed Fig6 analysis artifacts under `outputs/analysis/fig6_multiseed_*_seeds42_46/` are legacy 5-seed reruns

These legacy snapshots are retained for artifact integrity.
They should not be cited as the benchmark-standard rerun configuration going forward.
