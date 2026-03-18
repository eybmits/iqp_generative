# Experiment 6 Fixed-Beta n-Sweep

This directory contains the fixed-beta n-sweep scaling study for IQP parity versus the classical baselines.

Scope:

- fixed beta: `0.9`
- n values: `8,9,10,11,12`
- seeds: `101,102,103,104,105`
- model set: `IQP parity`, `Ising+fields (NN+NNN)`, `Dense Ising+fields (xent)`, `AR Transformer (MLE)`, `MaxEnt parity`
- this driver is a scaling/robustness study and is not benchmark-standard unless explicitly rerun with the benchmark seed set

Fixed protocol:

- holdout mode: `global`
- holdout seed: `46`
- train sample count: `m=200`
- parity band: `sigma=1`, `K=512`
- budgets: `iqp_steps=300`, `artr_epochs=300`, `maxent_steps=300`
- selected Transformer config: `d_model=32`, `layers=1`, `heads=4`, `dim_ff=64`

Saved artifacts:

- data bundle: `outputs/analysis/experiment6_beta0p90_nsweep_n8_12_medium_transformer_quick/experiment6_fixed_beta_nsweep_all_baselines_data.npz`
- per-seed metrics: `outputs/analysis/experiment6_beta0p90_nsweep_n8_12_medium_transformer_quick/experiment6_fixed_beta_nsweep_all_baselines_metrics.csv`
- per-n summary: `outputs/analysis/experiment6_beta0p90_nsweep_n8_12_medium_transformer_quick/experiment6_fixed_beta_nsweep_all_baselines_summary.csv`
- overview figure: `experiment6_fixed_beta_nsweep_all_baselines.pdf` / `experiment6_fixed_beta_nsweep_all_baselines.png`
- KL-only figure: `experiment6_fixed_beta_nsweep_all_baselines_kl_vs_n.pdf` / `experiment6_fixed_beta_nsweep_all_baselines_kl_vs_n.png`
- support figures: `experiment6_fixed_beta_nsweep_all_baselines_qholdout_vs_n.pdf`, `experiment6_fixed_beta_nsweep_all_baselines_q80_vs_n.pdf`

Primary metrics:

- `KL_pstar_to_q`: exact forward KL `D_KL(p* || q)`
- `TV`: total variation distance between `p*` and `q`
- `qH`: model mass assigned to the fixed holdout set
- `Q80`: first `Q` such that holdout recovery reaches `0.8`
- `R_Q10000`: holdout recovery at `Q=10000`

Plot semantics:

- overview left: mean±std forward KL versus `n`
- overview right: mean±std `R(10000)` versus `n`
- support plots show `q(H)` and `Q80` versus `n` with optional seed points
