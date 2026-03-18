# Analysis Scripts

This folder contains the self-contained rerun drivers for the curated post-freeze analysis set.
The active analysis standard in this folder uses `10` matched seeds (`101..110`) and a shared training budget of `600`; see `experiments/analysis/STANDARD_TRAINING_PROTOCOL.md`.

Kept analysis drivers:

- `plot_fig2_recovery_summary_panels.py`
- `plot_fig3_kl_bshs_dual_axis_boxplot.py`
- `plot_experiment6_fixed_beta_nsweep_all_baselines.py`
- `plot_fig6_beta_q80_summary.py`
- `plot_fig6_beta_sweep_recovery_grid_multiseed.py`
- `plot_transformer_capacity_ablation_fixed_beta.py`

These analysis scripts are self-contained and only reuse frozen final plotting assets where that is appropriate for style consistency.

## Environment

Install:

```bash
pip install -r requirements-analysis.txt
```

Documented local verification environment:

- macOS CPU
- Python `3.13.2`

## Curated Rerun Commands

### 1. Fig2 recovery-summary companion figure

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig2_recovery_summary_panels.py \
  --outdir outputs/analysis/fig2_recovery_summary_panels
```

### 2. Fig6 multiseed recovery rerun for `beta = 0.5..1.2`

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py \
  --recompute 1 \
  --q80-search-max 1000000000 \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --holdout-seed 46 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600 \
  --outdir outputs/analysis/fig6_multiseed_all600_seeds101_110
```

### 3. Fig3 KL-BSHS dual-axis boxplot

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py \
  --recompute 1 \
  --outdir outputs/analysis/fig3_kl_bshs_seedmean_scatter_10seeds_all600 \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600
```

### 4. Fig6 compact beta-vs-Q80 summary

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_q80_summary.py \
  --metrics-csv outputs/analysis/fig6_multiseed_all600_seeds101_110/fig6_beta_sweep_recovery_grid_multiseed_metrics.csv \
  --data-npz outputs/analysis/fig6_multiseed_all600_seeds101_110/fig6_beta_sweep_recovery_grid_multiseed_data.npz \
  --outdir outputs/analysis/fig6_beta_q80_summary
```

### 5. Fig6 wide multiseed recovery rerun for `beta = 0.1..2.0`

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py \
  --recompute 1 \
  --betas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0 \
  --q80-search-max 1000000000000000000 \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --holdout-seed 46 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600 \
  --outdir outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_110
```

### 6. Fig6 wide beta-vs-Q80 summaries

Recommended robust summary:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_q80_summary.py \
  --metrics-csv outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_110/fig6_beta_sweep_recovery_grid_multiseed_metrics.csv \
  --data-npz outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_110/fig6_beta_sweep_recovery_grid_multiseed_data.npz \
  --outdir outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr \
  --band-stat iqr \
  --show-seed-traces 0 \
  --show-band 1
```

Companion variants:

- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/`

### 7. Experiment 6 fixed-beta n-sweep scaling study

This driver evaluates a fixed `beta` slice across larger `n` using the shared active protocol.
It compares `IQP parity` against the classical baselines and the paper-facing plot should use the KL-only output.

Representative command:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_experiment6_fixed_beta_nsweep_all_baselines.py \
  --beta 0.9 \
  --n-values 8,10,12,14,16,18 \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --holdout-mode global \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600
```

### 8. Transformer capacity ablation at fixed beta

This quick study sweeps four Transformer sizes against `IQP parity` at fixed `beta`.
It reports validation NLL on an independent sample from `p_train` and exact test KL on `p*`.

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_transformer_capacity_ablation_fixed_beta.py \
  --beta 0.9 \
  --n 12 \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --train-m 200 \
  --val-m 1000 \
  --holdout-mode global \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --outdir outputs/analysis/transformer_capacity_ablation_beta0p90
```

## Runtime Notes

- Fig2 is lightweight and should finish quickly.
- Fig6 and Fig3 are heavier recomputations and should be treated as CPU jobs rather than instant re-renders.
- The wide `beta = 0.1..2.0` Fig6 sweep is materially heavier than the base `beta = 0.5..1.2` Fig6 sweep.
- The Experiment 6 `n`-sweep is a scaling pilot and can become expensive quickly for `n >= 16`; start with a short seed list before running the full slice.
- The Transformer capacity ablation is moderate in cost at `n=12`, but the `large` configuration dominates runtime; use `OMP_NUM_THREADS=1` if the local BLAS stack oversubscribes CPU threads.
- Curated outputs are deterministic at the level of saved seeds and run metadata, but absolute wall time depends on the local numeric stack.

## Curated Output Directories

- active-standard reruns are expected to land in new `...seeds101_110/` output directories
- the already committed `...seeds42_46/` Fig6 directories are legacy 5-seed snapshots kept for artifact continuity
- each trained experiment directory should contain its own `TRAINING_PROTOCOL.md`
- `outputs/analysis/fig2_recovery_summary_panels/`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_10seeds_all600/`
- `outputs/analysis/experiment6_fixed_beta0p90_nsweep_all_baselines/`
- `outputs/analysis/fig6_beta_q80_summary/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/`
- `outputs/analysis/transformer_capacity_ablation_beta0p90/`
- `outputs/analysis/fig6_multiseed_all600_seeds101_110/`
- `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_110/`
