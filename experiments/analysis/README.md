# Analysis Scripts

This folder contains the self-contained rerun drivers for the curated post-freeze analysis set.
The benchmark-standard reruns in this folder use `20` matched seeds (`101..120`); see `docs/benchmark_reporting_protocol.md`.

Kept analysis drivers:

- `plot_fig2_recovery_summary_panels.py`
- `plot_fig3_kl_bshs_dual_axis_boxplot.py`
- `plot_fig6_beta_q80_summary.py`
- `plot_fig6_beta_sweep_recovery_grid_multiseed.py`

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
  --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --holdout-seed 46 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600 \
  --outdir outputs/analysis/fig6_multiseed_all600_seeds101_120
```

### 3. Fig3 KL-BSHS dual-axis boxplot

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py \
  --recompute 1 \
  --outdir outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600 \
  --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600
```

### 4. Fig6 compact beta-vs-Q80 summary

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_q80_summary.py \
  --metrics-csv outputs/analysis/fig6_multiseed_all600_seeds101_120/fig6_beta_sweep_recovery_grid_multiseed_metrics.csv \
  --data-npz outputs/analysis/fig6_multiseed_all600_seeds101_120/fig6_beta_sweep_recovery_grid_multiseed_data.npz \
  --outdir outputs/analysis/fig6_beta_q80_summary
```

### 5. Fig6 wide multiseed recovery rerun for `beta = 0.1..2.0`

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py \
  --recompute 1 \
  --betas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0 \
  --q80-search-max 1000000000000000000 \
  --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --holdout-seed 46 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600 \
  --outdir outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_120
```

### 6. Fig6 wide beta-vs-Q80 summaries

Recommended robust summary:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_q80_summary.py \
  --metrics-csv outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_120/fig6_beta_sweep_recovery_grid_multiseed_metrics.csv \
  --data-npz outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_120/fig6_beta_sweep_recovery_grid_multiseed_data.npz \
  --outdir outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr \
  --band-stat iqr \
  --show-seed-traces 0 \
  --show-band 1
```

Companion variants:

- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/`

## Runtime Notes

- Fig2 is lightweight and should finish quickly.
- Fig6 and Fig3 are heavier recomputations and should be treated as CPU jobs rather than instant re-renders.
- The wide `beta = 0.1..2.0` Fig6 sweep is materially heavier than the base `beta = 0.5..1.2` Fig6 sweep.
- Curated outputs are deterministic at the level of saved seeds and run metadata, but absolute wall time depends on the local numeric stack.

## Curated Output Directories

- benchmark-standard reruns are expected to land in new `...seeds101_120/` output directories
- the already committed `...seeds42_46/` Fig6 directories are legacy 5-seed snapshots kept for artifact continuity
- `outputs/analysis/fig2_recovery_summary_panels/`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/`
- `outputs/analysis/fig6_beta_q80_summary/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/`
- `outputs/analysis/fig6_multiseed_all600_seeds42_46/`
- `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/`
