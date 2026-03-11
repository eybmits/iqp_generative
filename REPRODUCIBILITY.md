# Reproducibility Guide

This repository contains:
- the frozen final plotting package in `experiments/final_scripts/` and `outputs/final_plots/`
- documented recomputed analysis reruns in `experiments/analysis/` and `outputs/analysis/`

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional deterministic hash seed:

```bash
export PYTHONHASHSEED=0
```

Additional dependencies for the documented analysis reruns:

```bash
pip install -r requirements-analysis.txt
```

## 2) Rerender frozen final figures

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

## 3) Rebuild frozen final artifact manifest

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md
```

## 4) Verify frozen final artifact manifest

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## 5) Recompute the documented Fig2 recovery-summary companion figure

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig2_recovery_summary_panels.py
```

The current script defaults reproduce:

- `outputs/analysis/fig2_recovery_summary_panels/fig2_recovery_summary_panels.pdf`

Exact run metadata is stored in:

- `outputs/analysis/fig2_recovery_summary_panels/RUN_CONFIG.json`

## 6) Recompute the documented Fig6 multiseed rerun

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py \
  --q80-search-max 1000000000
```

The documented base rerun command produces:

- `outputs/analysis/fig6_multiseed_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed.pdf`

Exact run metadata is stored in:

- `outputs/analysis/fig6_multiseed_all600_seeds42_46/RUN_CONFIG.json`

The kept metrics were then extended for four remaining `classical_maxent_parity` rows with a targeted post-pass at `Qmax = 1e18`; this is documented in the output directory metadata.

## 7) Recompute the documented Fig3 KL-BSHS rerun

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py
```

The current script defaults reproduce:

- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`

The per-seed values and run metadata are stored in:

- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/kl_bshs_points_multiseed_beta_q1000_beta0p90_newseeds20.csv`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/kl_bshs_summary_multiseed_beta_q1000_beta0p90_newseeds20.json`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/RUN_CONFIG.json`

## 8) Recompute the documented Fig6 beta-vs-Q80 summary

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_q80_summary.py
```

The current script defaults reproduce:

- `outputs/analysis/fig6_beta_q80_summary/fig6_beta_q80_summary.pdf`

The aggregated summary values and run metadata are stored in:

- `outputs/analysis/fig6_beta_q80_summary/fig6_beta_q80_summary_metrics.csv`
- `outputs/analysis/fig6_beta_q80_summary/RUN_CONFIG.json`

## 9) Recompute the wide Fig6 multiseed rerun

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py \
  --recompute 1 \
  --betas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0 \
  --q80-search-max 1000000000000000000 \
  --seeds 42,43,44,45,46 \
  --holdout-seed 46 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600 \
  --outdir outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46
```

The documented wide rerun produces:

- `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed.pdf`

Exact run metadata is stored in:

- `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/RUN_CONFIG.json`

## 10) Recompute the wide Fig6 beta-vs-Q80 summary family

Recommended robust summary:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_q80_summary.py \
  --metrics-csv outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed_metrics.csv \
  --data-npz outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed_data.npz \
  --outdir outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr \
  --band-stat iqr \
  --show-seed-traces 0 \
  --show-band 1
```

Companion variants:

- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
  rendered with `--band-stat iqr --show-seed-traces 1 --show-band 1`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/`
  rendered with `--band-stat mean_std --show-seed-traces 0 --show-band 1`

The recommended output is:

- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/fig6_beta_q80_summary.pdf`

## 11) Verify the documented analysis artifacts

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/analysis/ARTIFACT_MANIFEST.csv \
  --root outputs/analysis \
  --strict 1
```

## 12) Notes

- The frozen 7-figure package rerenders deterministically from frozen final data.
- The Fig6 multiseed rerun and the Fig3 KL-BSHS rerun are recomputations, not frozen-data rerenders.
- The Fig6 beta-vs-Q80 summaries are companion plots derived from stored Fig6 multiseed artifacts.
- For the wide `beta=0.1..2.0` summary family, `Median + IQR` is the recommended default because the `Q80` distribution is strongly heavy-tailed at large `beta`.
- All documented analysis scripts are implemented as standalone scripts and do not require `git show` or repo-history lookups at runtime.
