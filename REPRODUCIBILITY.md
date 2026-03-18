# Reproducibility Guide

This repository separates two reproducibility modes:

- frozen final-figure re-renders from committed inputs in `outputs/final_plots/`
- curated analysis reruns in `outputs/analysis/` that recompute selected results with fixed run metadata

The benchmark-side rerun standard uses `20` matched seeds (`101..120`).
The full disclosure contract is documented in `docs/benchmark_reporting_protocol.md`, and the exact seed schedule is committed in `docs/benchmark_seed_schedule_20seeds.csv`.
The live paper-side benchmark ledger is generated at `docs/paper_benchmark_ledger.md` whenever the benchmark-standard 20-seed analysis scripts are run.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-analysis.txt
```

Documented local verification environment:

- macOS CPU
- Python `3.13.2`
- `numpy==2.3.3`
- `pandas==2.3.2`
- `matplotlib==3.10.6`
- `torch==2.10.0`
- `pennylane==0.42.3`

An explicit pinned Conda environment is also provided in [environment.yml](/Users/superposition/Coding/iqp_generative/environment.yml).

Optional deterministic hash seed:

```bash
export PYTHONHASHSEED=0
```

## 2. Re-render Frozen Final Figures

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

These scripts are deterministic re-renders from frozen inputs and should repopulate `outputs/final_plots/`.

## 3. Rebuild and Verify the Frozen Final Manifest

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md

python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## 4. Recompute the Curated Analysis Runs

### Fig2 recovery-summary companion figure

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig2_recovery_summary_panels.py \
  --outdir outputs/analysis/fig2_recovery_summary_panels
```

### Fig6 multiseed rerun for `beta = 0.5..1.2`

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

### Fig3 KL-BSHS dual-axis rerun

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py \
  --recompute 1 \
  --outdir outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600 \
  --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600
```

### Fig6 compact beta-vs-Q80 summary

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_q80_summary.py \
  --metrics-csv outputs/analysis/fig6_multiseed_all600_seeds101_120/fig6_beta_sweep_recovery_grid_multiseed_metrics.csv \
  --data-npz outputs/analysis/fig6_multiseed_all600_seeds101_120/fig6_beta_sweep_recovery_grid_multiseed_data.npz \
  --outdir outputs/analysis/fig6_beta_q80_summary
```

### Fig6 wide multiseed rerun for `beta = 0.1..2.0`

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

### Fig6 wide beta-vs-Q80 summary family

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

For the heavy Fig6 and Fig3 reruns, budget minutes to tens of minutes on CPU depending on local BLAS or Torch threading.

## 5. Rebuild and Verify the Curated Analysis Manifest

```bash
python tools/build_analysis_manifest.py \
  --root outputs/analysis \
  --output-csv outputs/analysis/ARTIFACT_MANIFEST.csv \
  --output-md outputs/analysis/ARTIFACT_MANIFEST.md

python tools/verify_analysis_manifest.py \
  --manifest outputs/analysis/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## 6. Notes

- The frozen final package rerenders deterministically from committed data files.
- The curated analysis reruns are recomputations and rely on fixed seeds and saved `RUN_CONFIG.json` files.
- The benchmark-standard reruns use `20` matched seeds (`101..120`); historical committed artifacts under `outputs/analysis/fig6_multiseed_*_seeds42_46/` remain legacy 5-seed snapshots.
- The frozen Fig3 and Fig7 final inputs remain historical `12`-seed and `5`-seed snapshots respectively; see `docs/benchmark_reporting_protocol.md` for the benchmark-standard disclosure.
- The Fig6 beta-vs-Q80 summaries are companion plots derived from stored Fig6 multiseed artifacts.
- For the wide `beta=0.1..2.0` summary family, `Median + IQR` is the recommended default because the `Q80` distribution is strongly heavy-tailed at large `beta`.
- Any new publishable analysis directory should include `README.md`, `RUN_CONFIG.json`, rendered outputs, and manifest coverage.
- Dated reruns, replots, and chunked scratch directories are intentionally excluded from the curated artifact policy.
