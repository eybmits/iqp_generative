# Reproducibility Guide

This document defines a deterministic rebuild and verification workflow for the final-paper figure package (Fig1-Fig7).

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

## 2) Rerender final figures (Fig1-Fig7)

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

## 3) Rebuild final artifact manifest

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md
```

## 4) Verify final artifact manifest

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## 5) Expected outputs

- Final figures and frozen inputs: `outputs/final_plots/...`
- Final manifest files:
  - `outputs/final_plots/ARTIFACT_MANIFEST.csv`
  - `outputs/final_plots/ARTIFACT_MANIFEST.md`

## 6) Scope note

This release is intentionally scoped to final plotting reproducibility.
Training/evaluation claim stacks and extended baseline experiments are not part of this package.
