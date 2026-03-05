# Reproducibility Guide

This document defines a deterministic rebuild and verification workflow for both final-paper figures and claim-level comparison artifacts.

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For deterministic Python hashing in local runs:

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

## 3) Reproduce D3PM vs IQP pilot (n=12, beta=0.9)

Default script settings are scientific-run settings:
- `seeds=101..112`
- `train_m=200`
- `holdout_protocol=global_smart`
- `holdout_k=20`, `holdout_pool=400`, `holdout_m=5000`

Run:

```bash
python experiments/d3pm/eval_d3pm_vs_iqp_beta0p9_n12.py
python experiments/d3pm/plot_d3pm_vs_iqp_beta0p9_n12.py
python experiments/d3pm/plot_d3pm_beta_ablation_study.py
```

## 4) Reproduce Transformer vs IQP pilot (n=12)

Default script settings are strong-baseline settings:
- `seeds=101..112`
- `train_m=200`, `val_m=200`
- `holdout_protocol=global_smart`
- architecture `d_model=192`, `n_layers=8`, `n_heads=6`

Run:

```bash
python experiments/transformer/eval_transformer_vs_iqp_n12.py
python experiments/transformer/plot_transformer_vs_iqp_ablation.py
```

## 5) Verify artifact manifests

```bash
python scripts/verify_artifacts.py \
  outputs/final_plots/ARTIFACT_MANIFEST.csv \
  outputs/claims/ARTIFACT_MANIFEST.csv
```

If artifacts were intentionally rebuilt, regenerate manifest(s) first:

```bash
python scripts/build_artifact_manifest.py outputs/claims
python scripts/build_artifact_manifest.py outputs/final_plots --output outputs/final_plots/ARTIFACT_MANIFEST.csv
```

## 6) Expected outputs

- Final figures: `outputs/final_plots/...`
- D3PM artifacts: `outputs/claims/d3pm_*`
- Transformer artifacts: `outputs/claims/transformer_*`
- Aggregate claim manifest: `outputs/claims/ARTIFACT_MANIFEST.csv`

## 7) Practical runtime note

The evaluation scripts in `experiments/d3pm/` and `experiments/transformer/` can be compute-heavy on CPU; GPU (`cuda` or `mps`) is supported via `--device`.
