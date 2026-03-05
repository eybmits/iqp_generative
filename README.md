# IQP Generative Final-Plots Reproducible Package

This repository is curated as a reproducible final-figure package for the IQP generative study.

It includes:
- final paper plotting scripts (Fig1-Fig7)
- frozen final figure inputs and rendered artifacts under `outputs/final_plots/`
- deterministic manifest build/verify utilities for publication integrity

## Repository layout

- `experiments/final_scripts/`: final figure scripts (Fig1-Fig7)
- `outputs/final_plots/`: frozen final inputs and rendered PDF/PNG artifacts
- `tools/build_final_manifest.py`: deterministic final-manifest builder
- `tools/verify_final_manifest.py`: final-manifest verification utility

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild commands

Rerender final figures:

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

## Artifact integrity

Verify final artifacts against the manifest:

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1
```

Rebuild the final manifest after intentional artifact updates:

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md
```

## Citation links

- Main repository: `https://github.com/eybmits/iqp_generative`
- Frozen snapshot tag: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`
- Final artifacts folder: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`
