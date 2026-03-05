# IQP Generative Reproducible Research Repository

This repository is curated as a reproducible research package for the IQP generative study.

It includes:
- final paper plotting scripts and frozen figure data
- extended D3PM and Transformer comparison experiments
- claim-level artifacts under `outputs/claims/`
- deterministic artifact manifests with SHA256 verification

## Repository layout

- `experiments/final_scripts/`: final figure scripts (Fig1-Fig7)
- `experiments/d3pm/`: D3PM evaluation and plotting workflows
- `experiments/transformer/`: Transformer baseline evaluation and ablation plots
- `outputs/final_plots/`: frozen final-figure data and rendered artifacts
- `outputs/claims/`: claim-level outputs and auxiliary comparison artifacts
- `scripts/build_artifact_manifest.py`: deterministic manifest builder
- `scripts/verify_artifacts.py`: manifest verification utility

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild commands

Final figures:

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

D3PM pilot + ablation plotting:

```bash
python experiments/d3pm/eval_d3pm_vs_iqp_beta0p9_n12.py
python experiments/d3pm/plot_d3pm_vs_iqp_beta0p9_n12.py
python experiments/d3pm/plot_d3pm_beta_ablation_study.py
```

Transformer pilot + ablation plotting:

```bash
python experiments/transformer/eval_transformer_vs_iqp_n12.py
python experiments/transformer/plot_transformer_vs_iqp_ablation.py
```

## Artifact integrity

Verify both final and claim artifacts:

```bash
python scripts/verify_artifacts.py \
  outputs/final_plots/ARTIFACT_MANIFEST.csv \
  outputs/claims/ARTIFACT_MANIFEST.csv
```

Regenerate a manifest (if artifacts are intentionally rebuilt):

```bash
python scripts/build_artifact_manifest.py outputs/claims
```

## Citation links

- Main repository: `https://github.com/eybmits/iqp_generative`
- Frozen snapshot tag: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`
- Final artifacts folder: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`
