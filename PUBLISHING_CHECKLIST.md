# Publishing Checklist

## Repository integrity

- [x] `main` branch has no unmerged paths.
- [x] Docs (`README.md`, `REPRODUCIBILITY.md`) match the current repository scope.
- [x] New experiment scripts are documented with runnable commands.

## Artifact scope

- [x] Final figures (Fig1-Fig7) are present under `outputs/final_plots/`.
- [x] Claim artifacts for D3PM and Transformer comparisons are present under `outputs/claims/`.
- [x] Both manifests exist:
  - `outputs/final_plots/ARTIFACT_MANIFEST.csv`
  - `outputs/claims/ARTIFACT_MANIFEST.csv`

## Rebuild checks

1. Rerender final figures:

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

2. Reproduce extended baselines:

```bash
python experiments/d3pm/eval_d3pm_vs_iqp_beta0p9_n12.py
python experiments/transformer/eval_transformer_vs_iqp_n12.py
```

3. Verify manifests:

```bash
python scripts/verify_artifacts.py \
  outputs/final_plots/ARTIFACT_MANIFEST.csv \
  outputs/claims/ARTIFACT_MANIFEST.csv
```

## Publication links

Use frozen tag links in manuscript references:

- `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`
- `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`
