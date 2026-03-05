# Publishing Checklist

## Repository integrity

- [x] `main` branch has no unmerged paths.
- [x] Docs (`README.md`, `REPRODUCIBILITY.md`) match the current repository scope.
- [x] Final plotting scripts are documented with runnable commands.

## Artifact scope

- [x] Final figures (Fig1-Fig7) are present under `outputs/final_plots/`.
- [x] Final manifest files exist:
  - `outputs/final_plots/ARTIFACT_MANIFEST.csv`
  - `outputs/final_plots/ARTIFACT_MANIFEST.md`

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

2. Rebuild final manifest:

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md
```

3. Verify final manifest:

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## Publication links

Use frozen tag links in manuscript references:

- `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`
- `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`
