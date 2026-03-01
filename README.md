# IQP Generative Final Plots (Minimal Release + Appendix Ablation)

This repository is reduced to the final paper plotting package.

## Scope

Only the final reproducible plotting pipeline is kept:
- exactly 7 final plot scripts in `experiments/final_scripts/`
- final outputs and frozen input data in `outputs/final_plots/`
- minimal docs for rerendering and publication checks

No legacy experiment pipeline is required for rerendering.

## Final scripts

- `plot_target_sharpness_beta_sweep.py` -> Fig1
- `plot_iqp_sigmak_ablation_recovery.py` -> Fig2
- `plot_tv_bshs_seedmean_scatter.py` -> Fig3
- `plot_visibility_mechanistic_recovery.py` -> Fig4
- `plot_visibility_visible_invisible_recovery.py` -> Fig5
- `plot_beta_sweep_recovery_grid.py` -> Fig6
- `plot_appendix_ablation_beta0p8_nsweep.py` -> Fig7 (appendix ablation)

All scripts are standalone and contain their plotting style locally.

## Data policy

- Fig2/Fig4/Fig5/Fig6 load frozen `.npz` data snapshots located next to their outputs.
- Fig3 loads the frozen multiseed points CSV (`beta=0.90`, seeds `101..112`).
- Fig7 loads a frozen ablation snapshot (`beta=0.8`, `n={12,14,16,18}`, 5 seeds):
  - exact evaluation for `n<=14`
  - shot-based evaluation (`100k` shots) for `n>=16`
  - matched optimization budget: `iqp_steps=300` for parity and mse
- Fig1 is generated directly from its internal deterministic construction.

## Quick run

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

## Documentation

- `PUBLISHING_CHECKLIST.md`
- `experiments/final_scripts/FINAL_6_PLOTS_RUNBOOK.md`
- `experiments/final_scripts/FINAL_SCRIPTS_SETTINGS_LOCK.md`

## Artifact integrity

- `outputs/final_plots/ARTIFACT_MANIFEST.csv`

## Paper links (stable)

Use these links directly in the manuscript:

- Repository (main): `https://github.com/eybmits/iqp_generative`
- Frozen paper snapshot (tag): `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`
- Exact commit used for this release: `https://github.com/eybmits/iqp_generative/tree/5f5b723`
- Final figure artifacts folder: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`

BibTeX example:

```bibtex
@misc{iqp_generative_final_2026,
  author       = {Eybmits},
  title        = {IQP Generative Final Plots (Minimal Release + Appendix Ablation)},
  year         = {2026},
  howpublished = {\url{https://github.com/eybmits/iqp_generative/tree/paper-final-v1}},
  note         = {Accessed: 2026-03-01}
}
```
