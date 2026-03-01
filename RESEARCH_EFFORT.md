# Research Effort: IQP Generative Final Plot Package

This repository is released as a publication-focused, reproducible artifact for the final manuscript figures, including the appendix ablation.

## Release status

As of March 1, 2026, this package is finalized and published with:
- a minimal 7-script plotting surface
- frozen figure input artifacts (`.npz` and `.csv`)
- checksum manifest for auditability

## Goal

Provide a minimal and auditable release that can:
- regenerate all final figures used in the manuscript
- verify artifact integrity via SHA-256 checksums
- avoid dependence on the legacy training pipeline during rerendering

## What was done

1. Reduced the repository to final figure production assets only.
2. Kept exactly seven standalone scripts in `experiments/final_scripts`:
   - Fig1 `plot_target_sharpness_beta_sweep.py`
   - Fig2 `plot_iqp_sigmak_ablation_recovery.py`
   - Fig3 `plot_tv_bshs_seedmean_scatter.py`
   - Fig4 `plot_visibility_mechanistic_recovery.py`
   - Fig5 `plot_visibility_visible_invisible_recovery.py`
   - Fig6 `plot_beta_sweep_recovery_grid.py`
   - Fig7 `plot_appendix_ablation_beta0p8_nsweep.py`
3. Embedded plotting style settings directly in each script (no shared style helper dependency).
4. Added frozen data snapshots:
   - Fig2/4/5/6 use frozen `.npz` default inputs
   - Fig3 uses frozen multiseed points CSV (`beta=0.90`, seeds `101..112`)
   - Fig7 uses frozen ablation NPZ/CSV (`beta=0.8`, `n={12,14,16,18,20}`, 5 seeds)
5. Regenerated and stored final PDF/PNG outputs in `outputs/final_plots/`.
6. Added locked execution and style docs:
   - `experiments/final_scripts/FINAL_6_PLOTS_RUNBOOK.md`
   - `experiments/final_scripts/FINAL_PLOT_STYLE.md`
   - `experiments/final_scripts/FINAL_SCRIPTS_SETTINGS_LOCK.md`
7. Generated artifact manifest with file size, SHA-256, and PDF dimensions:
   - `outputs/final_plots/ARTIFACT_MANIFEST.csv`
   - `outputs/final_plots/ARTIFACT_MANIFEST.md`

## Reproducibility model

- Fig1: deterministic internal construction from script logic.
- Fig2/Fig4/Fig5/Fig6: deterministic rerender from frozen `.npz` snapshots.
- Fig3: deterministic rerender from frozen multiseed points CSV.
- Fig7: deterministic rerender from frozen NPZ/CSV ablation snapshot.

## Appendix ablation note (Fig7)

The appendix ablation is fixed to `beta=0.8` (middle-regime setting) and compares IQP parity vs IQP MSE over `n={12,14,16,18,20}` with 5 seeds.

Evaluation protocol:
- exact evaluation for `n<=14`
- shot-based evaluation for `n>=16` with `100000` shots

The qualitative ranking remains stable under this mixed exact/shot protocol.

## Verification path

1. Install minimal dependencies from `requirements.txt`.
2. Run the seven final scripts (see `REPRODUCIBILITY.md`).
3. Verify checksums using `outputs/final_plots/ARTIFACT_MANIFEST.csv`.

## Intended use

This release is intended for:
- manuscript figure verification
- archival reproducibility checks
- external audit of final figure artifacts

This is intentionally not a full training-pipeline release.
