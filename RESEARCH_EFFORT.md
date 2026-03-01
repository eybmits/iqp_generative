# Research Effort: IQP Generative Final Plot Package

This repository is published as a reproducible research artifact focused on the final six manuscript figures.

## Goal

Provide a minimal, auditable package that can:
- regenerate all final figures used in the paper
- verify artifact integrity with checksums
- avoid dependence on legacy training code during final rerendering

## What was done

1. Reduced the codebase to the final plotting surface.
2. Kept exactly six final scripts (Fig1..Fig6).
3. Embedded plotting style settings directly in each final script.
4. Frozen final data snapshots for data-driven panels (`.npz` and `.csv`).
5. Added locked runbook, style lock, and settings lock documentation.
6. Added artifact manifest (`SHA-256`, file size, PDF dimensions).

## Reproducibility model

- Fig1: deterministic internal construction.
- Fig2/Fig4/Fig5/Fig6: deterministic rerender from frozen `.npz` snapshots.
- Fig3: deterministic rerender from frozen multiseed points CSV (`beta=0.90`, seeds `101..112`).

## Entry points

- Runbook: `experiments/final_scripts/FINAL_6_PLOTS_RUNBOOK.md`
- Reproduction guide: `REPRODUCIBILITY.md`
- Publishing checklist: `PUBLISHING_CHECKLIST.md`
- Artifact checksums: `outputs/final_plots/ARTIFACT_MANIFEST.csv`

## Intended use

This package is intended for:
- manuscript figure verification
- archival reproducibility checks
- external audit of final plotting outputs

It is intentionally not a full training pipeline release.
