# Budgetlaw Compact Post-Process (Claim 35)

This note documents the deterministic touch-up workflow for:

- `outputs/paper_even_final/35_claim_budgetlaw_global_m200_custom/budgetlaw_scatter_dual_holdout_m200_beta06to14_all_experiments_compact.pdf`
- `outputs/paper_even_final/35_claim_budgetlaw_global_m200_custom/budgetlaw_scatter_dual_holdout_m200_beta06to14_all_experiments_compact_annotated.pdf`

`outputs/` is git-ignored, so reproducibility relies on a script, not committed artifacts.

## Script

Use:

```bash
python experiments/legacy/postprocess_budgetlaw_compact.py --help
```

The script can:

- move the `best: IQP parity (beta=1)` label by `(text_dx, text_dy)`,
- re-align the white rounded underlay to the moved label baseline,
- apply a subtle global zoom,
- re-render corresponding PNG files via Ghostscript.

## Default Targets

If `--pdf` is omitted, the script patches both compact claim-35 PDFs above.

## Example Commands

Apply a mild zoom only:

```bash
python experiments/legacy/postprocess_budgetlaw_compact.py \
  --zoom 1.03 \
  --render-png
```

Move label right/down and keep underlay correctly coupled:

```bash
python experiments/legacy/postprocess_budgetlaw_compact.py \
  --text-dx 6 \
  --text-dy -3 \
  --render-png
```

Move label only (not recommended for this figure):

```bash
python experiments/legacy/postprocess_budgetlaw_compact.py \
  --text-dx 6 \
  --text-dy -3 \
  --no-align-underlay \
  --render-png
```

## Notes

- `--underlay-bottom-offset` controls the baseline-to-underlay relation.
  The default is tuned for this compact style.
- Re-running the script is allowed, but shifts are cumulative.
- PNG rendering requires `gs` (Ghostscript) in `PATH`.

## Session Record (2026-02-18)

The working compact artifact in this repo session was normalized so that:

- label transform is at approximately `(x=98.7697, y=95.1446)`,
- underlay bottom offset is exactly `-3.311875` relative to label baseline,
- compact PDFs were rendered back to their sibling PNG files.
