# Reproducibility Runbook

## 1) Environment

```bash
pip install -r requirements.txt
```

## 2) Run Claims

Execute the 8 claim runners in `experiments/claims/`.
Each run is self-contained and writes claim-scoped artifacts into `outputs/`.

## 3) Core Result Archives

- `outputs/paper_even_final/`: curated final analysis artifacts
- `outputs/paper_figures/`: selected paper figures

## 4) Expected Workflow

1. Run one or more claim scripts.
2. Inspect generated PDFs/CSVs in the corresponding output folder.
3. Use `docs/claims_overview.md` for claim phrasing consistency.

## 5) Scope Guardrails

- Dataset family in this repo: paper-even only.
- Primary model: IQP-QCBM plus fair classical baselines defined in claim scripts.
- Keep ad-hoc exploratory scripts outside the main repo history.

## 6) Figure Micro-Layout Fixes (Claim 35 Compact Scatter)

For final label/underlay touch-ups on the compact budgetlaw scatter, use:

```bash
python experiments/legacy/postprocess_budgetlaw_compact.py --render-png
```

Full options and examples:

- `docs/budgetlaw_compact_postprocess.md`
