# Reproducibility Runbook

## 1) Environment

```bash
pip install -r requirements.txt
pytest -q
```

## 2) Public Entry Points

Use claim runners only (`experiments/claims/claim01_...claim11_...`).
Each runner writes to a claim-scoped folder in `outputs/claims/`.

Example:

```bash
python experiments/claims/claim07_iqp_vs_strong_baselines_global.py
```

## 3) Recommended Reproduction Order

1. Run core claims first (`claim01` to `claim08`).
2. Run theory/diagnostic claims (`claim09` to `claim11`).
3. Inspect artifacts in each claim output folder (PDF, CSV, JSON).

## 4) Protocol Interpretation

- Primary protocol for model comparison: `global+smart`.
- Sensitivity protocol: `global+random`.
- Primary discovery endpoint: exact recovery curve `R(Q)` and measured `Q80`.
- Budget-law formulas are reported as auxiliary approximations.

## 5) Key Output Roots

- `outputs/claims/`: claim-scoped reproducible artifacts
- `outputs/paper_even_final/`: curated final analysis artifacts
- `outputs/paper_figures/`: paper-facing selected exports

## 6) Paper Method Text Assets

- `docs/methodology_section_full.tex`
- `docs/methodology_claim_mapping.tex`
- `docs/methodology_at_a_glance.tex`

## 7) Scope Guardrails

- Dataset family in this repository: `paper_even` only.
- Keep claim runners stable; implement changes in legacy backend scripts.
- Avoid committing ad-hoc one-off analysis scripts as public entry points.

## 8) Optional Figure Postprocessing

For claim-35 compact scatter label/underlay cleanup:

```bash
python experiments/legacy/postprocess_budgetlaw_compact.py --render-png
```

Details: `docs/budgetlaw_compact_postprocess.md`
