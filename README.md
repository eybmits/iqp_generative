# iqp_generative

IQP-QCBM discovery experiments on the paper-even dataset family.

## Setup

```bash
pip install -r requirements.txt
```

## Main Entry Points (Claims)

Run claim scripts from `experiments/claims/`:

```bash
python experiments/claims/claim01_fit_not_discovery.py
python experiments/claims/claim02_budget_law.py
python experiments/claims/claim03_visibility_invisibility.py
python experiments/claims/claim04_spectral_reconstruction.py
python experiments/claims/claim05_iqp_vs_strong_baselines_high_score.py
python experiments/claims/claim06_fair_baseline_protocol.py
python experiments/claims/claim07_iqp_vs_strong_baselines_global.py
python experiments/claims/claim08_iqp_vs_strong_baselines_beta_sweep.py
```

## Repository Structure

- `iqp_generative/core.py`: shared methods (targets, training, metrics, plotting)
- `experiments/claims/`: clean claim-level runners (recommended)
- `experiments/legacy/`: underlying experiment implementations
- `docs/claims_overview.md`: compact claim wording
- `docs/reproducibility_runbook.md`: reproducibility guide
- `outputs/paper_even_final/`: curated final result archive
- `outputs/paper_figures/`: paper-facing figure exports

## Notes

- Temporary/quick scripts and ad-hoc sweeps were removed to keep the repo paper-focused.
- `outputs/` is generated content and is ignored by git.
