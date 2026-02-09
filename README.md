# iqp_generative

IQP-QCBM holdout-discovery experiments: validating that parity-moment trained quantum generative models discover unseen states more efficiently than strong classical baselines.

## Quick Start

```bash
pip install -r requirements.txt
python experiments/claims/claim01_fit_not_discovery.py
```

Run any claim directly via the dedicated claim runner:

```bash
python experiments/claims/claim0X_*.py
```

All claim runners write to `outputs/claims/<claim_name>/`.

## Claim-First Structure

The repository now includes 8 dedicated claim entrypoints under `experiments/claims/`.
Each script writes to its own claim-specific output folder under `outputs/claims/`.

| Claim | Script | Default Output Folder |
|---|---|---|
| 1 — Fit =/= Discovery | `experiments/claims/claim01_fit_not_discovery.py` | `outputs/claims/claim01_fit_not_discovery/` |
| 2 — Budget Law | `experiments/claims/claim02_budget_law.py` | `outputs/claims/claim02_budget_law/` |
| 3 — Visibility/Invisibility | `experiments/claims/claim03_visibility_invisibility.py` | `outputs/claims/claim03_visibility_invisibility/` |
| 4 — Spectral Reconstruction | `experiments/claims/claim04_spectral_reconstruction.py` | `outputs/claims/claim04_spectral_reconstruction/` |
| 5 — IQP > strong classical baselines (high-score holdout) | `experiments/claims/claim05_iqp_vs_strong_baselines_high_score.py` | `outputs/claims/claim05_iqp_vs_strong_baselines_high_score/` |
| 6 — Fair baseline protocol (paired controls) | `experiments/claims/claim06_fair_baseline_protocol.py` | `outputs/claims/claim06_fair_baseline_protocol/` |
| 7 — IQP > strong classical baselines (global holdout) | `experiments/claims/claim07_iqp_vs_strong_baselines_global.py` | `outputs/claims/claim07_iqp_vs_strong_baselines_global/` |
| 8 — IQP > strong classical baselines across beta sweep | `experiments/claims/claim08_iqp_vs_strong_baselines_beta_sweep.py` | `outputs/claims/claim08_iqp_vs_strong_baselines_beta_sweep/` |

## Structure

- `iqp_generative/core.py` — shared core: target distributions, IQP/Ising training, metrics, plotting
- `experiments/claims/` — claim-first wrappers (one script per claim)
- `experiments/legacy/` — internal legacy experiment implementations (not primary entrypoints)
- `outputs/claims/` — claim-first output archive
- `outputs/paper_even_final/` — legacy evidence archive
- `docs/cover_letter_discovery.md` — cover letter draft

## Dependencies

```
numpy matplotlib scipy pennylane torch
```

## Claims

See `docs/claims_overview.md` for the current claim wording.
