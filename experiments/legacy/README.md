# Legacy Experiments

This folder contains the original experiment implementations that power the
claim runners in `../claims/`.

These scripts are kept as internal execution modules. The recommended public
entry points are the 11 claim scripts under:

- `experiments/claims/claim01_fit_not_discovery.py`
- `experiments/claims/claim02_budget_law.py`
- `experiments/claims/claim03_visibility_invisibility.py`
- `experiments/claims/claim04_spectral_reconstruction.py`
- `experiments/claims/claim05_iqp_vs_strong_baselines_high_score.py`
- `experiments/claims/claim06_fair_baseline_protocol.py`
- `experiments/claims/claim07_iqp_vs_strong_baselines_global.py`
- `experiments/claims/claim08_iqp_vs_strong_baselines_beta_sweep.py`
- `experiments/claims/claim09_expected_visibility_scaling.py`
- `experiments/claims/claim10_global_visibility_predicts_discovery.py`
- `experiments/claims/claim11_spectral_proxy_validation.py`

Guideline:

- Add new reusable logic here.
- Keep runner scripts in `../claims` thin and stable.
- Prefer adding tests in `tests/` for any new protocol/metric behavior.
