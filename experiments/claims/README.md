# Claim Runners

This folder contains the supported claim-level entrypoints.

## Scripts

- `claim01_fit_not_discovery.py`
- `claim02_budget_law.py`
- `claim03_visibility_invisibility.py`
- `claim04_spectral_reconstruction.py`
- `claim05_iqp_vs_strong_baselines_high_score.py`
- `claim06_fair_baseline_protocol.py`
- `claim07_iqp_vs_strong_baselines_global.py`
- `claim08_iqp_vs_strong_baselines_beta_sweep.py`
- `claim09_expected_visibility_scaling.py`
- `claim10_global_visibility_predicts_discovery.py`

Each script writes outputs to a claim-specific folder under `outputs/`.

## Recommended Usage

Use these scripts as the public interface for reproducing results.
Avoid editing legacy experiment modules unless implementation changes are required.
