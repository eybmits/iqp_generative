# Claim Runners

This folder contains the supported claim-level entry points.

## Available Runners

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
- `claim11_spectral_proxy_validation.py`

## Usage Pattern

Each runner forwards unknown CLI flags to its backend legacy experiment module.

Example:

```bash
python experiments/claims/claim11_spectral_proxy_validation.py \
  --betas 0.7,0.8,0.9 \
  --seeds 42,43,44 \
  --train-m 200
```

Outputs are written to:

- `outputs/claims/<claim_name>/`

## Conventions

- Treat this folder as the stable public interface.
- Implement algorithm/protocol changes in `experiments/legacy/`.
- Keep claim runner defaults aligned with paper/default protocol settings.
