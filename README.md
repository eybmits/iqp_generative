# iqp_generative

IQP-QCBM discovery experiments on the `paper_even` target family.

## Quickstart

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Optional shortcuts:

```bash
make test
make claim07
make claim11
```

## Claim Entry Points

Use claim runners in `experiments/claims/` as the public interface.

```bash
python experiments/claims/claim01_fit_not_discovery.py
python experiments/claims/claim02_budget_law.py
python experiments/claims/claim03_visibility_invisibility.py
python experiments/claims/claim04_spectral_reconstruction.py
python experiments/claims/claim05_iqp_vs_strong_baselines_high_score.py
python experiments/claims/claim06_fair_baseline_protocol.py
python experiments/claims/claim07_iqp_vs_strong_baselines_global.py
python experiments/claims/claim08_iqp_vs_strong_baselines_beta_sweep.py
python experiments/claims/claim09_expected_visibility_scaling.py
python experiments/claims/claim10_global_visibility_predicts_discovery.py
python experiments/claims/claim11_spectral_proxy_validation.py
```

All claim runs write to `outputs/claims/<claim_name>/`.

## Repository Layout

- `iqp_generative/core.py`: shared target/model/metric/plot utilities
- `experiments/claims/`: reproducible claim-level runner scripts
- `experiments/legacy/`: backend implementations used by claim runners
- `tests/`: unit and smoke checks for metrics/protocol logic
- `docs/reproducibility_runbook.md`: reproducibility instructions and workflow
- `docs/methodology_section_full.tex`: full methodology section (paper text)
- `docs/methodology_claim_mapping.tex`: claim-to-implementation mapping table

## Protocol Notes

- Primary model comparison protocol: `global+smart` holdout selection.
- Sensitivity protocol: `global+random`.
- Primary endpoint: exact holdout recovery via `R(Q)` and measured `Q80`.
- Budget-law terms are treated as auxiliary approximations, not sole decision criteria.
