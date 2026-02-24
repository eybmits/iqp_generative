PYTHON ?= python

.PHONY: test claim01 claim02 claim03 claim04 claim05 claim06 claim07 claim08 claim09 claim10 claim11

test:
	pytest -q

claim01:
	$(PYTHON) experiments/claims/claim01_fit_not_discovery.py

claim02:
	$(PYTHON) experiments/claims/claim02_budget_law.py

claim03:
	$(PYTHON) experiments/claims/claim03_visibility_invisibility.py

claim04:
	$(PYTHON) experiments/claims/claim04_spectral_reconstruction.py

claim05:
	$(PYTHON) experiments/claims/claim05_iqp_vs_strong_baselines_high_score.py

claim06:
	$(PYTHON) experiments/claims/claim06_fair_baseline_protocol.py

claim07:
	$(PYTHON) experiments/claims/claim07_iqp_vs_strong_baselines_global.py

claim08:
	$(PYTHON) experiments/claims/claim08_iqp_vs_strong_baselines_beta_sweep.py

claim09:
	$(PYTHON) experiments/claims/claim09_expected_visibility_scaling.py

claim10:
	$(PYTHON) experiments/claims/claim10_global_visibility_predicts_discovery.py

claim11:
	$(PYTHON) experiments/claims/claim11_spectral_proxy_validation.py
