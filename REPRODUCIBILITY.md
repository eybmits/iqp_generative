# Reproducibility Guide

This package contains only final plotting scripts and the frozen final data needed by those scripts.

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimal dependencies are:
- `numpy`
- `pandas`
- `matplotlib`

## 2) Rerender all final plots

Run the seven scripts directly:

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

Outputs are written to:
- `outputs/final_plots/fig1_target_sharpness/`
- `outputs/final_plots/fig2_iqp_sigmak_ablation_recovery/`
- `outputs/final_plots/fig3_tv_bshs_seedmean_scatter/`
- `outputs/final_plots/fig4_visibility_mechanistic_recovery/`
- `outputs/final_plots/fig5_visibility_visible_invisible_recovery/`
- `outputs/final_plots/fig6_beta_sweep_recovery_grid/`
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/`

## 3) Verify artifacts by checksum

```bash
python - <<'PY'
from pathlib import Path
import csv, hashlib
manifest = Path('outputs/final_plots/ARTIFACT_MANIFEST.csv')
rows = list(csv.DictReader(manifest.open('r', encoding='utf-8')))
for r in rows:
    p = Path(r['path'])
    if not p.exists():
        raise SystemExit(f"MISSING: {p}")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != r['sha256']:
        raise SystemExit(f"HASH MISMATCH: {p}")
print(f"OK: {len(rows)} files verified")
PY
```

## 4) Fig3 provenance in this minimal package

The final Fig3 plot uses the frozen CSV:
- `outputs/final_plots/fig3_tv_bshs_seedmean_scatter/tv_bshs_points_multiseed_beta_q1000_no_iqp_mse_beta0p9_newseeds12.csv`

Characteristics:
- fixed `beta = 0.90`
- `Q_eval = 1000`
- seeds `101..112` (12 seeds)
- 5 models x 12 seeds = 60 points

## 5) Fig7 appendix ablation provenance

The final Fig7 plot uses:
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_data_default.npz`
- `outputs/final_plots/fig7_appendix_ablation_beta0p8_nsweep/fig7_seed_table.csv`

Characteristics:
- fixed `beta = 0.8`
- `n in {12,14,16,18}`
- 5 seeds: `42..46`
- models: `IQP parity` vs `IQP MSE`
- exact evaluation up to `n=14`
- shot-based evaluation for `n>=16` with `100000` shots
- matched optimization budget: `iqp_steps=300` for parity and mse

## Notes

- Legacy training scripts are intentionally not part of this minimal release.
- Reproducibility here means deterministic rerender from frozen final data + final plotting scripts.
