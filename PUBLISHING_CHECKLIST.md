# Publishing Checklist

## Package scope

- [x] Only final plot package is present.
- [x] Exactly 7 final scripts in `experiments/final_scripts/`.
- [x] Final outputs and frozen data in `outputs/final_plots/`.
- [x] Reproducibility and style lock docs are included.

## Final figures present

- [x] Fig1 PDF/PNG
- [x] Fig2 PDF/PNG
- [x] Fig3 PDF/PNG
- [x] Fig4 PDF/PNG
- [x] Fig5 PDF/PNG
- [x] Fig6 PDF/PNG
- [x] Fig7 PDF/PNG

## Final data snapshots present

- [x] `fig2_data_default.npz`
- [x] `fig4_data_default.npz`
- [x] `fig5_data_default.npz`
- [x] `fig6_data_default.npz`
- [x] `fig7_data_default.npz`
- [x] Fig3 frozen multiseed CSV/summary files
- [x] Fig7 frozen per-seed CSV

## Pre-release technical checks

1. Rerender all figures:
```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

2. Verify artifact checksums:
```bash
python - <<'PY'
from pathlib import Path
import csv, hashlib
rows = list(csv.DictReader(Path('outputs/final_plots/ARTIFACT_MANIFEST.csv').open('r', encoding='utf-8')))
for r in rows:
    p = Path(r['path'])
    if not p.exists():
        raise SystemExit(f"MISSING {p}")
    if hashlib.sha256(p.read_bytes()).hexdigest() != r['sha256']:
        raise SystemExit(f"HASH MISMATCH {p}")
print('OK')
PY
```

3. Confirm manuscript assembly uses these exact final PDFs.

## Paper linking block

Use a frozen tag link (not `main`) in the paper:

- `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`

Optional direct artifact link:

- `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`
