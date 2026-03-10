# Reproducibility Guide

This repository contains:
- the frozen final plotting package in `experiments/final_scripts/` and `outputs/final_plots/`
- documented recomputed analysis reruns in `experiments/analysis/` and `outputs/analysis/`

## 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional deterministic hash seed:

```bash
export PYTHONHASHSEED=0
```

Additional dependencies for the documented analysis reruns:

```bash
pip install -r requirements-analysis.txt
```

## 2) Rerender frozen final figures

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

## 3) Rebuild frozen final artifact manifest

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md
```

## 4) Verify frozen final artifact manifest

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## 5) Recompute the documented Fig6 multiseed rerun

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py
```

The current script defaults reproduce:

- `outputs/analysis/fig6_multiseed_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed.pdf`

Exact run metadata is stored in:

- `outputs/analysis/fig6_multiseed_all600_seeds42_46/RUN_CONFIG.json`

## 6) Recompute the documented Fig3 KL-BSHS rerun

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig3_kl_bshs_dual_axis_boxplot.py
```

The current script defaults reproduce:

- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`

The per-seed values and run metadata are stored in:

- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/kl_bshs_points_multiseed_beta_q1000_beta0p90_newseeds20.csv`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/kl_bshs_summary_multiseed_beta_q1000_beta0p90_newseeds20.json`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/RUN_CONFIG.json`

## 7) Verify the documented analysis artifacts

```bash
python - <<'PY'
from pathlib import Path
import csv, hashlib
manifest = Path('outputs/analysis/ARTIFACT_MANIFEST.csv')
with manifest.open('r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    if header != ['path', 'bytes', 'sha256']:
        raise SystemExit(f'Unexpected header: {header}')
    count = 0
    for path_s, bytes_s, sha_s in reader:
        p = Path(path_s)
        data = p.read_bytes()
        if len(data) != int(bytes_s):
            raise SystemExit(f'BYTE MISMATCH: {p}')
        if hashlib.sha256(data).hexdigest() != sha_s:
            raise SystemExit(f'HASH MISMATCH: {p}')
        count += 1
print(f'OK: {count} analysis files verified')
PY
```

## 8) Notes

- The frozen 7-figure package rerenders deterministically from frozen final data.
- The Fig6 multiseed rerun and the Fig3 KL-BSHS rerun are recomputations, not frozen-data rerenders.
- Both analysis reruns are implemented as standalone scripts and do not require `git show` or repo-history lookups at runtime.
