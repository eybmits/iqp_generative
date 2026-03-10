# IQP Generative Final Plots + Documented Analysis Reruns

This repository contains the frozen final paper plotting package plus selected, self-contained analysis reruns.

It includes:
- final paper plotting scripts for Fig1-Fig7
- frozen final figure inputs and rendered artifacts under `outputs/final_plots/`
- deterministic manifest build and verification utilities for the frozen final package
- standalone analysis reruns under `experiments/analysis/` and `outputs/analysis/`

## Repository layout

- `experiments/final_scripts/`: frozen final figure scripts
- `outputs/final_plots/`: frozen final inputs and rendered PDF/PNG artifacts
- `tools/build_final_manifest.py`: deterministic final-manifest builder
- `tools/verify_final_manifest.py`: final-manifest verification utility
- `experiments/analysis/`: self-contained analysis rerun scripts
- `outputs/analysis/`: documented analysis artifacts and analysis manifest

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the documented analysis reruns, also install:

```bash
pip install -r requirements-analysis.txt
```

## Rebuild frozen final figures

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

## Selected documented analysis reruns

1. Fig2 recovery-summary companion figure
- built directly from the frozen Fig2 snapshot
- no retraining; same sigma-K settings and same recovery curves
- selected output:
  `outputs/analysis/fig2_recovery_summary_panels/fig2_recovery_summary_panels.pdf`
- exact run metadata:
  `outputs/analysis/fig2_recovery_summary_panels/RUN_CONFIG.json`

2. Fig6 multiseed recovery rerun
- betas `0.5..1.2`
- seeds `42..46`
- holdout seed `46`
- IQP `600` steps
- AR Transformer `600` epochs
- MaxEnt `600` steps
- selected output:
  `outputs/analysis/fig6_multiseed_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed.pdf`
- exact run metadata:
  `outputs/analysis/fig6_multiseed_all600_seeds42_46/RUN_CONFIG.json`

3. Fig3 KL-BSHS dual-axis boxplot
- `beta = 0.9`
- seeds `101..120`
- `Q_eval = 1000`
- `holdout_mode = global`
- IQP `600` steps
- AR Transformer `600` epochs
- MaxEnt `600` steps
- right axis metric: forward KL `D_KL(p^* || q)`
- selected output:
  `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/fig3_kl_bshs_seedmean_scatter_beta_0p90_dual_axis_boxplot.pdf`
- exact run metadata:
  `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/RUN_CONFIG.json`

The selected analysis reruns are implemented locally and do not load training logic from git history at runtime.

## Artifact integrity

Verify frozen final artifacts:

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1
```

Rebuild the frozen final manifest after intentional artifact updates:

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md
```

Verify the documented analysis artifacts:

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

## Documentation

- `REPRODUCIBILITY.md`
- `experiments/final_scripts/FINAL_SCRIPTS_SETTINGS_LOCK.md`
- `experiments/analysis/README.md`
- `outputs/analysis/README.md`

## Citation links

- Main repository: `https://github.com/eybmits/iqp_generative`
- Frozen snapshot tag: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`
- Frozen final artifacts folder: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`

The documented reruns in `outputs/analysis/` are post-freeze analysis artifacts tracked in this repository state, not part of the frozen `paper-final-v1` snapshot.
