# Contributing

## Scope

This repository mixes two artifact classes:

- frozen final figures in `outputs/final_plots/`
- curated analysis reruns in `outputs/analysis/`

Please keep that distinction intact. Do not mix local scratch outputs into the curated tracked sets.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-analysis.txt
```

## Expected Change Discipline

For changes under `experiments/final_scripts/` or `outputs/final_plots/`:

- treat default settings as frozen unless you are intentionally updating the canonical figure package
- rerender the affected outputs
- rebuild the final artifact manifest
- verify the final artifact manifest in strict mode

For changes under `experiments/analysis/` or `outputs/analysis/`:

- keep only publishable curated analysis directories under version control
- add or update `README.md` and `RUN_CONFIG.json` for any curated output directory
- rebuild the analysis artifact manifest
- verify the analysis artifact manifest in strict mode

## Recommended Verification

```bash
python -m py_compile $(git ls-files '*.py')

python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1

python tools/verify_analysis_manifest.py \
  --manifest outputs/analysis/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## Pull Request Checklist

- the worktree is clean apart from intentional changes
- no dated rerun or scratch directories remain
- READMEs and run metadata match the committed outputs
- manifests were rebuilt if artifact contents changed
- `RUN_CONFIG.json` matches the documented command lines
