# Publishing Checklist

Use this checklist before tagging or publishing a public snapshot.

## 1. Clean the tree

- remove local scratch outputs under `outputs/analysis/`
- remove empty chunk directories and aborted reruns
- confirm `git status` only shows intentional tracked changes

## 2. Verify documentation

- top-level `README.md` reflects the current curated artifact policy
- `REPRODUCIBILITY.md` matches the current commands
- each curated analysis directory has `README.md` and `RUN_CONFIG.json`
- `CITATION.cff` and `LICENSE` are present and current

## 3. Rebuild manifests

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md

python tools/build_analysis_manifest.py \
  --root outputs/analysis \
  --output-csv outputs/analysis/ARTIFACT_MANIFEST.csv \
  --output-md outputs/analysis/ARTIFACT_MANIFEST.md
```

## 4. Verify manifests

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1

python tools/verify_analysis_manifest.py \
  --manifest outputs/analysis/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## 5. Release hygiene

- ensure only curated artifact directories remain in `outputs/analysis/`
- ensure figure stems and filenames are internally consistent
- confirm the snapshot tag or release notes explain whether `outputs/analysis/` is included
- run or inspect the GitHub Actions verification workflow before publishing
