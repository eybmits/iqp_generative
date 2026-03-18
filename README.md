# IQP Generative: Frozen Paper Figures and Documented Analysis Reruns

This repository contains a frozen paper-figure package plus a curated set of post-freeze analysis reruns.

The publishable artifact policy is intentional:

- `outputs/final_plots/` stores the canonical frozen figure package and its manifest.
- `outputs/analysis/` stores curated, documented reruns that are kept under version control.
- dated reruns, replots, and chunked scratch outputs are local-only artifacts and are ignored.

## Quickstart

1. Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-analysis.txt
```

2. Re-render the frozen final figures:

```bash
python experiments/final_scripts/plot_target_sharpness_beta_sweep.py
python experiments/final_scripts/plot_iqp_sigmak_ablation_recovery.py
python experiments/final_scripts/plot_tv_bshs_seedmean_scatter.py
python experiments/final_scripts/plot_visibility_mechanistic_recovery.py
python experiments/final_scripts/plot_visibility_visible_invisible_recovery.py
python experiments/final_scripts/plot_beta_sweep_recovery_grid.py
python experiments/final_scripts/plot_appendix_ablation_beta0p8_nsweep.py
```

3. Verify both curated artifact sets:

```bash
python tools/verify_final_manifest.py \
  --manifest outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --strict 1

python tools/verify_analysis_manifest.py \
  --manifest outputs/analysis/ARTIFACT_MANIFEST.csv \
  --strict 1
```

## 20-Seed Benchmark Standard

The benchmark-side rerun standard now uses `20` matched seeds: `101..120`.
The full reporting contract is documented in [docs/benchmark_reporting_protocol.md](/Users/markus/Documents/Projekte/iqp_generative/docs/benchmark_reporting_protocol.md), and the exact seed schedule is committed in [docs/benchmark_seed_schedule_20seeds.csv](/Users/markus/Documents/Projekte/iqp_generative/docs/benchmark_seed_schedule_20seeds.csv).
The live paper-side run ledger is generated at [docs/paper_benchmark_ledger.md](/Users/markus/Documents/Projekte/iqp_generative/docs/paper_benchmark_ledger.md) whenever benchmark-standard 20-seed analysis drivers are executed.

Recommended benchmark-standard reruns:

1. Fig6 multiseed recovery rerun
- betas `0.5..1.2`
- seeds `101..120`
- holdout seed `46`
- IQP `600` steps
- AR Transformer `600` epochs
- MaxEnt `600` steps
- output directory:
  `outputs/analysis/fig6_multiseed_all600_seeds101_120/`

2. Fig3 KL-BSHS dual-axis boxplot
- `beta = 0.9`
- seeds `101..120`
- `Q_eval = 1000`
- `holdout_mode = global`
- IQP `600` steps
- AR Transformer `600` epochs
- MaxEnt `600` steps
- output directory:
  `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/`

3. Fig6 wide multiseed recovery rerun
- betas `0.1..2.0`
- seeds `101..120`
- holdout seed `46`
- IQP `600` steps
- AR Transformer `600` epochs
- MaxEnt `600` steps
- output directory:
  `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds101_120/`

4. Fig6 beta-vs-Q80 summary family
- derived from the benchmark-standard Fig6 multiseed artifacts
- recommended summary output:
  `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/`

## Legacy Tracked Analysis Artifacts

1. Fig2 recovery-summary companion figure
- built directly from the frozen Fig2 snapshot
- no retraining; same sigma-K settings and same recovery curves
- selected output:
  `outputs/analysis/fig2_recovery_summary_panels/fig2_recovery_summary_panels.pdf`
- exact run metadata:
  `outputs/analysis/fig2_recovery_summary_panels/RUN_CONFIG.json`

2. Fig6 multiseed recovery rerun
- betas `0.5..1.2`
- legacy seeds `42..46`
- holdout seed `46`
- IQP `600` steps
- AR Transformer `600` epochs
- MaxEnt `600` steps
- `q80_search_max = 1e9`
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

4. Fig6 beta-vs-Q80 compact summary
- derived from the documented Fig6 multiseed artifacts
- no retraining; same model family, betas, and recovery threshold
- selected output:
  `outputs/analysis/fig6_beta_q80_summary/fig6_beta_q80_summary.pdf`
- exact run metadata:
  `outputs/analysis/fig6_beta_q80_summary/RUN_CONFIG.json`

5. Fig6 wide multiseed recovery rerun
- betas `0.1..2.0`
- legacy seeds `42..46`
- holdout seed `46`
- IQP `600` steps
- AR Transformer `600` epochs
- MaxEnt `600` steps
- `q80_search_max = 1e18`
- selected output:
  `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/fig6_beta_sweep_recovery_grid_multiseed.pdf`
- exact run metadata:
  `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/RUN_CONFIG.json`

6. Fig6 wide beta-vs-Q80 summary family
- derived from the documented wide Fig6 multiseed artifacts
- no retraining; all values are aggregated from stored per-seed `Q80` outputs
- recommended robust view:
  `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/fig6_beta_q80_summary.pdf`
- companion detail view with all seed traces:
  `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/fig6_beta_q80_summary.pdf`
- comparison view using mean ± std:
  `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/fig6_beta_q80_summary.pdf`

## Tested Environment

The curated artifacts currently in this repository were regenerated and verified locally with:

- macOS on CPU
- Python `3.13.2`
- `numpy==2.3.3`
- `pandas==2.3.2`
- `matplotlib==3.10.6`
- `torch==2.10.0`
- `pennylane==0.42.3`

The frozen final figure scripts only require `requirements.txt`. The analysis reruns require `requirements-analysis.txt` in addition.
For a tested pinned environment, use [environment.yml](/Users/superposition/Coding/iqp_generative/environment.yml).

## Repository Layout

- `experiments/final_scripts/`: standalone frozen final plotting scripts
- `experiments/analysis/`: self-contained analysis rerun drivers
- `outputs/final_plots/`: canonical frozen figure outputs and manifests
- `outputs/analysis/`: curated post-freeze analysis artifacts and manifests
- `tools/`: manifest build and verification utilities
- `docs/`: paper-side supporting material

## Curated Analysis Outputs

The tracked publishable analysis set currently contains the historical curated artifacts below.
The benchmark-standard 20-seed reruns are defined by the source scripts and reproducibility commands, and can be regenerated into new output directories.

- `outputs/analysis/fig2_recovery_summary_panels/`
- `outputs/analysis/fig3_kl_bshs_seedmean_scatter_20seeds_all600/`
- `outputs/analysis/fig6_beta_q80_summary/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
- `outputs/analysis/fig6_beta_q80_summary_beta0p1_2p0_mean_std/`
- `outputs/analysis/fig6_multiseed_all600_seeds42_46/`
- `outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46/`

Each curated analysis directory is expected to include:

- `README.md`
- `RUN_CONFIG.json`
- rendered figures
- any kept machine-readable data products such as `.npz`, `.csv`, or `.json`

## Rebuild Frozen Final Figures

See [experiments/final_scripts/README.md](/Users/superposition/Coding/iqp_generative/experiments/final_scripts/README.md) for the script-by-script mapping and [FINAL_SCRIPTS_SETTINGS_LOCK.md](/Users/superposition/Coding/iqp_generative/experiments/final_scripts/FINAL_SCRIPTS_SETTINGS_LOCK.md) for the frozen defaults and artifact policy.

## Run Curated Analysis Reruns

See [experiments/analysis/README.md](/Users/superposition/Coding/iqp_generative/experiments/analysis/README.md) for the exact commands and runtime notes.

## Artifact Integrity

Rebuild the frozen final manifest after intentional updates:

```bash
python tools/build_final_manifest.py \
  --root outputs/final_plots \
  --output-csv outputs/final_plots/ARTIFACT_MANIFEST.csv \
  --output-md outputs/final_plots/ARTIFACT_MANIFEST.md
```

Rebuild the curated analysis manifest after intentional updates:

```bash
python tools/build_analysis_manifest.py \
  --root outputs/analysis \
  --output-csv outputs/analysis/ARTIFACT_MANIFEST.csv \
  --output-md outputs/analysis/ARTIFACT_MANIFEST.md
```

## Documentation

- [REPRODUCIBILITY.md](/Users/superposition/Coding/iqp_generative/REPRODUCIBILITY.md)
- [CONTRIBUTING.md](/Users/superposition/Coding/iqp_generative/CONTRIBUTING.md)
- [PUBLISHING.md](/Users/superposition/Coding/iqp_generative/PUBLISHING.md)
- [docs/benchmark_reporting_protocol.md](/Users/superposition/Coding/iqp_generative/docs/benchmark_reporting_protocol.md)
- [docs/paper_benchmark_ledger.md](/Users/superposition/Coding/iqp_generative/docs/paper_benchmark_ledger.md)
- [experiments/final_scripts/README.md](/Users/superposition/Coding/iqp_generative/experiments/final_scripts/README.md)
- [experiments/final_scripts/FINAL_SCRIPTS_SETTINGS_LOCK.md](/Users/superposition/Coding/iqp_generative/experiments/final_scripts/FINAL_SCRIPTS_SETTINGS_LOCK.md)
- [experiments/analysis/README.md](/Users/superposition/Coding/iqp_generative/experiments/analysis/README.md)
- [outputs/analysis/README.md](/Users/superposition/Coding/iqp_generative/outputs/analysis/README.md)
- [CITATION.cff](/Users/superposition/Coding/iqp_generative/CITATION.cff)

## Citation and Snapshot Links

- Main repository: `https://github.com/eybmits/iqp_generative`
- Frozen snapshot tag: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1`
- Frozen final artifacts: `https://github.com/eybmits/iqp_generative/tree/paper-final-v1/outputs/final_plots`

The curated analysis directories under `outputs/analysis/` are post-freeze artifacts tracked in the current repository state and are not part of the frozen `paper-final-v1` snapshot unless explicitly tagged later.
