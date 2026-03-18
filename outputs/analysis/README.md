# Analysis Outputs

This directory is the curated post-freeze analysis artifact set.
The benchmark-standard rerun configuration is now the 20-seed schedule `101..120` documented in `docs/benchmark_reporting_protocol.md`.
Some committed subdirectories remain legacy snapshots from older seed schedules and are retained for artifact integrity.

Tracked publishable subdirectories are:

- `fig2_recovery_summary_panels/`
- `fig3_kl_bshs_seedmean_scatter_20seeds_all600/`
- `fig6_beta_q80_summary/`
- `fig6_beta_q80_summary_beta0p1_2p0_iqr/`
- `fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
- `fig6_beta_q80_summary_beta0p1_2p0_mean_std/`
- `fig6_multiseed_all600_seeds42_46/`
- `fig6_multiseed_beta0p1_2p0_all600_seeds42_46/`

Legacy note:

- `fig6_multiseed_all600_seeds42_46/` and `fig6_multiseed_beta0p1_2p0_all600_seeds42_46/` are historical 5-seed reruns
- frozen final Fig3 and Fig7 artifacts under `outputs/final_plots/` likewise predate the 20-seed benchmark standard

Each curated subdirectory should contain:

- `README.md`
- `RUN_CONFIG.json`
- the kept rendered outputs
- any kept `.npz`, `.csv`, or `.json` files needed for downstream inspection

Scratch policy:

- dated reruns, replots, and chunked working directories are not part of the publishable artifact set
- those scratch patterns are ignored in `.gitignore`

Integrity metadata for this curated analysis set is stored in:

- `ARTIFACT_MANIFEST.csv`
- `ARTIFACT_MANIFEST.md`
