# Analysis Outputs

This directory is the curated post-freeze analysis artifact set.
The active analysis standard is the 10-seed schedule `101..110` with shared training budget `600`, documented in `experiments/analysis/STANDARD_TRAINING_PROTOCOL.md`.
Some committed subdirectories remain legacy snapshots from older seed schedules and are retained for artifact integrity.

Tracked publishable subdirectories are:

- `fig2_recovery_summary_panels/`
- `fig3_kl_bshs_seedmean_scatter_20seeds_all600/`
- `experiment6_fixed_beta0p90_nsweep_all_baselines/`
- `fig6_beta_q80_summary/`
- `fig6_beta_q80_summary_beta0p1_2p0_iqr/`
- `fig6_beta_q80_summary_beta0p1_2p0_iqr_seed_traces/`
- `fig6_beta_q80_summary_beta0p1_2p0_mean_std/`
- `transformer_capacity_ablation_beta0p90_quick/`
- `fig6_multiseed_all600_seeds42_46/`
- `fig6_multiseed_beta0p1_2p0_all600_seeds42_46/`

Legacy note:

- `fig6_multiseed_all600_seeds42_46/` and `fig6_multiseed_beta0p1_2p0_all600_seeds42_46/` are historical 5-seed reruns
- frozen final Fig3 and Fig7 artifacts under `outputs/final_plots/` likewise predate the active 10-seed analysis standard
- `fig3_kl_bshs_seedmean_scatter_20seeds_all600/` is an older committed artifact and does not define the current default seed schedule
- `transformer_capacity_ablation_beta0p90_quick/` is an earlier quick run and predates the active 10-seed / 600-budget default

Each curated subdirectory should contain:

- `README.md`
- `RUN_CONFIG.json`
- `TRAINING_PROTOCOL.md` for any directory produced by a training script under the active standard
- the kept rendered outputs
- any kept `.npz`, `.csv`, or `.json` files needed for downstream inspection

Scratch policy:

- dated reruns, replots, and chunked working directories are not part of the publishable artifact set
- those scratch patterns are ignored in `.gitignore`

Integrity metadata for this curated analysis set is stored in:

- `ARTIFACT_MANIFEST.csv`
- `ARTIFACT_MANIFEST.md`
