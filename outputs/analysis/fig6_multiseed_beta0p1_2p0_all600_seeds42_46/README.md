# Fig6 Multiseed Beta 0.1-2.0 All600

This directory contains the documented wide Fig6-style rerun with recomputed curves and seed bands.
The generating script is self-contained and performs all model training locally without loading code from git history at runtime.

Chosen rerun command:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiments/analysis/plot_fig6_beta_sweep_recovery_grid_multiseed.py \
  --recompute 1 \
  --betas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0 \
  --q80-search-max 1000000000000000000 \
  --seeds 42,43,44,45,46 \
  --holdout-seed 46 \
  --iqp-steps 600 \
  --artr-epochs 600 \
  --maxent-steps 600 \
  --outdir outputs/analysis/fig6_multiseed_beta0p1_2p0_all600_seeds42_46
```

Fixed protocol:

- `betas = 0.1,0.2,...,2.0`
- `n = 12`
- training seeds `42..46`
- holdout seed `46`
- `holdout_mode = global`
- `holdout_m_train = 5000`
- `holdout_k = 20`
- `holdout_pool = 400`
- `train_m = 200`
- `sigma = 1.0`
- `K = 512`
- `layers = 1`
- `good_frac = 0.05`
- `iqp_steps = 600`
- `iqp_lr = 0.05`
- `iqp_eval_every = 20`
- `artr_epochs = 600`
- `artr_d_model = 64`
- `artr_heads = 4`
- `artr_layers = 2`
- `artr_ff = 128`
- `artr_lr = 1e-3`
- `artr_batch_size = 256`
- `maxent_steps = 600`
- `maxent_lr = 5e-2`
- `q80_thr = 0.8`
- `q80_search_max = 1e18`

Interpretation notes:

- The holdout is regenerated separately for each `beta` using the same historical protocol. It is not one single holdout shared across all panels.
- The colored model bands in the recovery-grid plot are `mean +/- std` over seeds `42..46`.
- `Target` and `Uniform` are recomputed from the same per-`beta` holdout used for the trained models.
- The kept metrics file contains `20 * 5 * 5 = 500` rows and all `Q80` values are finite at this search budget.

Kept files:

- `fig6_beta_sweep_recovery_grid_multiseed.png`
- `fig6_beta_sweep_recovery_grid_multiseed.pdf`
- `fig6_beta_sweep_recovery_grid_multiseed_data.npz`
- `fig6_beta_sweep_recovery_grid_multiseed_metrics.csv`
- `README.md`
- `RUN_CONFIG.json`
