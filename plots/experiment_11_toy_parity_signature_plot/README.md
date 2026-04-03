# Experiment 11 toy parity signature plot

This directory contains a three-card n=4 explainer for why a parity-based fit can
assign positive mass to an unseen state such as `1001`.

Story:

- card 1: target mass versus `D_train`, highlighting that `1001` has zero train support
- card 2: `1001` and `1100` share the same sampled parity signature
- card 3: `IQP MSE` stays on the seen state, while `IQP Parity` restores mass on `1001`

Key values:

- toy sample seed: `53`
- sample size: `12`
- `1001` displayed `D_train` mass: `0.0000`
- `1001` `IQP Parity` mass: `0.1400`
- shared signature: `(-1, -1, +1)`

Saved artifacts:

- PDF: `plots/experiment_11_toy_parity_signature_plot/experiment_11_toy_parity_signature_plot.pdf`
- PNG: `plots/experiment_11_toy_parity_signature_plot/experiment_11_toy_parity_signature_plot.png`
- SVG: `plots/experiment_11_toy_parity_signature_plot/experiment_11_toy_parity_signature_plot.svg`
- data NPZ: `plots/experiment_11_toy_parity_signature_plot/experiment_11_toy_parity_signature_plot_data.npz`
- run config: `plots/experiment_11_toy_parity_signature_plot/RUN_CONFIG.json`

Reproduce:

- from repo root: `python experiment_11_toy_parity_signature_plot.py --outdir plots/experiment_11_toy_parity_signature_plot --train-m 12`

- source driver: `experiment_11_toy_parity_signature_plot.py`
- outdir: `plots/experiment_11_toy_parity_signature_plot`
