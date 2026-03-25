# Experiment 11 toy parity signature plot

This directory contains a small n=4 explainer for why a parity-based fit can assign
positive mass to an unseen state such as `1001`.

Story:

- panel 1: target mass versus `D_train`, with only the seen state shown at its target height
- panel 2: both states share the same sampled parity signature, but only `1100` comes from `D_train`
- panel 3: `IQP MSE` stays on the seen state, while `IQP Parity` puts positive mass on `1001`

Key values:

- toy sample seed: `53`
- sample size: `12`
- `1001` displayed `D_train` mass: `0.0000`
- `1001` `IQP Parity` mass: `0.1400`
- shared signature: `(-1, -1, +1)`

Saved artifacts:

- PDF: `plots/experiment_11_toy_parity_signature_plot/experiment_11_toy_parity_signature_plot.pdf`
- PNG: `plots/experiment_11_toy_parity_signature_plot/experiment_11_toy_parity_signature_plot.png`
- data NPZ: `plots/experiment_11_toy_parity_signature_plot/experiment_11_toy_parity_signature_plot_data.npz`
- run config: `plots/experiment_11_toy_parity_signature_plot/RUN_CONFIG.json`

- source driver: `experiment_11_toy_parity_signature_plot.py`
- outdir: `plots/experiment_11_toy_parity_signature_plot`
