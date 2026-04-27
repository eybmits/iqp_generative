# Experiment 1 Locked Visual Standard

This folder contains the approved Experiment 1 figure style as of `2026-03-23`.

The source of truth is `experiment_1_kl_diagnostics.py`. The values below are not suggestions; they are the current locked defaults that the script rerenders.

## Panel A: Heatmap

- keep the heatmap cell layout in the original non-stretched Experiment 1 style
- top axis shows plain `K` values with a separate top axis label `K`
- white minor-grid cell boundaries
- horizontal colorbar with the existing IQP parity KL label

## Panel B: Rank Ordering

- use the dot-plot layout, not horizontal bars
- each row gets a thin guide line that stops at the point
- no title on the panel
- x-axis uses reduced major ticks for readability
- color scale runs from baby blue for worse values through light red to black as the best-direction limit
- pure black is reserved for the `KL = 0` limit, not for the best observed nonzero point

## Panel C: Benchmark Comparison

- use the zoomed single-axis lollipop layout
- keep the axis continuous and simply tighten the right edge around the plotted benchmark values
- `Target p*` is dark neutral text/color
- `Best IQP Parity` is now a fixed global-best `(sigma, K)` over the saved panel-C seeds, not a seedwise-best selection
- `IQP MSE` remains plotted from the saved matched-seed data
- `Uniform` stays excluded

## Regeneration

Use the committed rerender config or run:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python experiment_1_kl_diagnostics.py --rerender-only 1 --data-npz plots/experiment_1_kl_diagnostics/experiment_1_data.npz --outdir plots/experiment_1_kl_diagnostics
```
