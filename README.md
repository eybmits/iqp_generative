# IQP Generative Benchmark

This repository contains the scripts and cached artifacts for the IQP
generative-model benchmark used in the manuscript figures. The active reporting
standard is documented in `STANDARD_TRAINING_PROTOCOL.md` and
`docs/benchmark_reporting_protocol.md`.

## Active Protocol

- System size: `n=12`
- Target family: even-parity score-tilted distributions
- Beta sweep: `0.1, 0.2, ..., 2.0`
- Matched seeds: `111..120`
- Training samples per matched instance: `m=200`
- Full sweep size: `20 betas x 10 seeds = 200` matched instances
- Shared budget: `600` optimizer updates or epochs per model
- Reference parity band: `sigma=1.0`, `K=512`

The current full-sweep raw summary is:

| Model | mean KL | median KL | KL wins | mean `C_q(1000)` |
| --- | ---: | ---: | ---: | ---: |
| IQP-parity | `0.385 +/- 0.021` | `0.414` | `190/200` | `0.053 +/- 0.004` |
| Ising+fields (NN+NNN) | `0.923 +/- 0.062` | `0.929` | `0/200` | `0.038 +/- 0.003` |
| Dense Ising+fields | `0.947 +/- 0.025` | `0.978` | `0/200` | `0.035 +/- 0.003` |
| AR Transformer | `0.744 +/- 0.054` | `0.737` | `10/200` | `0.036 +/- 0.002` |
| MaxEnt-parity | `1.804 +/- 0.108` | `1.689` | `0/200` | `0.018 +/- 0.002` |

## Reproducibility

The final figures are regenerated from cached CSV/NPZ artifacts unless
`--recompute 1` is explicitly passed.

Common commands:

```bash
python experiment_2_beta_kl_summary.py \
  --outdir plots/experiment_2_beta_kl_summary \
  --series-csv plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary_series.csv

python experiment_3_beta_quality_coverage.py \
  --outdir plots/experiment_3_beta_quality_coverage \
  --series-q1000 plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q1000_series.csv

python make_aligned_kl_triptych.py
python make_aligned_recovery_fourpanel.py
python make_aligned_cross_class_diagnostics.py
```

Hardware sampling artifacts are already cached under
`plots/experiment_15_ibm_hardware_seedwise_best_coverage/`. Re-running the
hardware scripts requires configured IBM Quantum credentials and current backend
availability.

## Dependencies

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The checked-in requirements are pinned to the environment used for the current
publication rerender pass (`Python 3.13.2`).

Exact LaTeX-sized figure rendering uses Matplotlib with `text.usetex=True`.
A local TeX installation with `newtx` fonts is therefore required for exact
paper rendering; cached PDFs are included for inspection without rerendering.
