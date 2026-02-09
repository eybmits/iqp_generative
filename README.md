# iqp_generative

IQP-QCBM holdout-discovery experiments: validating that parity-moment trained quantum generative models discover unseen states more efficiently than strong classical baselines.

## Quick Start

```bash
pip install -r requirements.txt
python experiments/make_paper_figures.py
```

This generates 4 publication-ready composite figures in `outputs/paper_figures/`:

| Figure | Content | Claims |
|--------|---------|--------|
| `fig1.pdf` | Recovery curves + visibility mechanism | Discovery + Claim 2 |
| `fig2.pdf` | Robustness heatmaps + budget-law scatter | Claims 3 + 4 |
| `fig3.pdf` | Fit vs discovery (parity vs prob-MSE) | Claim 1 |
| `fig4.pdf` | Fair classical baseline comparison | Claims 5 + 6 + 7 |

## Structure

- `iqp_generative/core.py` — shared core: target distributions, IQP/Ising training, metrics, plotting
- `experiments/make_paper_figures.py` — master script generating all 4 paper figures
- `experiments/exp02_budget_law.py` — reference: sweep computation for Figs 1+2
- `experiments/exp03_visibility_minvis.py` — reference: visibility computation for Fig 1(c)
- `experiments/exp05_discovery_axis.py` — reference: parity vs prob comparison for Fig 3
- `experiments/exp09_fair_baseline_global_holdout.py` — reference: baseline data for Fig 4
- `experiments/exp10_strong_classical_recovery.py` — reference: strong baseline computation for Fig 4
- `outputs/paper_even_final/` — evidence archive with per-claim directories and CLAIMS.md
- `docs/cover_letter_discovery.md` — cover letter draft

## Dependencies

```
numpy matplotlib scipy pennylane torch
```

## Claims

See `outputs/paper_even_final/CLAIMS.md` for the full 7-claim summary with quantitative results.
