# Project: Fair Classical Baseline + Global Holdout Extension

Date: 2026-02-06

## Goal
Extend the current `paper_even` evidence package with:

1. A fair, matched classical baseline comparison against IQP-QCBM.
2. A holdout protocol extension from high-value-only holdouts to full-support holdouts.

## New Claims

### Claim 6: Fair Baseline Claim
Under matched protocol and budget, IQP-QCBM (`parity_mse`) outperforms a classical Ising control on holdout discovery.

### Claim 7: Global Holdout Claim
The IQP-vs-classical discovery advantage remains visible when holdout states are selected from the full target support, not only high-value states.

## Experiment Implementation

Script:
- `/Users/superposition/Coding/iqp_generative/experiments/exp09_fair_baseline_global_holdout.py`

Run used (final):
```bash
python experiments/exp09_fair_baseline_global_holdout.py   --mode run+analyze   --seeds 42,43,44,45,46,47   --outdir outputs/.exp09_fair_global_full
```

Core configuration:
- target family: `paper_even`
- holdout modes: `high_value`, `global`
- seeds: `42..47` (6 seeds)
- train m: `1000,5000`
- layers: `1,2`
- sigma: `2.0`
- K: `128,256,512`
- IQP loss: `parity_mse`
- optimizer/budget: matched (`Adam`, same steps/lr across IQP and classical)

## Main Numerical Outcome (from `summary.json`)

High-value holdout:
- `n_pairs = 72`
- `q(H)` win fraction (IQP > classical): `0.6389`
- `Q80` win fraction (IQP better): `0.9444`
- median `Q80_iqp/Q80_class`: `0.1123`

Global holdout:
- `n_pairs = 72`
- `q(H)` win fraction (IQP > classical): `0.5556`
- `Q80` win fraction (IQP better): `0.8333`
- median `Q80_iqp/Q80_class`: `0.4008`

Interpretation:
- IQP remains strongly better on discovery (`Q80`) in both protocols.
- The effect is weaker in `global` than in `high_value`, which is expected.

## Output Locations

Final claim folders:
- `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/06_claim_fair_classical_baseline`
- `/Users/superposition/Coding/iqp_generative/outputs/paper_even_final/07_claim_global_holdout_full_distribution`

## Notes on Scope

- This extension still focuses on fixed-distribution unseen-state discovery.
- It is not an OOD/domain-shift claim.
- This package keeps the `paper_even`-only final scope.

## Claim Wording Lock (for paper drafting)

Use this claim language:
- "In this parity-structured benchmark, IQP-QCBM with parity loss shows an empirical advantage over the tested strong classical baselines on unseen-state holdout generalization under fixed target distribution."

Do not use this language:
- "IQP is better than all classical methods."
- "General quantum advantage."
- "OOD/domain-shift generalization."

Scope sentence to keep:
- "This is an in-distribution unseen-state discovery result (fixed target distribution), not a universal superiority claim."
