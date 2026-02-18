# Threats to Validity (Nature-Comms Package)

## External Validity

- Current evidence is scoped to the `paper_even` distribution family.
- Claims are not universal over arbitrary data distributions.
- Interpretation: evidence for parity-structured discovery tasks.

## In-Distribution Scope

- Holdout protocol tests unseen-state recovery under fixed target family.
- This is in-distribution unseen-state generalization, not domain shift / OOD transfer.

## Optimization Bias

- Different losses can have different optimization landscapes.
- Mitigation: matched seeds, matched budgets, and paired statistics.

## Hyperparameter Selection Bias

- Risk: post-hoc hyperparameter cherry-picking.
- Mitigation: pre-registered matrix + inner validation split for tuning.

## Statistical Reliability

- Single-seed results may be unstable.
- Mitigation: fixed 5-seed protocol, permutation tests, bootstrap CIs, Holm correction.

## Model Fairness

- Architecture families have different inductive biases and capacities.
- Mitigation: report parameter counts, training budgets, and sample/shot costs explicitly.

