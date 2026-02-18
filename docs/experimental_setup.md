# V. Experimental Setup

This section fixes the protocol for our main discovery experiments so results are directly reproducible and comparable across model classes. All models use the same target family, holdout construction, seed schedule, and matched training-budget classes. The primary endpoint is unseen-state discovery on holdout sets, not global fit alone.

## V.A Targets and Holdout Protocols

We work on the `paper_even` family with `n = 12` binary variables (`x in {0,1}^12`). The target support is the even-parity sector
`Omega_even = {x : sum_i x_i mod 2 = 0}`.
For each state `x`, the score is
`s(x) = 1 + longest_zero_run_between_ones(x)`.
The target distribution is a score tilt on this support:

`p*(x) = exp(beta * s(x)) / Z(beta)` for `x in Omega_even`, and `0` otherwise.

We sweep `beta in {0.6, 0.7, ..., 1.4}`.

Our main training budget is fixed to `train_m = 200` samples. We evaluate two holdout protocols:

- `global`: holdout candidates are drawn from all states in `Omega_even`.
- `high_value`: holdout candidates are drawn from a high-score subset defined by `alpha = 0.05` (top 5% by score within support).

Holdout construction is fixed as follows. For each `(holdout_mode, beta, seed)`, we build one holdout set `H` with `|H| = 20` from a candidate pool of size `400`. Candidates are filtered by a probability floor sequence `tau in {1/m_holdout, 0.5/m_holdout, 0.25/m_holdout, 0}`, with `m_holdout = 5000`, then ranked by `p*(x)`. Final states are chosen by farthest-point selection in Hamming distance (tie-break: larger `p*(x)`). This yields diverse but still target-relevant unseen states.

We use paired seeds `42, 43, 44, 45, 46` for all model comparisons. Holdouts are fixed before training and never used for tuning.

## V.B Models

### IQP model

The primary model is an IQP-QCBM with ZZ-only interactions on a ring topology with nearest-neighbor (NN) and next-nearest-neighbor (NNN) couplings, depth `L = 1`. The output distribution `q_theta` is trained from the same non-holdout sample set as all baselines. Our primary IQP objective is parity-moment MSE (`parity_mse`), with IQP `prob_mse` included as an additional IQP reference.

### Classical controls and strong baselines

Classical comparators:

- Ising+fields (NN+NNN), trained with parity-moment MSE.
- Dense Ising+fields (all-to-all), trained with cross-entropy (`xent`).
- Autoregressive Transformer, trained with MLE.
- MaxEnt parity model, trained by moment matching on the same parity feature information.

## V.C Training

For each `(holdout_mode, beta, seed)`, we define the train distribution by masking out holdout states and renormalizing:
`p_train(x) = p*(x) / (1 - p*(H))` for `x not in H`, else `0`.
We sample `m = 200` training indices with replacement from `p_train`.

Parity supervision uses a random feature set of size `K` with bandwidth `sigma`. Each parity mask `alpha_k in {0,1}^n` is sampled i.i.d. with Bernoulli rate
`p(sigma) = 0.5 * (1 - exp(-1/(2*sigma^2)))`,
excluding the all-zero mask. The mode-specific settings are fixed:

- `global`: `sigma = 1`, `K = 512`
- `high_value`: `sigma = 2`, `K = 256`

Given sampled masks, we form parity features and empirical target moments from the same training data used by all models.

Optimizer budgets:

- IQP (parity/prob branches): Adam, `steps = 300`, `lr = 0.05`, evaluation cadence every `50` steps.
- Ising+fields (NN+NNN): matched Adam budget (`steps = 300`, `lr = 0.05`).
- Dense Ising+fields: same step/lr budget (`steps = 300`, `lr = 0.05`).
- AR Transformer: `epochs = 300`, `d_model = 64`, `heads = 4`, `layers = 2`, `ff = 128`, `lr = 1e-3`, batch size `256`.
- MaxEnt parity: `steps = 2500`, `lr = 5e-2`.

Holdout information is never used in fitting or hyperparameter choice.

## V.D Evaluation Protocol

The main discovery quantity is holdout recovery:

`R(Q) = (1/|H|) * sum_{x in H} [1 - (1 - q(x))^Q]`.

We also report `q(H) = sum_{x in H} q(x)` and enrichment `q(H)/q_unif(H)`.
Our primary scalar endpoint is `Q80` (lower is better): the smallest integer `Q` such that `R(Q) >= 0.8`, searched up to `Q_max_search = 200000` (return `inf` if unmet). We additionally report `R(Q=1000)` and `R(Q=10000)`, plus the analytic predictor
`Q80_pred approx (|H| / q(H)) * ln(5)`.

Recovery curves are evaluated on a fixed grid up to `Q = 10000`, defined as the union of:

- log-spaced points from `10^0` to `10^3.5` (120 points, integer-cast),
- linear points from `1000` to `10000` (160 points, integer-cast),
- anchor points `{0,1,2,3,4,5,10,20,50,100}`,

followed by sorting and deduplication.

Primary ranking is done at expectation level from model distributions `q` (not finite-shot sampling). Optional finite-shot analyses are diagnostics only. Fit metrics (TV, KL) are secondary checks, not the discovery objective.

Across seeds (`42..46`), we aggregate per `(holdout_mode, train_m, beta, model)` using mean, standard deviation, median, and IQR. Paired comparisons are computed per `(beta, seed)`, including parity-vs-best-classical `Q80` ratios, to preserve matched-condition inference.
