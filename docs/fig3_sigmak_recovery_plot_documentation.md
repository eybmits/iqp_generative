# Unified Documentation: 4-Plot Experimental Story

This document consolidates the four plot families used in the current manuscript draft into one reproducible, coherent storyline.

## 0) Story at a glance

| Plot | Question answered | Main script(s) | Main output(s) |
|---|---|---|---|
| 1. IQP parity vs IQP MSE | Does parity loss improve discovery vs standard IQP MSE under same setup? | `experiments/legacy/exp36_iqp_sigmak_ablation_parity_vs_prob.py` | `outputs/paper_even_final/36_claim_iqp_sigmak_ablation_parity_vs_prob_m200_beta08/sigmak_ablation_recovery_parity_vs_prob.pdf` |
| 2. Spectral recall/completion | Is observed recovery consistent with band-limited spectral completion signal? | `iqp_generative/core.py` (Claim-4 style recovery panel) | `outputs/paper_even_final/01_claim_discovery_metric/4_recovery_best.pdf` (and manuscript-clean variant with simplified legend) |
| 3. TV vs BSHS + Pareto | Across 5 baselines, who best trades off bucket-mass fit vs support discovery? | `experiments/legacy/exp46_compare_iqp_vs_5baselines_tv_bshs_multiseed_beta.py` | `outputs/paper_even_final/46_compare_iqp_vs_5baselines_tv_bshs_multiseed_beta/tv_bshs_scatter_budgetlaw_style_multiseed_beta_seedmean_q1000_no_iqp_mse_pareto_only.pdf` |
| 4. Overall beta-sweep performance | How do all models compare across beta under one fixed global protocol? | `experiments/legacy/exp11_professional_recovery_strip.py` | `outputs/paper_even_final/34_claim_beta_sweep_bestparams/global_m200_sigma1_k512_multiseed5_steps600/collage/recovery_horizontal_professional_with_tv_encoding_multiseed_n5_color_baselines.pdf` |

## 1) Plot 1: IQP parity vs IQP MSE (same data, same budget)

### Purpose
A controlled in-family IQP ablation:
- red family: IQP with `parity_mse` over `(sigma, K)` grid,
- blue reference: IQP with `prob_mse`.

### Repro command (exact configuration used for the shown variant)

```bash
python3 experiments/legacy/exp36_iqp_sigmak_ablation_parity_vs_prob.py \
  --n 12 \
  --beta 0.8 \
  --seed 46 \
  --train-m 200 \
  --sigmas 0.5,1.0,2.0,3.0 \
  --Ks 128,256,512 \
  --layers 1 \
  --good-frac 0.05 \
  --holdout-mode global \
  --holdout-k 20 \
  --holdout-pool 400 \
  --holdout-m-train 5000 \
  --iqp-steps 300 \
  --iqp-lr 0.05 \
  --iqp-eval-every 50 \
  --reference-loss prob_mse \
  --reference-mmd-tau 2.0 \
  --q80-thr 0.8 \
  --q80-search-max 200000
```

### Protocol and metrics
- Target: `paper_even` (`n=12`, even-parity support, score tilt with `beta=0.8`).
- Holdout: `global` candidate set + smart selection (`select_holdout_smart`), `|H|=20`.
- Recovery curve:
  `R(Q) = (1/|H|) * sum_{x in H} [1 - (1 - q(x))^Q]`.
- `Q80`: smallest `Q` with `R(Q) >= 0.8`.
- Best parity curve: run with minimal finite `Q80`.

### Key artifacts
- `outputs/paper_even_final/36_claim_iqp_sigmak_ablation_parity_vs_prob_m200_beta08/sigmak_ablation_metrics.csv`
- `outputs/paper_even_final/36_claim_iqp_sigmak_ablation_parity_vs_prob_m200_beta08/sigmak_ablation_recovery_parity_vs_prob.pdf`

## 2) Plot 2: Spectral recall/completion diagnostic

### Purpose
Mechanistic check: compare target recovery, learned IQP recovery, and spectral-completion recovery under the same holdout.

### Source experiment family
- Core pipeline: `iqp_generative/core.py` recovery panel (`4_recovery_best.pdf`).
- Plot contains target, best IQP, spectral completion, uniform (plus Ising controls in the full raw variant).
- Manuscript variant is a cleaned rendering of this same experiment family (simplified legend, optional axis crop).

### Repro command (core default claim-style spectral reconstruction)

```bash
python3 iqp_generative/core.py \
  --outdir outputs/paper_even_final/01_claim_discovery_metric \
  --target-family paper_even
```

### Core settings in this family
- `n=12`, `beta=0.9`, `train_m=1000`, `seed=42`.
- `sigmas=0.5,1,2,3`, `Ks=128,256,512`.
- Holdout is smart-selected from high-score ROI (`good_frac=0.05`) in the core default flow.
- Best IQP setting chosen by measured `Q80` in the sweep; spectral curve uses the same `(sigma,K)`.

### Key artifact
- `outputs/paper_even_final/01_claim_discovery_metric/4_recovery_best.pdf`

## 3) Plot 3: TV-score vs BSHS(Q) scatter + Pareto front

### Purpose
Cross-model comparison on full-support bucket metrics:
- x-axis: `TV_score` (bucket-mass mismatch, lower better),
- y-axis: `BSHS(Q)` (bucket-weighted support-hit, higher better),
- Pareto front: nondominated points minimizing x and maximizing y.

### Metrics (from `exp45.compute_support_bucket_metrics`)
- Let score buckets be `s` over support.
- Bucket mass shares: `target_share_s`, `model_share_s`.
- `TV_score = 0.5 * sum_s |target_share_s - model_share_s|`.
- `bucket_hit_s(Q) = R_s(Q)` inside bucket `s`.
- `BSHS(Q) = sum_s target_share_s * bucket_hit_s(Q)`.

### Final documented run (q=1000, no IQP-prob)

```bash
python experiments/legacy/exp46_compare_iqp_vs_5baselines_tv_bshs_multiseed_beta.py \
  --outdir outputs/paper_even_final/46_compare_iqp_vs_5baselines_tv_bshs_multiseed_beta \
  --exclude-model-keys iqp_prob_mse \
  --q-eval 1000 \
  --single-beta 0.9 \
  --make-raw-scatter 0 \
  --make-bucket-profile 0 \
  --draw-pareto-front 1 \
  --pareto-only 0 \
  --draw-model-trends 0 \
  --hide-titles 1 \
  --tag q1000_no_iqp_mse_pareto_only
```

### Data scope
- Models: 5 (`iqp_parity_mse`, NNN+fields, dense-xent, AR, MaxEnt).
- Betas: `0.6..1.4`.
- Seeds: `42..46`.
- Raw points: 225; seed-mean points: 45.

### Key artifacts
- `outputs/paper_even_final/46_compare_iqp_vs_5baselines_tv_bshs_multiseed_beta/tv_bshs_scatter_budgetlaw_style_multiseed_beta_seedmean_q1000_no_iqp_mse_pareto_only.pdf`
- `outputs/paper_even_final/46_compare_iqp_vs_5baselines_tv_bshs_multiseed_beta/RUN_DOC_q1000_no_iqp_mse_pareto_front.md`

## 4) Plot 4: Overall multi-beta performance strip (4x2)

### Purpose
Single-view performance overview over beta values with fixed protocol and multiseed averaging.

### Exact run command (documented provenance)

```bash
python -u experiments/legacy/exp11_professional_recovery_strip.py \
  --outdir /Users/superposition/Coding/iqp_generative/outputs/paper_even_final/34_claim_beta_sweep_bestparams/global_m200_sigma1_k512_multiseed5_steps600 \
  --holdout-mode global \
  --betas 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2 \
  --grid-cols 4 \
  --train-m 200 \
  --sigma 1.0 \
  --K 512 \
  --seeds 42,43,44,45,46 \
  --iqp-steps 600 \
  --legend-in-first-panel 1 \
  --colorful-baselines 1
```

### Settings
- Global holdout protocol, smart selection.
- `n=12`, `m=200`, `sigma=1`, `K=512`, `layers=1`.
- 8 betas, 5 seeds.
- Per panel: recovery curves averaged over seeds.
- Vertical dashed `Q80` marker indicates the first baseline crossing the threshold in that panel.

### Key artifacts
- `outputs/paper_even_final/34_claim_beta_sweep_bestparams/global_m200_sigma1_k512_multiseed5_steps600/collage/recovery_horizontal_professional_with_tv_encoding_multiseed_n5_color_baselines.pdf`
- `outputs/paper_even_final/34_claim_beta_sweep_bestparams/global_m200_sigma1_k512_multiseed5_steps600/RUN_CONFIG.md`

## 5) Cohesive narrative across the 4 plots

1. Plot 1 isolates loss-function effect inside IQP (parity vs MSE) under matched conditions.
2. Plot 2 adds mechanism-level interpretation via spectral completion signal.
3. Plot 3 moves to cross-model benchmarking with two orthogonal axes: mass fit (`TV_score`) and discovery coverage (`BSHS`).
4. Plot 4 shows robustness across beta with multiseed aggregation in one global comparison figure.

This gives a full chain from in-model ablation -> mechanistic explanation -> cross-model tradeoff -> overall robustness sweep.

## 6) Methodology (paper-ready unified version)

### 6.1 Objective
Our objective is **discovery quality under distribution shift induced by holdout masking**, not only global fit.  
Given a target distribution `p*`, a holdout set `H`, and model distribution `q`, we evaluate how efficiently a generator rediscovers unseen states in `H`.

### 6.2 Discovery formalism (shared by all 4 plots)
For sampling budget `Q`, expected holdout recovery is:

`R(Q) = (1/|H|) * sum_{x in H} [1 - (1 - q(x))^Q]`.

Primary scalar endpoint:

`Q80 = min{Q : R(Q) >= 0.8}`  (with capped search).

Interpretation:
- lower `Q80` means faster discovery,
- higher `R(Q)` at fixed `Q` means better discovery efficiency.

### 6.3 Two complementary evaluation axes
All plots are organized around two orthogonal questions:

1. **Discovery speed on unseen states**: measured by `R(Q)` and `Q80`.
2. **Distributional faithfulness on support structure**: measured by score-bucket metrics:
   - `TV_score` (bucket-mass mismatch; lower better),
   - `BSHS(Q)` (bucket-weighted support-hit; higher better).

This separation avoids conflating “matching bucket mass” with “actually covering diverse states”.

### 6.4 Mechanistic lens: spectral completion
We use spectral completion as a mechanism diagnostic: parity moments define a linear reconstruction (`q_lin`) and simplex completion (`q_tilde`), yielding a **predicted discovery profile** under the same holdout formalism.
In the unified story, spectral curves are not treated as a separate model class but as a mechanistic probe explaining when parity constraints induce useful unseen-state mass.

### 6.5 Unified narrative mapping to plots
- **Plot 1**: in-family causal ablation (IQP parity loss vs IQP MSE) under matched setup.
- **Plot 2**: mechanism alignment (target vs IQP vs spectral completion vs uniform).
- **Plot 3**: cross-model tradeoff (mass-fit vs support-hit) with Pareto front.
- **Plot 4**: robustness panel over beta and seeds under fixed global protocol.

## 7) Experimental Setup (paper-ready unified version)

### 7.1 Common backbone
The four plots are aligned to one backbone protocol:
- target family: `paper_even`,
- system size: `n=12`,
- holdout size: `|H|=20`,
- global candidate region with smart selection (`global+smart`) for primary comparisons,
- fixed recovery grid up to `Q=10000`,
- paired multiseed evaluation (`seeds=42..46`) for aggregate comparisons.

### 7.2 Target distribution
Support is the even-parity sector.  
State score is the longest-zero-run-derived discrete score.  
Target is a score-tilted Gibbs-like distribution:

`p*(x) propto exp(beta * score(x))` on even-parity support.

Beta controls sharpness; we sweep beta for robustness and use fixed beta slices for diagnostics.

### 7.3 Holdout protocol
Primary protocol: **Global+Smart**.
- Candidate set is global support (`Omega_even`).
- Smart selection enforces nontrivial mass and state diversity (probability floor + pool filtering + Hamming farthest-point choice).
- Holdout states are removed from training distribution and never seen during optimization.

Sensitivity variants (used in side analyses) keep the same formalism but alter selection policy (e.g., pure random).  
Primary conclusions in this story are anchored to Global+Smart.

### 7.4 Models
Compared model family:
- IQP (parity loss) [primary model],
- IQP (prob-MSE) [in-family reference],
- Ising+fields (NN+NNN),
- Dense Ising+fields (xent),
- AR Transformer (MLE),
- MaxEnt parity.

All models are trained from the same masked train data per `(beta, seed, holdout)` instance.

### 7.5 Configuration by plot
- **Plot 1 (loss ablation)**:
  - `beta=0.8`, `m=200`, sigma-K grid for parity (`sigma={0.5,1,2,3}`, `K={128,256,512}`),
  - reference IQP branch with `prob_mse`,
  - best parity setting selected by minimal finite `Q80`.
- **Plot 2 (spectral diagnostic)**:
  - same discovery formalism; target/IQP/spectral/uniform overlay,
  - rendered as mechanism panel consistent with Plot-1 definitions.
- **Plot 3 (TV-BSHS Pareto)**:
  - multiseed, multibeta model comparison in `TV_score`-`BSHS` plane,
  - seed-mean points with Pareto front; canonical paper variant excludes IQP-prob branch.
- **Plot 4 (overall performance strip)**:
  - fixed global setting (`m=200`, `sigma=1`, `K=512`),
  - beta sweep (`0.5..1.2`) and 5-seed mean recovery panels.

## 8) Evaluation (paper-ready unified version)

### 8.1 Primary endpoints
Primary endpoint family:
- recovery curves `R(Q)`,
- `Q80`.

These directly quantify discovery efficiency in “samples required to recover unseen structure”.

### 8.2 Secondary endpoints
Secondary endpoints for structural fidelity:
- `TV_score` on score buckets,
- `BSHS(Q)` for bucket-weighted support discovery,
- optional composite summaries for ranking convenience.

The Pareto view (Plot 3) is used to compare models without collapsing both axes prematurely.

### 8.3 Aggregation and fairness
- Metrics are computed per matched `(beta, seed)` instance.
- Reported points are either raw per-run values or seed means, depending on panel purpose.
- Comparisons are paired by construction (same target, holdout protocol, seed family, and training budget class).

### 8.4 Figure-specific evaluation logic
- **Plot 1** asks: does parity supervision improve unseen-state discovery inside IQP?
- **Plot 2** asks: is the observed recovery behavior mechanistically consistent with spectral completion?
- **Plot 3** asks: across model classes, who lies on the best fit-vs-discovery frontier?
- **Plot 4** asks: do conclusions persist across beta and seeds under one fixed protocol?

### 8.5 Decision principle for the manuscript
Model quality is judged by:
1. Discovery efficiency first (`Q80`, `R(Q)`),
2. then mass-shape fidelity (`TV_score`),
3. then support-coverage quality (`BSHS` / Pareto position).

This ordering keeps the evaluation aligned with the paper’s core goal: **discovering unseen but relevant states, not only matching seen marginals**.
