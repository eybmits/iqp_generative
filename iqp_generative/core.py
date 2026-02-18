#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hero Spectral Holdout-Discovery Master Script (FULL VALIDATION + CLASSICAL BASELINE)
====================================================================================

This script is a **paper-faithful** implementation to validate **Result 1** and **Result 2**
from the provided paper, while keeping your **paper-grade plotting style** and output naming.

In addition, it includes a **classical baseline** that is trained with the
**same objective**, **same data**, and **same optimizer budget** as the IQP-QCBM:
  - Classical Ising/Boltzmann model on the **same NN + NNN ring topology**.
  - Optimizes the same parity-moment MSE loss:  mean((z_data - P @ q)^2).

Core validated math (paper mapping):
  - Result 1:
      Recovery R(Q) from expected unique fraction U_H(Q) (Eq. (4)-(5)),
      Q80 threshold, and the budget law Q80 ≈ (|H|/q(H)) ln 5 (Eq. (13)).
      Also checks Proposition-1 bound numerically (optional diagnostic).

  - Result 2:
      Linear band reconstruction q_lin via Walsh inversion (Eq. (14)),
      canonical completion q_tilde via positivity+normalization (Eq. (15)),
      spectral visibility Vis_B(H) (Def. 1), and Lemma-1 identity
          q_lin(H) = |H|/2^n + Vis_B(H)
      which is checked as an algebraic invariant (should hold to numerical precision).

IQP Architecture (fixed, as requested):
  - **ZZ-only** interactions on a **ring** with **nearest-neighbor (NN)** and
    **next-nearest-neighbor (NNN)** couplings.
  - No 4-body terms, no all-to-all.

Outputs (<outdir>/):
  config.json
  sweep_results.csv
  holdout_strings_smart.txt
  0_story_overview.pdf
  1_heatmap_qH_ratio.pdf                  (IQP qθ(H)/q_unif(H))
  1b_heatmap_qH_ratio_classical.pdf       (Classical baseline q(H)/q_unif(H))
  2_heatmap_Q80.pdf                       (IQP Q80, log color)
  2b_heatmap_Q80_classical.pdf            (Classical baseline Q80, log color)
  3_Q80_pred_vs_meas.pdf                  (IQP predicted vs measured Q80)
  3b_Q80_pred_vs_meas_classical.pdf       (Classical predicted vs measured Q80)
  4_recovery_best.pdf                     (Target vs IQP vs Classical vs Spectral vs Uniform)
  5_moment_mse_vs_qH.pdf                  (Spectral post-proj MSE vs q(H)/q_unif(H))
  6a_adversarial_curves.pdf               (visibility control; spectral completions; paper-target only)
  6b_adversarial_bars.pdf                 (paper-target only)
  7_iqp_training_dynamics.pdf             (best setting; optional)
  7b_classical_training_dynamics.pdf      (best setting; optional)
  9_column_triptych.pdf

Dependencies:
  pip install numpy matplotlib scipy

Pennylane is required if --use-iqp 1 or --use-classical 1:
  pip install pennylane

Run examples:
  # Paper-even target (Appendix-A style)
  python3 -m iqp_generative.core --outdir outputs/exp00_full_validation_paper_even --target-family paper_even

Notes:
  - Full sweeps with n=16, K up to 512, steps=600 are computationally heavy.
    For smoke tests reduce n, Ks, sigmas, or iqp-steps.

Author: you (+ ChatGPT helper)
"""

import os
import json
import math
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Optional: SciPy
try:
    import scipy  # noqa: F401
except Exception:
    scipy = None

# Optional: Pennylane (required for IQP and classical baseline training here)
HAS_PENNYLANE = False
try:
    import pennylane as qml  # type: ignore
    from pennylane import numpy as np  # type: ignore
    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False


# ------------------------------------------------------------------------------
# 1) Visual Style (paper-grade, column/full widths)
# ------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = str(ROOT / "outputs" / "exp00_full_validation")

COL_W = 3.37   # single-column width (inches)
FULL_W = 6.95  # full width (two-column) (inches)

# UNIFORM HEIGHT matching your prior single-column plots
UNIFORM_FIG_HEIGHT = 2.6

COLORS = {
    "target": "#222222",   # almost black
    "model":  "#D62728",   # deep red (IQP)
    "model_xent": "#FF7F0E",  # orange (IQP xent)
    "model_prob_mse": "#2CA02C",  # green (IQP prob MSE)
    "model_mmd": "#1F77B4",  # blue (IQP MMD)
    "gray":   "#666666",
    "blue":   "#1F77B4",   # classical baseline (blue)
}

def _iqp_loss_label(loss: str) -> str:
    name = str(loss).lower()
    if name == "parity_mse":
        return "IQP-QCBM (parity MSE)"
    if name == "xent":
        return "IQP-QCBM (xent)"
    if name == "prob_mse":
        return "IQP-QCBM (prob MSE)"
    if name == "mmd":
        return "IQP-QCBM (MMD)"
    return f"IQP-QCBM ({name})"

def _iqp_loss_color(loss: str) -> str:
    name = str(loss).lower()
    if name == "parity_mse":
        return COLORS["model"]
    if name == "xent":
        return COLORS["model_xent"]
    if name == "prob_mse":
        return COLORS["model_prob_mse"]
    if name == "mmd":
        return COLORS["model_mmd"]
    return COLORS["model"]

def fig_size(mode: str, h: float = None) -> Tuple[float, float]:
    """Returns (width, height)."""
    if h is None:
        h = UNIFORM_FIG_HEIGHT
    if mode not in ("col", "full"):
        raise ValueError("mode must be 'col' or 'full'")
    w = COL_W if mode == "col" else FULL_W
    return (w, h)

def set_style(base: int = 8) -> None:
    """Compact serif style + embedded fonts in PDF."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": base,
        "axes.labelsize": base + 1,
        "axes.titlesize": base + 1,
        "legend.fontsize": base - 1,
        "legend.frameon": False,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _end_label(ax, x, y, txt, color, dy: float = 0.0, fs: int = 8) -> None:
    ax.text(x, y + dy, txt, color=color, fontweight="bold",
            ha="right", va="bottom", fontsize=fs)

def _panel_label(ax, lab: str) -> None:
    ax.text(-0.15, 1.05, lab, transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold")


# ------------------------------------------------------------------------------
# 2) Bit utilities
# ------------------------------------------------------------------------------

def int2bits(k: int, n: int) -> onp.ndarray:
    return onp.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=onp.int8)

def bits_str(bits: onp.ndarray) -> str:
    return "".join("1" if int(b) else "0" for b in bits)

def parity_even(bits: onp.ndarray) -> bool:
    return (int(onp.sum(bits)) % 2) == 0

def make_bits_table(n: int) -> onp.ndarray:
    N = 2 ** n
    return onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)


# ------------------------------------------------------------------------------
# 3) Score-tilted target distributions (paper-style family)
# ------------------------------------------------------------------------------

def longest_zero_run_between_ones(bits: onp.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0

def build_target_distribution_score_tilt(
    n: int,
    beta: float,
    even_parity_only: bool,
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    Shared score-tilted target family:
      - score: s(x)=1+longest zero-run between ones
      - p*(x) ∝ exp(beta * s(x)) on selected support
      - in this project, support is the even-parity sector
    """
    N = 2 ** n
    support = onp.zeros(N, dtype=bool)
    scores = onp.zeros(N, dtype=onp.float64)

    for k in range(N):
        b = int2bits(k, n)
        if (not even_parity_only) or parity_even(b):
            support[k] = True
            scores[k] = 1.0 + float(longest_zero_run_between_ones(b))

    if not support.any():
        raise RuntimeError("Target support is empty.")

    logits = onp.full(N, -onp.inf, dtype=onp.float64)
    logits[support] = beta * scores[support]
    m = float(onp.max(logits[support]))
    unnorm = onp.zeros(N, dtype=onp.float64)
    unnorm[support] = onp.exp(logits[support] - m)
    z = float(unnorm.sum())
    if z <= 0:
        raise RuntimeError("Failed to normalize target distribution.")
    p_star = unnorm / z
    return p_star.astype(onp.float64), support, scores.astype(onp.float64)

def build_target_distribution_paper(n: int, beta: float):
    """
    Legacy paper target alias:
      even-parity support + score tilt.
    """
    return build_target_distribution_score_tilt(n=n, beta=beta, even_parity_only=True)


# ------------------------------------------------------------------------------
# 4) IQP-native "hardness-aligned" target distribution
# ------------------------------------------------------------------------------

def get_iqp_pairs_nn_nnn(n: int) -> List[Tuple[int, int]]:
    """
    Standard ring topology with:
      - nearest-neighbor pairs (i, i+1 mod n)
      - next-nearest-neighbor pairs (i, i+2 mod n)
    Returned as sorted unique undirected pairs.
    """
    pairs = []
    for i in range(n):
        pairs.append(tuple(sorted((i, (i + 1) % n))))
        pairs.append(tuple(sorted((i, (i + 2) % n))))
    pairs = sorted(list(set(pairs)))
    return pairs

def iqp_circuit_zz_only(W, wires, pairs, layers: int = 1):
    """
    IQP circuit family:
      H^{⊗n} Π_{l=1}^L ( Π_{(i,j)∈pairs} exp(-i W_{l,ij} Z_i Z_j / 2) ) H^{⊗n}
    implemented via IsingZZ gates and Hadamard layers.
    """
    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)
    for _ in range(layers):
        for (i, j) in pairs:
            qml.IsingZZ(W[idx], wires=[wires[i], wires[j]])
            idx += 1
        for w in wires:
            qml.Hadamard(wires=w)

# ------------------------------------------------------------------------------
# 5) Good-set selection and sampling
# ------------------------------------------------------------------------------

def topk_mask_by_scores(scores: onp.ndarray, support: onp.ndarray, frac: float = 0.05) -> onp.ndarray:
    valid = onp.where(support)[0]
    k = max(1, int(onp.floor(frac * valid.size)))
    order = onp.argsort(-scores[valid])
    top_indices = valid[order[:k]]
    mask = onp.zeros_like(support, dtype=bool)
    mask[top_indices] = True
    return mask

def sample_indices(probs: onp.ndarray, m: int, seed: int = 7) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    p = probs / probs.sum()
    return rng.choice(len(p), size=m, replace=True, p=p)

def empirical_dist(idxs: onp.ndarray, N: int) -> onp.ndarray:
    c = onp.bincount(idxs, minlength=N)
    return (c / max(1, c.sum())).astype(onp.float64)


# ------------------------------------------------------------------------------
# 6) Holdout selection + export (Appendix A style)
# ------------------------------------------------------------------------------

def _min_hamming_to_set(bit_vec: onp.ndarray, sel_bits: onp.ndarray) -> int:
    if sel_bits.shape[0] == 0:
        return bit_vec.shape[0]
    d = onp.sum(sel_bits != bit_vec[None, :], axis=1)
    return int(onp.min(d))

def select_holdout_smart(
    p_star: onp.ndarray,
    good_mask: onp.ndarray,
    bits_table: onp.ndarray,
    m_train: int,
    holdout_k: int,
    pool_size: int,
    seed: int,
) -> onp.ndarray:
    """
    Smart holdout selection (paper Appendix A):
      1) candidate pool restricted to G
      2) probability floor tau ~ 1/m and relax until enough candidates
      3) keep top M by p*
      4) farthest-point sampling in Hamming distance; tie-break by larger p*
    """
    if holdout_k <= 0:
        return onp.zeros_like(good_mask, dtype=bool)

    rng = onp.random.default_rng(seed)

    good_idxs = onp.where(good_mask)[0]
    taus = [1.0 / max(1, m_train), 0.5 / max(1, m_train), 0.25 / max(1, m_train), 0.0]

    cand = None
    for tau in taus:
        cand = good_idxs[p_star[good_idxs] >= tau]
        if cand.size >= holdout_k:
            break
    if cand is None or cand.size == 0:
        raise RuntimeError("No candidates for holdout selection. Check G or m_train.")

    # Sort by probability, then lightly shuffle to break ties deterministically.
    order = onp.argsort(-p_star[cand])
    cand = cand[order]
    pool = cand[:min(pool_size, cand.size)].copy()
    rng.shuffle(pool)

    pool_sorted = pool[onp.argsort(-p_star[pool])]
    selected = [int(pool_sorted[0])]
    selected_bits = bits_table[[selected[-1]]].copy()

    while len(selected) < holdout_k and len(selected) < pool.size:
        best_idx = None
        best_d = -1
        best_p = -1.0
        for idx in pool:
            idx = int(idx)
            if idx in selected:
                continue
            d = _min_hamming_to_set(bits_table[idx], selected_bits)
            p = float(p_star[idx])
            if (d > best_d) or (d == best_d and p > best_p):
                best_d = d
                best_p = p
                best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))
        selected_bits = onp.vstack([selected_bits, bits_table[best_idx]])

    holdout = onp.zeros_like(good_mask, dtype=bool)
    holdout[onp.array(selected, dtype=int)] = True
    return holdout

def save_holdout_list(
    holdout_mask: onp.ndarray,
    bits_table: onp.ndarray,
    p_star: onp.ndarray,
    scores: onp.ndarray,
    outdir: str,
    name: str = "holdout_strings_smart.txt",
) -> None:
    idxs = onp.where(holdout_mask)[0]
    if idxs.size == 0:
        return
    sorted_idxs = idxs[onp.argsort(-p_star[idxs])]
    path = os.path.join(outdir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Holdout Strings (k={len(idxs)})\n")
        f.write(f"# {'Index':<8} {'Bitstring':<18} {'Score':<12} {'Prob p*(x)':<16}\n")
        f.write("-" * 64 + "\n")
        for idx in sorted_idxs:
            b_str = bits_str(bits_table[int(idx)])
            s_val = float(scores[int(idx)])
            prob = float(p_star[int(idx)])
            f.write(f"{int(idx):<8d} {b_str:<18s} {s_val:<12.4f} {prob:.6e}\n")
    print(f"[Export] Holdout list saved to: {path}")


# ------------------------------------------------------------------------------
# 7) Random parity (Walsh/Fourier) features (Appendix D)
# ------------------------------------------------------------------------------

def p_sigma(sigma: float) -> float:
    # Paper Appendix D: p(σ) = 1/2 (1 - exp(-1/(2σ^2)))
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma ** 2))) if sigma > 0 else 0.5

def sample_alphas(n: int, sigma: float, K: int, seed: int) -> onp.ndarray:
    """
    Sample parity masks α_k ∈ {0,1}^n with α_i ~ Bernoulli(p(σ)),
    excluding the all-zero mask (α=0), since the constant term is handled
    separately in Eq. (14).
    """
    rng = onp.random.default_rng(seed)
    p = p_sigma(sigma)
    alphas = rng.binomial(1, p, size=(K, n)).astype(onp.int8)

    # Resample any all-zero rows (rare but possible)
    zero = onp.where(onp.sum(alphas, axis=1) == 0)[0]
    while zero.size > 0:
        alphas[zero] = rng.binomial(1, p, size=(zero.size, n)).astype(onp.int8)
        zero = onp.where(onp.sum(alphas, axis=1) == 0)[0]

    return alphas

def select_alphas_active(
    n: int,
    sigma: float,
    K: int,
    seed: int,
    bits_table: onp.ndarray,
    roi_mask: onp.ndarray,
    candidate_pool: int = 10000,
    diversity_gamma: float = 0.15,
    normalize_by_full_space: bool = True,
) -> Tuple[onp.ndarray, Dict[str, float]]:
    """
    Active parity-feature acquisition for ROI visibility:
      1) sample candidate alpha masks
      2) score each mask by |1hat_ROI(alpha)|
      3) greedy select top-K with diversity penalty on parity overlap

    Objective (greedy approximation):
      score(alpha) - gamma * mean_{beta in selected} corr(alpha, beta)
    where corr is cosine overlap over binary mask supports.
    """
    N = 2 ** int(n)
    roi_mask = onp.asarray(roi_mask, dtype=bool)
    if roi_mask.size != N:
        raise ValueError(f"roi_mask size mismatch: got {roi_mask.size}, expected {N}.")
    roi_idxs = onp.where(roi_mask)[0]
    if roi_idxs.size == 0:
        raise ValueError("roi_mask is empty; cannot run active selection.")

    target_pool = max(int(candidate_pool), int(K))
    if target_pool <= 0:
        raise ValueError("candidate_pool must be > 0.")

    # Build a deduplicated candidate set (all-zero masks are already excluded by sample_alphas).
    cand = onp.zeros((0, n), dtype=onp.int8)
    need = target_pool
    for trial in range(8):
        draw = max(need * 2, 256)
        new = sample_alphas(n=n, sigma=sigma, K=draw, seed=int(seed) + 7919 * trial)
        cand = onp.vstack([cand, new]) if cand.size else new
        cand = onp.unique(cand, axis=0)
        if cand.shape[0] >= target_pool:
            break
        need = target_pool - cand.shape[0]

    if cand.shape[0] < K:
        raise RuntimeError(
            f"Active candidate generation failed: only {cand.shape[0]} unique masks for K={K}. "
            "Try larger candidate_pool or different sigma."
        )
    cand = cand[:target_pool]

    # ROI visibility score: hat{1_ROI}(alpha) = (1/denom) sum_{x in ROI} (-1)^{alpha.x}
    # Using parity bits directly avoids materializing full P over all states.
    bits_roi_t = bits_table[roi_idxs].astype(onp.int16).T  # n x |ROI|
    A = cand.astype(onp.int16)  # M x n
    odd = onp.sum((A @ bits_roi_t) & 1, axis=1).astype(onp.float64)
    roi_size = float(roi_idxs.size)
    denom = float(N) if normalize_by_full_space else roi_size
    hat_roi = ((roi_size - 2.0 * odd) / max(1.0, denom)).astype(onp.float64)
    score = onp.abs(hat_roi)

    # Greedy selection with incremental diversity penalty.
    weights = onp.sum(cand, axis=1).astype(onp.float64)
    avail = onp.ones(cand.shape[0], dtype=bool)
    penalty_acc = onp.zeros(cand.shape[0], dtype=onp.float64)
    selected: List[int] = []

    for t in range(int(K)):
        if t == 0:
            obj = score.copy()
        else:
            obj = score - float(diversity_gamma) * (penalty_acc / float(t))
        obj = obj.astype(onp.float64)
        obj[~avail] = -onp.inf
        j = int(onp.argmax(obj))
        if not onp.isfinite(obj[j]):
            break
        selected.append(j)
        avail[j] = False

        # Update diversity penalties for remaining candidates.
        s = cand[j].astype(onp.float64)
        wj = max(1.0, float(weights[j]))
        inter = cand.astype(onp.float64) @ s  # support overlap with selected mask
        corr = inter / onp.sqrt(onp.maximum(1.0, weights * wj))
        penalty_acc += corr

    if len(selected) < K:
        raise RuntimeError(f"Active selection produced only {len(selected)} masks for requested K={K}.")

    sel = onp.array(selected, dtype=int)
    alphas_sel = cand[sel].astype(onp.int8)

    # Diagnostics
    sel_scores = score[sel]
    sel_hat = hat_roi[sel]
    avg_abs_corr = 0.0
    if len(sel) > 1:
        S = alphas_sel.astype(onp.float64)
        ws = onp.sum(S, axis=1)
        G = S @ S.T
        denom_mat = onp.sqrt(onp.maximum(1.0, ws[:, None] * ws[None, :]))
        C = G / denom_mat
        triu = onp.triu_indices(len(sel), k=1)
        if triu[0].size > 0:
            avg_abs_corr = float(onp.mean(onp.abs(C[triu])))

    info = {
        "candidate_pool": int(cand.shape[0]),
        "roi_size": int(roi_idxs.size),
        "score_mean_all": float(onp.mean(score)),
        "score_mean_selected": float(onp.mean(sel_scores)),
        "score_max_selected": float(onp.max(sel_scores)),
        "signed_hat_mean_selected": float(onp.mean(sel_hat)),
        "avg_abs_pair_corr_selected": float(avg_abs_corr),
    }
    return alphas_sel, info

def build_parity_matrix(alphas: onp.ndarray, bits_table: onp.ndarray) -> onp.ndarray:
    """P[k, x] = φ_{α_k}(x) = (-1)^{α_k · x} in {+1,-1}."""
    A = alphas.astype(onp.int16)
    X = bits_table.astype(onp.int16).T  # n x N
    par = (A @ X) & 1
    return onp.where(par == 0, 1.0, -1.0).astype(onp.float64)

def alpha_stats(alphas: onp.ndarray) -> Tuple[int, float]:
    """(K_unique, mean Hamming weight)."""
    a_view = alphas.view([("", alphas.dtype)] * alphas.shape[1])
    K_unique = int(onp.unique(a_view).shape[0])
    mean_wt = float(onp.mean(onp.sum(alphas, axis=1)))
    return K_unique, mean_wt


# ------------------------------------------------------------------------------
# 8) Result-2 core: q_lin, completion q_tilde, visibility, Lemma-1 checks
# ------------------------------------------------------------------------------

def linear_band_reconstruction(P: onp.ndarray, z: onp.ndarray, n: int) -> onp.ndarray:
    """Eq. (14): q_lin = (1/N) * (1 + P^T z)."""
    N = 2 ** n
    return (1.0 / N) * (1.0 + (P.T @ z))

def completion_by_axioms(q_lin: onp.ndarray) -> onp.ndarray:
    """Eq. (15): q_tilde(x) ∝ max(0, q_lin(x))."""
    q = onp.clip(q_lin, 0.0, None)
    s = float(q.sum())
    if s <= 0:
        q = onp.ones_like(q, dtype=onp.float64) / q.size
    else:
        q = q / s
    return q.astype(onp.float64)

def reconstruct_bandlimited(P: onp.ndarray, z: onp.ndarray, n: int, return_q_lin: bool = False):
    """Wrapper: build q_lin (Eq. 14) then completion (Eq. 15)."""
    q_lin = linear_band_reconstruction(P, z, n)
    q_tilde = completion_by_axioms(q_lin)
    return (q_tilde, q_lin) if return_q_lin else (q_tilde, None)

def indicator_walsh_coeffs(P: onp.ndarray, holdout_mask: onp.ndarray, n: int) -> onp.ndarray:
    """Eq. (16): 1̂_H(α_k) = (1/N) Σ_{x∈H} φ_{α_k}(x)."""
    N = 2 ** n
    H_idxs = onp.where(holdout_mask)[0]
    if H_idxs.size == 0:
        return onp.zeros(P.shape[0], dtype=onp.float64)
    return (onp.sum(P[:, H_idxs], axis=1) / N).astype(onp.float64)

def visibility_score(P: onp.ndarray, z: onp.ndarray, holdout_mask: onp.ndarray, n: int) -> float:
    """Def. 1 Eq. (17): Vis_B(H) = Σ_k z_k * 1̂_H(α_k)."""
    hat1H = indicator_walsh_coeffs(P, holdout_mask, n)
    return float(onp.dot(z.astype(onp.float64), hat1H))

def lemma1_residual(P: onp.ndarray, z: onp.ndarray, q_lin: onp.ndarray, holdout_mask: onp.ndarray, n: int) -> Dict[str, float]:
    """Lemma 1 Eq. (18): q_lin(H) = |H|/N + Vis_B(H)."""
    N = 2 ** n
    H_size = int(onp.sum(holdout_mask))
    if H_size == 0:
        return dict(q_lin_H=0.0, rhs=0.0, Vis=0.0, abs_err=0.0)
    q_lin_H = float(onp.sum(q_lin[holdout_mask]))
    Vis = visibility_score(P, z, holdout_mask, n)
    rhs = float(H_size / N) + Vis
    abs_err = float(abs(q_lin_H - rhs))
    return dict(q_lin_H=q_lin_H, rhs=rhs, Vis=Vis, abs_err=abs_err)


# ------------------------------------------------------------------------------
# 9) Result-1 discovery metrics: recovery, Q80, predictor, bounds
# ------------------------------------------------------------------------------

def expected_unique_fraction(probs: onp.ndarray, mask: onp.ndarray, Q_vals: onp.ndarray) -> onp.ndarray:
    """R(Q) = (1/|H|) Σ_{x∈H} (1 - (1 - q(x))^Q)."""
    Q_vals = onp.array(Q_vals, dtype=int)
    H = int(onp.sum(mask))
    if H == 0:
        return onp.zeros_like(Q_vals, dtype=onp.float64)
    pS = probs[mask].astype(onp.float64)[:, None]  # H x 1
    return onp.sum(1.0 - onp.power(1.0 - pS, Q_vals[None, :]), axis=0) / H

def find_Q_threshold(probs: onp.ndarray, mask: onp.ndarray, thr: float = 0.8, Qmax: int = 200000) -> float:
    """Smallest integer Q such that expected_unique_fraction >= thr."""
    H = int(onp.sum(mask))
    if H == 0:
        return float("nan")

    def frac(Q: int) -> float:
        return float(expected_unique_fraction(probs, mask, onp.array([Q]))[0])

    if frac(1) >= thr:
        return 1.0

    lo, hi = 1, 1
    while hi < Qmax and frac(hi) < thr:
        hi *= 2
    if hi >= Qmax and frac(Qmax) < thr:
        return float("inf")
    hi = min(hi, Qmax)

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if frac(mid) >= thr:
            hi = mid
        else:
            lo = mid
    return float(hi)

def Q80_prediction_from_qH(qH: float, H_size: int) -> float:
    """Eq. (13): Q80 ≈ |H|/q(H) * ln 5."""
    if H_size <= 0 or qH <= 0:
        return float("inf")
    return float((H_size / qH) * math.log(5.0))

def Q80_bestcase_lower_bound(qH: float, H_size: int) -> float:
    """Prop. 1 Eq. (8): Q80 >= ln(0.2)/ln(1 - q(H)/|H|)."""
    if H_size <= 0:
        return float("nan")
    mu = float(qH)
    if mu <= 0:
        return float("inf")
    eps = mu / float(H_size)
    if eps >= 1.0:
        return 1.0
    denom = math.log(1.0 - eps)
    if denom >= 0:
        return float("nan")
    return float(math.log(0.2) / denom)

def moment_mse(P: onp.ndarray, q: onp.ndarray, z: onp.ndarray) -> float:
    """Appendix B Eq. (B2): mean((P@q - z)^2)."""
    r = (P @ q) - z
    return float(onp.mean(r ** 2))

# ------------------------------------------------------------------------------
# 10) Plotting helpers
# ------------------------------------------------------------------------------

def _safe_log10_for_heatmap(data: onp.ndarray, clip_min: float = 1e-12) -> onp.ndarray:
    data = onp.array(data, dtype=onp.float64)
    plot_data = onp.log10(onp.clip(data, clip_min, None))
    finite = onp.isfinite(plot_data)
    if finite.any():
        vmax = float(onp.max(plot_data[finite]))
        plot_data = plot_data.copy()
        plot_data[~finite] = vmax + 1.0  # push inf to top color
    else:
        plot_data = onp.zeros_like(plot_data)
    return plot_data

def plot_heatmap(mat, row_labels, col_labels, cbar_label, outpath,
                 log10: bool = False, fmt: str = "{:.0f}", cmap="magma", mode: str = "col"):
    data = onp.array(mat, dtype=onp.float64)
    plot_data = _safe_log10_for_heatmap(data) if log10 else data

    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    im = ax.imshow(plot_data, aspect="auto", cmap=cmap)

    ax.set_xticks(onp.arange(len(col_labels)))
    ax.set_yticks(onp.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("K (number of parity features)")
    ax.set_ylabel(r"$\sigma$ (feature sparsity)")

    # Subtle cell grid
    ax.set_xticks(onp.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(onp.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.18)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(cbar_label)

    n_cells = data.shape[0] * data.shape[1]
    fs = 7 if n_cells > 12 else 8
    max_pd = float(onp.max(plot_data[onp.isfinite(plot_data)])) if onp.isfinite(plot_data).any() else 1.0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = r"$\infty$" if onp.isinf(val) else fmt.format(val)

            text_col = "white"
            if isinstance(cmap, str) and cmap in ("Reds", "YlOrRd", "Oranges", "Greys"):
                if plot_data[i, j] < max_pd * 0.5:
                    text_col = "black"

            ax.text(j, i, txt, ha="center", va="center",
                    color=text_col, fontsize=fs, fontweight="bold")

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_prediction_scatter(Q80_meas, Q80_pred, outpath, color_edge, mode: str = "col", label_line: str = r"$y=x$"):
    x = onp.array(Q80_pred, dtype=onp.float64)
    y = onp.array(Q80_meas, dtype=onp.float64)
    m = onp.isfinite(x) & onp.isfinite(y) & (x > 0) & (y > 0)
    xf, yf = x[m], y[m]
    if xf.size == 0:
        return

    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    ax.scatter(xf, yf, s=45, facecolors="white",
               edgecolors=color_edge, linewidths=1.6, alpha=0.9, zorder=3)

    lo = float(min(onp.min(xf), onp.min(yf)))
    hi = float(max(onp.max(xf), onp.max(yf)))
    xs = onp.logspace(onp.log10(lo), onp.log10(hi), 100)
    ax.plot(xs, xs, color=COLORS["target"], linestyle="--", linewidth=1.6, alpha=0.7, label=label_line)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Predicted $Q_{80} \approx \frac{|H|}{q(H)}\ln 5$")
    ax.set_ylabel(r"Measured $Q_{80}$")

    lx = onp.log10(xf)
    ly = onp.log10(yf)
    if lx.size >= 2:
        r = onp.corrcoef(lx, ly)[0, 1]
        ax.text(0.05, 0.95, f"log-corr = {r:.3f}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

    ax.legend(loc="lower right")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_moment_mse_vs_qH(results, outpath, mode: str = "col"):
    xs = onp.array([r["moment_mse_spec"] for r in results], dtype=onp.float64)
    ys = onp.array([r["qH_ratio_spec"] for r in results], dtype=onp.float64)

    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    ax.scatter(xs, ys, s=45, facecolors="white",
               edgecolors=COLORS["model"], linewidths=1.6, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Post-projection moment MSE (spectral)")
    ax.set_ylabel(r"$q(H)/q_{\mathrm{unif}}(H)$ (spectral)")

    top = onp.argsort(-ys)[:3]
    for i in top:
        ax.annotate(results[int(i)]["label"],
                    (xs[int(i)], ys[int(i)]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color=COLORS["target"])

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_recovery_best_comparison(
    p_star: onp.ndarray,
    q_iqp: Optional[onp.ndarray],
    q_class: Optional[onp.ndarray],
    q_spec: onp.ndarray,
    holdout_mask: onp.ndarray,
    outpath: str,
    Qmax: int = 10000,
    mode: str = "col",
):
    """Recovery comparison: Target, IQP, Classical, Spectral completion, Uniform."""
    H = int(onp.sum(holdout_mask))
    if H == 0:
        return

    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 120).astype(int)),
        onp.linspace(1000, Qmax, 160).astype(int),
    ]))
    Q = Q[Q <= Qmax]

    y_star = expected_unique_fraction(p_star, holdout_mask, Q)
    y_spec = expected_unique_fraction(q_spec, holdout_mask, Q)

    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"Target $p^*$", zorder=6)

    if q_iqp is not None:
        y_iqp = expected_unique_fraction(q_iqp, holdout_mask, Q)
        ax.plot(Q, y_iqp, color=COLORS["model"], linewidth=2.2, label=r"IQP-QCBM $q_\theta$", zorder=7)

    if q_class is not None:
        y_c = expected_unique_fraction(q_class, holdout_mask, Q)
        ax.plot(Q, y_c, color=COLORS["blue"], linestyle=":", linewidth=2.0,
                label=r"Classical baseline $q_{\mathrm{cl}}$", zorder=5)

    ax.plot(Q, y_spec, color="#555555", linestyle="-.", linewidth=1.9, label=r"Spectral completion $\~q$", zorder=4)

    u = onp.ones_like(p_star) / p_star.size
    y_u = expected_unique_fraction(u, holdout_mask, Q)
    ax.plot(Q, y_u, color=COLORS["gray"], linewidth=1.6, linestyle="--", alpha=0.9, label="Uniform", zorder=3)

    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_recovery_iqp_loss_compare(
    p_star: onp.ndarray,
    q_iqp_parity: Optional[onp.ndarray],
    q_iqp_xent: Optional[onp.ndarray],
    q_spec: onp.ndarray,
    holdout_mask: onp.ndarray,
    outpath: str,
    Qmax: int = 10000,
    mode: str = "col",
    label_a: str = "IQP-QCBM (parity MSE)",
    label_b: str = "IQP-QCBM (xent)",
    color_a: str = COLORS["model"],
    color_b: str = COLORS["model_xent"],
):
    """Recovery comparison: Target, two IQP losses, Spectral completion, Uniform."""
    H = int(onp.sum(holdout_mask))
    if H == 0:
        return

    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 120).astype(int)),
        onp.linspace(1000, Qmax, 160).astype(int),
    ]))
    Q = Q[Q <= Qmax]

    y_star = expected_unique_fraction(p_star, holdout_mask, Q)
    y_spec = expected_unique_fraction(q_spec, holdout_mask, Q)

    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"Target $p^*$", zorder=6)

    if q_iqp_parity is not None:
        y_iqp_p = expected_unique_fraction(q_iqp_parity, holdout_mask, Q)
        ax.plot(Q, y_iqp_p, color=color_a, linewidth=2.2,
                label=label_a, zorder=7)

    if q_iqp_xent is not None:
        y_iqp_x = expected_unique_fraction(q_iqp_xent, holdout_mask, Q)
        ax.plot(Q, y_iqp_x, color=color_b, linewidth=2.1,
                label=label_b, zorder=6)

    ax.plot(Q, y_spec, color="#555555", linestyle="-.", linewidth=1.9,
            label=r"Spectral completion $\~q$", zorder=4)

    u = onp.ones_like(p_star) / p_star.size
    y_u = expected_unique_fraction(u, holdout_mask, Q)
    ax.plot(Q, y_u, color=COLORS["gray"], linewidth=1.6, linestyle="--", alpha=0.9,
            label="Uniform", zorder=3)

    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_story_overview(
    qH_ratio_mat_iqp,
    Q80_mat_iqp,
    sigmas,
    Ks,
    Q80_meas_iqp,
    Q80_pred_iqp,
    p_star,
    q_spec_best,
    q_iqp_best,
    q_class_best,
    holdout_mask,
    outpath,
    cmap_custom,
):
    """2x2 full-width summary (main panels are IQP; recovery panel includes classical)."""
    fig, axes = plt.subplots(2, 2, figsize=fig_size("full", 4.8), constrained_layout=True)

    # (1) IQP qH ratio heatmap
    ax = axes[0, 0]
    im = ax.imshow(qH_ratio_mat_iqp, aspect="auto", cmap="Reds", vmin=0.0)
    ax.set_xticks(onp.arange(len(Ks)))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_yticks(onp.arange(len(sigmas)))
    ax.set_yticklabels([str(s) for s in sigmas])
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\sigma$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$q_\theta(H)/q_{\mathrm{unif}}(H)$")
    mmax = float(onp.max(qH_ratio_mat_iqp)) if onp.isfinite(qH_ratio_mat_iqp).any() else 1.0
    for i in range(qH_ratio_mat_iqp.shape[0]):
        for j in range(qH_ratio_mat_iqp.shape[1]):
            tcol = "black" if qH_ratio_mat_iqp[i, j] < mmax * 0.5 else "white"
            ax.text(j, i, f"{qH_ratio_mat_iqp[i, j]:.1f}",
                    ha="center", va="center", color=tcol,
                    fontsize=7, fontweight="bold")

    # (2) IQP Q80 heatmap (Red->Black, log10)
    ax = axes[0, 1]
    plot_data = _safe_log10_for_heatmap(Q80_mat_iqp, clip_min=1e-9)
    im = ax.imshow(plot_data, aspect="auto", cmap=cmap_custom)
    ax.set_xticks(onp.arange(len(Ks)))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_yticks(onp.arange(len(sigmas)))
    ax.set_yticklabels([str(s) for s in sigmas])
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\sigma$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\log_{10} Q_{80}$")
    for i in range(Q80_mat_iqp.shape[0]):
        for j in range(Q80_mat_iqp.shape[1]):
            v = Q80_mat_iqp[i, j]
            if onp.isinf(v):
                txt = r"$\infty$"
            elif not onp.isfinite(v):
                txt = "nan"
            else:
                txt = f"{int(v):d}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")

    # (3) IQP Pred vs Meas scatter
    ax = axes[1, 0]
    x = onp.array(Q80_pred_iqp, dtype=onp.float64)
    y = onp.array(Q80_meas_iqp, dtype=onp.float64)
    m = onp.isfinite(x) & onp.isfinite(y) & (x > 0) & (y > 0)
    xf, yf = x[m], y[m]
    if xf.size > 0:
        ax.scatter(xf, yf, s=38, facecolors="white",
                   edgecolors=COLORS["model"], linewidths=1.4, alpha=0.9)
        lo = float(min(onp.min(xf), onp.min(yf)))
        hi = float(max(onp.max(xf), onp.max(yf)))
        xs = onp.logspace(onp.log10(lo), onp.log10(hi), 100)
        ax.plot(xs, xs, color=COLORS["target"], linestyle="--", linewidth=1.6, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lx = onp.log10(xf)
        ly = onp.log10(yf)
        if lx.size >= 2:
            r = onp.corrcoef(lx, ly)[0, 1]
            ax.text(0.05, 0.95, f"log-corr = {r:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    ax.set_xlabel("Predicted $Q_{80}$")
    ax.set_ylabel("Measured $Q_{80}$")

    # (4) Recovery curve comparison (includes classical)
    ax = axes[1, 1]
    H = int(onp.sum(holdout_mask))
    if H > 0:
        Q = onp.unique(onp.concatenate([
            onp.unique(onp.logspace(0, 3.5, 110).astype(int)),
            onp.linspace(1000, 10000, 120).astype(int),
        ]))
        Q = Q[Q <= 10000]
        y_star = expected_unique_fraction(p_star, holdout_mask, Q)
        y_spec = expected_unique_fraction(q_spec_best, holdout_mask, Q)
        y_iqp = expected_unique_fraction(q_iqp_best, holdout_mask, Q) if q_iqp_best is not None else None
        y_cl = expected_unique_fraction(q_class_best, holdout_mask, Q) if q_class_best is not None else None
        u = onp.ones_like(p_star) / p_star.size
        y_u = expected_unique_fraction(u, holdout_mask, Q)

        ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.8, label=r"$p^*$")
        if y_iqp is not None:
            ax.plot(Q, y_iqp, color=COLORS["model"], linewidth=2.1, label=r"$q_\theta$")
        if y_cl is not None:
            ax.plot(Q, y_cl, color=COLORS["blue"], linestyle=":", linewidth=1.9, label=r"$q_{\mathrm{cl}}$")
        ax.plot(Q, y_spec, color="#555555", linestyle="-.", linewidth=1.7, label=r"$\~q$")
        ax.plot(Q, y_u, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="uniform")
        ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$Q$")
        ax.set_ylabel(r"$R(Q)$")
        ax.legend(loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_column_triptych(results, p_star, q_spec_best, q_iqp_best, q_class_best, holdout_mask, outpath):
    """Triptych: (a) IQP pred vs meas, (b) spectral MSE vs qH_ratio, (c) recovery curve with classical."""
    fig, axs = plt.subplots(1, 3, figsize=fig_size("full", 2.5), constrained_layout=True)

    # (a) IQP scatter
    ax = axs[0]
    _panel_label(ax, "(a)")
    x = onp.array([r["Q80_pred_iqp"] for r in results])
    y = onp.array([r["Q80_iqp"] for r in results])
    m = onp.isfinite(x) & onp.isfinite(y) & (x > 0) & (y > 0)
    xf, yf = x[m], y[m]
    if xf.size > 0:
        ax.scatter(xf, yf, s=38, facecolors="white", edgecolors=COLORS["model"], linewidths=1.4, alpha=0.9)
        lo = float(min(onp.min(xf), onp.min(yf)))
        hi = float(max(onp.max(xf), onp.max(yf)))
        xs = onp.logspace(onp.log10(lo), onp.log10(hi), 100)
        ax.plot(xs, xs, color=COLORS["target"], linestyle="--", linewidth=1.4, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        lx = onp.log10(xf)
        ly = onp.log10(yf)
        if lx.size >= 2:
            r = onp.corrcoef(lx, ly)[0, 1]
            ax.text(0.05, 0.95, f"log-corr = {r:.3f}",
                    transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    ax.set_xlabel("Predicted $Q_{80}$")
    ax.set_ylabel("Measured $Q_{80}$")

    # (b) Spectral MSE vs qH_ratio
    ax = axs[1]
    _panel_label(ax, "(b)")
    xs = onp.array([r["moment_mse_spec"] for r in results])
    ys = onp.array([r["qH_ratio_spec"] for r in results])
    ax.scatter(xs, ys, s=38, facecolors="white", edgecolors=COLORS["model"], linewidths=1.4, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Moment MSE (spectral)")
    ax.set_ylabel(r"$q(H)/q_{\mathrm{unif}}(H)$ (spectral)")

    # (c) Recovery curve
    ax = axs[2]
    _panel_label(ax, "(c)")
    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 120).astype(int)),
        onp.linspace(1000, 10000, 160).astype(int),
    ]))
    Q = Q[Q <= 10000]
    y_star = expected_unique_fraction(p_star, holdout_mask, Q)
    y_spec = expected_unique_fraction(q_spec_best, holdout_mask, Q)
    u = onp.ones_like(p_star) / p_star.size
    y_u = expected_unique_fraction(u, holdout_mask, Q)

    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"$p^*$")
    if q_iqp_best is not None:
        y_iqp = expected_unique_fraction(q_iqp_best, holdout_mask, Q)
        ax.plot(Q, y_iqp, color=COLORS["model"], linewidth=2.2, label=r"$q_\theta$")
    if q_class_best is not None:
        y_cl = expected_unique_fraction(q_class_best, holdout_mask, Q)
        ax.plot(Q, y_cl, color=COLORS["blue"], linestyle=":", linewidth=2.0, label=r"$q_{\mathrm{cl}}$")
    ax.plot(Q, y_spec, color="#555555", linestyle="-.", linewidth=1.7, label=r"$\~q$")
    ax.plot(Q, y_u, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="uniform")
    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$R(Q)$")
    ax.legend(loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_training_dynamics_generic(hist: Dict[str, List[float]], outpath: str, color: str, ylab: str = "Loss"):
    st = onp.array(hist["step"], dtype=int)
    loss = onp.array(hist["loss"], dtype=onp.float64)

    fig, ax = plt.subplots(figsize=fig_size("col"), constrained_layout=True)
    ax.plot(st, loss, color=color, linewidth=2.1)
    ax.set_yscale("log")
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylab)
    _end_label(ax, st[-1], loss[-1], f"{loss[-1]:.2e}", color, dy=0.05 * loss[-1], fs=8)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")


# ------------------------------------------------------------------------------
# 11) Adversarial visibility plots (paper-target only)
# ------------------------------------------------------------------------------

def plot_adversarial_visibility_split(p_star, holdout_visible, holdout_invisible, q_vis, q_inv, outdir):
    """Two separate panels (6a curves, 6b bars) with requested final styling."""
    H = int(onp.sum(holdout_visible))
    assert H == int(onp.sum(holdout_invisible)), "Holdout sizes must match."

    # --- Plot 6a: Curves ---
    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 110).astype(int)),
        onp.linspace(1000, 10000, 120).astype(int),
    ]))
    Q = Q[Q <= 10000]

    y_star = expected_unique_fraction(p_star, holdout_visible, Q)
    y_vis = expected_unique_fraction(q_vis, holdout_visible, Q)
    y_inv = expected_unique_fraction(q_inv, holdout_invisible, Q)

    u = onp.ones_like(p_star) / p_star.size
    y_u = expected_unique_fraction(u, holdout_visible, Q)

    fig, ax = plt.subplots(figsize=fig_size("col"), constrained_layout=True)
    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"Target $p^*$", zorder=4)
    ax.plot(Q, y_vis,  color=COLORS["model"],  linewidth=2.2, label=r"Visible $H_{\mathrm{vis}}$", zorder=6)

    # Invisible curve: dark gray + dash-dot
    ax.plot(Q, y_inv,  color="#555555", linestyle="-.", linewidth=1.9,
            label=r"Invisible $H_{\mathrm{inv}}$", zorder=5)

    ax.plot(Q, y_u, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="Uniform", zorder=3)
    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$R(Q)$")

    # Legend in free space
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78), frameon=False)

    outpath_a = os.path.join(outdir, "6a_adversarial_curves.pdf")
    fig.savefig(outpath_a)
    plt.close(fig)
    print(f"[Saved] {outpath_a}")

    # --- Plot 6b: Bars ---
    Qbars = onp.array([1000, 10000], dtype=int)
    b_star = expected_unique_fraction(p_star, holdout_visible, Qbars)
    b_vis = expected_unique_fraction(q_vis, holdout_visible, Qbars)
    b_inv = expected_unique_fraction(q_inv, holdout_invisible, Qbars)
    b_u = expected_unique_fraction(u, holdout_visible, Qbars)

    fig, ax = plt.subplots(figsize=fig_size("col"), constrained_layout=True)
    x = onp.arange(len(Qbars))
    w = 0.23

    ax.bar(x - w, b_star, w, color="white", edgecolor="#333333", hatch="///",
           alpha=0.9, label=r"Target $p^*$")
    ax.bar(x, b_vis, w, color=COLORS["model"], edgecolor="black",
           alpha=0.85, label=r"$H_{\mathrm{vis}}$")
    ax.bar(x + w, b_inv, w, color="white", edgecolor="black", hatch="XX",
           alpha=1.0, label=r"$H_{\mathrm{inv}}$")

    ax.axhline(float(b_u[1]), color=COLORS["gray"], linestyle="--", alpha=0.6)
    ax.text(1.05, float(b_u[1]) + 0.03, "Uniform", color=COLORS["gray"],
            va="bottom", ha="left", fontsize=7)

    def _lbl(xpos, val, color):
        ax.text(xpos, float(val) + 0.03, f"{100*float(val):.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)

    for i in range(len(Qbars)):
        _lbl(x[i] - w, b_star[i], "#333333")
        _lbl(x[i],     b_vis[i],  "#800000")
        _lbl(x[i] + w, b_inv[i],  "black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q={int(q)}" for q in Qbars])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recovered fraction")
    ax.legend(loc="upper left", frameon=False)

    outpath_b = os.path.join(outdir, "6b_adversarial_bars.pdf")
    fig.savefig(outpath_b)
    plt.close(fig)
    print(f"[Saved] {outpath_b}")


# ------------------------------------------------------------------------------
# 12) Training: IQP-QCBM and Classical baseline (same objective)
# ------------------------------------------------------------------------------

def _hamming_rbf_kernel_matrix(n: int, tau: float) -> onp.ndarray:
    """
    Build kernel matrix K[x,x'] = exp(-d_H(x,x') / tau) over all 2^n bitstrings.
    """
    if tau <= 0.0:
        raise ValueError("mmd_tau must be > 0.")
    N = 2 ** int(n)
    # n=12 in this project, so uint16 is sufficient and memory-safe.
    dtype = onp.uint16 if n <= 16 else onp.uint32
    states = onp.arange(N, dtype=dtype)
    xor = onp.bitwise_xor(states[:, None], states[None, :])
    max_x = int(onp.max(xor))
    pop_lut = onp.array([int(i).bit_count() for i in range(max_x + 1)], dtype=onp.float64)
    d_h = pop_lut[xor]
    return onp.exp(-d_h / float(tau)).astype(onp.float64)


def _build_mmd_kernel_matrix(n: int, kernel: str, tau: float) -> onp.ndarray:
    name = str(kernel).strip().lower()
    if name == "hamming_rbf":
        return _hamming_rbf_kernel_matrix(n=n, tau=tau)
    raise ValueError(f"Unsupported mmd kernel: {kernel}")


def train_iqp_qcbm(
    n: int,
    layers: int,
    steps: int,
    lr: float,
    P: Optional[onp.ndarray],
    z_data: Optional[onp.ndarray],
    seed_init: int,
    eval_every: int = 50,
    return_hist: bool = False,
    loss_mode: str = "parity_mse",
    xent_emp: Optional[onp.ndarray] = None,
    xent_eps: float = 1e-12,
    mmd_tau: float = 2.0,
    mmd_kernel: str = "hamming_rbf",
):
    """Train IQP-QCBM with parity-moment MSE, prob-MSE, xent, or MMD loss."""
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is not installed. Install with `pip install pennylane` or set --use-iqp 0.")

    dev = qml.device("default.qubit", wires=n)
    pairs = get_iqp_pairs_nn_nnn(n)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit_zz_only(W, range(n), pairs, layers=layers)
        return qml.probs(wires=range(n))

    loss_mode = str(loss_mode).lower()
    K_t = None
    if loss_mode == "parity_mse":
        if P is None or z_data is None:
            raise ValueError("parity_mse loss requires P and z_data.")
        P_t = np.array(P, requires_grad=False)
        z_t = np.array(z_data, requires_grad=False)
    elif loss_mode == "prob_mse":
        if xent_emp is None:
            raise ValueError("prob_mse loss requires xent_emp (empirical distribution).")
        emp_t = np.array(xent_emp, requires_grad=False)
        emp_t = emp_t / np.sum(emp_t)
    elif loss_mode == "xent":
        if xent_emp is None:
            raise ValueError("xent loss requires xent_emp (empirical distribution).")
        emp_t = np.array(xent_emp, requires_grad=False)
        emp_t = emp_t / np.sum(emp_t)
    elif loss_mode == "mmd":
        if xent_emp is None:
            raise ValueError("mmd loss requires xent_emp (empirical distribution).")
        emp_t = np.array(xent_emp, requires_grad=False)
        emp_t = emp_t / np.sum(emp_t)
        K_np = _build_mmd_kernel_matrix(n=n, kernel=mmd_kernel, tau=float(mmd_tau))
        # Keep K as an autograd tensor to ensure dot-products remain differentiable wrt q.
        K_t = np.array(K_np, requires_grad=True)
    else:
        raise ValueError("loss_mode must be 'parity_mse', 'prob_mse', 'xent', or 'mmd'.")

    num_params = len(pairs) * layers
    rng = onp.random.default_rng(seed_init)
    W = np.array(0.01 * rng.standard_normal(num_params), requires_grad=True)

    opt = qml.AdamOptimizer(lr)
    hist = {"step": [], "loss": []} if return_hist else None

    def loss_fn(w):
        q = circuit(w)
        if loss_mode == "parity_mse":
            return np.mean((z_t - P_t @ q) ** 2)
        if loss_mode == "prob_mse":
            return np.mean((q - emp_t) ** 2)
        if loss_mode == "mmd":
            assert K_t is not None
            d = q - emp_t
            return np.dot(d, np.dot(K_t, d))
        q_clip = np.clip(q, xent_eps, 1.0)
        return -np.sum(emp_t * np.log(q_clip))

    loss_val = None
    for t in range(1, steps + 1):
        W, loss_val = opt.step_and_cost(loss_fn, W)
        if return_hist and ((t % eval_every == 0) or (t == 1) or (t == steps)):
            hist["step"].append(int(t))
            hist["loss"].append(float(loss_val))

    q_final = onp.array(circuit(W), dtype=onp.float64)
    q_final = onp.clip(q_final, 0.0, 1.0)
    q_final /= max(1e-15, float(q_final.sum()))

    return q_final.astype(onp.float64), float(loss_val) if loss_val is not None else float("nan"), hist

def train_classical_ising_baseline(
    n: int,
    layers: int,
    steps: int,
    lr: float,
    P: onp.ndarray,
    z_data: onp.ndarray,
    seed_init: int,
    eval_every: int = 50,
    return_hist: bool = False,
):
    """
    Classical baseline with full Ising expressivity:
      q_cl(x) ∝ exp( Σ_{(i,j)∈NN+NNN} J_{ij} s_i s_j  +  Σ_i h_i s_i ),
      with s_i = (-1)^{x_i} ∈ {+1,-1}.

    Includes local fields (h_i) for full single-layer expressivity.
    The `layers` parameter is accepted for API compatibility but does not
    multiply the parameter count: an Ising model has no meaningful depth
    (summing coupling constants across layers is equivalent to a single layer).

    Trained with EXACT same moment-MSE loss as IQP-QCBM:
      loss = mean((z_data - P @ q_cl)^2)
    using the same Adam optimizer, steps, lr.
    """
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for the classical baseline (we reuse qml.AdamOptimizer).")

    pairs = get_iqp_pairs_nn_nnn(n)
    num_pairs = len(pairs)

    # Precompute spins for all bitstrings
    bits = make_bits_table(n)  # N x n
    spins = 1.0 - 2.0 * bits.astype(onp.float64)  # {+1,-1}, N x n
    N = spins.shape[0]

    # Precompute pair products F_pair[k,x] = s_i s_j
    pair_feats = onp.zeros((num_pairs, N), dtype=onp.float64)
    for k, (i, j) in enumerate(pairs):
        pair_feats[k] = spins[:, i] * spins[:, j]

    # Local field features F_field[i,x] = s_i
    field_feats = spins.T.copy()  # n x N

    # Combined feature matrix: pair interactions + local fields
    F = onp.concatenate([pair_feats, field_feats], axis=0)  # (num_pairs + n) x N
    num_features = F.shape[0]

    F_t = np.array(F, requires_grad=False)
    P_t = np.array(P, requires_grad=False)
    z_t = np.array(z_data, requires_grad=False)

    rng = onp.random.default_rng(seed_init)
    J = np.array(0.01 * rng.standard_normal(num_features), requires_grad=True)

    opt = qml.AdamOptimizer(lr)
    hist = {"step": [], "loss": []} if return_hist else None

    def softmax(logits):
        m = np.max(logits)
        ex = np.exp(logits - m)
        return ex / np.sum(ex)

    def loss_fn(J_flat):
        logits = np.dot(J_flat, F_t)            # (N,)
        q = softmax(logits)
        return np.mean((z_t - P_t @ q) ** 2)

    loss_val = None
    for t in range(1, steps + 1):
        J, loss_val = opt.step_and_cost(loss_fn, J)
        if return_hist and ((t % eval_every == 0) or (t == 1) or (t == steps)):
            hist["step"].append(int(t))
            hist["loss"].append(float(loss_val))

    # Final q
    J_np = onp.array(J, dtype=onp.float64)
    logits = J_np @ F  # (N,)
    logits = logits - float(logits.max())
    q = onp.exp(logits)
    q = q / float(q.sum())

    return q.astype(onp.float64), float(loss_val) if loss_val is not None else float("nan"), hist


# ------------------------------------------------------------------------------
# 13) Main experiment logic
# ------------------------------------------------------------------------------

@dataclass
class Config:
    n: int = 12
    beta: float = 0.9
    train_m: int = 1000
    holdout_k: int = 20
    holdout_pool: int = 400
    seed: int = 42

    good_frac: float = 0.05

    sigmas: List[float] = None
    Ks: List[int] = None

    Qmax: int = 10000
    Q80_thr: float = 0.8
    Q80_search_max: int = 200000

    # target family (fixed in this project scope)
    target_family: str = "paper_even"

    # Adversarial demo (paper-target only)
    adversarial: bool = True
    adv_score_level: int = 7
    adv_sigma: float = 1.0
    adv_K: int = 512

    # IQP training (Result 1 validation)
    use_iqp: bool = True
    iqp_steps: int = 600
    iqp_lr: float = 0.05
    iqp_eval_every: int = 50
    iqp_layers: int = 1
    iqp_loss: str = "parity_mse"  # "parity_mse", "prob_mse", "xent", or "mmd"
    iqp_mmd_tau: float = 2.0      # Hamming-RBF bandwidth for mmd
    iqp_mmd_kernel: str = "hamming_rbf"

    # Classical baseline (same budget)
    use_classical: bool = True

    outdir: str = DEFAULT_OUTDIR

def compute_metrics_for_q(q: onp.ndarray, holdout_mask: onp.ndarray, qH_unif: float, H_size: int,
                          Q80_thr: float, Q80_search_max: int) -> Dict[str, float]:
    qH = float(q[holdout_mask].sum()) if H_size > 0 else 0.0
    qH_ratio = float(qH / qH_unif) if qH_unif > 0 else float("nan")
    R1000 = float(expected_unique_fraction(q, holdout_mask, onp.array([1000]))[0]) if H_size > 0 else 0.0
    R10000 = float(expected_unique_fraction(q, holdout_mask, onp.array([10000]))[0]) if H_size > 0 else 0.0
    Q80 = find_Q_threshold(q, holdout_mask, thr=Q80_thr, Qmax=Q80_search_max) if H_size > 0 else float("nan")
    Q80_pred = Q80_prediction_from_qH(qH, H_size) if H_size > 0 else float("inf")
    Q80_lb = Q80_bestcase_lower_bound(qH, H_size) if H_size > 0 else float("nan")
    return dict(qH=qH, qH_ratio=qH_ratio, R_Q1000=R1000, R_Q10000=R10000, Q80=Q80, Q80_pred=Q80_pred, Q80_lb=Q80_lb)

def run_sweep(cfg: Config, p_star: onp.ndarray, holdout_mask: onp.ndarray, good_mask: onp.ndarray, bits_table: onp.ndarray) -> List[Dict]:
    """Full sweep over (σ,K): spectral completion + IQP + classical baseline."""
    if (cfg.use_iqp or cfg.use_classical) and not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP/classical training. Install with `pip install pennylane`.")

    N = p_star.size
    H_size = int(onp.sum(holdout_mask))

    # Training distribution (holdout removed): Eq. (3)
    p_train = p_star.copy()
    if H_size > 0:
        p_train[holdout_mask] = 0.0
        p_train /= p_train.sum()

    # Fixed training dataset
    idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.seed + 7)
    emp = empirical_dist(idxs_train, N)

    # Uniform baseline
    q_unif = onp.ones(N, dtype=onp.float64) / N
    qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0

    maxK = int(max(cfg.Ks))
    results: List[Dict] = []

    for si, sigma in enumerate(cfg.sigmas):
        # Sample a superset of masks for nesting across K
        alphas_sup = sample_alphas(cfg.n, float(sigma), maxK, seed=cfg.seed + 222 + 13 * si)
        P_sup = build_parity_matrix(alphas_sup, bits_table)

        for K in cfg.Ks:
            alphas = alphas_sup[:K]
            P = P_sup[:K]

            # Target moments from data (finite-sample)
            z = P @ emp

            # --- Spectral completion (Result 2) ---
            q_spec, q_lin = reconstruct_bandlimited(P, z, cfg.n, return_q_lin=True)
            assert q_lin is not None

            lemma = lemma1_residual(P, z, q_lin, holdout_mask, cfg.n)
            if lemma["abs_err"] > 1e-9:
                raise RuntimeError(f"Lemma 1 check failed (abs_err={lemma['abs_err']:.3e}) at sigma={sigma}, K={K}.")

            spec_metrics = compute_metrics_for_q(q_spec, holdout_mask, qH_unif, H_size, cfg.Q80_thr, cfg.Q80_search_max)
            mse_spec = moment_mse(P, q_spec, z)

            # --- IQP training (Result 1) ---
            q_iqp = None
            iqp_metrics = dict(qH=float("nan"), qH_ratio=float("nan"), R_Q1000=float("nan"),
                               R_Q10000=float("nan"), Q80=float("nan"), Q80_pred=float("nan"), Q80_lb=float("nan"))
            loss_iqp = float("nan")

            loss_mode = str(cfg.iqp_loss).lower()
            seed_term_k = 7 * int(K) if loss_mode == "parity_mse" else 0
            seed_init = cfg.seed + 10000 + 97 * si + seed_term_k

            if cfg.use_iqp:
                q_iqp, loss_iqp, _ = train_iqp_qcbm(
                    n=cfg.n,
                    layers=cfg.iqp_layers,
                    steps=cfg.iqp_steps,
                    lr=cfg.iqp_lr,
                    P=P,
                    z_data=z,
                    seed_init=seed_init,
                    eval_every=cfg.iqp_eval_every,
                    return_hist=False,
                    loss_mode=loss_mode,
                    xent_emp=emp,
                    mmd_tau=cfg.iqp_mmd_tau,
                    mmd_kernel=cfg.iqp_mmd_kernel,
                )
                iqp_metrics = compute_metrics_for_q(q_iqp, holdout_mask, qH_unif, H_size, cfg.Q80_thr, cfg.Q80_search_max)

            # --- Classical baseline training (same objective/budget) ---
            q_class = None
            class_metrics = dict(qH=float("nan"), qH_ratio=float("nan"), R_Q1000=float("nan"),
                                 R_Q10000=float("nan"), Q80=float("nan"), Q80_pred=float("nan"), Q80_lb=float("nan"))
            loss_class = float("nan")

            if cfg.use_classical:
                q_class, loss_class, _ = train_classical_ising_baseline(
                    n=cfg.n,
                    layers=cfg.iqp_layers,
                    steps=cfg.iqp_steps,
                    lr=cfg.iqp_lr,
                    P=P,
                    z_data=z,
                    seed_init=seed_init + 999,  # deterministic offset for baseline
                    eval_every=cfg.iqp_eval_every,
                    return_hist=False,
                )
                class_metrics = compute_metrics_for_q(q_class, holdout_mask, qH_unif, H_size, cfg.Q80_thr, cfg.Q80_search_max)

            Ku, mean_wt = alpha_stats(alphas)

            res = dict(
                sigma=float(sigma),
                K=int(K),
                K_unique=int(Ku),
                mean_alpha_wt=float(mean_wt),

                # IQP (Result 1)
                iqp_loss_mode=str(cfg.iqp_loss),
                qH_iqp=float(iqp_metrics["qH"]),
                qH_ratio_iqp=float(iqp_metrics["qH_ratio"]),
                R_iqp_Q1000=float(iqp_metrics["R_Q1000"]),
                R_iqp_Q10000=float(iqp_metrics["R_Q10000"]),
                Q80_iqp=float(iqp_metrics["Q80"]),
                Q80_pred_iqp=float(iqp_metrics["Q80_pred"]),
                Q80_lb_iqp=float(iqp_metrics["Q80_lb"]),
                train_loss_iqp=float(loss_iqp),

                # Classical baseline
                qH_class=float(class_metrics["qH"]),
                qH_ratio_class=float(class_metrics["qH_ratio"]),
                R_class_Q1000=float(class_metrics["R_Q1000"]),
                R_class_Q10000=float(class_metrics["R_Q10000"]),
                Q80_class=float(class_metrics["Q80"]),
                Q80_pred_class=float(class_metrics["Q80_pred"]),
                Q80_lb_class=float(class_metrics["Q80_lb"]),
                train_loss_class=float(loss_class),

                # Spectral completion (Result 2)
                qH_spec=float(spec_metrics["qH"]),
                qH_ratio_spec=float(spec_metrics["qH_ratio"]),
                R_spec_Q1000=float(spec_metrics["R_Q1000"]),
                R_spec_Q10000=float(spec_metrics["R_Q10000"]),
                Q80_spec=float(spec_metrics["Q80"]),
                Q80_pred_spec=float(spec_metrics["Q80_pred"]),
                Q80_lb_spec=float(spec_metrics["Q80_lb"]),
                moment_mse_spec=float(mse_spec),

                # Visibility + Lemma 1 diagnostics
                Vis=float(lemma["Vis"]),
                q_lin_H=float(lemma["q_lin_H"]),
                lemma1_rhs=float(lemma["rhs"]),
                lemma1_abs_err=float(lemma["abs_err"]),

                label=f"s={sigma:g},K={K:d}",
            )
            results.append(res)

            msg_iqp = f"IQP qH={res['qH_iqp']:.2e} (x{res['qH_ratio_iqp']:.2f}) Q80={res['Q80_iqp']:.0f}" if cfg.use_iqp else ""
            msg_cl = f"CL qH={res['qH_class']:.2e} (x{res['qH_ratio_class']:.2f}) Q80={res['Q80_class']:.0f}" if cfg.use_classical else ""
            print(
                f"[Sweep] sigma={sigma:g}, K={K:4d} | "
                f"Spec qH={res['qH_spec']:.2e} (x{res['qH_ratio_spec']:.2f}) Q80={res['Q80_spec']:.0f} | "
                f"{msg_iqp} | {msg_cl}"
            )

    return results

def save_results_csv(results: List[Dict], outpath: str):
    keys = list(results[0].keys())
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[Saved] {outpath}")

def build_result_matrix(results: List[Dict], sigmas: List[float], Ks: List[int], key: str) -> onp.ndarray:
    sigma_to_i = {float(s): i for i, s in enumerate(sigmas)}
    K_to_j = {int(K): j for j, K in enumerate(Ks)}
    mat = onp.full((len(sigmas), len(Ks)), onp.nan, dtype=onp.float64)
    for r in results:
        i = sigma_to_i[float(r["sigma"])]
        j = K_to_j[int(r["K"])]
        mat[i, j] = float(r[key])
    return mat

def pick_best_setting(results: List[Dict], prefer: str = "iqp") -> Dict:
    """
    Choose best by minimal finite Q80 on a preferred model.
    prefer in {"iqp","classical","spectral"}.
    Tie-break by larger qH_ratio.
    """
    if prefer == "iqp":
        finite = [r for r in results if onp.isfinite(r["Q80_iqp"])]
        if finite:
            finite.sort(key=lambda r: (r["Q80_iqp"], -r["qH_ratio_iqp"]))
            return finite[0]
    if prefer == "classical":
        finite = [r for r in results if onp.isfinite(r["Q80_class"])]
        if finite:
            finite.sort(key=lambda r: (r["Q80_class"], -r["qH_ratio_class"]))
            return finite[0]
    finite = [r for r in results if onp.isfinite(r["Q80_spec"])]
    if finite:
        finite.sort(key=lambda r: (r["Q80_spec"], -r["qH_ratio_spec"]))
        return finite[0]
    return results[0]

def rerun_single_setting(cfg: Config, p_star: onp.ndarray, holdout_mask: onp.ndarray, bits_table: onp.ndarray,
                        sigma: float, K: int, return_hist: bool = True, iqp_loss: Optional[str] = None) -> Dict[str, object]:
    """Rebuild P,z and return q_spec, q_iqp, q_class plus optional training histories."""
    if (cfg.use_iqp or cfg.use_classical) and not HAS_PENNYLANE:
        raise RuntimeError("Pennylane required. Install or disable IQP/classical baselines.")

    N = p_star.size
    H_size = int(onp.sum(holdout_mask))

    p_train = p_star.copy()
    if H_size > 0:
        p_train[holdout_mask] = 0.0
        p_train /= p_train.sum()

    idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.seed + 7)
    emp = empirical_dist(idxs_train, N)

    si = [i for i, s in enumerate(cfg.sigmas) if float(s) == float(sigma)][0]
    maxK = int(max(cfg.Ks))
    alphas_sup = sample_alphas(cfg.n, float(sigma), maxK, seed=cfg.seed + 222 + 13 * si)
    P_sup = build_parity_matrix(alphas_sup, bits_table)

    alphas = alphas_sup[:K]
    P = P_sup[:K]
    z = P @ emp

    q_spec, q_lin = reconstruct_bandlimited(P, z, cfg.n, return_q_lin=True)
    assert q_lin is not None
    lemma = lemma1_residual(P, z, q_lin, holdout_mask, cfg.n)
    if lemma["abs_err"] > 1e-9:
        raise RuntimeError(f"Lemma 1 check failed at rerun (abs_err={lemma['abs_err']:.3e}).")

    loss_mode = cfg.iqp_loss if iqp_loss is None else str(iqp_loss)
    loss_mode = str(loss_mode).lower()
    seed_term_k = 7 * int(K) if loss_mode == "parity_mse" else 0
    seed_init = cfg.seed + 10000 + 97 * si + seed_term_k

    q_iqp, loss_iqp, hist_iqp = (None, float("nan"), None)
    if cfg.use_iqp:
        q_iqp, loss_iqp, hist_iqp = train_iqp_qcbm(
            n=cfg.n, layers=cfg.iqp_layers, steps=cfg.iqp_steps, lr=cfg.iqp_lr,
            P=P, z_data=z, seed_init=seed_init, eval_every=cfg.iqp_eval_every, return_hist=return_hist,
            loss_mode=loss_mode, xent_emp=emp, mmd_tau=cfg.iqp_mmd_tau, mmd_kernel=cfg.iqp_mmd_kernel
        )

    q_class, loss_class, hist_class = (None, float("nan"), None)
    if cfg.use_classical:
        q_class, loss_class, hist_class = train_classical_ising_baseline(
            n=cfg.n, layers=cfg.iqp_layers, steps=cfg.iqp_steps, lr=cfg.iqp_lr,
            P=P, z_data=z, seed_init=seed_init + 999, eval_every=cfg.iqp_eval_every, return_hist=return_hist
        )

    return dict(
        alphas=alphas, P=P, z=z,
        q_spec=q_spec, q_lin=q_lin, lemma=lemma,
        q_iqp=q_iqp, loss_iqp=loss_iqp, hist_iqp=hist_iqp,
        q_class=q_class, loss_class=loss_class, hist_class=hist_class,
    )


# ------------------------------------------------------------------------------
# 14) Adversarial demo (paper-target only)
# ------------------------------------------------------------------------------

def run_adversarial_demo_paper_target(cfg: Config, p_star: onp.ndarray, scores: onp.ndarray, good_mask: onp.ndarray,
                                      bits_table: onp.ndarray, outdir: str):
    """
    Adversarial: pick H_vis vs H_inv with same p*(H) by choosing within one score level.
    This relies on the discrete score structure of the paper target (Appendix A).
    """
    N = p_star.size
    s_int = scores.astype(int)

    from collections import Counter
    score_counts = Counter(s_int[good_mask].tolist())
    preferred = int(cfg.adv_score_level)

    if score_counts.get(preferred, 0) < cfg.holdout_k:
        viable = [(s, c) for (s, c) in score_counts.items() if c >= cfg.holdout_k]
        if not viable:
            raise RuntimeError(
                "Adversarial demo: no score level in the good set has enough states for holdout_k. "
                "Try smaller --holdout-k or change the target distribution / n."
            )
        preferred = max(viable, key=lambda t: t[1])[0]
        print(f"[Adv] adv_score_level={cfg.adv_score_level} has only {score_counts.get(int(cfg.adv_score_level), 0)} states; "
              f"using score={preferred} (count={score_counts[preferred]}).")
    else:
        print(f"[Adv] using score level {preferred} (count={score_counts[preferred]}).")

    cand = onp.where(good_mask & (s_int == preferred))[0]

    # Fixed feature set
    alphas = sample_alphas(cfg.n, float(cfg.adv_sigma), int(cfg.adv_K), seed=cfg.seed + 999)
    P = build_parity_matrix(alphas, bits_table)

    # Reference moments from full target p*
    z_ref = P @ p_star
    q_lin_ref = linear_band_reconstruction(P, z_ref, cfg.n)

    q_c = q_lin_ref[cand]
    order = onp.argsort(q_c)  # ascending
    H_inv_idxs = cand[order[:cfg.holdout_k]]
    H_vis_idxs = cand[order[-cfg.holdout_k:]]

    holdout_inv = onp.zeros(N, dtype=bool); holdout_inv[H_inv_idxs] = True
    holdout_vis = onp.zeros(N, dtype=bool); holdout_vis[H_vis_idxs] = True

    pH_vis = float(p_star[holdout_vis].sum())
    pH_inv = float(p_star[holdout_inv].sum())
    print(f"[Adv] p*(H_vis)={pH_vis:.6f}, p*(H_inv)={pH_inv:.6f} (should match)")

    # Train spectral completions for each holdout set
    def train_and_reconstruct(holdout_mask_local: onp.ndarray, seed_offset: int) -> onp.ndarray:
        p_train = p_star.copy()
        p_train[holdout_mask_local] = 0.0
        p_train /= p_train.sum()
        idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.seed + 7 + seed_offset)
        emp = empirical_dist(idxs_train, N)
        z = P @ emp
        q_tilde, q_lin = reconstruct_bandlimited(P, z, cfg.n, return_q_lin=True)
        assert q_lin is not None
        lemma = lemma1_residual(P, z, q_lin, holdout_mask_local, cfg.n)
        if lemma["abs_err"] > 1e-9:
            raise RuntimeError(f"[Adv] Lemma1 failed (abs_err={lemma['abs_err']:.3e}).")
        return q_tilde

    q_vis = train_and_reconstruct(holdout_vis, seed_offset=0)
    q_inv = train_and_reconstruct(holdout_inv, seed_offset=123)

    save_holdout_list(holdout_vis, bits_table, p_star, scores, outdir, name="holdout_strings_visible.txt")
    save_holdout_list(holdout_inv, bits_table, p_star, scores, outdir, name="holdout_strings_invisible.txt")

    plot_adversarial_visibility_split(p_star, holdout_vis, holdout_inv, q_vis, q_inv, outdir)


# ------------------------------------------------------------------------------
# 15) CLI
# ------------------------------------------------------------------------------

def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)

    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--train-m", type=int, default=1000)

    parser.add_argument("--good-frac", type=float, default=0.05)

    parser.add_argument("--holdout-k", type=int, default=20)
    parser.add_argument("--holdout-pool", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigmas", type=str, default="0.5,1,2,3")
    parser.add_argument("--Ks", type=str, default="128,256,512")

    parser.add_argument("--Qmax", type=int, default=10000)
    parser.add_argument("--Q80-thr", type=float, default=0.8)
    parser.add_argument("--Q80-search-max", type=int, default=200000)

    # Target family selection
    parser.add_argument(
        "--target-family",
        type=str,
        default="paper_even",
        choices=["paper_even", "paper"],
    )

    # Adversarial (paper_even target only)
    parser.add_argument("--adversarial", type=int, default=1)
    parser.add_argument("--adv-score-level", type=int, default=7)
    parser.add_argument("--adv-sigma", type=float, default=1.0)
    parser.add_argument("--adv-K", type=int, default=512)

    # IQP training
    parser.add_argument("--use-iqp", type=int, default=1)
    parser.add_argument("--use-classical", type=int, default=1)
    parser.add_argument("--iqp-steps", type=int, default=600)
    parser.add_argument("--iqp-lr", type=float, default=0.05)
    parser.add_argument("--iqp-eval-every", type=int, default=50)
    parser.add_argument("--iqp-layers", type=int, default=1)
    parser.add_argument("--iqp-loss", type=str, default="parity_mse", choices=["parity_mse", "prob_mse", "xent", "mmd"])
    parser.add_argument("--iqp-mmd-tau", type=float, default=2.0)
    parser.add_argument("--iqp-mmd-kernel", type=str, default="hamming_rbf", choices=["hamming_rbf"])
    parser.add_argument("--compare-iqp-losses", type=int, default=0)
    parser.add_argument("--compare-sigma", type=float, default=None)
    parser.add_argument("--compare-K", type=int, default=None)
    parser.add_argument("--compare-loss-a", type=str, default="parity_mse",
                        choices=["parity_mse", "prob_mse", "xent", "mmd"])
    parser.add_argument("--compare-loss-b", type=str, default="xent",
                        choices=["parity_mse", "prob_mse", "xent", "mmd"])

    args = parser.parse_args()

    target_family = str(args.target_family).strip().lower()
    # Backward-compatible alias used in older configs/scripts.
    if target_family == "paper":
        target_family = "paper_even"
    if target_family != "paper_even":
        raise ValueError("Only target-family=paper_even is supported in this codebase.")

    set_style(base=8)
    outdir = ensure_outdir(args.outdir)

    cfg = Config(
        n=args.n,
        beta=args.beta,
        train_m=args.train_m,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
        good_frac=args.good_frac,
        sigmas=_parse_list_floats(args.sigmas),
        Ks=_parse_list_ints(args.Ks),
        Qmax=args.Qmax,
        Q80_thr=args.Q80_thr,
        Q80_search_max=args.Q80_search_max,

        target_family=target_family,

        adversarial=bool(args.adversarial),
        adv_score_level=args.adv_score_level,
        adv_sigma=args.adv_sigma,
        adv_K=args.adv_K,

        use_iqp=bool(args.use_iqp),
        use_classical=bool(args.use_classical),
        iqp_steps=args.iqp_steps,
        iqp_lr=args.iqp_lr,
        iqp_eval_every=args.iqp_eval_every,
        iqp_layers=args.iqp_layers,
        iqp_loss=args.iqp_loss,
        iqp_mmd_tau=args.iqp_mmd_tau,
        iqp_mmd_kernel=args.iqp_mmd_kernel,

        outdir=outdir,
    )

    # Save config
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print("[Config]", cfg)

    bits_table = make_bits_table(cfg.n)

    # --- Build target distribution ---
    if cfg.target_family != "paper_even":
        raise ValueError(f"Unsupported target family: {cfg.target_family}")
    p_star, support, scores = build_target_distribution_paper(cfg.n, cfg.beta)
    print("[Target] family=paper_even (even parity + score tilt)")

    # Good set selection
    good_mask = topk_mask_by_scores(scores, support, frac=cfg.good_frac)
    print(f"[Target] |G|={int(onp.sum(good_mask))} | p*(G)={float(p_star[good_mask].sum()):.6f}")

    # Smart holdout
    holdout_mask = select_holdout_smart(
        p_star=p_star,
        good_mask=good_mask,
        bits_table=bits_table,
        m_train=cfg.train_m,
        holdout_k=cfg.holdout_k,
        pool_size=cfg.holdout_pool,
        seed=cfg.seed + 111,
    )
    print(f"[Holdout] k={int(onp.sum(holdout_mask))} | p*(H)={float(p_star[holdout_mask].sum()):.6f}")
    save_holdout_list(holdout_mask, bits_table, p_star, scores, outdir, name="holdout_strings_smart.txt")

    # --- Sweep ---
    results = run_sweep(cfg, p_star, holdout_mask, good_mask, bits_table)

    # Save CSV
    csv_path = os.path.join(outdir, "sweep_results.csv")
    save_results_csv(results, csv_path)

    # Matrices
    qH_ratio_mat_iqp = build_result_matrix(results, cfg.sigmas, cfg.Ks, key="qH_ratio_iqp")
    Q80_mat_iqp = build_result_matrix(results, cfg.sigmas, cfg.Ks, key="Q80_iqp")

    qH_ratio_mat_cl = build_result_matrix(results, cfg.sigmas, cfg.Ks, key="qH_ratio_class")
    Q80_mat_cl = build_result_matrix(results, cfg.sigmas, cfg.Ks, key="Q80_class")

    # Custom red-black colormap for Q80 heatmap
    cmap_rb = LinearSegmentedColormap.from_list("RedBlack", ["#FF0000", "#000000"])

    # PLOT 1: IQP qH ratio
    plot_heatmap(
        qH_ratio_mat_iqp,
        row_labels=[str(s) for s in cfg.sigmas],
        col_labels=[str(k) for k in cfg.Ks],
        cbar_label=r"$q_\theta(H)/q_{\mathrm{unif}}(H)$",
        outpath=os.path.join(outdir, "1_heatmap_qH_ratio.pdf"),
        log10=False,
        fmt="{:.1f}",
        cmap="Reds",
        mode="col",
    )

    # PLOT 1b: Classical qH ratio
    plot_heatmap(
        qH_ratio_mat_cl,
        row_labels=[str(s) for s in cfg.sigmas],
        col_labels=[str(k) for k in cfg.Ks],
        cbar_label=r"$q_{\mathrm{cl}}(H)/q_{\mathrm{unif}}(H)$",
        outpath=os.path.join(outdir, "1b_heatmap_qH_ratio_classical.pdf"),
        log10=False,
        fmt="{:.1f}",
        cmap="Blues",
        mode="col",
    )

    # PLOT 2: IQP Q80
    plot_heatmap(
        Q80_mat_iqp,
        row_labels=[str(s) for s in cfg.sigmas],
        col_labels=[str(k) for k in cfg.Ks],
        cbar_label=r"$\log_{10} Q_{80}$",
        outpath=os.path.join(outdir, "2_heatmap_Q80.pdf"),
        log10=True,
        fmt="{:.0f}",
        cmap=cmap_rb,
        mode="col",
    )

    # PLOT 2b: Classical Q80
    plot_heatmap(
        Q80_mat_cl,
        row_labels=[str(s) for s in cfg.sigmas],
        col_labels=[str(k) for k in cfg.Ks],
        cbar_label=r"$\log_{10} Q_{80}$",
        outpath=os.path.join(outdir, "2b_heatmap_Q80_classical.pdf"),
        log10=True,
        fmt="{:.0f}",
        cmap=cmap_rb,
        mode="col",
    )

    # PLOT 3: IQP predicted vs measured
    plot_prediction_scatter(
        Q80_meas=[r["Q80_iqp"] for r in results],
        Q80_pred=[r["Q80_pred_iqp"] for r in results],
        outpath=os.path.join(outdir, "3_Q80_pred_vs_meas.pdf"),
        color_edge=COLORS["model"],
        mode="col",
    )

    # PLOT 3b: Classical predicted vs measured
    plot_prediction_scatter(
        Q80_meas=[r["Q80_class"] for r in results],
        Q80_pred=[r["Q80_pred_class"] for r in results],
        outpath=os.path.join(outdir, "3b_Q80_pred_vs_meas_classical.pdf"),
        color_edge=COLORS["blue"],
        mode="col",
    )

    # PLOT 5: spectral moment MSE vs spectral qH_ratio
    plot_moment_mse_vs_qH(
        results,
        os.path.join(outdir, "5_moment_mse_vs_qH.pdf"),
        mode="col",
    )

    # Best setting (prefer IQP if enabled; else classical; else spectral)
    prefer = "iqp" if cfg.use_iqp else ("classical" if cfg.use_classical else "spectral")
    best = pick_best_setting(results, prefer=prefer)
    best_sigma = float(best["sigma"])
    best_K = int(best["K"])
    print(f"[Best] sigma={best_sigma:g}, K={best_K} | Q80_iqp={best['Q80_iqp']:.0f} | Q80_class={best['Q80_class']:.0f} | Q80_spec={best['Q80_spec']:.0f}")

    best_art = rerun_single_setting(cfg, p_star, holdout_mask, bits_table, best_sigma, best_K, return_hist=True)
    q_spec_best = best_art["q_spec"]      # type: ignore
    q_iqp_best = best_art["q_iqp"]        # type: ignore
    q_class_best = best_art["q_class"]    # type: ignore
    hist_iqp = best_art["hist_iqp"]       # type: ignore
    hist_class = best_art["hist_class"]   # type: ignore

    # Training dynamics plots (best)
    if cfg.use_iqp and hist_iqp is not None:
        if cfg.iqp_loss == "parity_mse":
            iqp_ylab = "Moment MSE loss"
        elif cfg.iqp_loss == "prob_mse":
            iqp_ylab = "Prob MSE loss"
        elif cfg.iqp_loss == "mmd":
            iqp_ylab = "MMD loss"
        else:
            iqp_ylab = "NLL loss"
        plot_training_dynamics_generic(hist_iqp, os.path.join(outdir, "7_iqp_training_dynamics.pdf"),
                                       color=COLORS["model"], ylab=iqp_ylab)
    if cfg.use_classical and hist_class is not None:
        plot_training_dynamics_generic(hist_class, os.path.join(outdir, "7b_classical_training_dynamics.pdf"),
                                       color=COLORS["blue"], ylab="Moment MSE loss")

    # PLOT 4: recovery curve comparison
    plot_recovery_best_comparison(
        p_star=p_star,
        q_iqp=q_iqp_best,
        q_class=q_class_best,
        q_spec=q_spec_best,
        holdout_mask=holdout_mask,
        outpath=os.path.join(outdir, "4_recovery_best.pdf"),
        Qmax=cfg.Qmax,
        mode="col",
    )

    # Optional: compare IQP losses (parity MSE vs xent) at a single (sigma, K)
    if bool(args.compare_iqp_losses):
        if not cfg.use_iqp:
            print("[Compare] Skipping IQP loss comparison because --use-iqp 0.")
        else:
            sigma_cmp = float(args.compare_sigma) if args.compare_sigma is not None else best_sigma
            K_cmp = int(args.compare_K) if args.compare_K is not None else best_K
            loss_a = args.compare_loss_a
            loss_b = args.compare_loss_b
            print(f"[Compare] IQP losses at sigma={sigma_cmp:g}, K={K_cmp:d} | {loss_a} vs {loss_b}")

            art_a = rerun_single_setting(cfg, p_star, holdout_mask, bits_table, sigma_cmp, K_cmp,
                                         return_hist=True, iqp_loss=loss_a)
            art_b = rerun_single_setting(cfg, p_star, holdout_mask, bits_table, sigma_cmp, K_cmp,
                                         return_hist=True, iqp_loss=loss_b)

            q_spec_cmp = art_a["q_spec"]  # type: ignore
            q_iqp_a = art_a["q_iqp"]  # type: ignore
            q_iqp_b = art_b["q_iqp"]  # type: ignore

            plot_recovery_iqp_loss_compare(
                p_star=p_star,
                q_iqp_parity=q_iqp_a,
                q_iqp_xent=q_iqp_b,
                q_spec=q_spec_cmp,
                holdout_mask=holdout_mask,
                outpath=os.path.join(outdir, "8_iqp_loss_compare_recovery.pdf"),
                Qmax=cfg.Qmax,
                mode="col",
                label_a=_iqp_loss_label(loss_a),
                label_b=_iqp_loss_label(loss_b),
                color_a=_iqp_loss_color(loss_a),
                color_b=_iqp_loss_color(loss_b),
            )

            # Save quick metrics snapshot
            N = p_star.size
            H_size = int(onp.sum(holdout_mask))
            q_unif = onp.ones(N, dtype=onp.float64) / N
            qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0
            metrics_a = compute_metrics_for_q(q_iqp_a, holdout_mask, qH_unif, H_size,
                                              cfg.Q80_thr, cfg.Q80_search_max) if q_iqp_a is not None else {}
            metrics_b = compute_metrics_for_q(q_iqp_b, holdout_mask, qH_unif, H_size,
                                              cfg.Q80_thr, cfg.Q80_search_max) if q_iqp_b is not None else {}
            compare_out = {
                "sigma": sigma_cmp,
                "K": K_cmp,
                "loss_a": loss_a,
                "loss_b": loss_b,
                loss_a: metrics_a,
                loss_b: metrics_b,
            }
            with open(os.path.join(outdir, "8_iqp_loss_compare_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(compare_out, f, indent=2)

    # PLOT 0: story overview (IQP panels + recovery includes classical)
    plot_story_overview(
        qH_ratio_mat_iqp=qH_ratio_mat_iqp,
        Q80_mat_iqp=Q80_mat_iqp,
        sigmas=cfg.sigmas,
        Ks=cfg.Ks,
        Q80_meas_iqp=[r["Q80_iqp"] for r in results],
        Q80_pred_iqp=[r["Q80_pred_iqp"] for r in results],
        p_star=p_star,
        q_spec_best=q_spec_best,
        q_iqp_best=q_iqp_best,
        q_class_best=q_class_best,
        holdout_mask=holdout_mask,
        outpath=os.path.join(outdir, "0_story_overview.pdf"),
        cmap_custom=cmap_rb,
    )

    # PLOT 9: column triptych
    plot_column_triptych(
        results,
        p_star,
        q_spec_best,
        q_iqp_best,
        q_class_best,
        holdout_mask,
        os.path.join(outdir, "9_column_triptych.pdf"),
    )

    # --- Adversarial visibility demo (paper-target only) ---
    if cfg.adversarial:
        if cfg.target_family != "paper_even":
            print("[Adv] Skipping adversarial demo: it is defined for the paper_even target with discrete score levels.")
        else:
            run_adversarial_demo_paper_target(cfg, p_star, scores, good_mask, bits_table, outdir)

    print(f"Done. Results in ./{outdir}/")

if __name__ == "__main__":
    main()
