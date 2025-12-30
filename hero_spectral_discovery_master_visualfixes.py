#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hero Spectral Holdout-Discovery Master Script
=============================================

This is the **working master script** with the **final visual fixes** from your
"FINAL VISUAL FIXES - GRAY CURVES" version integrated:

Visual updates (as requested):
  - Plot 2 (Q80 heatmap): **Red→Black** gradient colormap.
  - Plot 6a (Curves): "Invisible" curve is **Dark Gray** with **dash-dot** style.
  - Plot 6b (Bars): "Invisible" bars are **White** with **Karo/XX** hatch pattern.
  - Overall style: compact, paper-grade (Times/Serif), column/full figure sizes,
    lighter grid, embedded fonts in PDF.
  - **UPDATE**: Single-column plots strictly match '2_heatmap_Q80' size (Height=2.6).
  - **UPDATE**: Plot 6a Legend moved to "free space" (top-right area, but below the saturated curves).

Outputs (<outdir>/):
  sweep_results.csv
  holdout_strings_smart.txt
  0_story_overview.pdf
  1_heatmap_qH_ratio.pdf
  2_heatmap_Q80.pdf
  3_Q80_pred_vs_meas.pdf
  4_recovery_best.pdf
  5_moment_mse_vs_qH.pdf
  6a_adversarial_curves.pdf      (if --adversarial 1)
  6b_adversarial_bars.pdf        (if --adversarial 1)
  7_iqp_training_dynamics.pdf    (if --use-iqp 1 and pennylane available)
  8_iqp_vs_spectral_recovery.pdf (if --use-iqp 1)
  9_column_triptych.pdf

Dependencies (spectral):
  pip install numpy matplotlib scipy

Optional (IQP):
  pip install pennylane

Author: you (+ ChatGPT helper)
"""

import os
import json
import math
import csv
import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Optional: SciPy
try:
    import scipy  # noqa: F401
except Exception:
    scipy = None

# Optional: Pennylane (only if --use-iqp 1)
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

COL_W = 3.37   # single-column width (inches)
FULL_W = 6.95  # full width (two-column) (inches)

# UNIFORM HEIGHT matching '2_heatmap_Q80' (2.6 inches)
UNIFORM_FIG_HEIGHT = 2.6

COLORS = {
    "target": "#222222",   # almost black
    "model":  "#D62728",   # deep red
    "gray":   "#666666",
    "blue":   "#1F77B4",   # kept for IQP vs spectral plot
}

def fig_size(mode: str, h: float = None) -> Tuple[float, float]:
    """
    Returns (width, height).
    If h is not provided, defaults to UNIFORM_FIG_HEIGHT (2.6) for consistency.
    """
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
# 2) Target distribution + utilities
# ------------------------------------------------------------------------------

def int2bits(k: int, n: int) -> onp.ndarray:
    return onp.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=onp.int8)

def bits_str(bits: onp.ndarray) -> str:
    return "".join("1" if int(b) else "0" for b in bits)

def parity_even(bits: onp.ndarray) -> bool:
    return (int(onp.sum(bits)) % 2) == 0

def longest_zero_run_between_ones(bits: onp.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0

def make_bits_table(n: int) -> onp.ndarray:
    N = 2 ** n
    return onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)

def build_target_distribution(n: int, beta: float):
    """Even parity support, scores from longest 0-run, softmax with beta."""
    N = 2 ** n
    scores = onp.full(N, -100.0, dtype=onp.float64)
    support = onp.zeros(N, dtype=bool)
    for k in range(N):
        b = int2bits(k, n)
        if parity_even(b):
            support[k] = True
            scores[k] = 1.0 + float(longest_zero_run_between_ones(b))

    logits = onp.full(N, -onp.inf, dtype=onp.float64)
    logits[support] = beta * scores[support]
    m = onp.max(logits[support])
    unnorm = onp.zeros(N, dtype=onp.float64)
    unnorm[support] = onp.exp(logits[support] - m)
    p_star = unnorm / unnorm.sum()
    return p_star.astype(onp.float64), support, scores.astype(onp.float64)

def topk_mask(scores: onp.ndarray, support: onp.ndarray, frac: float = 0.05) -> onp.ndarray:
    valid = onp.where(support)[0]
    k = max(1, int(onp.floor(frac * valid.size)))
    local_order = onp.argsort(-scores[valid])
    top_indices = valid[local_order[:k]]
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
# 3) Holdout selection + export
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
    'Smart' holdout: prefer higher p*(x) (but not ultra-rare),
    then spread out in Hamming distance.
    """
    if holdout_k <= 0:
        return onp.zeros_like(good_mask, dtype=bool)

    good_idxs = onp.where(good_mask)[0]
    taus = [1.0 / max(1, m_train), 0.5 / max(1, m_train), 0.25 / max(1, m_train), 0.0]

    cand = None
    for tau in taus:
        cand = good_idxs[p_star[good_idxs] >= tau]
        if cand.size >= holdout_k:
            break

    cand = cand[onp.argsort(-p_star[cand])]
    pool = cand[:min(pool_size, cand.size)]

    selected = [int(pool[0])]
    selected_bits = bits_table[[selected[-1]]].copy()

    while len(selected) < holdout_k and len(selected) < pool.size:
        best_idx = None
        best_d = -1
        for idx in pool:
            idx = int(idx)
            if idx in selected:
                continue
            d = _min_hamming_to_set(bits_table[idx], selected_bits)
            if d > best_d:
                best_d = d
                best_idx = idx
            elif d == best_d and best_idx is not None:
                if p_star[idx] > p_star[best_idx]:
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
        f.write(f"# {'Index':<8} {'Bitstring':<18} {'Score':<8} {'Prob p*(x)':<16}\n")
        f.write("-" * 60 + "\n")
        for idx in sorted_idxs:
            b_str = bits_str(bits_table[int(idx)])
            s_val = float(scores[int(idx)])
            prob = float(p_star[int(idx)])
            f.write(f"{int(idx):<8d} {b_str:<18s} {s_val:<8.1f} {prob:.6e}\n")
    print(f"[Export] Holdout list saved to: {path}")


# ------------------------------------------------------------------------------
# 4) Random parity (Fourier) features
# ------------------------------------------------------------------------------

def p_sigma(sigma: float) -> float:
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma ** 2))) if sigma > 0 else 0.5

def sample_alphas(n: int, sigma: float, K: int, seed: int) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas: onp.ndarray, bits_table: onp.ndarray) -> onp.ndarray:
    """
    P[k, x] = χ_{α_k}(x) = (-1)^{α_k · x} in {+1,-1}.
    """
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
# 5) Bandlimited reconstruction + discovery metrics
# ------------------------------------------------------------------------------

def reconstruct_bandlimited(P: onp.ndarray, z: onp.ndarray, alphas: onp.ndarray, n: int) -> onp.ndarray:
    """
    Truncated Walsh inversion on selected characters + positivity projection.
    """
    N = 2 ** n
    zero_rows = onp.all(alphas == 0, axis=1)
    if onp.any(zero_rows):
        P_use = P[~zero_rows]
        z_use = z[~zero_rows]
    else:
        P_use = P
        z_use = z

    q_raw = (1.0 / N) * (1.0 + (P_use.T @ z_use))
    q = onp.clip(q_raw, 0.0, None)
    s = float(q.sum())
    if s <= 0:
        q = onp.ones(N, dtype=onp.float64) / N
    else:
        q = q / s
    return q.astype(onp.float64)

def expected_unique_fraction(probs: onp.ndarray, mask: onp.ndarray, Q_vals: onp.ndarray) -> onp.ndarray:
    """
    U_H(Q)/|H| where U_H(Q)=sum_{x in H} (1-(1-q(x))^Q).
    """
    Q_vals = onp.array(Q_vals, dtype=int)
    H = int(onp.sum(mask))
    if H == 0:
        return onp.zeros_like(Q_vals, dtype=onp.float64)
    pS = probs[mask].astype(onp.float64)[:, None]  # H x 1
    return onp.sum(1.0 - onp.power(1.0 - pS, Q_vals[None, :]), axis=0) / H

def find_Q_threshold(
    probs: onp.ndarray,
    mask: onp.ndarray,
    thr: float = 0.8,
    Qmax: int = 200000,
) -> float:
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
    """
    R(Q) ≈ 1 - (1 - q(H)/|H|)^Q  =>  Q80 ≈ |H|/q(H)*ln(5)
    """
    if H_size <= 0 or qH <= 0:
        return float("inf")
    return float((H_size / qH) * math.log(5.0))

def moment_mse(P: onp.ndarray, q: onp.ndarray, z: onp.ndarray) -> float:
    r = (P @ q) - z
    return float(onp.mean(r ** 2))


# ------------------------------------------------------------------------------
# 6) Plotting functions (FIXED SIZE = 2.6 inches)
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

def plot_heatmap(
    mat,
    row_labels,
    col_labels,
    title,
    cbar_label,
    outpath,
    log10: bool = False,
    fmt: str = "{:.0f}",
    cmap="magma",
    mode: str = "col",
):
    # Enforce size: COL_W x 2.6
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
            # Adjust text color for readability
            if isinstance(cmap, str) and cmap in ("Reds", "YlOrRd", "Oranges", "Greys"):
                 if plot_data[i, j] < max_pd * 0.5:
                     text_col = "black"
            if not isinstance(cmap, str):
                pass

            ax.text(j, i, txt, ha="center", va="center",
                    color=text_col, fontsize=fs, fontweight="bold")

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_prediction_scatter(Q80_meas, Q80_pred, labels, title, outpath, mode: str = "col"):
    x = onp.array(Q80_pred, dtype=onp.float64)
    y = onp.array(Q80_meas, dtype=onp.float64)
    m = onp.isfinite(x) & onp.isfinite(y) & (x > 0) & (y > 0)
    xf, yf = x[m], y[m]
    if xf.size == 0:
        return

    # Enforce size: COL_W x 2.6
    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    ax.scatter(xf, yf, s=45, facecolors="white",
               edgecolors=COLORS["model"], linewidths=1.6, alpha=0.9, zorder=3)

    lo = float(min(onp.min(xf), onp.min(yf)))
    hi = float(max(onp.max(xf), onp.max(yf)))
    xs = onp.logspace(onp.log10(lo), onp.log10(hi), 100)
    ax.plot(xs, xs, color=COLORS["target"], linestyle="--", linewidth=1.6, alpha=0.7, label=r"$y=x$")

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

def plot_recovery_curve(p_star, q_model, holdout_mask, outpath, title, Qmax: int = 10000, add_uniform: bool = True, mode: str = "col"):
    H = int(onp.sum(holdout_mask))
    if H == 0:
        return

    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 120).astype(int)),
        onp.linspace(1000, Qmax, 160).astype(int),
    ]))
    Q = Q[Q <= Qmax]

    y_star = expected_unique_fraction(p_star, holdout_mask, Q)
    y_mod = expected_unique_fraction(q_model, holdout_mask, Q)

    # Enforce size: COL_W x 2.6
    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"Target $p^*$", zorder=4)
    ax.plot(Q, y_mod,  color=COLORS["model"],  linewidth=2.2, label=r"Reconstruction $q$", zorder=5)

    if add_uniform:
        N = p_star.size
        u = onp.ones(N, dtype=onp.float64) / N
        y_u = expected_unique_fraction(u, holdout_mask, Q)
        ax.plot(Q, y_u, color=COLORS["gray"], linewidth=1.6, linestyle="--",
                alpha=0.9, label="Uniform", zorder=3)

    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_moment_mse_vs_qH(results, outpath, title, mode: str = "col"):
    xs = onp.array([r["moment_mse"] for r in results], dtype=onp.float64)
    ys = onp.array([r["qH_ratio"] for r in results], dtype=onp.float64)

    # Enforce size: COL_W x 2.6
    fig, ax = plt.subplots(figsize=fig_size(mode), constrained_layout=True)
    ax.scatter(xs, ys, s=45, facecolors="white",
               edgecolors=COLORS["model"], linewidths=1.6, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Moment MSE")
    ax.set_ylabel(r"$q(H) / q_{\mathrm{unif}}(H)$")

    top = onp.argsort(-ys)[:3]
    for i in top:
        ax.annotate(results[int(i)]["label"],
                    (xs[int(i)], ys[int(i)]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color=COLORS["target"])

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_story_overview(
    qH_ratio_mat,
    Q80_mat,
    sigmas,
    Ks,
    Q80_meas,
    Q80_pred,
    best_title,
    p_star,
    q_best,
    holdout_mask,
    outpath,
    cmap_custom,
):
    # Full width figure (kept at 4.8 as before)
    fig, axes = plt.subplots(2, 2, figsize=fig_size("full", 4.8), constrained_layout=True)

    # (1) qH ratio heatmap (Reds)
    ax = axes[0, 0]
    im = ax.imshow(qH_ratio_mat, aspect="auto", cmap="Reds", vmin=0.0)
    ax.set_xticks(onp.arange(len(Ks)))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_yticks(onp.arange(len(sigmas)))
    ax.set_yticklabels([str(s) for s in sigmas])
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\sigma$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$q(H)/q_{\mathrm{unif}}(H)$")
    mmax = float(onp.max(qH_ratio_mat)) if onp.isfinite(qH_ratio_mat).any() else 1.0
    for i in range(qH_ratio_mat.shape[0]):
        for j in range(qH_ratio_mat.shape[1]):
            tcol = "black" if qH_ratio_mat[i, j] < mmax * 0.5 else "white"
            ax.text(j, i, f"{qH_ratio_mat[i, j]:.1f}",
                    ha="center", va="center", color=tcol,
                    fontsize=7, fontweight="bold")

    # (2) Q80 heatmap (Custom Red-Black, log-color)
    ax = axes[0, 1]
    plot_data = _safe_log10_for_heatmap(Q80_mat, clip_min=1e-9)
    im = ax.imshow(plot_data, aspect="auto", cmap=cmap_custom)
    ax.set_xticks(onp.arange(len(Ks)))
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_yticks(onp.arange(len(sigmas)))
    ax.set_yticklabels([str(s) for s in sigmas])
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\sigma$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\log_{10} Q_{80}$")
    for i in range(Q80_mat.shape[0]):
        for j in range(Q80_mat.shape[1]):
            v = Q80_mat[i, j]
            txt = r"$\infty$" if onp.isinf(v) else f"{int(v):d}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")

    # (3) Scatter (pred vs meas)
    ax = axes[1, 0]
    x = onp.array(Q80_pred, dtype=onp.float64)
    y = onp.array(Q80_meas, dtype=onp.float64)
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

    # (4) Recovery curve (best)
    ax = axes[1, 1]
    H = int(onp.sum(holdout_mask))
    if H > 0:
        Q = onp.unique(onp.concatenate([
            onp.unique(onp.logspace(0, 3.5, 110).astype(int)),
            onp.linspace(1000, 10000, 120).astype(int),
        ]))
        Q = Q[Q <= 10000]
        y_star = expected_unique_fraction(p_star, holdout_mask, Q)
        y_best = expected_unique_fraction(q_best, holdout_mask, Q)
        u = onp.ones_like(p_star) / p_star.size
        y_u = expected_unique_fraction(u, holdout_mask, Q)
        ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.8, label=r"$p^*$")
        ax.plot(Q, y_best, color=COLORS["model"], linewidth=2.1, label=r"best $q$")
        ax.plot(Q, y_u, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="uniform")
        ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$Q$")
        ax.set_ylabel(r"$R(Q)$")
        ax.legend(loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_column_triptych(results, p_star, q_best, holdout_mask, best_label, outpath):
    fig, axs = plt.subplots(1, 3, figsize=fig_size("full", 2.5), constrained_layout=True)

    # (a) Scatter
    ax = axs[0]
    _panel_label(ax, "(a)")
    x = onp.array([r["Q80_pred"] for r in results])
    y = onp.array([r["Q80"] for r in results])
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

    # (b) MSE vs qH
    ax = axs[1]
    _panel_label(ax, "(b)")
    xs = onp.array([r["moment_mse"] for r in results])
    ys = onp.array([r["qH_ratio"] for r in results])
    ax.scatter(xs, ys, s=38, facecolors="white", edgecolors=COLORS["model"], linewidths=1.4, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Moment MSE")
    ax.set_ylabel(r"$q(H)/q_{\mathrm{unif}}(H)$")

    # (c) Recovery curve
    ax = axs[2]
    _panel_label(ax, "(c)")
    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 120).astype(int)),
        onp.linspace(1000, 10000, 160).astype(int),
    ]))
    Q = Q[Q <= 10000]
    y_star = expected_unique_fraction(p_star, holdout_mask, Q)
    y_best = expected_unique_fraction(q_best, holdout_mask, Q)
    u = onp.ones_like(p_star) / p_star.size
    y_u = expected_unique_fraction(u, holdout_mask, Q)
    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"$p^*$")
    ax.plot(Q, y_best, color=COLORS["model"], linewidth=2.2, label="best $q$")
    ax.plot(Q, y_u, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="uniform")
    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$R(Q)$")
    ax.legend(loc="lower right", frameon=False)

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")

def plot_adversarial_visibility_split(p_star, holdout_visible, holdout_invisible, q_vis, q_inv, outdir):
    """Two separate panels (6a curves, 6b bars) with the requested final styling."""
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

    # Enforce size: COL_W x 2.6
    fig, ax = plt.subplots(figsize=fig_size("col"), constrained_layout=True)
    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"Target $p^*$", zorder=4)
    ax.plot(Q, y_vis,  color=COLORS["model"],  linewidth=2.2, label=r"Visible $H_{\mathrm{vis}}$", zorder=6)

    # CHANGED: Invisible curve is dark gray + dash-dot
    ax.plot(Q, y_inv,  color="#555555", linestyle="-.", linewidth=1.9,
            label=r"Invisible $H_{\mathrm{inv}}$", zorder=5)

    ax.plot(Q, y_u, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="Uniform", zorder=3)
    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"$R(Q)$")
    
    # MODIFIED LEGEND LOCATION
    # "oben rechts sein aber nicht ganz oben rechts sondern das es im freien space ist"
    # This places the top-right corner of the legend at y=0.78, putting the legend body in the gap
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

    # Enforce size: COL_W x 2.6
    fig, ax = plt.subplots(figsize=fig_size("col"), constrained_layout=True)
    x = onp.arange(len(Qbars))
    w = 0.23

    # Target: White with diagonal hatch
    ax.bar(x - w, b_star, w, color="white", edgecolor="#333333", hatch="///",
           alpha=0.9, label=r"Target $p^*$")
    # Visible: Red
    ax.bar(x, b_vis, w, color=COLORS["model"], edgecolor="black",
           alpha=0.85, label=r"$H_{\mathrm{vis}}$")
    # Invisible: White with Karo (XX)
    ax.bar(x + w, b_inv, w, color="white", edgecolor="black", hatch="XX",
           alpha=1.0, label=r"$H_{\mathrm{inv}}$")

    # Uniform line at Q=10k baseline (same for both holdouts since |H| matches)
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
# 7) Optional: IQP-QCBM training (Hero architecture D)
# ------------------------------------------------------------------------------

def get_iqp_topology(n: int, arch: str = "D"):
    pairs, quads = [], []

    def clean(l): 
        return sorted(list(set(l)))

    if arch in ["A", "B", "C", "D"]:
        pairs.extend([tuple(sorted((i, (i + 1) % n))) for i in range(n)])
    if arch in ["C", "D"]:
        pairs.extend([tuple(sorted((i, (i + 2) % n))) for i in range(n)])
    if arch in ["B", "D"]:
        quads.extend([tuple(sorted((i, (i + 1) % n, (i + 2) % n, (i + 3) % n))) for i in range(n)])
    if arch == "E":
        pairs = list(itertools.combinations(range(n), 2))
    return clean(pairs), clean(quads)

def iqp_circuit(W, wires, pairs, quads, layers: int = 1):
    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)
    for _ in range(layers):
        for (i, j) in pairs:
            qml.IsingZZ(W[idx], wires=[wires[i], wires[j]])
            idx += 1
        for (a, b, c, d) in quads:
            qml.MultiRZ(W[idx], wires=[wires[a], wires[b], wires[c], wires[d]])
            idx += 1
        for w in wires:
            qml.Hadamard(wires=w)

def train_iqp_qcbm(
    n: int,
    layers: int,
    arch: str,
    steps: int,
    lr: float,
    P: onp.ndarray,
    z_data: onp.ndarray,
    seed_init: int,
    eval_every: int,
) -> Tuple[onp.ndarray, Dict[str, List[float]]]:
    """
    Train an IQP-QCBM to match parity moments (MSE loss in feature space).
    """
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is not installed. Run `pip install pennylane` or disable --use-iqp.")

    dev = qml.device("default.qubit", wires=n)
    pairs, quads = get_iqp_topology(n, arch)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(n), pairs, quads, layers=layers)
        return qml.probs(wires=range(n))

    P_t = np.array(P, requires_grad=False)
    z_t = np.array(z_data, requires_grad=False)

    num_params = (len(pairs) + len(quads)) * layers
    rng = onp.random.default_rng(seed_init)
    W = np.array(0.01 * rng.standard_normal(num_params), requires_grad=True)

    opt = qml.AdamOptimizer(lr)
    hist: Dict[str, List[float]] = {"step": [], "loss": []}

    def loss_fn(w):
        q = circuit(w)
        return np.mean((z_t - P_t @ q) ** 2)

    print(f"[IQP] Training arch={arch}, L={layers}, params={num_params}, steps={steps}, lr={lr}")
    for t in range(1, steps + 1):
        W, l_val = opt.step_and_cost(loss_fn, W)
        if (t % eval_every == 0) or (t == 1) or (t == steps):
            hist["step"].append(int(t))
            hist["loss"].append(float(l_val))
            print(f"[IQP] Step {t:4d} | Loss {float(l_val):.3e}")

    q_final = onp.array(circuit(W), dtype=onp.float64)
    q_final = onp.clip(q_final, 0.0, 1.0)
    q_final /= max(1e-15, float(q_final.sum()))
    return q_final.astype(onp.float64), hist

def plot_iqp_training_dynamics(hist: Dict[str, List[float]], outpath: str, title: str):
    st = onp.array(hist["step"], dtype=int)
    loss = onp.array(hist["loss"], dtype=onp.float64)

    # Enforce size: COL_W x 2.6
    fig, ax = plt.subplots(figsize=fig_size("col"), constrained_layout=True)
    ax.plot(st, loss, color=COLORS["target"], linewidth=2.1, label="MMD loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")
    _end_label(ax, st[-1], loss[-1], f"{loss[-1]:.2e}", COLORS["target"], dy=0.05 * loss[-1], fs=8)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")


# ------------------------------------------------------------------------------
# 8) Main experiment logic
# ------------------------------------------------------------------------------

@dataclass
class Config:
    n: int = 14
    beta: float = 0.9
    train_m: int = 1000
    holdout_k: int = 20
    holdout_pool: int = 400
    seed: int = 42

    sigmas: List[float] = None
    Ks: List[int] = None

    Qmax: int = 10000
    Q80_thr: float = 0.8
    Q80_search_max: int = 200000

    # Adversarial demo
    adversarial: bool = True
    adv_score_level: int = 7
    adv_sigma: float = 1.0
    adv_K: int = 512

    # Optional IQP
    use_iqp: bool = False
    iqp_steps: int = 600
    iqp_lr: float = 0.05
    iqp_eval_every: int = 50
    iqp_layers: int = 1
    iqp_arch: str = "D"

    outdir: str = "hero_spectral_discovery"

def run_sweep(cfg: Config, p_star: onp.ndarray, holdout_mask: onp.ndarray, good_mask: onp.ndarray, bits_table: onp.ndarray) -> List[Dict]:
    """Sweep over sigma and K. Returns list of result dicts."""
    N = p_star.size
    H_size = int(onp.sum(holdout_mask))

    # Training distribution (holdout removed)
    p_train = p_star.copy()
    if H_size > 0:
        p_train[holdout_mask] = 0.0
        p_train /= p_train.sum()

    # Fixed training dataset
    idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.seed + 7)
    emp = empirical_dist(idxs_train, N)

    # Uniform baseline for ratio
    q_unif = onp.ones(N, dtype=onp.float64) / N
    qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0

    maxK = int(max(cfg.Ks))
    results: List[Dict] = []

    for si, sigma in enumerate(cfg.sigmas):
        # sample a superset of alphas for nesting across K
        alphas_sup = sample_alphas(cfg.n, float(sigma), maxK, seed=cfg.seed + 222 + 13 * si)
        P_sup = build_parity_matrix(alphas_sup, bits_table)

        for K in cfg.Ks:
            alphas = alphas_sup[:K]
            P = P_sup[:K]

            z = P @ emp
            q = reconstruct_bandlimited(P, z, alphas, cfg.n)

            qH = float(q[holdout_mask].sum()) if H_size > 0 else 0.0
            qG = float(q[good_mask].sum())
            R1000 = float(expected_unique_fraction(q, holdout_mask, onp.array([1000]))[0]) if H_size > 0 else 0.0
            R10000 = float(expected_unique_fraction(q, holdout_mask, onp.array([10000]))[0]) if H_size > 0 else 0.0
            Q80 = find_Q_threshold(q, holdout_mask, thr=cfg.Q80_thr, Qmax=cfg.Q80_search_max) if H_size > 0 else float("nan")
            Q80_pred = Q80_prediction_from_qH(qH, H_size) if H_size > 0 else float("inf")

            Ku, mean_wt = alpha_stats(alphas)

            res = dict(
                sigma=float(sigma),
                K=int(K),
                K_unique=int(Ku),
                mean_alpha_wt=float(mean_wt),
                qH=float(qH),
                qH_ratio=float(qH / qH_unif) if qH_unif > 0 else float("nan"),
                qG=float(qG),
                R_Q1000=float(R1000),
                R_Q10000=float(R10000),
                Q80=float(Q80),
                Q80_pred=float(Q80_pred),
                moment_mse=float(moment_mse(P, q, z)),
                label=f"s={sigma:g},K={K:d}",
            )
            results.append(res)
            print(f"[Sweep] sigma={sigma:g}, K={K:4d} | qH={qH:.4e} (x{res['qH_ratio']:.2f}) | R10k={R10000:.3f} | Q80={Q80:.0f}")

    return results

def save_results_csv(results: List[Dict], outpath: str):
    keys = list(results[0].keys())
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[Saved] {outpath}")

def build_result_matrices(results: List[Dict], sigmas: List[float], Ks: List[int]) -> Tuple[onp.ndarray, onp.ndarray]:
    """Return matrices (qH_ratio, Q80) with rows=sigmas, cols=Ks."""
    sigma_to_i = {float(s): i for i, s in enumerate(sigmas)}
    K_to_j = {int(K): j for j, K in enumerate(Ks)}

    qH_ratio = onp.full((len(sigmas), len(Ks)), onp.nan, dtype=onp.float64)
    Q80 = onp.full((len(sigmas), len(Ks)), onp.inf, dtype=onp.float64)

    for r in results:
        i = sigma_to_i[float(r["sigma"])]
        j = K_to_j[int(r["K"])]
        qH_ratio[i, j] = float(r["qH_ratio"])
        Q80[i, j] = float(r["Q80"])

    return qH_ratio, Q80

def pick_best_setting(results: List[Dict]) -> Dict:
    """Pick best by minimal finite Q80; tie-break by higher qH_ratio."""
    finite = [r for r in results if onp.isfinite(r["Q80"])]
    if not finite:
        return results[0]
    finite.sort(key=lambda r: (r["Q80"], -r["qH_ratio"]))
    return finite[0]

def rerun_single_setting(
    cfg: Config,
    p_star: onp.ndarray,
    holdout_mask: onp.ndarray,
    good_mask: onp.ndarray,
    bits_table: onp.ndarray,
    sigma: float,
    K: int,
) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Rebuild nested alphas deterministically and return (alphas, P, q_recon)."""
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
    q = reconstruct_bandlimited(P, z, alphas, cfg.n)
    return alphas, P, q

def run_adversarial_demo(
    cfg: Config,
    p_star: onp.ndarray,
    support: onp.ndarray,
    scores: onp.ndarray,
    good_mask: onp.ndarray,
    bits_table: onp.ndarray,
    outdir: str,
):
    """
    Adversarial: pick H_vis vs H_inv with same p*(H) by choosing within one score level.
    Visibility is defined by a bandlimited approximation of p* under the same (sigma,K) features.
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

    # Build feature set once
    alphas = sample_alphas(cfg.n, float(cfg.adv_sigma), int(cfg.adv_K), seed=cfg.seed + 999)
    P = build_parity_matrix(alphas, bits_table)

    # Bandlimited approximation of p* => defines "spectral visibility"
    z_star = P @ p_star
    q_band_star = reconstruct_bandlimited(P, z_star, alphas, cfg.n)

    # Select visible/invisible by q_band_star probability within this score band
    q_c = q_band_star[cand]
    order = onp.argsort(q_c)  # ascending
    H_inv_idxs = cand[order[:cfg.holdout_k]]
    H_vis_idxs = cand[order[-cfg.holdout_k:]]

    holdout_inv = onp.zeros(N, dtype=bool)
    holdout_inv[H_inv_idxs] = True
    holdout_vis = onp.zeros(N, dtype=bool)
    holdout_vis[H_vis_idxs] = True

    pH_vis = float(p_star[holdout_vis].sum())
    pH_inv = float(p_star[holdout_inv].sum())
    print(f"[Adv] p*(H_vis)={pH_vis:.6f}, p*(H_inv)={pH_inv:.6f} (should match)")

    def train_and_reconstruct(holdout_mask_local: onp.ndarray, seed_offset: int) -> onp.ndarray:
        p_train = p_star.copy()
        p_train[holdout_mask_local] = 0.0
        p_train /= p_train.sum()
        idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.seed + 7 + seed_offset)
        emp = empirical_dist(idxs_train, N)
        z = P @ emp
        return reconstruct_bandlimited(P, z, alphas, cfg.n)

    q_vis = train_and_reconstruct(holdout_vis, seed_offset=0)
    q_inv = train_and_reconstruct(holdout_inv, seed_offset=123)

    save_holdout_list(holdout_vis, bits_table, p_star, scores, outdir, name="holdout_strings_visible.txt")
    save_holdout_list(holdout_inv, bits_table, p_star, scores, outdir, name="holdout_strings_invisible.txt")

    plot_adversarial_visibility_split(p_star, holdout_vis, holdout_inv, q_vis, q_inv, outdir)

def maybe_run_iqp_comparison(
    cfg: Config,
    p_star: onp.ndarray,
    holdout_mask: onp.ndarray,
    good_mask: onp.ndarray,
    bits_table: onp.ndarray,
    best_sigma: float,
    best_K: int,
    outdir: str,
):
    if not cfg.use_iqp:
        return
    if not HAS_PENNYLANE:
        raise RuntimeError("You set --use-iqp 1 but pennylane is not installed. Run `pip install pennylane`.")

    N = p_star.size
    p_train = p_star.copy()
    if onp.any(holdout_mask):
        p_train[holdout_mask] = 0.0
        p_train /= p_train.sum()
    idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.seed + 7)
    emp = empirical_dist(idxs_train, N)

    # Build parity features for (best_sigma, best_K)
    alphas = sample_alphas(cfg.n, float(best_sigma), int(best_K), seed=cfg.seed + 2024)
    P = build_parity_matrix(alphas, bits_table)
    z = P @ emp

    # Train IQP-QCBM
    q_iqp, hist = train_iqp_qcbm(
        n=cfg.n,
        layers=cfg.iqp_layers,
        arch=cfg.iqp_arch,
        steps=cfg.iqp_steps,
        lr=cfg.iqp_lr,
        P=P,
        z_data=z,
        seed_init=cfg.seed + 42,
        eval_every=cfg.iqp_eval_every,
    )

    plot_iqp_training_dynamics(
        hist,
        outpath=os.path.join(outdir, "7_iqp_training_dynamics.pdf"),
        title=f"IQP-QCBM training dynamics (Hero arch={cfg.iqp_arch}, L={cfg.iqp_layers})",
    )

    # Spectral reconstruction for same (sigma,K) and same training set
    q_spec = reconstruct_bandlimited(P, z, alphas, cfg.n)

    # Combined comparison plot (keep blue for spectral here)
    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 120).astype(int)),
        onp.linspace(1000, cfg.Qmax, 180).astype(int),
    ]))
    Q = Q[Q <= cfg.Qmax]

    y_star = expected_unique_fraction(p_star, holdout_mask, Q)
    y_iqp = expected_unique_fraction(q_iqp, holdout_mask, Q)
    y_spec = expected_unique_fraction(q_spec, holdout_mask, Q)
    u = onp.ones_like(p_star) / p_star.size
    y_u = expected_unique_fraction(u, holdout_mask, Q)

    # Enforce size: COL_W x 2.6
    fig, ax = plt.subplots(figsize=fig_size("col"), constrained_layout=True)
    ax.plot(Q, y_star, color=COLORS["target"], linewidth=1.9, label=r"Target $p^*$")
    ax.plot(Q, y_iqp,  color=COLORS["model"],  linewidth=2.2, label=r"IQP-QCBM $q_\theta$")
    ax.plot(Q, y_spec, color="#555555", linestyle="-.", linewidth=1.9, label=r"Spectral recon $q$")
    ax.plot(Q, y_u, color=COLORS["gray"], linestyle="--", linewidth=1.5, label="Uniform")
    ax.axhline(1.0, color=COLORS["gray"], linestyle=":", alpha=0.7)

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.legend(loc="lower right", frameon=False)

    outpath = os.path.join(outdir, "8_iqp_vs_spectral_recovery.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[Saved] {outpath}")


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="hero_spectral_discovery")

    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--train-m", type=int, default=1000)

    parser.add_argument("--holdout-k", type=int, default=20)
    parser.add_argument("--holdout-pool", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigmas", type=str, default="0.5,1,2,3")
    parser.add_argument("--Ks", type=str, default="128,256,512")

    parser.add_argument("--Qmax", type=int, default=10000)
    parser.add_argument("--Q80-thr", type=float, default=0.8)
    parser.add_argument("--Q80-search-max", type=int, default=200000)

    parser.add_argument("--adversarial", type=int, default=1)
    parser.add_argument("--adv-score-level", type=int, default=7)
    parser.add_argument("--adv-sigma", type=float, default=1.0)
    parser.add_argument("--adv-K", type=int, default=512)

    parser.add_argument("--use-iqp", type=int, default=0)
    parser.add_argument("--iqp-steps", type=int, default=600)
    parser.add_argument("--iqp-lr", type=float, default=0.05)
    parser.add_argument("--iqp-eval-every", type=int, default=50)
    parser.add_argument("--iqp-layers", type=int, default=1)
    parser.add_argument("--iqp-arch", type=str, default="D", choices=["A", "B", "C", "D", "E"])

    args = parser.parse_args()

    set_style(base=8)
    outdir = ensure_outdir(args.outdir)

    cfg = Config(
        n=args.n,
        beta=args.beta,
        train_m=args.train_m,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
        sigmas=_parse_list_floats(args.sigmas),
        Ks=_parse_list_ints(args.Ks),
        Qmax=args.Qmax,
        Q80_thr=args.Q80_thr,
        Q80_search_max=args.Q80_search_max,
        adversarial=bool(args.adversarial),
        adv_score_level=args.adv_score_level,
        adv_sigma=args.adv_sigma,
        adv_K=args.adv_K,
        use_iqp=bool(args.use_iqp),
        iqp_steps=args.iqp_steps,
        iqp_lr=args.iqp_lr,
        iqp_eval_every=args.iqp_eval_every,
        iqp_layers=args.iqp_layers,
        iqp_arch=args.iqp_arch,
        outdir=outdir,
    )

    # Save config
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print("[Config]", cfg)

    # Build target distribution + tables
    bits_table = make_bits_table(cfg.n)
    p_star, support, scores = build_target_distribution(cfg.n, cfg.beta)
    good_mask = topk_mask(scores, support, frac=0.05)

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

    # --- Main sweep ---
    results = run_sweep(cfg, p_star, holdout_mask, good_mask, bits_table)

    # Save CSV
    csv_path = os.path.join(outdir, "sweep_results.csv")
    save_results_csv(results, csv_path)

    # Matrices
    qH_ratio_mat, Q80_mat = build_result_matrices(results, cfg.sigmas, cfg.Ks)

    # Custom red-black colormap for Q80 heatmap
    cmap_rb = LinearSegmentedColormap.from_list("RedBlack", ["#FF0000", "#000000"])

    # PLOT 1: qH ratio (Reds)
    plot_heatmap(
        qH_ratio_mat,
        row_labels=[str(s) for s in cfg.sigmas],
        col_labels=[str(k) for k in cfg.Ks],
        title="qH ratio",
        cbar_label=r"$q(H)/q_{\mathrm{unif}}(H)$",
        outpath=os.path.join(outdir, "1_heatmap_qH_ratio.pdf"),
        log10=False,
        fmt="{:.1f}",
        cmap="Reds",
        mode="col",
    )

    # PLOT 2: Q80 (Red→Black)
    plot_heatmap(
        Q80_mat,
        row_labels=[str(s) for s in cfg.sigmas],
        col_labels=[str(k) for k in cfg.Ks],
        title="Q80",
        cbar_label=r"$\log_{10} Q_{80}$",
        outpath=os.path.join(outdir, "2_heatmap_Q80.pdf"),
        log10=True,
        fmt="{:.0f}",
        cmap=cmap_rb,
        mode="col",
    )

    # PLOT 3: predicted vs measured
    plot_prediction_scatter(
        [r["Q80"] for r in results],
        [r["Q80_pred"] for r in results],
        [r["label"] for r in results],
        "Pred vs Meas",
        os.path.join(outdir, "3_Q80_pred_vs_meas.pdf"),
        mode="col",
    )

    # PLOT 5: MSE vs qH
    plot_moment_mse_vs_qH(
        results,
        os.path.join(outdir, "5_moment_mse_vs_qH.pdf"),
        "MSE vs qH",
        mode="col",
    )

    # Best setting
    best = pick_best_setting(results)
    best_sigma = float(best["sigma"])
    best_K = int(best["K"])
    _, _, q_best = rerun_single_setting(cfg, p_star, holdout_mask, good_mask, bits_table, best_sigma, best_K)

    # PLOT 4: recovery curve
    plot_recovery_curve(
        p_star,
        q_best,
        holdout_mask,
        os.path.join(outdir, "4_recovery_best.pdf"),
        "Best Recovery",
        Qmax=cfg.Qmax,
        add_uniform=True,
        mode="col",
    )

    # PLOT 0: story overview
    plot_story_overview(
        qH_ratio_mat=qH_ratio_mat,
        Q80_mat=Q80_mat,
        sigmas=cfg.sigmas,
        Ks=cfg.Ks,
        Q80_meas=[r["Q80"] for r in results],
        Q80_pred=[r["Q80_pred"] for r in results],
        best_title="Best",
        p_star=p_star,
        q_best=q_best,
        holdout_mask=holdout_mask,
        outpath=os.path.join(outdir, "0_story_overview.pdf"),
        cmap_custom=cmap_rb,
    )

    # PLOT 9: column triptych
    plot_column_triptych(
        results,
        p_star,
        q_best,
        holdout_mask,
        "Best",
        os.path.join(outdir, "9_column_triptych.pdf"),
    )

    # --- Adversarial visibility demo ---
    if cfg.adversarial:
        run_adversarial_demo(cfg, p_star, support, scores, good_mask, bits_table, outdir)

    # --- Optional IQP comparison ---
    maybe_run_iqp_comparison(cfg, p_star, holdout_mask, good_mask, bits_table, best_sigma, best_K, outdir)

    print(f"Done. Results in ./{outdir}/")

if __name__ == "__main__":
    main()