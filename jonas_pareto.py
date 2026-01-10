#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IQP-QCBM (Parity-MMD): Jonas Pareto Navigator
(Publishable Plots: Clean Scatter + Perfectly Attached Heatmap Colorbar)

================================================================

Goal
----
Pareto navigation for the "training-unseen" generalization setup.

Metrics:
  1) KL(p* || q)                     (global fit, ↓)
  2) U_unseen(Q0)                    (yield, ↑)
  3) spread = U_unseen / (Q0*q_unseen) (dispersion, ↑)
  4) unseen_score_TVD                (calibration, ↓)
  5) q(score=1 | unseen)             (leakage diagnostic -> Heatmap)

Visualization Style Changes:
  - Scatter Plots: 
      * Rotated X-Axis labels.
      * External legend.
      * K -> Hatch Patterns.
      * Sigma -> Marker Shapes.
  - Heatmap:
      * 1:1 Aspect Ratio (Square).
      * Colorbar uses 'make_axes_locatable' to attach perfectly to the plot.
      * No whitespace gap between plot and legend.

Outputs (outdir/)
-----------------
- jonas_pareto_raw.csv
- jonas_pareto_summary.csv
- pareto_points_m{m}.txt
- pareto_fit_yield_m{m}.pdf
- pareto_dispersion_m{m}.pdf
- heatmap_qS1_m{m}.pdf

Dependencies
------------
pip install pennylane numpy matplotlib
"""

import os
import csv
import json
import math
import argparse
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Suppress PennyLane warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pennylane")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

import numpy as onp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable  # <--- CRITICAL FIX

try:
    import pennylane as qml
    from pennylane import numpy as np
except Exception as _e:
    qml = None  # type: ignore
    np = None   # type: ignore


# ------------------------------------------------------------------------------
# Style & Layout Constants
# ------------------------------------------------------------------------------

COLORS = {
    "target": "#222222",  # Black-ish
    "model":  "#D62728",  # Standard Matplotlib Red
    "gray":   "#888888",  # Neutral Gray
    "light":  "#F0F0F0",  # Light background if needed
}

# Physical Figure Dimensions (Inches)
COL_W  = 3.37
FULL_W = 6.95

# Fixed heights
FIXED_H_COL  = 3.2 
FIXED_H_FULL = 3.8

def fig_size(fig_target: str) -> Tuple[float, float]:
    """Returns (width, height) tuple based on target column width."""
    if fig_target not in ("col", "full"):
        raise ValueError("fig_target must be 'col' or 'full'")
    return (COL_W, FIXED_H_COL) if fig_target == "col" else (FULL_W, FIXED_H_FULL)

def set_style(fig_target: str) -> None:
    """Sets matplotlib rcParams for publication-quality figures."""
    base_size = 8 if fig_target == "col" else 11
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "legend.fontsize": base_size - 1,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 6.0,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 0.8,
        "hatch.color": COLORS["target"]
    })

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
# Physics / Domain Logic (Parity Problem)
# ------------------------------------------------------------------------------

def int2bits(k: int, n: int) -> onp.ndarray:
    """Converts integer k to bit array of length n (Big Endian)."""
    return onp.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=onp.int8)

def parity_even(bits: onp.ndarray) -> bool:
    """Returns True if sum of bits is even."""
    return (int(onp.sum(bits)) % 2) == 0

def longest_zero_run_between_ones(bits: onp.ndarray) -> int:
    """Calculates the score metric: Max consecutive zeros between ones."""
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0

def build_target_distribution(n: int, beta: float) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Constructs the target distribution p*(x)."""
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

def make_bits_table(n: int) -> onp.ndarray:
    """Pre-computes bit representations for all 2^n states."""
    N = 2 ** n
    return onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)

def sample_indices(probs: onp.ndarray, m: int, seed: int) -> onp.ndarray:
    """Draws m samples (indices) from probs."""
    rng = onp.random.default_rng(seed)
    p = probs / probs.sum()
    return rng.choice(len(p), size=m, replace=True, p=p)

def empirical_dist(idxs: onp.ndarray, N: int) -> onp.ndarray:
    """Converts sample indices to a histogram/probability vector."""
    c = onp.bincount(idxs, minlength=N)
    return (c / max(1, c.sum())).astype(onp.float64)

def seen_mask_from_indices(idxs: onp.ndarray, N: int) -> onp.ndarray:
    """Boolean mask of states present in the training data."""
    c = onp.bincount(idxs, minlength=N)
    return (c > 0)

# ------------------------------------------------------------------------------
# Parity-MMD & Circuit Implementation
# ------------------------------------------------------------------------------

def p_sigma(sigma: float) -> float:
    """Bernoulli probability for drawing parity feature components."""
    return 0.5 * (1.0 - math.exp(-1.0/(2.0*sigma**2))) if sigma > 0 else 0.5

def sample_alphas(n: int, sigma: float, K: int, seed: int) -> onp.ndarray:
    """Samples the feature matrix 'alpha'."""
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas: onp.ndarray, bits_table: onp.ndarray) -> onp.ndarray:
    """Computes the parity feature matrix P where P_kx = (-1)^(a_k * x)."""
    A = alphas.astype(onp.int16)
    X = bits_table.astype(onp.int16).T
    par = (A @ X) & 1
    # Map 0 -> +1, 1 -> -1
    return onp.where(par == 0, 1.0, -1.0).astype(onp.float64)

def get_iqp_topology(n: int, arch: str):
    """Defines connectivity (Pairs/Quads) for the IQP circuit."""
    pairs, quads = [], []
    def clean(l): return sorted(list(set(l)))
    
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
    """The standard IQP circuit ansatz."""
    idx = 0
    for w in wires: qml.Hadamard(wires=w)
    for _ in range(layers):
        for (i, j) in pairs:
            qml.IsingZZ(W[idx], wires=[wires[i], wires[j]])
            idx += 1
        for (a, b, c, d) in quads:
            qml.MultiRZ(W[idx], wires=[wires[a], wires[b], wires[c], wires[d]])
            idx += 1
        for w in wires: qml.Hadamard(wires=w)

# ------------------------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------------------------

def kl_divergence(p_star, q, support, eps=1e-12):
    q_clip = onp.clip(q, eps, 1.0)
    q_clip /= max(1e-15, float(q_clip.sum()))
    p = p_star[support]
    q_s = q_clip[support]
    return float(onp.sum(p * (onp.log(p) - onp.log(q_s))))

def score_spectrum_masses(dist, scores, support, mask=None):
    scores_int = scores.astype(int)
    base_mask = support.copy()
    if mask is not None: base_mask &= mask
    valid_scores = scores_int[base_mask]
    unique_s = sorted(onp.unique(valid_scores).tolist())
    masses = [float(dist[(scores_int == s) & base_mask].sum()) for s in unique_s]
    return unique_s, masses

def conditional_score_tvd(p_star, q, scores, support, mask):
    """TVD between p* and q restricted to a specific mask (e.g. unseen)."""
    _, p_mass = score_spectrum_masses(p_star, scores, support, mask)
    _, q_mass = score_spectrum_masses(q, scores, support, mask)
    p_tot, q_tot = sum(p_mass), sum(q_mass)
    if p_tot <= 0 or q_tot <= 0: return float("nan")
    p_cond = onp.array(p_mass)/p_tot
    q_cond = onp.array(q_mass)/q_tot
    return float(0.5 * onp.sum(onp.abs(p_cond - q_cond)))

def expected_unique_set(probs, mask, Q):
    """Calculates 'Yield': Expected number of unique valid samples in Q draws."""
    pS = probs[mask].astype(onp.float64)
    if pS.size == 0: return 0.0
    return float(onp.sum(1.0 - onp.power(1.0 - pS, int(Q))))

def q_score1_given_unseen(q, scores, support, unseen_mask):
    """The leakage metric: q(score=1 | x is in support AND unseen)."""
    m = support & unseen_mask
    denom = float(q[m].sum())
    if denom <= 0: return float("nan")
    num = float(q[m & (scores.astype(int) == 1)].sum())
    return num / denom


# ------------------------------------------------------------------------------
# Training Logic
# ------------------------------------------------------------------------------

@dataclass
class Config:
    n: int = 12
    arch: str = "D"
    layers: int = 1
    beta: float = 0.9
    steps: int = 600
    lr: float = 0.05
    train_m_list: Tuple[int, ...] = (2000,)
    seeds: Tuple[int, ...] = (42,)
    sigma_list: Tuple[float, ...] = (0.8, 1.0, 1.2)
    K_list: Tuple[int, ...] = (256, 512, 1024)
    Q0: int = 5000
    nested_data: bool = True
    outdir: str = "jonas_pareto"
    fig_target: str = "full"

def build_qnode(cfg: Config):
    if qml is None: raise ImportError("Pennylane missing")
    dev = qml.device("default.qubit", wires=cfg.n)
    pairs, quads = get_iqp_topology(cfg.n, cfg.arch)
    
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(cfg.n), pairs, quads, layers=cfg.layers)
        return qml.probs(wires=range(cfg.n))
    
    return circuit, (len(pairs) + len(quads)) * cfg.layers

def train_one_run(cfg, circuit, num_params, P_mat_tensor, p_star, support, scores, idxs_train, init_seed):
    """Executes gradient descent for one config + computes final metrics."""
    N = 2 ** cfg.n
    emp_dist = empirical_dist(idxs_train, N)
    z_data = np.array(onp.dot(onp.array(P_mat_tensor), emp_dist), requires_grad=False)
    
    rng = onp.random.default_rng(init_seed)
    W = np.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(cfg.lr)

    # Training Loop
    for _ in range(cfg.steps):
        def cost(w):
            q = circuit(w)
            z_model = P_mat_tensor @ q
            return np.mean((z_data - z_model) ** 2)
        W, _ = opt.step_and_cost(cost, W)

    # Evaluation
    q_final = onp.clip(onp.array(circuit(W)), 0.0, 1.0)
    q_final /= max(1e-15, float(q_final.sum()))
    seen = seen_mask_from_indices(idxs_train, N)
    unseen = support & (~seen)
    
    q_unseen = float(q_final[unseen].sum())
    U_unseen = expected_unique_set(q_final, unseen, cfg.Q0)
    
    return {
        "KL": kl_divergence(p_star, q_final, support),
        "q_unseen": q_unseen,
        "U_unseen": U_unseen,
        "spread": U_unseen / (float(cfg.Q0)*q_unseen) if q_unseen>0 else float("nan"),
        "unseen_score_TVD": conditional_score_tvd(p_star, q_final, scores, support, unseen),
        "q_score1_unseen": q_score1_given_unseen(q_final, scores, support, unseen),
    }

# ------------------------------------------------------------------------------
# Plotting Helpers
# ------------------------------------------------------------------------------

def _hatch_map(K_list: List[int]) -> Dict[int, str]:
    """Maps K values to hatch patterns instead of size."""
    patterns = ['', '////', 'xxxx', '....', '||||', '++++']
    Ks_sorted = sorted(list(set(K_list)))
    return {int(K): patterns[i % len(patterns)] for i, K in enumerate(Ks_sorted)}

def _marker_map(sigmas: List[float]) -> Dict[float, str]:
    """Maps Sigma to marker shapes."""
    markers = ["o", "s", "^", "D", "v", "<", ">"]
    sig_sorted = sorted(list(set(sigmas)))
    return {float(s): markers[i % len(markers)] for i, s in enumerate(sig_sorted)}

def _pad_limits(vmin: float, vmax: float, pad_frac: float = 0.18) -> Tuple[float, float]:
    """Adds padding to axis limits."""
    if not onp.isfinite(vmin) or not onp.isfinite(vmax): return vmin, vmax
    r = vmax - vmin if vmax > vmin else 1.0
    return vmin - pad_frac * r, vmax + pad_frac * r

def _make_fig_layout_scatter(fig_target: str):
    """Layout for SCATTER Plots: Wide legend panel."""
    fig = plt.figure(figsize=fig_size(fig_target))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.3, 1.25], wspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    axl = fig.add_subplot(gs[0, 1])
    axl.axis("off")
    
    if fig_target == "full":
        fig.subplots_adjust(left=0.10, right=0.96, bottom=0.22, top=0.92)
    else:
        fig.subplots_adjust(left=0.18, right=0.96, bottom=0.28, top=0.92)
    return fig, ax, axl

def _draw_legend_panel(axl, sigma_list, K_list, marker_map, hatch_map, fig_target):
    """Draws custom legend for Scatter Plots."""
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    
    fs = 10 if fig_target == "full" else 8
    fs_small = fs - 1
    x_mark = 0.15
    x_text = 0.32
    current_y = 0.95

    # Sigma
    axl.text(0.02, current_y, "Shape: σ", fontsize=fs, fontweight="bold", va="top")
    current_y -= 0.08
    for s in sorted(sigma_list):
        axl.scatter([x_mark], [current_y], s=60, marker=marker_map[float(s)],
                    facecolors="white", edgecolors=COLORS["target"], linewidths=1)
        axl.text(x_text, current_y, f"{float(s):g}", fontsize=fs_small, va="center")
        current_y -= 0.07

    # K
    current_y -= 0.02
    axl.text(0.02, current_y, "Pattern: K", fontsize=fs, fontweight="bold", va="top")
    current_y -= 0.08
    for K in sorted(K_list):
        h = hatch_map[int(K)]
        axl.scatter([x_mark], [current_y], s=90, marker="s", 
                    facecolors="white", edgecolors=COLORS["target"], hatch=h, linewidths=1)
        axl.text(x_text, current_y, f"{int(K)}", fontsize=fs_small, va="center")
        current_y -= 0.07

    # Status
    current_y -= 0.02
    axl.text(0.02, current_y, "Status", fontsize=fs, fontweight="bold", va="top")
    current_y -= 0.08
    axl.scatter([x_mark], [current_y], s=60, marker="o", facecolors="white", 
                edgecolors=COLORS["gray"], linewidths=0.8)
    axl.text(x_text, current_y, "Dominated", fontsize=fs_small, va="center")
    current_y -= 0.07
    axl.scatter([x_mark], [current_y], s=60, marker="o", facecolors="white", 
                edgecolors=COLORS["model"], linewidths=2.5)
    axl.text(x_text, current_y, "Pareto", fontsize=fs_small, va="center")


# ------------------------------------------------------------------------------
# Visualization Functions
# ------------------------------------------------------------------------------

def plot_scatter_metric(
    points: List[Dict], pareto_mask: onp.ndarray, 
    xkey: str, ykey: str, 
    xlabel: str, ylabel: str,
    sigma_list, K_list, outpath, fig_target
):
    """Generates Scatter plots with Rotated X-Labels and Hatch patterns."""
    marker_map = _marker_map(sigma_list)
    hatch_map = _hatch_map(K_list)
    
    fig, ax, axl = _make_fig_layout_scatter(fig_target)
    
    for p, is_p in zip(points, pareto_mask):
        if not (onp.isfinite(p[xkey]) and onp.isfinite(p[ykey])): continue
        edge_col = COLORS["model"] if is_p else COLORS["gray"]
        lw = 2.5 if is_p else 0.8
        alpha = 1.0 if is_p else 0.6
        z = 4 if is_p else 3
        
        ax.scatter(
            p[xkey], p[ykey],
            s=100, 
            marker=marker_map[float(p["sigma"])],
            hatch=hatch_map[int(p["K"])],
            facecolors="white",
            edgecolors=edge_col,
            linewidths=lw,
            alpha=alpha,
            zorder=z
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Rotate X labels to prevent overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # Limits
    xs = [p[xkey] for p in points if onp.isfinite(p[xkey])]
    ys = [p[ykey] for p in points if onp.isfinite(p[ykey])]
    if xs and ys:
        ax.set_xlim(*_pad_limits(min(xs), max(xs), 0.2))
        ax.set_ylim(*_pad_limits(min(ys), max(ys), 0.22))

    _draw_legend_panel(axl, sigma_list, K_list, marker_map, hatch_map, fig_target)
    fig.savefig(outpath)
    plt.close(fig)


def plot_score1_heatmap(points, sigma_list, K_list, outpath, fig_target):
    """
    Generates Heatmap (Square 1:1) with TIGHTLY attached colorbar
    using make_axes_locatable. This removes the whitespace gap.
    """
    sigmas = sorted(list(set(sigma_list)))
    Ks = sorted(list(set(K_list)))
    
    # Grid construction
    grid = onp.full((len(sigmas), len(Ks)), onp.nan)
    sig_map = {s: i for i, s in enumerate(sigmas)}
    K_map = {k: i for i, k in enumerate(Ks)}
    
    for p in points:
        if p["sigma"] in sig_map and p["K"] in K_map:
            grid[sig_map[p["sigma"]], K_map[p["K"]]] = p["qS1_mean"]
            
    # --- Custom Layout for TIGHT Heatmap ---
    # We just create a single axes, and use make_axes_locatable to attach the cbar
    fig, ax = plt.subplots(figsize=fig_size(fig_target))
    
    # Adjust margins slightly to allow room for the attached colorbar labels
    if fig_target == "full":
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.22, top=0.92)
    else:
        fig.subplots_adjust(left=0.18, right=0.82, bottom=0.28, top=0.92)
    
    # Plot
    cmap = "Reds" 
    # aspect='equal' enforces square pixels. 
    im = ax.imshow(grid, cmap=cmap, origin="lower", aspect="equal")
    
    # --- THE FIX: make_axes_locatable ---
    divider = make_axes_locatable(ax)
    # Append axes to the right of 'ax', with 5% width and 0.05 padding
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("q(score=1|unseen)", rotation=270, labelpad=15, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Ticks & Labels
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f"{k}" for k in Ks], fontsize=9)
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f"{s:g}" for s in sigmas], fontsize=9)
    
    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel("σ", fontsize=10)
    
    # Grid Lines (Cell separators)
    ax.set_xticks(onp.arange(len(Ks) + 1) - 0.5, minor=True)
    ax.set_yticks(onp.arange(len(sigmas) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6, alpha=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Values
    for i in range(len(sigmas)):
        for j in range(len(Ks)):
            val = grid[i, j]
            if not onp.isnan(val):
                norm = (val - onp.nanmin(grid)) / (onp.nanmax(grid) - onp.nanmin(grid) + 1e-9)
                col = "white" if norm > 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=col, fontsize=8)

    fig.savefig(outpath)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--arch", type=str, default="D", choices=["A","B","C","D","E"])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--train-m-list", type=int, nargs="+", default=[2000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--sigma-list", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    ap.add_argument("--K-list", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--Q0", type=int, default=5000)
    ap.add_argument("--nested-data", action="store_true", default=True)
    ap.add_argument("--outdir", type=str, default="jonas_pareto")
    ap.add_argument("--fig-target", type=str, default="full", choices=["col","full"])
    args = ap.parse_args()

    cfg = Config(
        n=args.n, arch=args.arch, layers=args.layers, beta=args.beta,
        steps=args.steps, lr=args.lr,
        train_m_list=tuple(args.train_m_list), seeds=tuple(args.seeds),
        sigma_list=tuple(args.sigma_list), K_list=tuple(args.K_list),
        Q0=args.Q0, nested_data=args.nested_data, outdir=args.outdir, fig_target=args.fig_target
    )

    set_style(cfg.fig_target)
    outdir = ensure_outdir(cfg.outdir)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    p_star, support, scores = build_target_distribution(cfg.n, cfg.beta)
    bits_table = make_bits_table(cfg.n)
    circuit, num_params = build_qnode(cfg)
    
    m_max = max(cfg.train_m_list)
    rows = []

    print(f"[Start] n={cfg.n}, arch={cfg.arch}, layers={cfg.layers}, m={cfg.train_m_list}")
    print(f"[Sweep] Seeds={cfg.seeds}, Sigmas={cfg.sigma_list}, K={cfg.K_list}")

    for seed in cfg.seeds:
        if cfg.nested_data:
            idxs_all = sample_indices(p_star, m_max, seed + 7)
            train_sets = {m: idxs_all[:m] for m in cfg.train_m_list}
        else:
            train_sets = {m: sample_indices(p_star, m, seed + 7 + 10000*m) for m in cfg.train_m_list}

        for m, idxs_train in train_sets.items():
            for sigma in cfg.sigma_list:
                for K in cfg.K_list:
                    fseed = seed + 222 + int(1e5*sigma) + 10*K
                    alphas = sample_alphas(cfg.n, sigma, K, fseed)
                    P_mat = build_parity_matrix(alphas, bits_table)
                    res = train_one_run(cfg, circuit, num_params, P_mat, p_star, support, scores, idxs_train, seed)
                    
                    row = {"seed": seed, "m": m, "sigma": sigma, "K": K}
                    row.update(res)
                    rows.append(row)
                    print(f"   Done: s={seed} m={m:<4} sig={sigma:<4} K={K:<4} | "
                          f"KL={res['KL']:.3f} U={res['U_unseen']:.0f} q1={res['q_score1_unseen']:.3f}")

    # Summary
    summary = []
    keys = ["KL", "U_unseen", "spread", "unseen_score_TVD", "q_score1_unseen", "q_unseen"]
    map_keys = {"KL":"KL", "U_unseen":"U", "spread":"spread", 
                "unseen_score_TVD":"unseenTVD", "q_score1_unseen":"qS1", "q_unseen":"q_unseen"}
    
    groups = {}
    for r in rows:
        k = (r['m'], r['sigma'], r['K'])
        if k not in groups: groups[k] = []
        groups[k].append(r)
        
    for (m, sig, K), grp in groups.items():
        entry = {"m": m, "sigma": sig, "K": K}
        for k in keys:
            vals = [g[k] for g in grp]
            entry[f"{map_keys[k]}_mean"] = float(onp.mean(vals))
            entry[f"{map_keys[k]}_std"] = float(onp.std(vals, ddof=1)) if len(vals)>1 else 0.0
        summary.append(entry)

    save_csv(rows, os.path.join(outdir, "jonas_pareto_raw.csv"))
    save_csv(summary, os.path.join(outdir, "jonas_pareto_summary.csv"))
    print("\n[Data] CSV files saved.")

    # Plotting
    objectives = [("KL_mean", "min"), ("U_mean", "max"), ("spread_mean", "max"), ("unseenTVD_mean", "min")]
    unique_ms = sorted(list(set(s["m"] for s in summary)))
    
    for m in unique_ms:
        pts = [s for s in summary if s["m"] == m]
        
        # Pareto Logic
        n = len(pts)
        is_pareto = onp.ones(n, dtype=bool)
        vals = onp.array([[p[obj] for obj, _ in objectives] for p in pts])
        senses = np.array([1.0 if s=="min" else -1.0 for _, s in objectives])
        
        for i in range(n):
            for j in range(n):
                if i==j: continue
                diff = (vals[j] - vals[i]) * senses 
                if np.all(diff <= 1e-9) and np.any(diff < -1e-9):
                    is_pareto[i] = False
                    break
        
        pareto_list = [pts[i] for i in range(n) if is_pareto[i]]
        save_pareto_txt(
            sorted(pareto_list, key=lambda x: x["KL_mean"]), 
            os.path.join(outdir, f"pareto_points_m{m}.txt")
        )
        
        # Plot Calls
        plot_scatter_metric(
            pts, is_pareto, "KL_mean", "U_mean", 
            "KL(p* || q)  (↓)", f"U_unseen(Q={cfg.Q0})  (↑)",
            cfg.sigma_list, cfg.K_list, 
            os.path.join(outdir, f"pareto_fit_yield_m{m}.pdf"), cfg.fig_target
        )
        plot_scatter_metric(
            pts, is_pareto, "spread_mean", "unseenTVD_mean", 
            "spread = U / (Q·q_unseen) (↑)", "TVD(score | unseen) (↓)",
            cfg.sigma_list, cfg.K_list, 
            os.path.join(outdir, f"pareto_dispersion_m{m}.pdf"), cfg.fig_target
        )
        plot_score1_heatmap(
            pts, cfg.sigma_list, cfg.K_list, 
            os.path.join(outdir, f"heatmap_qS1_m{m}.pdf"), cfg.fig_target
        )

    print(f"[Done] All plots saved to ./{outdir}/")

def save_csv(rows, path):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

def save_pareto_txt(points, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Non-Dominated Configs\n")
        for p in points:
            f.write(f"sig={p['sigma']:g} K={p['K']:<4} | KL={p['KL_mean']:.4f} U={p['U_mean']:.1f}\n")

if __name__ == "__main__":
    main()