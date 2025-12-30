#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQP-QCBM: Hero Master Script (Architecture D default)
=====================================================

Paper-grade IQP-QCBM experiments + plots in a clean red/black style.

Update in this version:
  - Adds a *journal-aware* plotting style with consistent typography.
  - Adds a `--fig-target {col,full}` switch.
  - FORCE SIZE: All plots are now generated with IDENTICAL dimensions
    (2.8 inches height for col, 3.5 inches for full) to ensure perfect alignment.
  - ADJUSTMENTS:
      * Plot 1 Legend moved to upper center (whitespace).
      * Plot 3 Value label moved below the curve.
      * Plot 4 Zoom removed, Legend moved down (center right).
      * Plot 5 "Full recovery" label moved up.

Outputs (<outdir>/):
  - 1_dynamics_clean.pdf
  - 2_spectrum.pdf
  - 3_diversity.pdf
  - 4_holdout_unique_vs_Q.pdf   (if holdout enabled)
  - 5_holdout_recovery_bars.pdf (if holdout enabled)
  - holdout_strings.txt         (if holdout enabled)
  - run_A{arch}_L{layers}_S{seed}.json

Deps:
  pip install pennylane numpy matplotlib
"""

import os
import json
import math
import argparse
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Inset locator not needed anymore for Plot 4, but kept if you want to reuse later
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pennylane")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

import numpy as onp
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as np


# ------------------------------------------------------------------------------
# 1) Visual Style (journal-aware)
# ------------------------------------------------------------------------------

COLORS = {
    "target": "#222222",   # Almost Black
    "model":  "#D62728",   # Deep Red
    "loss":   "#1F77B4",   # Deep Blue (optional)
    "gray":   "#666666",
}

# Practical widths for APS/RevTeX, in inches
COL_W  = 3.37  # single-column
FULL_W = 6.95  # two-column (figure*)

# --- FORCE UNIFORM HEIGHTS ---
# Damit alle Plots exakt gleich groß sind:
FIXED_H_COL  = 2.8
FIXED_H_FULL = 3.5

def fig_size(fig_target: str, h_col: float, h_full: float) -> Tuple[float, float]:
    """Return (width,height) in inches for the chosen LaTeX target."""
    if fig_target not in ("col", "full"):
        raise ValueError("fig_target must be 'col' or 'full'")
    if fig_target == "col":
        return (COL_W, h_col)
    return (FULL_W, h_full)

def set_style(fig_target: str = "col"):
    """
    Two consistent styles:
      - col  : RevTeX column-ready (fonts ~8–9pt, thin lines)
      - full : larger "Nature-like" figure* vibe (fonts ~12–14pt, thicker lines)
    """
    if fig_target == "col":
        base = 8
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": base,
            "axes.labelsize": base + 1,
            "axes.titlesize": base + 1,
            "legend.fontsize": base - 1,
            "legend.frameon": False,
            "xtick.labelsize": base,
            "ytick.labelsize": base,

            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,

            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,

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
    else:
        # This matches your "bigger" master style
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "--",
            "lines.linewidth": 2.5,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
# 2) Physics & Logic
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

def build_target_distribution(n: int, beta: float):
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

def topk_mask(scores, support, frac=0.05):
    valid = onp.where(support)[0]
    k = max(1, int(onp.floor(frac * valid.size)))
    valid_scores = scores[valid]
    local_order = onp.argsort(-valid_scores)
    top_indices = valid[local_order[:k]]
    mask = onp.zeros_like(support, dtype=bool)
    mask[top_indices] = True
    return mask

def sample_indices(probs, m, seed=7):
    rng = onp.random.default_rng(seed)
    p = probs / probs.sum()
    return rng.choice(len(p), size=m, replace=True, p=p)

def empirical_dist(idxs, N):
    c = onp.bincount(idxs, minlength=N)
    return (c / max(1, c.sum())).astype(onp.float64)

def make_bits_table(n: int) -> onp.ndarray:
    N = 2 ** n
    return onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)


# ------------------------------------------------------------------------------
# 3) Smart Holdout & Export
# ------------------------------------------------------------------------------

def _min_hamming_to_set(bit_vec: onp.ndarray, sel_bits: onp.ndarray) -> int:
    if sel_bits.shape[0] == 0:
        return bit_vec.shape[0]
    d = onp.sum(sel_bits != bit_vec[None, :], axis=1)
    return int(onp.min(d))

def select_holdout_smart(p_star, good_mask, bits_table, m_train, holdout_k, pool_size, seed):
    if holdout_k <= 0:
        return onp.zeros_like(good_mask, dtype=bool)
    good_idxs = onp.where(good_mask)[0]
    if good_idxs.size == 0:
        raise RuntimeError("good_mask ist leer.")

    # prefer not-too-rare states given m_train
    taus = [1.0/max(1,m_train), 0.5/max(1,m_train), 0.25/max(1,m_train), 0.0]
    cand = None
    for tau in taus:
        cand = good_idxs[p_star[good_idxs] >= tau]
        if cand.size >= holdout_k:
            break
    cand = cand[onp.argsort(-p_star[cand])]
    pool = cand[:min(pool_size, cand.size)]

    selected = [int(pool[0])]
    selected_bits = onp.vstack([onp.zeros((0, bits_table.shape[1]), dtype=onp.int8), bits_table[selected[-1]]])

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
    print(f"[Holdout] Selected {int(holdout.sum())} states.")
    return holdout

def save_holdout_list(holdout_mask, bits_table, p_star, scores, outdir):
    idxs = onp.where(holdout_mask)[0]
    sorted_idxs = idxs[onp.argsort(-p_star[idxs])]

    path = os.path.join(outdir, "holdout_strings.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Holdout Strings (k={len(idxs)})\n")
        f.write(f"# {'Index':<8} {'Bitstring':<16} {'Score':<8} {'Prob p*(x)':<16}\n")
        f.write("-" * 55 + "\n")
        for idx in sorted_idxs:
            b_str = bits_str(bits_table[int(idx)])
            s_val = float(scores[int(idx)])
            prob  = float(p_star[int(idx)])
            f.write(f"{int(idx):<8d} {b_str:<16s} {s_val:<8.1f} {prob:.6e}\n")
    print(f"[Export] Holdout list saved to: {path}")


# ------------------------------------------------------------------------------
# 4) Circuit & MMD
# ------------------------------------------------------------------------------

def get_iqp_topology(n, arch):
    pairs, quads = [], []
    def clean(l): return sorted(list(set(l)))

    if arch in ["A","B","C","D"]:
        pairs.extend([tuple(sorted((i, (i+1)%n))) for i in range(n)])
    if arch in ["C","D"]:
        pairs.extend([tuple(sorted((i, (i+2)%n))) for i in range(n)])
    if arch in ["B","D"]:
        quads.extend([tuple(sorted((i, (i+1)%n, (i+2)%n, (i+3)%n))) for i in range(n)])
    if arch == "E":
        pairs = list(itertools.combinations(range(n), 2))

    return clean(pairs), clean(quads)

def iqp_circuit(W, wires, pairs, quads, layers=1):
    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)
    for _ in range(layers):
        for (i,j) in pairs:
            qml.IsingZZ(W[idx], wires=[wires[i],wires[j]])
            idx += 1
        for (a,b,c,d) in quads:
            qml.MultiRZ(W[idx], wires=[wires[a],wires[b],wires[c],wires[d]])
            idx += 1
        for w in wires:
            qml.Hadamard(wires=w)

def p_sigma(sigma):
    return 0.5 * (1.0 - math.exp(-1.0/(2.0*sigma**2))) if sigma > 0 else 0.5

def sample_alphas(n, sigma, K, seed):
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas, bits_table):
    """
    P[k, x] = (-1)^{alpha_k · x} in {+1,-1}.
    bits_table: (N,n)
    """
    A = alphas.astype(onp.int16)
    X = bits_table.astype(onp.int16).T  # n x N
    par = (A @ X) & 1
    return onp.where(par == 0, 1.0, -1.0).astype(onp.float64)

def expected_unique_set(probs, mask, Q_vals):
    pS = probs[mask].astype(onp.float64)[:, None]
    if pS.size == 0:
        return onp.zeros_like(Q_vals, dtype=onp.float64)
    return onp.sum(1.0 - onp.power(1.0 - pS, Q_vals[None, :]), axis=0)


# ------------------------------------------------------------------------------
# 5) Training
# ------------------------------------------------------------------------------

@dataclass
class Config:
    n: int = 14
    arch: str = "D"
    layers: int = 1
    beta: float = 0.9
    steps: int = 600
    lr: float = 0.05
    sigma: float = 1.0
    num_alpha: int = 512
    train_m: int = 1000
    eval_every: int = 50
    holdout_k: int = 0
    holdout_pool: int = 400
    seed: int = 42
    outdir: str = "results_clean"
    fig_target: str = "col"  # 'col' or 'full'

def train(cfg: Config):
    outdir = ensure_outdir(cfg.outdir)
    N = 2 ** cfg.n
    bits_table = make_bits_table(cfg.n)

    p_star, support, scores = build_target_distribution(cfg.n, cfg.beta)
    good_mask = topk_mask(scores, support, frac=0.05)
    pstar_top5 = float(p_star[good_mask].sum())
    print(f"[Target] p*(G)={pstar_top5:.6f}")

    holdout_enabled = (cfg.holdout_k > 0)
    if not holdout_enabled:
        ds, as_, is_ = 7, 42, 42
        print(f"[Mode] Baseline | arch={cfg.arch}")
    else:
        ds, as_, is_ = cfg.seed + 7, cfg.seed + 222, cfg.seed
        print(f"[Mode] Holdout ON (k={cfg.holdout_k}) | arch={cfg.arch}")

    holdout_mask = select_holdout_smart(
        p_star, good_mask, bits_table, cfg.train_m, cfg.holdout_k, cfg.holdout_pool, cfg.seed + 111
    )

    if holdout_enabled:
        save_holdout_list(holdout_mask, bits_table, p_star, scores, outdir)

    p_train = p_star.copy()
    if holdout_enabled and holdout_mask.any():
        p_train[holdout_mask] = 0.0
    p_train /= p_train.sum()

    idxs_train = sample_indices(p_train, cfg.train_m, seed=ds)
    emp_dist = empirical_dist(idxs_train, N)

    alphas = sample_alphas(cfg.n, cfg.sigma, cfg.num_alpha, seed=as_)
    P_mat = build_parity_matrix(alphas, bits_table)
    z_data = onp.dot(P_mat, emp_dist).astype(onp.float64)

    dev = qml.device("default.qubit", wires=cfg.n)
    pairs, quads = get_iqp_topology(cfg.n, cfg.arch)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(cfg.n), pairs, quads, layers=cfg.layers)
        return qml.probs(wires=range(cfg.n))

    P_mat_tensor = np.array(P_mat, requires_grad=False)
    z_data_tensor = np.array(z_data, requires_grad=False)

    num_params = (len(pairs) + len(quads)) * cfg.layers
    rng = onp.random.default_rng(is_)
    W = np.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(cfg.lr)

    history = {"step": [], "loss": [], "top5": []}

    print(f"--- Training ({cfg.steps} steps) ---")
    for t in range(1, cfg.steps + 1):
        def loss_fn(w):
            q = circuit(w)
            return np.mean((z_data_tensor - P_mat_tensor @ q) ** 2)
        W, l_val = opt.step_and_cost(loss_fn, W)

        if t % cfg.eval_every == 0 or t == 1 or t == cfg.steps:
            q_val = onp.clip(onp.array(circuit(W), dtype=onp.float64), 0.0, 1.0)
            q_sum = float(q_val.sum())
            if q_sum > 0:
                q_val /= q_sum
            t5 = float(q_val[good_mask].sum())
            history["step"].append(int(t))
            history["loss"].append(float(l_val))
            history["top5"].append(float(t5))
            print(f"Step {t:3d} | Loss {float(l_val):.2e} | q(G) {t5:.3f}")

    q_final = onp.clip(onp.array(circuit(W), dtype=onp.float64), 0.0, 1.0)
    q_sum = float(q_final.sum())
    if q_sum > 0:
        q_final /= q_sum

    with open(os.path.join(outdir, f"run_A{cfg.arch}_L{cfg.layers}_S{cfg.seed}.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "params": int(num_params)}, f, indent=2)

    return {
        "history": history,
        "p_star": p_star,
        "q": q_final,
        "scores": scores,
        "support": support,
        "good_mask": good_mask,
        "holdout_mask": holdout_mask,
        "pstar_top5": pstar_top5,
        "outdir": outdir,
        "holdout_enabled": holdout_enabled,
        "Qmax": 10000,
        "fig_target": cfg.fig_target,
    }


# ------------------------------------------------------------------------------
# 6) Plotting (sizes controlled by fig_target AND fixed constants)
# ------------------------------------------------------------------------------

def plot_dynamics_clean(history, pstar_top5, outdir, fig_target: str):
    steps = onp.array(history["step"], dtype=int)
    loss = onp.array(history["loss"], dtype=onp.float64)
    top5 = onp.array(history["top5"], dtype=onp.float64)

    # Use FIXED height
    fig, ax1 = plt.subplots(figsize=fig_size(fig_target, FIXED_H_COL, FIXED_H_FULL), constrained_layout=True)

    ax1.plot(steps, loss, color=COLORS["target"], linewidth=2.0, label=r"MMD Loss $\mathcal{L}$")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss (MMD)", color=COLORS["target"])
    ax1.tick_params(axis="y", labelcolor=COLORS["target"])
    ax1.set_yscale("log")
    ax1.text(steps[-1], loss[-1] * 1.12, f"{loss[-1]:.2e}", color=COLORS["target"],
             fontweight="bold", ha="right", va="bottom")

    ax2 = ax1.twinx()
    ax2.plot(steps, top5, color=COLORS["model"], linewidth=2.2, label=r"Model $q(G)$")
    h = ax2.axhline(pstar_top5, color=COLORS["target"], linestyle=":", linewidth=1.6, alpha=0.8,
                    label=r"Target $p^*(G)$")
    ax2.text(steps[-1], top5[-1] + 0.02, f"{top5[-1]:.3f}", color=COLORS["model"],
             fontweight="bold", ha="right", va="bottom")

    ax2.set_ylabel("Top-5% Probability Mass", color=COLORS["model"], fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=COLORS["model"])
    ax2.set_ylim(0, 1.1)

    # combined legend - MOVED TO UPPER CENTER (Whitespace)
    lines = ax1.get_lines() + ax2.get_lines() + [h]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", frameon=True, framealpha=0.9, edgecolor="white")

    fig.savefig(os.path.join(outdir, "1_dynamics_clean.pdf"))
    plt.close(fig)

def plot_score_spectrum_awesome(p_star, q, scores, support, outdir, fig_target: str):
    valid_scores = scores[support].astype(int)
    unique_s = sorted(onp.unique(valid_scores))

    y_star, y_q = [], []
    scores_int = scores.astype(int)
    for s in unique_s:
        mask = (scores_int == s) & support
        y_star.append(float(p_star[mask].sum()))
        y_q.append(float(q[mask].sum()))

    tvd_val = 0.5 * onp.sum(onp.abs(onp.array(y_star) - onp.array(y_q)))
    x = onp.arange(len(unique_s))
    width = 0.42

    # Use FIXED height
    fig, ax = plt.subplots(figsize=fig_size(fig_target, FIXED_H_COL, FIXED_H_FULL), constrained_layout=True)
    ax.bar(x - width/2, y_star, width, color="white", edgecolor="#333333", linewidth=1.0,
           hatch="///", alpha=0.9, label=r"Target $p^*$", zorder=2)
    ax.bar(x + width/2, y_q, width, color=COLORS["model"], edgecolor="black", linewidth=0.8,
           alpha=0.9, label=r"Model $q_\theta$", zorder=3)

    ax.text(0.02, 0.95, f"TVD = {tvd_val:.3f}", transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
            zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in unique_s])
    ax.set_xlabel("Score")
    ax.set_ylabel("Probability Mass")
    ax.set_ylim(0, max(max(y_star), max(y_q)) * 1.25)

    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.legend(loc="upper right")

    fig.savefig(os.path.join(outdir, "2_spectrum.pdf"))
    plt.close(fig)

def plot_diversity_curve(p_star, q, good_mask, outdir, fig_target: str, Qmax=5000):
    Q = onp.linspace(1, Qmax, 120).astype(int)
    ys = expected_unique_set(p_star, good_mask, Q)
    yq = expected_unique_set(q, good_mask, Q)
    k = int(good_mask.sum())

    # Use FIXED height
    fig, ax = plt.subplots(figsize=fig_size(fig_target, FIXED_H_COL, FIXED_H_FULL), constrained_layout=True)
    ax.plot(Q, ys, color=COLORS["target"], linewidth=1.8, label="Target")
    ax.plot(Q, yq, color=COLORS["model"], linewidth=2.0, label="Model")
    ax.axhline(k, color=COLORS["gray"], linestyle=":", linewidth=1.2, label=f"Max ({k})")

    final_val = float(yq[-1])
    # Shifted down to sit below curve
    ax.text(Q[-1], final_val - (k * 0.05), f"{final_val:.1f}", color=COLORS["model"],
            fontweight="bold", ha="right", va="top")

    ax.set_xlabel(r"$Q$")
    ax.set_ylabel("Expected unique in good set")
    ax.legend(loc="lower right")

    fig.savefig(os.path.join(outdir, "3_diversity.pdf"))
    plt.close(fig)

def plot_holdout_unique_vs_Q_awesome(p_star, q, holdout_mask, outdir, fig_target: str, Qmax=10000):
    H = int(onp.sum(holdout_mask))
    if H == 0:
        return

    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, 3.5, 120).astype(int)),
        onp.linspace(1000, Qmax, 140).astype(int)
    ]))
    Q = Q[Q <= Qmax]

    ys = expected_unique_set(p_star, holdout_mask, Q)
    yq = expected_unique_set(q, holdout_mask, Q)

    # Use FIXED height
    fig, ax = plt.subplots(figsize=fig_size(fig_target, FIXED_H_COL, FIXED_H_FULL), constrained_layout=True)
    ax.plot(Q, ys, color=COLORS["target"], linewidth=1.8, label=r"Target $U_H^*(Q)$", zorder=5)
    ax.plot(Q, yq, color=COLORS["model"], linewidth=2.2, label=r"Model $U_H(Q)$", zorder=6)
    ax.axhline(H, color=COLORS["gray"], linestyle="--", alpha=0.5, linewidth=1.0)
    ax.fill_between(Q, yq, ys, color=COLORS["model"], alpha=0.10, label="Discovery gap")

    ax.set_xlabel(r"$Q$")
    ax.set_ylabel(r"Expected unique in $H$")
    
    # MODIFICATION: Removed inset zoom, Moved legend to center right
    ax.legend(loc="center right", frameon=True, framealpha=0.95, edgecolor="white")

    fig.savefig(os.path.join(outdir, "4_holdout_unique_vs_Q.pdf"))
    plt.close(fig)

def plot_holdout_recovery_bars_sexy(p_star, q, holdout_mask, outdir, fig_target: str, Q_list=(1000, 5000, 10000)):
    H = int(onp.sum(holdout_mask))
    if H == 0:
        return

    Q = onp.array(list(Q_list), dtype=int)
    Us = expected_unique_set(p_star, holdout_mask, Q) / H
    Uq = expected_unique_set(q, holdout_mask, Q) / H
    x = onp.arange(len(Q))
    w = 0.36

    # Use FIXED height
    fig, ax = plt.subplots(figsize=fig_size(fig_target, FIXED_H_COL, FIXED_H_FULL), constrained_layout=True)
    bs = ax.bar(x - w/2, Us, w, color="white", edgecolor="#333333", hatch="///",
                alpha=0.9, label=r"Target $p^*$")
    bq = ax.bar(x + w/2, Uq, w, color=COLORS["model"], edgecolor="black",
                alpha=0.85, label=r"Model $q_\theta$")

    ax.axhline(1.0, color=COLORS["gray"], linestyle="--", alpha=0.6)
    
    # MODIFICATION: Moved text up from 1.05 to 1.12
    ax.text(x[-1] + w, 1.12, "Full recovery", color=COLORS["gray"],
            va="bottom", ha="right", style="italic")

    def _lbl(rects, color, bold=False):
        for r in rects:
            h = float(r.get_height())
            ax.annotate(f"{100*h:.0f}%",
                        xy=(r.get_x() + r.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=max(7, plt.rcParams["font.size"] - 1),
                        fontweight="bold" if bold else "normal",
                        color=color)

    _lbl(bs, color="#333333", bold=False)
    _lbl(bq, color="#800000", bold=True)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q={int(qv)}" for qv in Q])
    ax.set_ylabel("Fraction recovered")
    ax.set_ylim(0, 1.25)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.12)

    fig.savefig(os.path.join(outdir, "5_holdout_recovery_bars.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=14)
    parser.add_argument("--arch", type=str, default="D", choices=["A","B","C","D","E"])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-alpha", type=int, default=512)
    parser.add_argument("--train-m", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--holdout-k", type=int, default=0)
    parser.add_argument("--holdout-pool", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="results_clean")

    # NEW: plot target (column vs full-width)
    parser.add_argument("--fig-target", type=str, default="col", choices=["col", "full"],
                        help="col: RevTeX single-column-ready PDFs, full: figure*-ready PDFs")

    args = parser.parse_args()

    set_style(args.fig_target)

    if args.arch == "D":
        print("[Hero] Architecture D (Full Hybrid) selected.")
    else:
        print(f"[Info] Architecture {args.arch} selected.")

    cfg = Config(
        n=args.n,
        arch=args.arch,
        layers=args.layers,
        beta=args.beta,
        steps=args.steps,
        lr=args.lr,
        sigma=args.sigma,
        num_alpha=args.num_alpha,
        train_m=args.train_m,
        eval_every=args.eval_every,
        holdout_k=args.holdout_k,
        holdout_pool=args.holdout_pool,
        seed=args.seed,
        outdir=args.outdir,
        fig_target=args.fig_target,
    )

    ensure_outdir(cfg.outdir)
    with open(os.path.join(cfg.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    res = train(cfg)
    outdir = res["outdir"]
    fig_target = res["fig_target"]

    print("\nGenerating Plots...")
    plot_dynamics_clean(res["history"], res["pstar_top5"], outdir, fig_target)
    plot_score_spectrum_awesome(res["p_star"], res["q"], res["scores"], res["support"], outdir, fig_target)
    plot_diversity_curve(res["p_star"], res["q"], res["good_mask"], outdir, fig_target)

    if res["holdout_enabled"]:
        plot_holdout_unique_vs_Q_awesome(res["p_star"], res["q"], res["holdout_mask"], outdir, fig_target, res["Qmax"])
        plot_holdout_recovery_bars_sexy(res["p_star"], res["q"], res["holdout_mask"], outdir, fig_target)

    print(f"Done. Results in ./{outdir}/")
    print(f"[Tip] LaTeX: include these PDFs with width=\\columnwidth if --fig-target col, or width=\\textwidth if --fig-target full.")

if __name__ == "__main__":
    main()