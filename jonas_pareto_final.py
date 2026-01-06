#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jonas_pareto_final.py
====================

Pareto-Navigation for "training-unseen" generalization (Jonas sample-complexity setting)

Goal
----
For a *fixed architecture* (IQP-QCBM topology, depth), fixed training sample size m, and fixed optimizer
settings, we sweep only the *parity-MMD feature hyperparameters*:

  - sigma (kernel bandwidth)  -> which Walsh-Fourier band is enforced
  - K     (# random parity features) -> how many constraints are enforced

and evaluate a minimal-but-complete metric suite that makes the trade-offs scientific and operational:

  1) KL(p* || q)                         [global fit; lower is better]
  2) U_unseen(Q0)                        [yield of unique training-unseen states; higher is better]
  3) spread = U_unseen(Q0)/(Q0*q(unseen)) [dispersion of unseen mass; closer to 1 is better]
  4) unseen_score_TVD                    [calibration on the unseen tail; lower is better]
  (optional diagnostic) q(score=1 | unseen) [detects the score-1 spike]

The script writes:
  - outdir/jonas_pareto_raw.csv          : per seed x (m,sigma,K)
  - outdir/jonas_pareto_summary.csv      : mean±std aggregated over seeds
  - outdir/pareto_points_m{m}.txt        : readable summary + Pareto set
  - outdir/pareto_fit_yield_m{m}.pdf     : Plot 1 (KL vs Yield) with clean labels
  - outdir/pareto_dispersion_m{m}.pdf    : Plot 2 (unseenTVD vs spread) with clean labels

Notes
-----
- No architecture changes, no extra loss terms. This is pure parity-MMD.
- Uses full-statevector evaluation (default.qubit) and exact q(x), like the main scripts.
- The plots label only Pareto-optimal points to avoid clutter.

Example
-------
python -B jonas_pareto_final.py \
  --n 14 --arch D --layers 1 --beta 0.9 \
  --train-m-list 2000 \
  --seeds 42 43 44 \
  --steps 600 --lr 0.05 \
  --sigma-list 0.6 0.8 1.0 1.2 \
  --K-list 128 256 512 1024 \
  --Q0 5000 \
  --outdir jonas_pareto_sigK \
  --fig-target full

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

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pennylane")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

import numpy as onp
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as np

# ------------------------------------------------------------------------------
# Style (APS/RevTeX-friendly; clean Pareto plots)
# ------------------------------------------------------------------------------

COLORS = {
    "target": "#222222",   # almost black
    "model":  "#D62728",   # deep red
    "gray":   "#666666",
    "light":  "#D0D0D0",
    "dark":   "#111111",
}

COL_W  = 3.37  # single-column
FULL_W = 6.95  # two-column
FIXED_H_COL  = 2.75
FIXED_H_FULL = 3.45

def fig_size(fig_target: str) -> Tuple[float, float]:
    if fig_target not in ("col", "full"):
        raise ValueError("fig_target must be 'col' or 'full'")
    return (COL_W, FIXED_H_COL) if fig_target == "col" else (FULL_W, FIXED_H_FULL)

def set_style(fig_target: str) -> None:
    if fig_target == "col":
        base = 8
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": base,
            "axes.labelsize": base + 1,
            "legend.fontsize": base - 1,
            "legend.frameon": False,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

# ------------------------------------------------------------------------------
# Target distribution: even parity + longest-zero-run score
# ------------------------------------------------------------------------------

def int2bits(k: int, n: int) -> onp.ndarray:
    return onp.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=onp.int8)

def parity_even(bits: onp.ndarray) -> bool:
    return (int(onp.sum(bits)) % 2) == 0

def longest_zero_run_between_ones(bits: onp.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0

def build_target_distribution(n: int, beta: float) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    p*(x) ∝ exp(beta * score(x)) on even-parity sector, else 0.
    Returns:
      p_star:  (N,)
      support: bool mask (N,)
      scores:  (N,)
    """
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
    N = 2 ** n
    return onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)

# ------------------------------------------------------------------------------
# Sampling utilities (training sets)
# ------------------------------------------------------------------------------

def sample_indices(probs: onp.ndarray, m: int, seed: int) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    p = probs / probs.sum()
    return rng.choice(len(p), size=m, replace=True, p=p)

def empirical_dist(idxs: onp.ndarray, N: int) -> onp.ndarray:
    c = onp.bincount(idxs, minlength=N)
    return (c / max(1, c.sum())).astype(onp.float64)

def seen_mask_from_indices(idxs: onp.ndarray, N: int) -> onp.ndarray:
    c = onp.bincount(idxs, minlength=N)
    return (c > 0)

# ------------------------------------------------------------------------------
# Parity features (Walsh characters) + random-feature kernel approx
# ------------------------------------------------------------------------------

def p_sigma(sigma: float) -> float:
    # p(sigma) = (1 - exp(-1/(2*sigma^2)))/2
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma**2))) if sigma > 0 else 0.5

def sample_alphas(n: int, sigma: float, K: int, seed: int) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas: onp.ndarray, bits_table: onp.ndarray) -> onp.ndarray:
    """
    P[k, x] = (-1)^{alpha_k · x} in {+1,-1}.
    bits_table: (N,n)
    """
    A = alphas.astype(onp.int16)        # (K,n)
    X = bits_table.astype(onp.int16).T  # (n,N)
    par = (A @ X) & 1                   # (K,N) in {0,1}
    return onp.where(par == 0, 1.0, -1.0).astype(onp.float32)

# ------------------------------------------------------------------------------
# IQP-QCBM circuit family
# ------------------------------------------------------------------------------

def get_iqp_topology(n: int, arch: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
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

def iqp_circuit(W, wires, pairs, quads, layers: int):
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

def build_qnode(n: int, arch: str, layers: int):
    dev = qml.device("default.qubit", wires=n)
    pairs, quads = get_iqp_topology(n, arch)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(n), pairs, quads, layers=layers)
        return qml.probs(wires=range(n))

    num_params = (len(pairs) + len(quads)) * layers
    return circuit, num_params

# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

def kl_divergence(p_star: onp.ndarray, q: onp.ndarray, support: onp.ndarray, eps: float = 1e-12) -> float:
    q_clip = onp.clip(q, eps, 1.0)
    q_clip = q_clip / max(1e-15, float(q_clip.sum()))
    p = p_star[support]
    q_s = q_clip[support]
    return float(onp.sum(p * (onp.log(p) - onp.log(q_s))))

def score_spectrum_masses(dist: onp.ndarray, scores: onp.ndarray, support: onp.ndarray, mask: Optional[onp.ndarray] = None):
    scores_int = scores.astype(int)
    base_mask = support.copy()
    if mask is not None:
        base_mask = base_mask & mask
    valid_scores = scores_int[base_mask]
    unique_s = sorted(onp.unique(valid_scores).tolist())
    masses = []
    for s in unique_s:
        m = (scores_int == s) & base_mask
        masses.append(float(dist[m].sum()))
    return unique_s, masses

def tvd(a: onp.ndarray, b: onp.ndarray) -> float:
    return float(0.5 * onp.sum(onp.abs(onp.array(a) - onp.array(b))))

def conditional_score_tvd(p_star: onp.ndarray, q: onp.ndarray,
                          scores: onp.ndarray, support: onp.ndarray,
                          mask: onp.ndarray) -> float:
    unique_s, p_mass = score_spectrum_masses(p_star, scores, support, mask=mask)
    _, q_mass = score_spectrum_masses(q, scores, support, mask=mask)
    p_tot = sum(p_mass)
    q_tot = sum(q_mass)
    if p_tot <= 0 or q_tot <= 0:
        return float("inf")
    p_cond = onp.array(p_mass, dtype=onp.float64) / p_tot
    q_cond = onp.array(q_mass, dtype=onp.float64) / q_tot
    return tvd(p_cond, q_cond)

def expected_unique_set(probs: onp.ndarray, mask: onp.ndarray, Q: int) -> float:
    """Expected number of unique states from mask seen at least once in Q i.i.d. samples."""
    pS = probs[mask].astype(onp.float64)
    if pS.size == 0:
        return 0.0
    return float(onp.sum(1.0 - onp.power(1.0 - pS, Q)))

def q_score_given_mask(dist: onp.ndarray, scores: onp.ndarray, support: onp.ndarray, mask: onp.ndarray, score_value: int) -> float:
    scores_int = scores.astype(int)
    m = support & mask & (scores_int == int(score_value))
    return float(dist[m].sum())

# ------------------------------------------------------------------------------
# Pareto computation
# ------------------------------------------------------------------------------

def pareto_mask(points: List[Dict]) -> List[bool]:
    """
    Multi-objective Pareto set for:
      minimize KL
      maximize U
      maximize spread
      minimize unseenTVD

    A dominates B if it is >= as good in all objectives and strictly better in at least one.
    """
    mask = []
    for i, A in enumerate(points):
        dominated = False
        for j, B in enumerate(points):
            if i == j:
                continue
            # B dominates A?
            if (B["KL"] <= A["KL"] and
                B["U"] >= A["U"] and
                B["spread"] >= A["spread"] and
                B["unseenTVD"] <= A["unseenTVD"] and
                (B["KL"] < A["KL"] or B["U"] > A["U"] or B["spread"] > A["spread"] or B["unseenTVD"] < A["unseenTVD"])):
                dominated = True
                break
        mask.append(not dominated)
    return mask

# ------------------------------------------------------------------------------
# Plotting (clean: label only Pareto points; two legends; no title)
# ------------------------------------------------------------------------------

_SIGMA_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
_K_SIZE = {128: 38, 256: 55, 512: 75, 1024: 100, 2048: 130}

def _pad_limits(v: onp.ndarray, frac: float = 0.08) -> Tuple[float, float]:
    vmin = float(onp.min(v))
    vmax = float(onp.max(v))
    if vmax == vmin:
        return vmin - 1.0, vmax + 1.0
    r = vmax - vmin
    return vmin - frac * r, vmax + frac * r

def _annotate(ax, x, y, text, idx):
    # Deterministic offsets to avoid overlap (works well for <= ~8 labels)
    offsets = [(10, 10), (10, -14), (-12, 12), (-12, -14), (16, 0), (-18, 0), (0, 16), (0, -18)]
    dx, dy = offsets[idx % len(offsets)]
    ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left" if dx >= 0 else "right",
        va="bottom" if dy >= 0 else "top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=50,
        annotation_clip=False,
    )

def plot_fit_vs_yield(points: List[Dict], pareto: List[bool], outpath: str, fig_target: str) -> None:
    fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)

    # encode sigma -> marker, K -> size
    sigmas_sorted = sorted({p["sigma"] for p in points})
    Ks_sorted = sorted({p["K"] for p in points})
    sigma_to_marker = {s: _SIGMA_MARKERS[i % len(_SIGMA_MARKERS)] for i, s in enumerate(sigmas_sorted)}

    # base layer: all points (light)
    for p in points:
        ax.scatter(
            p["KL"], p["U"],
            s=_K_SIZE.get(int(p["K"]), 60),
            marker=sigma_to_marker[p["sigma"]],
            facecolor=COLORS["light"],
            edgecolor=COLORS["gray"],
            linewidth=0.7,
            alpha=0.85,
            zorder=10,
        )

    # pareto points highlighted
    pareto_points = [p for p, m in zip(points, pareto) if m]
    pareto_points_sorted = sorted(pareto_points, key=lambda d: d["KL"])
    for k, p in enumerate(pareto_points_sorted):
        ax.scatter(
            p["KL"], p["U"],
            s=_K_SIZE.get(int(p["K"]), 60),
            marker=sigma_to_marker[p["sigma"]],
            facecolor=COLORS["model"],
            edgecolor=COLORS["dark"],
            linewidth=1.0,
            alpha=0.95,
            zorder=30,
        )
        _annotate(ax, p["KL"], p["U"], text=rf"$\sigma={p['sigma']},\ K={p['K']}$", idx=k)

    # connect Pareto front (visual guide)
    if len(pareto_points_sorted) >= 2:
        ax.plot(
            [p["KL"] for p in pareto_points_sorted],
            [p["U"]  for p in pareto_points_sorted],
            color=COLORS["dark"],
            linewidth=1.2,
            alpha=0.6,
            zorder=20,
        )

    ax.set_xlabel(r"$D_{\mathrm{KL}}(p^*\Vert q)$  (lower is better)")
    ax.set_ylabel(r"$U_{\mathrm{unseen}}(Q_0)$  (higher is better)")
    ax.grid(True, which="both", linestyle="--", alpha=0.14)

    xs = onp.array([p["KL"] for p in points], dtype=float)
    ys = onp.array([p["U"] for p in points], dtype=float)
    ax.set_xlim(*_pad_limits(xs))
    ax.set_ylim(*_pad_limits(ys))

    # Two separate legends: sigma (markers), K (sizes)
    from matplotlib.lines import Line2D

    sigma_handles = [
        Line2D([0], [0], marker=sigma_to_marker[s], linestyle="None",
               markerfacecolor=COLORS["light"], markeredgecolor=COLORS["gray"],
               markersize=7, label=rf"$\sigma={s}$")
        for s in sigmas_sorted
    ]
    k_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=COLORS["light"], markeredgecolor=COLORS["gray"],
               markersize=math.sqrt(_K_SIZE.get(int(K), 60))/1.5,
               label=rf"$K={K}$")
        for K in Ks_sorted
    ]

    leg1 = ax.legend(handles=sigma_handles, loc="lower center", bbox_to_anchor=(0.30, 1.02),
                     ncol=len(sigma_handles), columnspacing=0.9, handletextpad=0.3, borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=k_handles, loc="lower center", bbox_to_anchor=(0.72, 1.02),
              ncol=len(k_handles), columnspacing=0.9, handletextpad=0.3, borderaxespad=0.0)

    fig.savefig(outpath)
    plt.close(fig)

def plot_dispersion(points: List[Dict], pareto: List[bool], outpath: str, fig_target: str) -> None:
    fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)

    sigmas_sorted = sorted({p["sigma"] for p in points})
    Ks_sorted = sorted({p["K"] for p in points})
    sigma_to_marker = {s: _SIGMA_MARKERS[i % len(_SIGMA_MARKERS)] for i, s in enumerate(sigmas_sorted)}

    # all points
    for p in points:
        ax.scatter(
            p["unseenTVD"], p["spread"],
            s=_K_SIZE.get(int(p["K"]), 60),
            marker=sigma_to_marker[p["sigma"]],
            facecolor=COLORS["light"],
            edgecolor=COLORS["gray"],
            linewidth=0.7,
            alpha=0.85,
            zorder=10,
        )

    # pareto highlighted + labels
    pareto_points = [p for p, m in zip(points, pareto) if m]
    pareto_points_sorted = sorted(pareto_points, key=lambda d: d["unseenTVD"])
    for k, p in enumerate(pareto_points_sorted):
        ax.scatter(
            p["unseenTVD"], p["spread"],
            s=_K_SIZE.get(int(p["K"]), 60),
            marker=sigma_to_marker[p["sigma"]],
            facecolor=COLORS["model"],
            edgecolor=COLORS["dark"],
            linewidth=1.0,
            alpha=0.95,
            zorder=30,
        )
        _annotate(ax, p["unseenTVD"], p["spread"], text=rf"$\sigma={p['sigma']},\ K={p['K']}$", idx=k)

    ax.set_xlabel(r"unseen score TVD  (lower is better)")
    ax.set_ylabel(r"spread $= U_{\mathrm{unseen}}(Q_0)/(Q_0\,q(\mathrm{unseen}))$  (higher is better)")
    ax.grid(True, which="both", linestyle="--", alpha=0.14)

    xs = onp.array([p["unseenTVD"] for p in points], dtype=float)
    ys = onp.array([p["spread"] for p in points], dtype=float)
    ax.set_xlim(*_pad_limits(xs))
    ax.set_ylim(*_pad_limits(ys))

    # Legends (same split)
    from matplotlib.lines import Line2D
    sigma_handles = [
        Line2D([0], [0], marker=sigma_to_marker[s], linestyle="None",
               markerfacecolor=COLORS["light"], markeredgecolor=COLORS["gray"],
               markersize=7, label=rf"$\sigma={s}$")
        for s in sigmas_sorted
    ]
    k_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor=COLORS["light"], markeredgecolor=COLORS["gray"],
               markersize=math.sqrt(_K_SIZE.get(int(K), 60))/1.5,
               label=rf"$K={K}$")
        for K in Ks_sorted
    ]
    leg1 = ax.legend(handles=sigma_handles, loc="lower center", bbox_to_anchor=(0.30, 1.02),
                     ncol=len(sigma_handles), columnspacing=0.9, handletextpad=0.3, borderaxespad=0.0)
    ax.add_artist(leg1)
    ax.legend(handles=k_handles, loc="lower center", bbox_to_anchor=(0.72, 1.02),
              ncol=len(k_handles), columnspacing=0.9, handletextpad=0.3, borderaxespad=0.0)

    fig.savefig(outpath)
    plt.close(fig)

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

@dataclass
class Config:
    n: int
    arch: str
    layers: int
    beta: float
    steps: int
    lr: float
    Q0: int
    train_m_list: Tuple[int, ...]
    seeds: Tuple[int, ...]
    sigma_list: Tuple[float, ...]
    K_list: Tuple[int, ...]
    nested_data: bool
    outdir: str
    fig_target: str

def train_one_run(
    cfg: Config,
    circuit,
    num_params: int,
    P_mat_tensor,
    p_star: onp.ndarray,
    support: onp.ndarray,
    scores: onp.ndarray,
    idxs_train: onp.ndarray,
    init_seed: int,
) -> Dict:
    N = 2 ** cfg.n

    emp_dist = empirical_dist(idxs_train, N)
    z_data = onp.dot(onp.array(P_mat_tensor, dtype=onp.float32), emp_dist.astype(onp.float32)).astype(onp.float32)
    z_data_tensor = np.array(z_data, requires_grad=False)

    rng = onp.random.default_rng(init_seed)
    W = np.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(cfg.lr)

    # Train: parity-MMD moment matching (no extra regularizers)
    for _ in range(cfg.steps):
        def loss_fn(w):
            q = circuit(w)
            return np.mean((z_data_tensor - P_mat_tensor @ q) ** 2)
        W, _ = opt.step_and_cost(loss_fn, W)

    q_final = onp.clip(onp.array(circuit(W), dtype=onp.float64), 0.0, 1.0)
    q_final /= max(1e-15, float(q_final.sum()))

    seen_mask = seen_mask_from_indices(idxs_train, N)
    unseen_mask = support & (~seen_mask)

    kl = kl_divergence(p_star, q_final, support)
    p_unseen_mass = float(p_star[unseen_mask].sum())
    q_unseen_mass = float(q_final[unseen_mask].sum())

    U_unseen_q = expected_unique_set(q_final, unseen_mask, cfg.Q0)
    U_unseen_p = expected_unique_set(p_star, unseen_mask, cfg.Q0)

    spread = (U_unseen_q / (cfg.Q0 * q_unseen_mass)) if q_unseen_mass > 0 else 0.0

    unseen_score_tvd = conditional_score_tvd(p_star, q_final, scores, support, unseen_mask)

    # score-1 spike diagnostic on unseen
    q_s1_unseen = (q_score_given_mask(q_final, scores, support, unseen_mask, 1) / q_unseen_mass) if q_unseen_mass > 0 else 0.0
    p_s1_unseen = (q_score_given_mask(p_star,  scores, support, unseen_mask, 1) / p_unseen_mass) if p_unseen_mass > 0 else 0.0

    return dict(
        q=q_final,
        kl=kl,
        p_unseen_mass=p_unseen_mass,
        q_unseen_mass=q_unseen_mass,
        U_unseen_q=U_unseen_q,
        U_unseen_p=U_unseen_p,
        spread=spread,
        unseen_score_TVD=unseen_score_tvd,
        q_s1_unseen=q_s1_unseen,
        p_s1_unseen=p_s1_unseen,
    )

# ------------------------------------------------------------------------------
# CSV utilities
# ------------------------------------------------------------------------------

def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = onp.array(vals, dtype=onp.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--arch", type=str, default="D", choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.9)

    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)

    ap.add_argument("--train-m-list", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)

    ap.add_argument("--sigma-list", type=float, nargs="+", required=True)
    ap.add_argument("--K-list", type=int, nargs="+", required=True)

    ap.add_argument("--Q0", type=int, default=5000)

    ap.add_argument("--nested-data", action="store_true", help="Use prefix of one long sample per seed")
    ap.add_argument("--independent-data", dest="nested_data", action="store_false")
    ap.set_defaults(nested_data=True)

    ap.add_argument("--outdir", type=str, default="jonas_pareto_out")
    ap.add_argument("--fig-target", type=str, default="full", choices=["col", "full"])

    args = ap.parse_args()

    cfg = Config(
        n=args.n, arch=args.arch, layers=args.layers, beta=args.beta,
        steps=args.steps, lr=args.lr,
        Q0=args.Q0,
        train_m_list=tuple(int(x) for x in args.train_m_list),
        seeds=tuple(int(s) for s in args.seeds),
        sigma_list=tuple(float(s) for s in args.sigma_list),
        K_list=tuple(int(k) for k in args.K_list),
        nested_data=bool(args.nested_data),
        outdir=args.outdir,
        fig_target=args.fig_target,
    )

    set_style(cfg.fig_target)
    outdir = ensure_outdir(cfg.outdir)

    # Save config for reproducibility
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Build target and circuit once
    p_star, support, scores = build_target_distribution(cfg.n, cfg.beta)
    bits_table = make_bits_table(cfg.n)
    N = 2 ** cfg.n

    circuit, num_params = build_qnode(cfg.n, cfg.arch, cfg.layers)

    print(f"[Setup] n={cfg.n} (N={N}) | arch={cfg.arch} | L={cfg.layers} | beta={cfg.beta}")
    print(f"[Sweep] m={list(cfg.train_m_list)} | sigmas={list(cfg.sigma_list)} | Ks={list(cfg.K_list)} | seeds={list(cfg.seeds)}")
    print(f"[Eval ] Q0={cfg.Q0} | steps={cfg.steps} | lr={cfg.lr} | nested_data={cfg.nested_data}")
    print("")

    rows_raw: List[Dict] = []

    # For nested-data: sample once per seed with m_max, then reuse prefixes for smaller m
    m_max = max(cfg.train_m_list)

    for seed in cfg.seeds:
        # dataset sampling per seed
        if cfg.nested_data:
            idxs_all = sample_indices(p_star, m_max, seed=seed + 7)
            train_sets = {m: idxs_all[:m] for m in cfg.train_m_list}
        else:
            train_sets = {m: sample_indices(p_star, m, seed=seed + 7 + 10000 * m) for m in cfg.train_m_list}

        for m in cfg.train_m_list:
            idxs_train = train_sets[int(m)]

            for sigma in cfg.sigma_list:
                for K in cfg.K_list:
                    # Fix feature masks per (seed, sigma, K) so repeats are reproducible
                    feature_seed = int(seed + 222 + 1000 * int(round(100 * sigma)) + 10 * int(K))
                    alphas = sample_alphas(cfg.n, sigma, int(K), seed=feature_seed)
                    P_mat = build_parity_matrix(alphas, bits_table)
                    P_mat_tensor = np.array(P_mat, requires_grad=False)

                    run = train_one_run(
                        cfg=cfg,
                        circuit=circuit,
                        num_params=num_params,
                        P_mat_tensor=P_mat_tensor,
                        p_star=p_star,
                        support=support,
                        scores=scores,
                        idxs_train=idxs_train,
                        init_seed=seed,
                    )

                    rows_raw.append({
                        "seed": int(seed),
                        "m": int(m),
                        "sigma": float(sigma),
                        "K": int(K),
                        "KL": float(run["kl"]),
                        "p_unseen_mass": float(run["p_unseen_mass"]),
                        "q_unseen_mass": float(run["q_unseen_mass"]),
                        f"U_unseen_q_Q{cfg.Q0}": float(run["U_unseen_q"]),
                        f"U_unseen_p_Q{cfg.Q0}": float(run["U_unseen_p"]),
                        "spread": float(run["spread"]),
                        "unseen_score_TVD": float(run["unseen_score_TVD"]),
                        "q_score1_given_unseen": float(run["q_s1_unseen"]),
                        "p_score1_given_unseen": float(run["p_s1_unseen"]),
                    })

                    print(
                        f"[Done] seed={seed} m={m:>5d} sigma={sigma:>4.2f} K={K:>4d} | "
                        f"KL={run['kl']:.3f} | U_unseen={run['U_unseen_q']:.1f} | spread={run['spread']:.3f} | "
                        f"unseenTVD={run['unseen_score_TVD']:.3f} | q(s=1|unseen)={run['q_s1_unseen']:.3f}"
                    )

    # Save raw
    raw_path = os.path.join(outdir, "jonas_pareto_raw.csv")
    save_csv(rows_raw, raw_path)
    print(f"\n[Saved] {raw_path}")

    # Aggregate over seeds per (m, sigma, K)
    summary_rows: List[Dict] = []
    for m in sorted(set(int(r["m"]) for r in rows_raw)):
        for sigma in sorted(set(float(r["sigma"]) for r in rows_raw if int(r["m"]) == m)):
            for K in sorted(set(int(r["K"]) for r in rows_raw if int(r["m"]) == m and float(r["sigma"]) == sigma)):
                subset = [r for r in rows_raw if int(r["m"]) == m and float(r["sigma"]) == sigma and int(r["K"]) == K]
                def agg(key):
                    mu, sd = mean_std([float(x[key]) for x in subset])
                    return mu, sd

                KL_mu, KL_sd = agg("KL")
                U_mu, U_sd = agg(f"U_unseen_q_Q{cfg.Q0}")
                spread_mu, spread_sd = agg("spread")
                unseen_mu, unseen_sd = agg("unseen_score_TVD")
                qs1_mu, qs1_sd = agg("q_score1_given_unseen")

                summary_rows.append({
                    "m": int(m),
                    "sigma": float(sigma),
                    "K": int(K),
                    "KL_mean": KL_mu, "KL_std": KL_sd,
                    f"U_unseen_mean_Q{cfg.Q0}": U_mu, f"U_unseen_std_Q{cfg.Q0}": U_sd,
                    "spread_mean": spread_mu, "spread_std": spread_sd,
                    "unseen_score_TVD_mean": unseen_mu, "unseen_score_TVD_std": unseen_sd,
                    "q_score1_given_unseen_mean": qs1_mu, "q_score1_given_unseen_std": qs1_sd,
                })

    summary_path = os.path.join(outdir, "jonas_pareto_summary.csv")
    save_csv(summary_rows, summary_path)
    print(f"[Saved] {summary_path}")

    # For each m: build Pareto set + plots + text summary
    for m in sorted(set(int(r["m"]) for r in summary_rows)):
        pts = [r for r in summary_rows if int(r["m"]) == m]
        points = []
        for r in pts:
            points.append({
                "m": m,
                "sigma": float(r["sigma"]),
                "K": int(r["K"]),
                "KL": float(r["KL_mean"]),
                "U": float(r[f"U_unseen_mean_Q{cfg.Q0}"]),
                "spread": float(r["spread_mean"]),
                "unseenTVD": float(r["unseen_score_TVD_mean"]),
                "qs1": float(r["q_score1_given_unseen_mean"]),
            })

        pareto = pareto_mask(points)

        # write readable points
        txt_path = os.path.join(outdir, f"pareto_points_m{m}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Pareto navigation summary (m={m}, Q0={cfg.Q0})\n")
            f.write("Objectives: minimize KL and unseenTVD; maximize U and spread.\n\n")
            f.write("All configs (mean over seeds):\n")
            for p, is_p in sorted(zip(points, pareto), key=lambda t: (t[0]["KL"], -t[0]["U"])):
                tag = "PARETO" if is_p else "     -"
                f.write(
                    f"{tag}  sigma={p['sigma']:<4}  K={p['K']:<4}  "
                    f"KL={p['KL']:.4f}  U={p['U']:.1f}  spread={p['spread']:.3f}  unseenTVD={p['unseenTVD']:.4f}  "
                    f"q(s=1|unseen)={p['qs1']:.3f}\n"
                )
            f.write("\nPareto-optimal configs:\n")
            for p, is_p in zip(points, pareto):
                if is_p:
                    f.write(
                        f"  sigma={p['sigma']}, K={p['K']}: "
                        f"KL={p['KL']:.4f}, U={p['U']:.1f}, spread={p['spread']:.3f}, unseenTVD={p['unseenTVD']:.4f}, "
                        f"q(s=1|unseen)={p['qs1']:.3f}\n"
                    )
        print(f"[Saved] {txt_path}")

        # plots
        pdf1 = os.path.join(outdir, f"pareto_fit_yield_m{m}.pdf")
        pdf2 = os.path.join(outdir, f"pareto_dispersion_m{m}.pdf")
        plot_fit_vs_yield(points, pareto, pdf1, cfg.fig_target)
        plot_dispersion(points, pareto, pdf2, cfg.fig_target)
        print(f"[Saved] {pdf1}")
        print(f"[Saved] {pdf2}")

        # print Pareto list to terminal
        print("\n[Pareto] Non-dominated configs (mean over seeds):")
        for p, is_p in zip(points, pareto):
            if is_p:
                print(
                    f"  sigma={p['sigma']}, K={p['K']}: "
                    f"KL={p['KL']:.4f}, U={p['U']:.1f}, spread={p['spread']:.3f}, unseenTVD={p['unseenTVD']:.4f}, "
                    f"q(s=1|unseen)={p['qs1']:.3f}"
                )
        print("")

    print("Done.")

if __name__ == "__main__":
    main()
