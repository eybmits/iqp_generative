#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQP-QCBM: Jonas Exploration + Sample-Complexity Study
====================================================

This script answers (empirically, under the same IQP-QCBM + parity-MMD pipeline):

  (1) Wie viele (random) Trainingspunkte m braucht man, um die Zielverteilung p*(x) gut zu lernen?
  (2) Kann das Modell *neue* Samples erzeugen, die im Training nie vorkamen, aber statistisch zur gleichen
      Verteilung passen – und zwar über das *gesamte* Score-Spektrum (nicht nur High-Score)?

Core idea (matching Jonas' interpretation):
- We do *no explicit high-score holdout*.
- For each m we draw m i.i.d. training samples from p*(x).
- Many states are therefore "unseen" simply because m is finite.
- After training we measure:
    • Full-distribution fit: KL(p* || q)
    • Score-spectrum agreement: TVD over score-binned probability mass
    • Novelty mass: q(Unseen) where Unseen = {x in support : x never appeared in training}
      and compare to the *true* novelty mass p*(Unseen).
    • Unseen score-spectrum agreement: TVD between p*(score | unseen) and q(score | unseen)
      (optional: also conditional KL on unseen states).
    • "How many new states do we expect to discover in Q samples?":
        Expected unique unseen states U_unseen(Q) = Σ_{x in Unseen} [1 - (1-q(x))^Q]
      compared against the ideal reference under p*.

Outputs (outdir/):
  - jonas_sample_complexity_summary.csv
  - 1_fit_KL_vs_m.pdf
  - 2_score_TVD_vs_m.pdf
  - 3_novelty_mass_vs_m.pdf
  - 4_unseen_score_TVD_vs_m.pdf
  - 5_unique_unseen_vs_m.pdf
  - 6_reference_score_spectrum_with_train.pdf
  - 7_reference_unseen_score_spectrum.pdf
  - 8_reference_unseen_unique_vs_Q.pdf
  - config.json

Typical usage:
  python jonas_full_exploration_fixed.py \
      --n 14 --arch D --layers 1 --beta 0.9 \
      --train-m-list 50 100 200 500 1000 2000 \
      --seeds 42 43 44 \
      --steps 600 --num-alpha 512 --sigma 1.0 \
      --Q0 5000 --Qmax 10000 \
      --outdir jonas_exploration --fig-target col

Dependencies:
  pip install pennylane numpy matplotlib

Plotting fixes in this version (paper-ready):
  - Value annotations are offset from the last datapoint (no touching markers/lines).
  - Long y-axis labels are automatically wrapped to avoid cropping at fixed figure height.
  - Legends in reference plots (6/7/8) are moved above the axes (not covering data).
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
# 0) Style (same spirit as your hero script: red/black, journal-aware, fixed height)
# ------------------------------------------------------------------------------

COLORS = {
    "target": "#222222",   # almost black
    "model":  "#D62728",   # deep red
    "gray":   "#666666",
    "light":  "#DDDDDD",
}

# Practical widths for APS/RevTeX, in inches
COL_W  = 3.37  # single-column
FULL_W = 6.95  # two-column (figure*)

# FORCE UNIFORM HEIGHTS (match hero style)
FIXED_H_COL  = 2.8
FIXED_H_FULL = 3.5

def fig_size(fig_target: str, h_col: float = FIXED_H_COL, h_full: float = FIXED_H_FULL) -> Tuple[float, float]:
    if fig_target not in ("col", "full"):
        raise ValueError("fig_target must be 'col' or 'full'")
    return (COL_W, h_col) if fig_target == "col" else (FULL_W, h_full)

def set_style(fig_target: str = "col") -> None:
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
            "lines.linewidth": 1.7,
            "lines.markersize": 4.0,

            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,

            # Keep tight bounding box, but avoid ultra-tiny padding
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
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
            "lines.linewidth": 2.6,
            "figure.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
# 1) Benchmark: target distribution + score
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

def build_target_distribution(n: int, beta: float) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """
    p*(x) ∝ exp(beta * score(x)) on even-parity sector, else 0.
    Returns:
      p_star: (N,)
      support: bool mask (N,)
      scores: (N,)
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

def topk_mask(scores: onp.ndarray, support: onp.ndarray, frac: float = 0.05) -> onp.ndarray:
    """Top-frac mask within support by score (used only for an auxiliary diagnostic)."""
    valid = onp.where(support)[0]
    k = max(1, int(onp.floor(frac * valid.size)))
    valid_scores = scores[valid]
    local_order = onp.argsort(-valid_scores)
    top_indices = valid[local_order[:k]]
    mask = onp.zeros_like(support, dtype=bool)
    mask[top_indices] = True
    return mask


# ------------------------------------------------------------------------------
# 2) Sampling utilities
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
# 3) Parity-feature MMD pieces
# ------------------------------------------------------------------------------

def p_sigma(sigma: float) -> float:
    return 0.5 * (1.0 - math.exp(-1.0/(2.0*sigma**2))) if sigma > 0 else 0.5

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
    return onp.where(par == 0, 1.0, -1.0).astype(onp.float64)


# ------------------------------------------------------------------------------
# 4) IQP-QCBM circuit family
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


# ------------------------------------------------------------------------------
# 5) Metrics (fit + "novelty beyond training")
# ------------------------------------------------------------------------------

def kl_divergence(p_star: onp.ndarray, q: onp.ndarray, support: onp.ndarray, eps: float = 1e-12) -> float:
    q_clip = onp.clip(q, eps, 1.0)
    q_clip = q_clip / max(1e-15, float(q_clip.sum()))
    p = p_star[support]
    q_s = q_clip[support]
    return float(onp.sum(p * (onp.log(p) - onp.log(q_s))))

def score_spectrum_masses(dist: onp.ndarray, scores: onp.ndarray, support: onp.ndarray, mask: Optional[onp.ndarray] = None):
    """
    Returns:
      unique_scores (sorted list of ints)
      masses_per_score (float list, same length)
    If mask is provided, it restricts to support & mask.
    """
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
    """
    TVD between conditional score distributions among states in (support & mask).
    """
    unique_s, p_mass = score_spectrum_masses(p_star, scores, support, mask=mask)
    _, q_mass = score_spectrum_masses(q, scores, support, mask=mask)
    p_tot = sum(p_mass)
    q_tot = sum(q_mass)
    if p_tot <= 0 or q_tot <= 0:
        return float("inf")
    p_cond = onp.array(p_mass, dtype=onp.float64) / p_tot
    q_cond = onp.array(q_mass, dtype=onp.float64) / q_tot
    return tvd(p_cond, q_cond)

def conditional_kl_on_mask(p_star: onp.ndarray, q: onp.ndarray,
                           support: onp.ndarray, mask: onp.ndarray,
                           eps: float = 1e-12) -> float:
    """
    KL between p*(x | mask) and q(x | mask), restricted to support & mask.
    """
    m = support & mask
    p = p_star[m]
    qv = q[m]
    p_sum = float(p.sum())
    q_sum = float(qv.sum())
    if p_sum <= 0 or q_sum <= 0:
        return float("inf")
    p = p / p_sum
    qv = onp.clip(qv / q_sum, eps, 1.0)
    return float(onp.sum(p * (onp.log(p) - onp.log(qv))))

def expected_unique_set(probs: onp.ndarray, mask: onp.ndarray, Q_vals: onp.ndarray) -> onp.ndarray:
    pS = probs[mask].astype(onp.float64)[:, None]
    if pS.size == 0:
        return onp.zeros_like(Q_vals, dtype=onp.float64)
    return onp.sum(1.0 - onp.power(1.0 - pS, Q_vals[None, :]), axis=0)


# ------------------------------------------------------------------------------
# 6) Training
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

    # evaluation / sweep
    train_m_list: Tuple[int, ...] = (50, 100, 200, 500, 1000)
    seeds: Tuple[int, ...] = (42, 43, 44)

    # novelty / unique
    Q0: int = 5000
    Qmax: int = 10000

    # misc
    nested_data: bool = True  # if True: for each seed, use prefix of a single long sample
    outdir: str = "jonas_exploration"
    fig_target: str = "col"
    ref_m: Optional[int] = None
    ref_seed: Optional[int] = None

def build_qnode(cfg: Config):
    dev = qml.device("default.qubit", wires=cfg.n)
    pairs, quads = get_iqp_topology(cfg.n, cfg.arch)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(cfg.n), pairs, quads, layers=cfg.layers)
        return qml.probs(wires=range(cfg.n))

    num_params = (len(pairs) + len(quads)) * cfg.layers
    return circuit, num_params

def train_one_run(
    cfg: Config,
    circuit,
    num_params: int,
    P_mat_tensor,
    bits_table: onp.ndarray,
    p_star: onp.ndarray,
    support: onp.ndarray,
    scores: onp.ndarray,
    idxs_train: onp.ndarray,
    init_seed: int,
) -> Dict:
    """
    Train on the empirical parity moments from idxs_train (no explicit holdout).
    Returns a dict with q and various metrics.
    """
    N = 2 ** cfg.n

    emp_dist = empirical_dist(idxs_train, N)
    z_data = onp.dot(onp.array(P_mat_tensor, dtype=onp.float64), emp_dist).astype(onp.float64)
    z_data_tensor = np.array(z_data, requires_grad=False)

    # init
    rng = onp.random.default_rng(init_seed)
    W = np.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(cfg.lr)

    # train
    for _ in range(cfg.steps):
        def loss_fn(w):
            q = circuit(w)
            return np.mean((z_data_tensor - P_mat_tensor @ q) ** 2)
        W, _ = opt.step_and_cost(loss_fn, W)

    q_final = onp.clip(onp.array(circuit(W), dtype=onp.float64), 0.0, 1.0)
    q_final /= max(1e-15, float(q_final.sum()))

    # training-seen vs unseen masks
    seen_mask = seen_mask_from_indices(idxs_train, N)
    unseen_mask = support & (~seen_mask)

    # fit metrics
    kl = kl_divergence(p_star, q_final, support)

    # score-spectrum TVD (overall)
    _, p_mass = score_spectrum_masses(p_star, scores, support)
    _, q_mass = score_spectrum_masses(q_final, scores, support)
    spectrum_tvd = tvd(onp.array(p_mass), onp.array(q_mass))

    # novelty mass: probability to sample a state that never appeared in training
    p_unseen_mass = float(p_star[unseen_mask].sum())
    q_unseen_mass = float(q_final[unseen_mask].sum())

    # how much of the target mass was "covered" by the unique training set?
    p_seen_mass = float(p_star[support & seen_mask].sum())
    q_seen_mass = float(q_final[support & seen_mask].sum())
    train_unique = int(onp.sum(support & seen_mask))
    support_size = int(onp.sum(support))

    # unseen score distribution agreement
    unseen_score_tvd = conditional_score_tvd(p_star, q_final, scores, support, unseen_mask)
    unseen_kl = conditional_kl_on_mask(p_star, q_final, support, unseen_mask)

    # expected #unique unseen states at budget Q0
    Q0 = int(cfg.Q0)
    U_unseen_q = float(expected_unique_set(q_final, unseen_mask, onp.array([Q0]))[0])
    U_unseen_p = float(expected_unique_set(p_star, unseen_mask, onp.array([Q0]))[0])

    # auxiliary (top5 mass) just to ensure we don't only look at it
    good_mask = topk_mask(scores, support, frac=0.05)
    p_top5 = float(p_star[good_mask].sum())
    q_top5 = float(q_final[good_mask].sum())

    return dict(
        q=q_final,
        kl=kl,
        spectrum_tvd=spectrum_tvd,
        train_unique=train_unique,
        support_size=support_size,
        p_seen_mass=p_seen_mass,
        q_seen_mass=q_seen_mass,
        p_unseen_mass=p_unseen_mass,
        q_unseen_mass=q_unseen_mass,
        unseen_score_tvd=unseen_score_tvd,
        unseen_kl=unseen_kl,
        U_unseen_q_Q0=U_unseen_q,
        U_unseen_p_Q0=U_unseen_p,
        p_top5=p_top5,
        q_top5=q_top5,
        idxs_train=idxs_train,         # for reference plots
        seen_mask=seen_mask,
        unseen_mask=unseen_mask,
    )


# ------------------------------------------------------------------------------
# 7) Plotting helpers (layout-safe)
# ------------------------------------------------------------------------------

def _save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _wrap_label(text: str, width: int = 28) -> str:
    """
    Wrap long plain-text axis labels to avoid cropping at fixed figure height.
    We keep mathtext (contains '$') as-is.
    """
    if not isinstance(text, str):
        return text
    if "\n" in text:
        return text
    if "$" in text:
        return text
    t = text.strip()
    if len(t) <= width:
        return t
    # split into two lines at the last space before width
    cut = t.rfind(" ", 0, width + 1)
    if cut <= 0:
        return t
    return t[:cut] + "\n" + t[cut + 1:]

def _set_ylabel(ax, text: str) -> None:
    ax.set_ylabel(_wrap_label(text))

def _xlim_with_margin(ax, x: onp.ndarray, xscale: str = "log") -> None:
    """Add a small right margin so annotations don't hit the frame."""
    if x.size == 0:
        return
    xmin = float(onp.min(x))
    xmax = float(onp.max(x))
    if xscale == "log" and xmin > 0:
        ax.set_xlim(xmin * 0.90, xmax * 1.25)
    else:
        r = xmax - xmin
        ax.set_xlim(xmin - 0.05 * r, xmax + 0.08 * r)

def _annotate_last_value(ax, fig, x_last: float, y_last: float, text: str, color: str) -> None:
    """
    Place a value label near (x_last, y_last) but offset so it does not touch the marker/line.
    Offset direction is chosen based on proximity to the right/top edge.
    """
    # Convert data position -> axes coords (0..1) to decide safe direction
    try:
        xa, ya = ax.transAxes.inverted().transform(ax.transData.transform((x_last, y_last)))
    except Exception:
        xa, ya = 0.9, 0.9

    dx = -28 if xa > 0.80 else 8
    dy = -14 if ya > 0.85 else 6
    ha = "right" if dx < 0 else "left"
    va = "top" if dy < 0 else "bottom"

    ax.annotate(
        text,
        xy=(x_last, y_last),
        xycoords="data",
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=30,
        annotation_clip=False,
    )

def plot_metric_vs_m(
    m_list: List[int],
    mean: List[float],
    std: List[float],
    outpath: str,
    ylabel: str,
    fig_target: str,
    yscale: Optional[str] = None,
    xscale: str = "log",
    annotate_last: bool = True,
):
    fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)

    x = onp.array(m_list, dtype=float)
    y = onp.array(mean, dtype=float)
    e = onp.array(std, dtype=float)

    ax.errorbar(x, y, yerr=e, color=COLORS["model"], marker="o",
                capsize=3.0, lw=2.0, label="Mean ± std")
    ax.plot(x, y, color=COLORS["model"], lw=2.0, alpha=0.95)

    ax.set_xlabel(r"Training samples $m$")
    _set_ylabel(ax, ylabel)

    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    _xlim_with_margin(ax, x, xscale=xscale)

    ax.grid(True, which="both", linestyle="--", alpha=0.12)
    ax.legend(loc="best")

    if annotate_last and len(m_list) > 0:
        _annotate_last_value(
            ax, fig, float(x[-1]), float(y[-1]),
            text=f"{float(y[-1]):.3g}",
            color=COLORS["model"]
        )

    fig.savefig(outpath)
    plt.close(fig)

def plot_two_lines_vs_m(
    m_list: List[int],
    y1_mean: List[float], y1_std: List[float], y1_label: str, y1_color: str,
    y2_mean: List[float], y2_std: List[float], y2_label: str, y2_color: str,
    outpath: str,
    ylabel: str,
    fig_target: str,
    xscale: str = "log",
    ylim: Optional[Tuple[float, float]] = None,
):
    fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)

    x = onp.array(m_list, dtype=float)

    ax.errorbar(x, y1_mean, yerr=y1_std, color=y1_color, marker="o", capsize=3.0, lw=2.0, label=y1_label)
    ax.errorbar(x, y2_mean, yerr=y2_std, color=y2_color, marker="s", capsize=3.0, lw=2.0, label=y2_label)

    ax.set_xlabel(r"Training samples $m$")
    _set_ylabel(ax, ylabel)
    ax.set_xscale(xscale)

    _xlim_with_margin(ax, x, xscale=xscale)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, which="both", linestyle="--", alpha=0.12)
    ax.legend(loc="best")

    fig.savefig(outpath)
    plt.close(fig)

def plot_reference_score_spectrum_with_train(p_star, q, emp_dist, scores, support, outdir, fig_target):
    """
    Score spectrum comparing:
      - Target p*
      - Model q
      - Empirical training distribution (histogram)
    """
    scores_int = scores.astype(int)
    valid_scores = scores_int[support]
    unique_s = sorted(onp.unique(valid_scores))

    def masses(dist):
        y = []
        for s in unique_s:
            m = (scores_int == s) & support
            y.append(float(dist[m].sum()))
        return y

    y_star = masses(p_star)
    y_q = masses(q)
    y_emp = masses(emp_dist)

    tvd_q = tvd(onp.array(y_star), onp.array(y_q))

    x = onp.arange(len(unique_s))
    width = 0.27

    fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)

    ax.bar(x - width, y_star, width, color="white", edgecolor="#333333",
           hatch="///", alpha=0.95, label=r"Target $p^*$", zorder=3)
    ax.bar(x, y_emp, width, color="white", edgecolor="#333333",
           hatch="..", alpha=0.85, label=r"Train empirical $\hat p$", zorder=2)
    ax.bar(x + width, y_q, width, color=COLORS["model"], edgecolor="black",
           alpha=0.90, label=r"Model $q_\theta$", zorder=4)

    ax.text(0.02, 0.95, f"TVD(p*, q) = {tvd_q:.3f}", transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
            zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in unique_s])
    ax.set_xlabel("Score")
    ax.set_ylabel("Probability Mass")
    ax.set_ylim(0, max(max(y_star), max(y_q), max(y_emp)) * 1.30)

    ax.grid(axis="y", linestyle=":", alpha=0.25)

    # Legend ABOVE the axes (no overlap with bars)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.0,
        handlelength=1.4,
        borderaxespad=0.0
    )

    fig.savefig(os.path.join(outdir, "6_reference_score_spectrum_with_train.pdf"))
    plt.close(fig)

def plot_reference_unseen_score_spectrum(p_star, q, scores, support, unseen_mask, outdir, fig_target):
    """
    Bar chart of conditional score distribution among *unseen* states.
    """
    scores_int = scores.astype(int)
    base = support & unseen_mask
    valid_scores = scores_int[base]
    unique_s = sorted(onp.unique(valid_scores)) if valid_scores.size > 0 else []

    if not unique_s:
        print("[RefPlot] No unseen states (mask empty). Skipping unseen score spectrum plot.")
        return

    def masses(dist):
        y = []
        for s in unique_s:
            m = (scores_int == s) & base
            y.append(float(dist[m].sum()))
        return y

    y_star = masses(p_star)
    y_q = masses(q)

    p_tot = sum(y_star)
    q_tot = sum(y_q)
    if p_tot <= 0 or q_tot <= 0:
        print("[RefPlot] Unseen mass is zero in p* or q. Skipping unseen score spectrum plot.")
        return

    y_star_c = [v / p_tot for v in y_star]
    y_q_c = [v / q_tot for v in y_q]
    tvd_cond = tvd(onp.array(y_star_c), onp.array(y_q_c))

    x = onp.arange(len(unique_s))
    width = 0.42

    fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)
    ax.bar(x - width/2, y_star_c, width, color="white", edgecolor="#333333", linewidth=1.0,
           hatch="///", alpha=0.9, label=r"Target $p^*(\mathrm{score}\mid \mathrm{unseen})$", zorder=2)
    ax.bar(x + width/2, y_q_c, width, color=COLORS["model"], edgecolor="black", linewidth=0.8,
           alpha=0.9, label=r"Model $q(\mathrm{score}\mid \mathrm{unseen})$", zorder=3)

    ax.text(0.02, 0.95, f"Cond. TVD = {tvd_cond:.3f}", transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
            zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in unique_s])
    ax.set_xlabel("Score")
    _set_ylabel(ax, "Conditional mass among unseen states")
    ax.set_ylim(0, max(max(y_star_c), max(y_q_c)) * 1.30)

    ax.grid(axis="y", linestyle=":", alpha=0.25)

    # Legend ABOVE the axes (no overlap with bars)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.4,
        borderaxespad=0.0
    )

    fig.savefig(os.path.join(outdir, "7_reference_unseen_score_spectrum.pdf"))
    plt.close(fig)

def plot_reference_unseen_unique_curve(p_star, q, unseen_mask, outdir, fig_target, Qmax: int):
    """
    Curve: expected number of unique unseen states vs Q, for p* and q.
    """
    if int(onp.sum(unseen_mask)) == 0:
        print("[RefPlot] No unseen states. Skipping unseen unique curve.")
        return

    Q = onp.unique(onp.concatenate([
        onp.unique(onp.logspace(0, math.log10(max(2, Qmax)), 160).astype(int)),
        onp.linspace(max(2, Qmax//10), Qmax, 140).astype(int)
    ]))
    Q = Q[(Q >= 1) & (Q <= Qmax)]

    ys = expected_unique_set(p_star, unseen_mask, Q)
    yq = expected_unique_set(q, unseen_mask, Q)
    k = int(onp.sum(unseen_mask))

    fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)
    ax.plot(Q, ys, color=COLORS["target"], linewidth=1.9, label=r"Target $U^*_{\mathrm{unseen}}(Q)$", zorder=6)
    ax.plot(Q, yq, color=COLORS["model"], linewidth=2.2, label=r"Model $U_{\mathrm{unseen}}(Q)$", zorder=7)

    # Max line (not in legend) + annotation near top
    ax.axhline(k, color=COLORS["gray"], linestyle=":", linewidth=1.2, zorder=2)
    ax.text(0.98, 0.95, f"Max ({k})", transform=ax.transAxes,
            ha="right", va="top", color=COLORS["gray"], style="italic",
            bbox=dict(boxstyle="round", pad=0.15, facecolor="white", edgecolor="none", alpha=0.90),
            zorder=10)

    ax.set_xscale("log")
    ax.set_xlabel(r"Sampling budget $Q$ (log)")
    _set_ylabel(ax, "Expected #unique unseen states")

    # small headroom so the max line is not flush with the border
    ax.set_ylim(0.0, max(k * 1.05, float(max(ys.max(), yq.max())) * 1.10))

    ax.grid(True, which="both", linestyle="--", alpha=0.12)

    # Legend ABOVE the axes (avoid covering curves at high Q)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.6,
        borderaxespad=0.0
    )

    fig.savefig(os.path.join(outdir, "8_reference_unseen_unique_vs_Q.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------
# 8) Main: sweep m, aggregate, plot
# ------------------------------------------------------------------------------

def mean_std(x: List[float]) -> Tuple[float, float]:
    arr = onp.array(x, dtype=onp.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=14)
    p.add_argument("--arch", type=str, default="D", choices=["A", "B", "C", "D", "E"])
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.9)

    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--num-alpha", type=int, default=512)

    p.add_argument("--train-m-list", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])

    p.add_argument("--Q0", type=int, default=5000, help="Budget for unique-unseen summary metric")
    p.add_argument("--Qmax", type=int, default=10000, help="Max budget for reference unseen curve")

    p.add_argument("--nested-data", action="store_true", help="Use prefix of one long sample per seed (default)")
    p.add_argument("--independent-data", dest="nested_data", action="store_false",
                   help="Use independent training samples for each m")
    p.set_defaults(nested_data=True)

    p.add_argument("--outdir", type=str, default="jonas_exploration")
    p.add_argument("--fig-target", type=str, default="col", choices=["col", "full"])

    p.add_argument("--ref-m", type=int, default=None, help="Reference m for detailed plots (default: max m)")
    p.add_argument("--ref-seed", type=int, default=None, help="Reference seed for detailed plots (default: first seed)")

    args = p.parse_args()

    cfg = Config(
        n=args.n, arch=args.arch, layers=args.layers, beta=args.beta,
        steps=args.steps, lr=args.lr, sigma=args.sigma, num_alpha=args.num_alpha,
        train_m_list=tuple(args.train_m_list),
        seeds=tuple(args.seeds),
        Q0=args.Q0, Qmax=args.Qmax,
        nested_data=bool(args.nested_data),
        outdir=args.outdir, fig_target=args.fig_target,
        ref_m=args.ref_m, ref_seed=args.ref_seed
    )

    set_style(cfg.fig_target)
    outdir = ensure_outdir(cfg.outdir)

    # Save config
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Build target once
    p_star, support, scores = build_target_distribution(cfg.n, cfg.beta)
    bits_table = make_bits_table(cfg.n)
    N = 2 ** cfg.n

    # Circuit/QNode once (arch fixed here)
    circuit, num_params = build_qnode(cfg)

    # Choose reference settings for detailed plots
    ref_m = cfg.ref_m if cfg.ref_m is not None else max(cfg.train_m_list)
    ref_seed = cfg.ref_seed if cfg.ref_seed is not None else int(cfg.seeds[0])

    print(f"[Setup] n={cfg.n} (N={N}), arch={cfg.arch}, layers={cfg.layers}, beta={cfg.beta}")
    print(f"[Sweep] m_list={list(cfg.train_m_list)} | seeds={list(cfg.seeds)} | nested_data={cfg.nested_data}")
    print(f"[Ref] ref_m={ref_m}, ref_seed={ref_seed}")

    # Collect per-run results (for CSV + aggregation)
    rows = []
    # Store one representative run for reference plots
    ref_payload = None

    m_max = max(cfg.train_m_list)

    for seed in cfg.seeds:
        # Fix feature masks per seed (so m-sweep isolates data size effect)
        feature_seed = seed + 222
        alphas = sample_alphas(cfg.n, cfg.sigma, cfg.num_alpha, seed=feature_seed)
        P_mat = build_parity_matrix(alphas, bits_table)
        P_mat_tensor = np.array(P_mat, requires_grad=False)

        # Data sampling
        if cfg.nested_data:
            data_seed = seed + 7
            idxs_all = sample_indices(p_star, m_max, seed=data_seed)  # sample once
            train_sets = {m: idxs_all[:m] for m in cfg.train_m_list}
        else:
            train_sets = {}
            for m in cfg.train_m_list:
                data_seed = seed + 7 + 10000 * int(m)
                train_sets[m] = sample_indices(p_star, int(m), seed=data_seed)

        # Init seed fixed per seed
        init_seed = seed

        for m in cfg.train_m_list:
            idxs_train = train_sets[int(m)]
            run = train_one_run(
                cfg=cfg,
                circuit=circuit,
                num_params=num_params,
                P_mat_tensor=P_mat_tensor,
                bits_table=bits_table,
                p_star=p_star,
                support=support,
                scores=scores,
                idxs_train=idxs_train,
                init_seed=init_seed,
            )

            # store row (no huge vectors)
            rows.append({
                "seed": int(seed),
                "m": int(m),
                "KL_pstar_q": float(run["kl"]),
                "ScoreTVD_pstar_q": float(run["spectrum_tvd"]),
                "train_unique": int(run["train_unique"]),
                "support_size": int(run["support_size"]),
                "p_seen_mass": float(run["p_seen_mass"]),
                "q_seen_mass": float(run["q_seen_mass"]),
                "p_unseen_mass": float(run["p_unseen_mass"]),
                "q_unseen_mass": float(run["q_unseen_mass"]),
                "unseen_score_TVD": float(run["unseen_score_tvd"]),
                "unseen_cond_KL": float(run["unseen_kl"]),
                f"U_unseen_q_Q{cfg.Q0}": float(run["U_unseen_q_Q0"]),
                f"U_unseen_p_Q{cfg.Q0}": float(run["U_unseen_p_Q0"]),
                "p_top5_mass": float(run["p_top5"]),
                "q_top5_mass": float(run["q_top5"]),
            })

            # keep representative run for detailed plots
            if int(seed) == int(ref_seed) and int(m) == int(ref_m) and ref_payload is None:
                # also compute empirical dist for ref plot
                emp = empirical_dist(idxs_train, N)
                ref_payload = {
                    "q": run["q"],
                    "emp": emp,
                    "idxs_train": idxs_train,
                    "seen_mask": run["seen_mask"],
                    "unseen_mask": run["unseen_mask"],
                }

            print(f"[Done] seed={seed} m={m:>5d} | KL={run['kl']:.3f} | TVD={run['spectrum_tvd']:.3f} | "
                  f"q_unseen={run['q_unseen_mass']:.3f} | unseenTVD={run['unseen_score_tvd']:.3f}")

    # Write CSV summary
    csv_path = os.path.join(outdir, "jonas_sample_complexity_summary.csv")
    _save_csv(rows, csv_path)
    print(f"[Saved] {csv_path}")

    # Aggregate over seeds for each m
    m_list = sorted(set(int(r["m"]) for r in rows))
    def collect(key):
        per_m = []
        per_m_std = []
        for m in m_list:
            vals = [float(r[key]) for r in rows if int(r["m"]) == int(m)]
            mu, sd = mean_std(vals)
            per_m.append(mu)
            per_m_std.append(sd)
        return per_m, per_m_std

    KL_mean, KL_std = collect("KL_pstar_q")
    TVD_mean, TVD_std = collect("ScoreTVD_pstar_q")
    p_unseen_mean, p_unseen_std = collect("p_unseen_mass")
    q_unseen_mean, q_unseen_std = collect("q_unseen_mass")
    unseen_score_tvd_mean, unseen_score_tvd_std = collect("unseen_score_TVD")
    Uq_mean, Uq_std = collect(f"U_unseen_q_Q{cfg.Q0}")
    Up_mean, Up_std = collect(f"U_unseen_p_Q{cfg.Q0}")

    # Plots answering Jonas' questions directly
    plot_metric_vs_m(
        m_list, KL_mean, KL_std,
        outpath=os.path.join(outdir, "1_fit_KL_vs_m.pdf"),
        ylabel=r"$D_{KL}(p^*\Vert q_\theta)$",
        fig_target=cfg.fig_target,
        yscale="log" if max(KL_mean) / max(1e-12, min([v for v in KL_mean if v > 0] + [1e-6])) > 20 else None,
        xscale="log",
        annotate_last=True,
    )

    plot_metric_vs_m(
        m_list, TVD_mean, TVD_std,
        outpath=os.path.join(outdir, "2_score_TVD_vs_m.pdf"),
        ylabel="Score-spectrum TVD",
        fig_target=cfg.fig_target,
        yscale=None,
        xscale="log",
        annotate_last=True,
    )

    # Novelty: mass on unseen states (target vs model)  -> answers Q2 directly
    plot_two_lines_vs_m(
        m_list,
        y1_mean=p_unseen_mean, y1_std=p_unseen_std, y1_label=r"Target $p^*(\mathrm{unseen})$", y1_color=COLORS["target"],
        y2_mean=q_unseen_mean, y2_std=q_unseen_std, y2_label=r"Model $q(\mathrm{unseen})$",  y2_color=COLORS["model"],
        outpath=os.path.join(outdir, "3_novelty_mass_vs_m.pdf"),
        ylabel="Probability mass on states not seen in training",
        fig_target=cfg.fig_target,
        xscale="log",
        ylim=(0.0, 1.05),
    )

    # Do novel samples match *statistics* across the entire score spectrum?
    plot_metric_vs_m(
        m_list, unseen_score_tvd_mean, unseen_score_tvd_std,
        outpath=os.path.join(outdir, "4_unseen_score_TVD_vs_m.pdf"),
        ylabel=r"Cond. score TVD on unseen states",
        fig_target=cfg.fig_target,
        yscale=None,
        xscale="log",
        annotate_last=True,
    )

    # "How many new states do we discover in Q0 samples?"
    plot_two_lines_vs_m(
        m_list,
        y1_mean=Up_mean, y1_std=Up_std, y1_label=rf"Target $U^*_{{\mathrm{{unseen}}}}(Q={cfg.Q0})$", y1_color=COLORS["target"],
        y2_mean=Uq_mean, y2_std=Uq_std, y2_label=rf"Model $U_{{\mathrm{{unseen}}}}(Q={cfg.Q0})$",  y2_color=COLORS["model"],
        outpath=os.path.join(outdir, "5_unique_unseen_vs_m.pdf"),
        ylabel=f"Expected #unique unseen states at Q={cfg.Q0}",
        fig_target=cfg.fig_target,
        xscale="log",
        ylim=None,
    )

    # Reference plots (one representative run) to make the story tangible
    if ref_payload is not None:
        plot_reference_score_spectrum_with_train(
            p_star=p_star, q=ref_payload["q"], emp_dist=ref_payload["emp"],
            scores=scores, support=support,
            outdir=outdir, fig_target=cfg.fig_target
        )
        plot_reference_unseen_score_spectrum(
            p_star=p_star, q=ref_payload["q"],
            scores=scores, support=support, unseen_mask=ref_payload["unseen_mask"],
            outdir=outdir, fig_target=cfg.fig_target
        )
        plot_reference_unseen_unique_curve(
            p_star=p_star, q=ref_payload["q"], unseen_mask=ref_payload["unseen_mask"],
            outdir=outdir, fig_target=cfg.fig_target, Qmax=int(cfg.Qmax)
        )
    else:
        print("[Warn] No reference payload captured (check ref_seed/ref_m).")

    print(f"\nDone. Results in ./{outdir}/")
    print("[Tip] LaTeX: use width=\\columnwidth for --fig-target col, or width=\\textwidth for --fig-target full.")

if __name__ == "__main__":
    main()
