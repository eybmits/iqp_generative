#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hero-D IQP-QCBM: Parity Root-Cause + Fix (Full Professional Analysis)
=====================================================================

This script is designed to be dropped next to your existing paper code and run as a
"professional evaluation" pipeline focused on the depth L=2 pathology you discovered.

What it does (for Hero architecture D by default)
-------------------------------------------------
A) Parity sanity check (NO TRAINING):
   - For the original ansatz (ZZ + ZZZZ only), empirically verifies:
       L odd  -> q(S_even) ≈ 1
       L even -> q(S_even) ≈ 0.5
     for random parameters.
   - Produces: 01_parity_sanity.pdf

B) Baseline trainability curves (ENSEMBLE over init seeds):
   - Trains L=1/2/3 with the paper-style spectral-MMD objective (random parity features).
   - Logs vs steps:
       KL(p*||q), q(G), ||∇L||, q(S_even)
     and plots mean ± std across init seeds.
   - Produces: 02_trainability_baseline_L123.pdf

C) Robustness sweeps for L=2 baseline:
   - LR sweep, K sweep, σ sweep (each with ensemble over init seeds).
   - Produces: 03_lr_sweep_L2.pdf, 04_K_sweep_L2.pdf, 05_sigma_sweep_L2.pdf

D) The fix (minimal intervention):
   - Add single-qubit Z terms per layer ("Z-fields" / singles=True)
   - Force the global parity feature α=1^n into the MMD feature set
     (and optionally upweight it).
   - Trains L=1/2/3 again and plots trainability.
   - Produces: 06_trainability_fixed_L123.pdf, 07_final_summary_bars.pdf

E) Architecture diagrams (baseline vs fixed):
   - Generates a clean ring schematic for Hero D (NN+NNN+4-body) and the fixed version
     (adds local Z-fields).
   - Produces: 00_architecture_D_and_fix.pdf

Outputs
-------
All outputs go into --outdir (default: heroD_parity_analysis).
Additionally saves a JSON bundle with all metrics:
  outdir/summary.json

Dependencies
------------
  pip install pennylane numpy matplotlib

Recommended quick run (small n)
-------------------------------
  python heroD_parity_full_analysis.py --n 8 --steps 400 --train-m 500 --num-alpha 256 \
      --init-seeds 0,1,2,3,4 --outdir heroD_parity_analysis_n8

Paper-like run (heavier)
------------------------
  python heroD_parity_full_analysis.py --n 14 --steps 600 --train-m 1000 --num-alpha 512 \
      --init-seeds 0,1,2,3,4 --outdir heroD_parity_analysis_n14

Notes
-----
- This script intentionally keeps the same clean red/black/white aesthetic as your master script.
- It is written for simulation/statevector (default.qubit) where q(x) is exact.
"""

import os
import json
import math
import argparse
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Any

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pennylane")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Patch
from matplotlib.lines import Line2D

import pennylane as qml
from pennylane import numpy as np


# ------------------------------------------------------------------------------
# 1) Paper Visual Style
# ------------------------------------------------------------------------------

COLORS = {
    "target": "#222222",   # Almost Black
    "model":  "#D62728",   # Deep Red
    "gray":   "#666666",
    "light":  "#DDDDDD",
}

def set_style():
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
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
# 2) Target distribution (same as your paper)
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


# ------------------------------------------------------------------------------
# 3) IQP topologies + circuit
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

def num_params(n, pairs, quads, layers: int, include_singles: bool) -> int:
    per_layer = len(pairs) + len(quads) + (n if include_singles else 0)
    return int(per_layer * layers)

def iqp_circuit(W, wires, pairs, quads, layers=1, include_singles=False):
    """
    IQP ansatz:
      H^n  Π_{l=1..L} ( D_l  H^n )
    If include_singles=True: add per-qubit RZ before ZZ/ZZZZ in each layer.
    """
    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)

    for _ in range(layers):
        if include_singles:
            for i in wires:
                qml.RZ(W[idx], wires=i)
                idx += 1
        for (i, j) in pairs:
            qml.IsingZZ(W[idx], wires=[wires[i], wires[j]])
            idx += 1
        for (a, b, c, d) in quads:
            qml.MultiRZ(W[idx], wires=[wires[a], wires[b], wires[c], wires[d]])
            idx += 1
        for w in wires:
            qml.Hadamard(wires=w)


# ------------------------------------------------------------------------------
# 4) Spectral MMD with random parity features
# ------------------------------------------------------------------------------

def p_sigma(sigma):
    return 0.5 * (1.0 - math.exp(-1.0/(2.0*sigma**2))) if sigma > 0 else 0.5

def sample_alphas(n, sigma, K, seed):
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas, n):
    N = 2**n
    xs = onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)
    return onp.where(((alphas @ xs.T) % 2) == 0, 1.0, -1.0).astype(onp.float64)


# ------------------------------------------------------------------------------
# 5) Metrics
# ------------------------------------------------------------------------------

def kl_divergence(p_star: onp.ndarray, q: onp.ndarray, eps: float = 1e-12) -> float:
    q_safe = onp.clip(q.astype(onp.float64), eps, 1.0)
    mask = (p_star > 0)
    return float(onp.sum(p_star[mask] * (onp.log(p_star[mask]) - onp.log(q_safe[mask]))))


# ------------------------------------------------------------------------------
# 6) Training utilities
# ------------------------------------------------------------------------------

@dataclass
class RunConfig:
    n: int = 8
    arch: str = "D"
    beta: float = 0.9
    steps: int = 400
    lr: float = 0.05
    sigma: float = 1.0
    num_alpha: int = 256
    train_m: int = 500
    eval_every: int = 50

    data_seed: int = 7
    alpha_seed: int = 42

    init_scale: float = 0.01
    kl_eps: float = 1e-12

    # Fix controls
    include_singles: bool = False
    force_parity_feature: bool = False
    parity_weight: float = 20.0  # used only when force_parity_feature=True

    outdir: str = "heroD_parity_analysis"
    verbose: bool = False

def build_loss_objects(cfg: RunConfig,
                       p_star: onp.ndarray,
                       support: onp.ndarray,
                       good_mask: onp.ndarray,
                       use_oracle_moments: bool = False) -> Dict[str, Any]:
    """
    Build:
      - empirical distribution or oracle distribution
      - random feature matrix P_mat
      - target moment vector z_data
      - optional per-feature weights
    """
    N = 2**cfg.n

    if use_oracle_moments:
        train_dist = p_star.copy()
    else:
        idxs_train = sample_indices(p_star, cfg.train_m, seed=cfg.data_seed)
        train_dist = empirical_dist(idxs_train, N)

    alphas = sample_alphas(cfg.n, cfg.sigma, cfg.num_alpha, seed=cfg.alpha_seed)
    if cfg.force_parity_feature and cfg.num_alpha > 0:
        alphas[0, :] = 1  # global parity feature α = 1^n

    P_mat = build_parity_matrix(alphas, cfg.n)  # (K,N)
    z_data = onp.dot(P_mat, train_dist).astype(onp.float64)  # (K,)

    weights = onp.ones(cfg.num_alpha, dtype=onp.float64)
    if cfg.force_parity_feature and cfg.num_alpha > 0:
        weights[0] = float(cfg.parity_weight)

    return dict(train_dist=train_dist, alphas=alphas, P_mat=P_mat, z_data=z_data, weights=weights)

def train_single_depth(cfg: RunConfig,
                       layers: int,
                       p_star: onp.ndarray,
                       support: onp.ndarray,
                       good_mask: onp.ndarray,
                       loss_objs: Dict[str, Any],
                       init_seed: int) -> Dict[str, Any]:
    """
    Train a single run and return history.
    """
    dev = qml.device("default.qubit", wires=cfg.n)
    pairs, quads = get_iqp_topology(cfg.n, cfg.arch)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(cfg.n), pairs, quads, layers=layers, include_singles=cfg.include_singles)
        return qml.probs(wires=range(cfg.n))

    P_mat_tensor = np.array(loss_objs["P_mat"], requires_grad=False)
    z_data_tensor = np.array(loss_objs["z_data"], requires_grad=False)
    w_tensor = np.array(loss_objs["weights"], requires_grad=False)

    def loss_fn(W):
        q = circuit(W)
        z_model = P_mat_tensor @ q
        diff = z_data_tensor - z_model
        return np.sum(w_tensor * diff**2) / np.sum(w_tensor)

    nump = num_params(cfg.n, pairs, quads, layers, cfg.include_singles)
    rng = onp.random.default_rng(init_seed)
    W = np.array(cfg.init_scale * rng.standard_normal(nump), requires_grad=True)

    opt = qml.AdamOptimizer(cfg.lr)

    hist = {"step": [], "loss": [], "kl": [], "top5": [], "grad_norm": [], "q_even": []}
    support_mask = support.astype(bool)

    def eval_and_log(step: int):
        l_val = float(loss_fn(W))
        g = qml.grad(loss_fn)(W)
        g_np = onp.array(g, dtype=onp.float64)
        g_norm = float(onp.linalg.norm(g_np))

        q_val = onp.clip(onp.array(circuit(W), dtype=onp.float64), 0.0, 1.0)
        q_val = q_val / max(1e-15, float(q_val.sum()))

        kl = kl_divergence(p_star, q_val, eps=cfg.kl_eps)
        top5 = float(q_val[good_mask].sum())
        q_even = float(q_val[support_mask].sum())

        hist["step"].append(int(step))
        hist["loss"].append(l_val)
        hist["kl"].append(kl)
        hist["top5"].append(top5)
        hist["grad_norm"].append(g_norm)
        hist["q_even"].append(q_even)

        if cfg.verbose:
            print(f"[seed={init_seed:02d} L={layers}] step {step:4d} | loss {l_val:.2e} | KL {kl:.3f} | q(G) {top5:.3f} | q(even) {q_even:.3f} | ||grad|| {g_norm:.2e}")

    eval_and_log(0)

    for t in range(1, cfg.steps + 1):
        W = opt.step(loss_fn, W)
        if (t % cfg.eval_every == 0) or (t == 1) or (t == cfg.steps):
            eval_and_log(t)

    return {"history": hist, "num_params": int(nump)}

def ensemble_depth(cfg: RunConfig,
                   layers: int,
                   p_star: onp.ndarray,
                   support: onp.ndarray,
                   good_mask: onp.ndarray,
                   loss_objs: Dict[str, Any],
                   init_seeds: List[int]) -> Dict[str, Any]:
    runs = []
    for s in init_seeds:
        if not cfg.verbose:
            print(f"  -> train seed={s:02d} (L={layers}, singles={cfg.include_singles}, parity_feat={cfg.force_parity_feature})")
        run = train_single_depth(cfg, layers, p_star, support, good_mask, loss_objs, init_seed=s)
        runs.append(run["history"])

    # Aggregate (mean ± std) over seeds at each eval point (assumes identical step grid)
    steps = onp.array(runs[0]["step"], dtype=int)
    keys = ["loss", "kl", "top5", "grad_norm", "q_even"]
    agg = {"step": steps.tolist()}

    for k in keys:
        M = onp.array([r[k] for r in runs], dtype=onp.float64)  # (S, T)
        agg[k] = {
            "mean": onp.mean(M, axis=0).tolist(),
            "std":  onp.std(M, axis=0).tolist(),
            "final_mean": float(onp.mean(M[:, -1])),
            "final_std":  float(onp.std(M[:, -1])),
        }

    return {"agg": agg, "runs": runs}

def print_final_summary(name: str, results_by_L: Dict[int, Dict[str, Any]]):
    print(f"\n=== Final-step summary: {name} (mean ± std across init seeds) ===")
    print("Depth |             KL |           q(G) |        q(even) |         loss |       ||grad||")
    print("-"*92)
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        klm, kls = agg["kl"]["final_mean"], agg["kl"]["final_std"]
        qgm, qgs = agg["top5"]["final_mean"], agg["top5"]["final_std"]
        qem, qes = agg["q_even"]["final_mean"], agg["q_even"]["final_std"]
        lom, los = agg["loss"]["final_mean"], agg["loss"]["final_std"]
        grm, grs = agg["grad_norm"]["final_mean"], agg["grad_norm"]["final_std"]
        print(f"{L:5d} | {klm:7.3f} ± {kls:6.3f} | {qgm:7.3f} ± {qgs:6.3f} | {qem:7.3f} ± {qes:6.3f} | {lom:9.2e} ± {los:6.1e} | {grm:8.2e} ± {grs:6.1e}")


# ------------------------------------------------------------------------------
# 7) Plotting helpers (mean ± std shading) in red/black/white style
# ------------------------------------------------------------------------------

def _shade(ax, x, mean, std, color, alpha=0.15, logy=False):
    m = onp.array(mean, dtype=float)
    s = onp.array(std, dtype=float)
    lo = m - s
    hi = m + s
    if logy:
        lo = onp.clip(lo, 1e-16, None)
        hi = onp.clip(hi, 1e-16, None)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)

def plot_architecture_D_and_fix(n: int, outdir: str):
    """
    Draws a clean schematic for hero D and the fixed version (adds Z fields).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")

    def draw(ax, include_singles: bool, title: str):
        # Node positions on circle
        angles = onp.linspace(0, 2*onp.pi, n, endpoint=False)
        pos = onp.stack([onp.cos(angles), onp.sin(angles)], axis=1)

        # 4-body patches for consecutive qubits
        for i in range(n):
            quad = [(i)%n, (i+1)%n, (i+2)%n, (i+3)%n]
            pts = pos[quad]
            poly = Polygon(pts, closed=True, facecolor=COLORS["light"], edgecolor="none", alpha=0.25, zorder=0)
            ax.add_patch(poly)

        # NN edges (black) and NNN edges (gray dashed)
        for i in range(n):
            j = (i+1)%n
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color=COLORS["target"], lw=2.2, zorder=1)
        for i in range(n):
            j = (i+2)%n
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color=COLORS["gray"], lw=1.8, ls="--", alpha=0.9, zorder=1)

        # Nodes
        for i in range(n):
            ax.add_patch(Circle((pos[i,0], pos[i,1]), 0.08, facecolor="white", edgecolor="black", lw=1.2, zorder=3))
            ax.text(pos[i,0], pos[i,1], str(i), ha="center", va="center", fontsize=10, zorder=4)

            if include_singles:
                ax.add_patch(Circle((pos[i,0], pos[i,1]), 0.105, facecolor="none", edgecolor=COLORS["model"], lw=2.0, zorder=2))

        # NO TITLES (removed)
        _ = title  # keep signature identical / unused

    draw(axes[0], include_singles=False, title="Hero D (baseline)\nNN + NNN + 4-body")
    draw(axes[1], include_singles=True, title="Hero D + Z-fields (fix)\nadds single-qubit RZ per layer")

    legend_items = [
        Line2D([0],[0], color=COLORS["target"], lw=2.2, label="NN ZZ"),
        Line2D([0],[0], color=COLORS["gray"], lw=1.8, ls="--", label="NNN ZZ"),
        Patch(facecolor=COLORS["light"], edgecolor="none", alpha=0.25, label="local 4-body ZZZZ"),
        Line2D([0],[0], color=COLORS["model"], lw=2.0, label="Z-field (RZ)"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=4, frameon=True, framealpha=0.95, edgecolor="white")
    plt.tight_layout()
    path = os.path.join(outdir, "00_architecture_D_and_fix.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

def plot_parity_sanity(L_vals: List[int], q_even_means: List[float], q_even_stds: List[float], outdir: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = onp.arange(len(L_vals))
    ax.bar(x, q_even_means, yerr=q_even_stds, capsize=4,
           color="white", edgecolor="#333333", hatch="///", alpha=0.9)
    for i, v in enumerate(q_even_means):
        ax.text(x[i], v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}" for L in L_vals])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(r"$q(\mathcal{S}_\mathrm{even})$")
    # NO TITLE (removed)
    plt.tight_layout()
    path = os.path.join(outdir, "01_parity_sanity.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

def plot_trainability_grid(results_by_L: Dict[int, Dict[str, Any]],
                           pstar_top5: float,
                           outdir: str,
                           fname: str,
                           title: str):
    """
    2x2 grid:
      KL vs steps
      q(G) vs steps
      ||grad|| vs steps (log)
      q(even) vs steps
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    axes = axes.ravel()

    # Styles: L=1 black, L=2 red, L=3 gray
    def style(L):
        if L == 1:
            return dict(color=COLORS["target"], lw=2.8)
        if L == 2:
            return dict(color=COLORS["model"], lw=3.2)
        return dict(color=COLORS["gray"], lw=2.8)

    # Panel 0: KL
    ax = axes[0]
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["kl"]["mean"])
        s = onp.array(agg["kl"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.15)
        ax.text(x[-1], y[-1], f"{y[-1]:.2f}", color=style(L)["color"], fontweight="bold", ha="right", va="bottom", fontsize=10)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^*\Vert q_\theta)$")
    # NO TITLE (removed)
    ax.grid(True, linestyle="--", alpha=0.15)

    # Panel 1: q(G)
    ax = axes[1]
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["top5"]["mean"])
        s = onp.array(agg["top5"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.15)
        ax.text(x[-1], y[-1], f"{y[-1]:.3f}", color=style(L)["color"], fontweight="bold", ha="right", va="bottom", fontsize=10)
    ax.axhline(pstar_top5, color=COLORS["target"], linestyle=":", linewidth=2.0, alpha=0.85, label=r"Target $p^*(G)$")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$q_\theta(G)$")
    ax.set_ylim(0, 1.1)
    # NO TITLE (removed)
    ax.grid(True, linestyle="--", alpha=0.15)

    # Panel 2: grad norm (log)
    ax = axes[2]
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["grad_norm"]["mean"])
        s = onp.array(agg["grad_norm"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.12, logy=True)
        ax.text(x[-1], y[-1], f"{y[-1]:.1e}", color=style(L)["color"], fontweight="bold", ha="right", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    # NO TITLE (removed)
    ax.grid(True, linestyle="--", alpha=0.15)

    # Panel 3: q(even)
    ax = axes[3]
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["q_even"]["mean"])
        s = onp.array(agg["q_even"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.15)
        ax.text(x[-1], y[-1], f"{y[-1]:.3f}", color=style(L)["color"], fontweight="bold", ha="right", va="bottom", fontsize=10)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$q(\mathcal{S}_\mathrm{even})$")
    ax.set_ylim(0, 1.1)
    # NO TITLE (removed)
    ax.grid(True, linestyle="--", alpha=0.15)

    # Legend (deduplicate)
    handles, labels = [], []
    for a in axes:
        h, l = a.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, framealpha=0.95, edgecolor="white")

    # NO SUPTITLE (removed)
    _ = title  # keep signature identical / unused

    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

def plot_sweep_final_metric(xvals, metric_means, metric_stds, xlabel, ylabel, title, outpath, logx=False):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    x = onp.array(xvals, dtype=float)
    y = onp.array(metric_means, dtype=float)
    s = onp.array(metric_stds, dtype=float)

    ax.plot(x, y, color=COLORS["model"], lw=3.0)
    ax.fill_between(x, y - s, y + s, color=COLORS["model"], alpha=0.15, linewidth=0)
    for i in range(len(x)):
        ax.text(x[i], y[i], f"{y[i]:.2f}", color=COLORS["model"], fontweight="bold", ha="center", va="bottom", fontsize=9)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # NO TITLE (removed)
    _ = title  # keep signature identical / unused
    ax.grid(True, linestyle="--", alpha=0.15)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[Saved] {outpath}")
def plot_final_summary_bars(baseline_by_L: Dict[int, Dict[str, Any]],
                            fixed_by_L: Dict[int, Dict[str, Any]],
                            outdir: str):
    """
    1x3 panel bar chart: KL, q(G), q(even), baseline vs fixed for L=1/2/3.
    Legend is placed BELOW the panels. Value labels are placed above errorbar caps.
    """
    Ls = sorted(baseline_by_L.keys())
    x = onp.arange(len(Ls))
    w = 0.35

    def finals(res_by_L, key):
        m = [res_by_L[L]["agg"][key]["final_mean"] for L in Ls]
        s = [res_by_L[L]["agg"][key]["final_std"] for L in Ls]
        return onp.array(m), onp.array(s)

    kl_b, kl_bs = finals(baseline_by_L, "kl")
    kl_f, kl_fs = finals(fixed_by_L, "kl")

    qg_b, qg_bs = finals(baseline_by_L, "top5")
    qg_f, qg_fs = finals(fixed_by_L, "top5")

    qe_b, qe_bs = finals(baseline_by_L, "q_even")
    qe_f, qe_fs = finals(fixed_by_L, "q_even")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    def barpanel(ax, yb, ybs, yf, yfs, ylabel, title, ylim=None):
        bb = ax.bar(x - w/2, yb, w, yerr=ybs, capsize=4,
                    color="white", edgecolor="#333333", hatch="///", alpha=0.9, label="Baseline")
        bf = ax.bar(x + w/2, yf, w, yerr=yfs, capsize=4,
                    color=COLORS["model"], edgecolor="black", alpha=0.85, label="Fix (Z-fields + parity feature)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"L={L}" for L in Ls])
        ax.set_ylabel(ylabel)

        # NO TITLE (removed)
        _ = title

        ax.grid(axis="y", linestyle=":", alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # --- Value labels: place above (bar height + errorbar + padding) ---
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        pad = 0.03 * yr  # increase to 0.04 if you want even more spacing

        for i, r in enumerate(bb):
            h = float(r.get_height())
            err = float(ybs[i]) if ybs is not None else 0.0
            y_text = h + err + pad
            y_text = min(y_text, y1 - 0.01 * yr)  # keep inside axis
            ax.text(
                r.get_x() + r.get_width()/2,
                y_text,
                f"{h:.2f}" if ylabel == "KL" else f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
                clip_on=False,
            )

        for i, r in enumerate(bf):
            h = float(r.get_height())
            err = float(yfs[i]) if yfs is not None else 0.0
            y_text = h + err + pad
            y_text = min(y_text, y1 - 0.01 * yr)  # keep inside axis
            ax.text(
                r.get_x() + r.get_width()/2,
                y_text,
                f"{h:.2f}" if ylabel == "KL" else f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#800000",
                fontweight="bold",
                clip_on=False,
            )

    barpanel(
        axes[0], kl_b, kl_bs, kl_f, kl_fs,
        ylabel="KL",
        title=r"$D_{\mathrm{KL}}(p^*\Vert q_\theta)$",
        ylim=(0, max(float(kl_b.max()), float(kl_f.max()))*1.25 if max(float(kl_b.max()), float(kl_f.max())) > 0 else 1.0),
    )
    barpanel(axes[1], qg_b, qg_bs, qg_f, qg_fs, ylabel=r"$q_\theta(G)$", title="Good-set mass", ylim=(0, 1.1))
    barpanel(axes[2], qe_b, qe_bs, qe_f, qe_fs, ylabel=r"$q(\mathcal{S}_\mathrm{even})$", title="Parity support mass", ylim=(0, 1.1))

    handles, labels = axes[0].get_legend_handles_labels()

    # Legend BELOW
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="white",
    )

    path = os.path.join(outdir, "07_final_summary_bars.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")



# ------------------------------------------------------------------------------
# 7b) EXTRA: also save every multi-panel plot as individual single-panel PDFs
# (Everything else remains identical; original combined PDFs are still produced.)
# Titles are removed here as well.
# ------------------------------------------------------------------------------

def plot_architecture_D_single(n: int, outdir: str, include_singles: bool, title: str, fname: str):
    """
    Same drawing as in plot_architecture_D_and_fix(), but saved as a single-panel PDF.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")
    ax.axis("off")

    angles = onp.linspace(0, 2*onp.pi, n, endpoint=False)
    pos = onp.stack([onp.cos(angles), onp.sin(angles)], axis=1)

    for i in range(n):
        quad = [(i)%n, (i+1)%n, (i+2)%n, (i+3)%n]
        pts = pos[quad]
        poly = Polygon(pts, closed=True, facecolor=COLORS["light"], edgecolor="none", alpha=0.25, zorder=0)
        ax.add_patch(poly)

    for i in range(n):
        j = (i+1)%n
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color=COLORS["target"], lw=2.2, zorder=1)
    for i in range(n):
        j = (i+2)%n
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color=COLORS["gray"], lw=1.8, ls="--", alpha=0.9, zorder=1)

    for i in range(n):
        ax.add_patch(Circle((pos[i,0], pos[i,1]), 0.08, facecolor="white", edgecolor="black", lw=1.2, zorder=3))
        ax.text(pos[i,0], pos[i,1], str(i), ha="center", va="center", fontsize=10, zorder=4)
        if include_singles:
            ax.add_patch(Circle((pos[i,0], pos[i,1]), 0.105, facecolor="none", edgecolor=COLORS["model"], lw=2.0, zorder=2))

    # NO TITLE (removed)
    _ = title

    legend_items = [
        Line2D([0],[0], color=COLORS["target"], lw=2.2, label="NN ZZ"),
        Line2D([0],[0], color=COLORS["gray"], lw=1.8, ls="--", label="NNN ZZ"),
        Patch(facecolor=COLORS["light"], edgecolor="none", alpha=0.25, label="local 4-body ZZZZ"),
        Line2D([0],[0], color=COLORS["model"], lw=2.0, label="Z-field (RZ)"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=4, frameon=True, framealpha=0.95, edgecolor="white")

    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

def plot_trainability_panels_individual(results_by_L: Dict[int, Dict[str, Any]],
                                        pstar_top5: float,
                                        outdir: str,
                                        fig_prefix_num: str,
                                        fig_prefix_name: str,
                                        suptitle: str):
    """
    Same content as plot_trainability_grid(), but saves each panel as its own PDF.
    Outputs:
      {fig_prefix_num}a_{fig_prefix_name}_KL.pdf
      {fig_prefix_num}b_{fig_prefix_name}_qG.pdf
      {fig_prefix_num}c_{fig_prefix_name}_grad.pdf
      {fig_prefix_num}d_{fig_prefix_name}_qEven.pdf
    """
    def style(L):
        if L == 1:
            return dict(color=COLORS["target"], lw=2.8)
        if L == 2:
            return dict(color=COLORS["model"], lw=3.2)
        return dict(color=COLORS["gray"], lw=2.8)

    # Panel KL
    fig, ax = plt.subplots(figsize=(6.25, 4.0))
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["kl"]["mean"])
        s = onp.array(agg["kl"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.15)
        ax.text(x[-1], y[-1], f"{y[-1]:.2f}", color=style(L)["color"], fontweight="bold",
                ha="right", va="bottom", fontsize=10)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^*\Vert q_\theta)$")
    # NO TITLE / NO SUPTITLE (removed)
    _ = suptitle
    ax.grid(True, linestyle="--", alpha=0.15)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=True, framealpha=0.95, edgecolor="white")
    plt.tight_layout()
    path = os.path.join(outdir, f"{fig_prefix_num}a_{fig_prefix_name}_KL.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

    # Panel q(G)
    fig, ax = plt.subplots(figsize=(6.25, 4.0))
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["top5"]["mean"])
        s = onp.array(agg["top5"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.15)
        ax.text(x[-1], y[-1], f"{y[-1]:.3f}", color=style(L)["color"], fontweight="bold",
                ha="right", va="bottom", fontsize=10)
    ax.axhline(pstar_top5, color=COLORS["target"], linestyle=":", linewidth=2.0, alpha=0.85,
               label=r"Target $p^*(G)$")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$q_\theta(G)$")
    ax.set_ylim(0, 1.1)
    # NO TITLE / NO SUPTITLE (removed)
    _ = suptitle
    ax.grid(True, linestyle="--", alpha=0.15)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=True, framealpha=0.95, edgecolor="white")
    plt.tight_layout()
    path = os.path.join(outdir, f"{fig_prefix_num}b_{fig_prefix_name}_qG.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

    # Panel grad norm (log)
    fig, ax = plt.subplots(figsize=(6.25, 4.0))
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["grad_norm"]["mean"])
        s = onp.array(agg["grad_norm"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.12, logy=True)
        ax.text(x[-1], y[-1], f"{y[-1]:.1e}", color=style(L)["color"], fontweight="bold",
                ha="right", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    # NO TITLE / NO SUPTITLE (removed)
    _ = suptitle
    ax.grid(True, linestyle="--", alpha=0.15)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=True, framealpha=0.95, edgecolor="white")
    plt.tight_layout()
    path = os.path.join(outdir, f"{fig_prefix_num}c_{fig_prefix_name}_grad.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

    # Panel q(even)
    fig, ax = plt.subplots(figsize=(6.25, 4.0))
    for L in sorted(results_by_L.keys()):
        agg = results_by_L[L]["agg"]
        x = onp.array(agg["step"])
        y = onp.array(agg["q_even"]["mean"])
        s = onp.array(agg["q_even"]["std"])
        ax.plot(x, y, label=f"L={L}", **style(L))
        _shade(ax, x, y, s, color=style(L)["color"], alpha=0.15)
        ax.text(x[-1], y[-1], f"{y[-1]:.3f}", color=style(L)["color"], fontweight="bold",
                ha="right", va="bottom", fontsize=10)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(r"$q(\mathcal{S}_\mathrm{even})$")
    ax.set_ylim(0, 1.1)
    # NO TITLE / NO SUPTITLE (removed)
    _ = suptitle
    ax.grid(True, linestyle="--", alpha=0.15)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=True, framealpha=0.95, edgecolor="white")
    plt.tight_layout()
    path = os.path.join(outdir, f"{fig_prefix_num}d_{fig_prefix_name}_qEven.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

def plot_final_summary_bars_individual(baseline_by_L: Dict[int, Dict[str, Any]],
                                       fixed_by_L: Dict[int, Dict[str, Any]],
                                       outdir: str):
    """
    Same content as plot_final_summary_bars(), but saves each of the 3 panels as its own PDF:
      07a_final_summary_KL.pdf
      07b_final_summary_qG.pdf
      07c_final_summary_qEven.pdf

    Legend is placed BELOW. Value labels are placed above errorbar caps.
    """
    Ls = sorted(baseline_by_L.keys())
    x = onp.arange(len(Ls))
    w = 0.35

    def finals(res_by_L, key):
        m = [res_by_L[L]["agg"][key]["final_mean"] for L in Ls]
        s = [res_by_L[L]["agg"][key]["final_std"] for L in Ls]
        return onp.array(m), onp.array(s)

    kl_b, kl_bs = finals(baseline_by_L, "kl")
    kl_f, kl_fs = finals(fixed_by_L, "kl")

    qg_b, qg_bs = finals(baseline_by_L, "top5")
    qg_f, qg_fs = finals(fixed_by_L, "top5")

    qe_b, qe_bs = finals(baseline_by_L, "q_even")
    qe_f, qe_fs = finals(fixed_by_L, "q_even")

    def barpanel(ax, yb, ybs, yf, yfs, ylabel, title, ylim=None):
        bb = ax.bar(x - w/2, yb, w, yerr=ybs, capsize=4,
                    color="white", edgecolor="#333333", hatch="///", alpha=0.9, label="Baseline")
        bf = ax.bar(x + w/2, yf, w, yerr=yfs, capsize=4,
                    color=COLORS["model"], edgecolor="black", alpha=0.85, label="Fix (Z-fields + parity feature)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"L={L}" for L in Ls])
        ax.set_ylabel(ylabel)

        # NO TITLE (removed)
        _ = title

        ax.grid(axis="y", linestyle=":", alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # --- Value labels: place above (bar height + errorbar + padding) ---
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        pad = 0.03 * yr  # increase to 0.04 if you want even more spacing

        for i, r in enumerate(bb):
            h = float(r.get_height())
            err = float(ybs[i]) if ybs is not None else 0.0
            y_text = h + err + pad
            y_text = min(y_text, y1 - 0.01 * yr)
            ax.text(
                r.get_x() + r.get_width()/2,
                y_text,
                f"{h:.2f}" if ylabel == "KL" else f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
                clip_on=False,
            )

        for i, r in enumerate(bf):
            h = float(r.get_height())
            err = float(yfs[i]) if yfs is not None else 0.0
            y_text = h + err + pad
            y_text = min(y_text, y1 - 0.01 * yr)
            ax.text(
                r.get_x() + r.get_width()/2,
                y_text,
                f"{h:.2f}" if ylabel == "KL" else f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#800000",
                fontweight="bold",
                clip_on=False,
            )

    # KL panel
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    barpanel(
        ax, kl_b, kl_bs, kl_f, kl_fs,
        ylabel="KL",
        title=r"$D_{\mathrm{KL}}(p^*\Vert q_\theta)$",
        ylim=(0, max(float(kl_b.max()), float(kl_f.max()))*1.25 if max(float(kl_b.max()), float(kl_f.max())) > 0 else 1.0),
    )
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="white",
    )
    path = os.path.join(outdir, "07a_final_summary_KL.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

    # q(G) panel
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    barpanel(ax, qg_b, qg_bs, qg_f, qg_fs,
             ylabel=r"$q_\theta(G)$", title="Good-set mass", ylim=(0, 1.1))
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="white",
    )
    path = os.path.join(outdir, "07b_final_summary_qG.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

    # q(even) panel
    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    barpanel(ax, qe_b, qe_bs, qe_f, qe_fs,
             ylabel=r"$q(\mathcal{S}_\mathrm{even})$", title="Parity support mass", ylim=(0, 1.1))
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="white",
    )
    path = os.path.join(outdir, "07c_final_summary_qEven.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")



# ------------------------------------------------------------------------------
# 8) Main analysis pipeline
# ------------------------------------------------------------------------------

def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--arch", type=str, default="D", choices=["A","B","C","D","E"])
    parser.add_argument("--beta", type=float, default=0.9)

    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--num-alpha", type=int, default=256)
    parser.add_argument("--train-m", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)

    parser.add_argument("--data-seed", type=int, default=7)
    parser.add_argument("--alpha-seed", type=int, default=42)
    parser.add_argument("--init-seeds", type=str, default="0,1,2,3,4")

    parser.add_argument("--parity-weight", type=float, default=20.0)

    parser.add_argument("--outdir", type=str, default="heroD_parity_analysis")
    parser.add_argument("--verbose", action="store_true")

    # sweep lists
    parser.add_argument("--lr-list", type=str, default="0.005,0.01,0.02,0.05")
    parser.add_argument("--K-list", type=str, default="64,128,256,512")
    parser.add_argument("--sigma-list", type=str, default="0.5,1.0,2.0")

    args = parser.parse_args()

    set_style()
    outdir = ensure_outdir(args.outdir)

    init_seeds = parse_int_list(args.init_seeds)
    lr_list = [float(x) for x in args.lr_list.split(",")]
    K_list  = [int(x) for x in args.K_list.split(",")]
    sigma_list = [float(x) for x in args.sigma_list.split(",")]

    cfg_base = RunConfig(
        n=args.n, arch=args.arch, beta=args.beta,
        steps=args.steps, lr=args.lr, sigma=args.sigma,
        num_alpha=args.num_alpha, train_m=args.train_m,
        eval_every=args.eval_every,
        data_seed=args.data_seed, alpha_seed=args.alpha_seed,
        include_singles=False, force_parity_feature=False,
        parity_weight=args.parity_weight,
        outdir=outdir, verbose=args.verbose
    )

    print("="*80)
    print(f"Hero parity analysis | arch={cfg_base.arch} | n={cfg_base.n} | beta={cfg_base.beta}")
    print(f"steps={cfg_base.steps} lr={cfg_base.lr} sigma={cfg_base.sigma} K={cfg_base.num_alpha} m={cfg_base.train_m}")
    print(f"init_seeds={init_seeds} | data_seed={cfg_base.data_seed} | alpha_seed={cfg_base.alpha_seed}")
    print("="*80)

    # Target
    p_star, support, scores = build_target_distribution(cfg_base.n, cfg_base.beta)
    good_mask = topk_mask(scores, support, frac=0.05)
    pstar_top5 = float(p_star[good_mask].sum())
    print(f"[Target] p*(G)={pstar_top5:.6f} | |G|={int(good_mask.sum())} | |S_even|={int(onp.sum(support))}")

    # Architecture diagram
    n_arch = min(cfg_base.n, 10)
    plot_architecture_D_and_fix(n=n_arch, outdir=outdir)

    # EXTRA: architecture panels individually (no titles)
    plot_architecture_D_single(n=n_arch, outdir=outdir, include_singles=False,
                              title="Hero D (baseline)\nNN + NNN + 4-body",
                              fname="00a_architecture_D_baseline.pdf")
    plot_architecture_D_single(n=n_arch, outdir=outdir, include_singles=True,
                              title="Hero D + Z-fields (fix)\nadds single-qubit RZ per layer",
                              fname="00b_architecture_D_fix.pdf")

    # (A) Parity sanity check: random parameters (no training) for baseline ansatz
    print("\n[A] Parity sanity check (random parameters, no training) ...")
    L_vals = [1, 2, 3]
    q_even_means, q_even_stds = [], []
    for L in L_vals:
        dev = qml.device("default.qubit", wires=cfg_base.n)
        pairs, quads = get_iqp_topology(cfg_base.n, cfg_base.arch)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit_tmp(W):
            iqp_circuit(W, range(cfg_base.n), pairs, quads, layers=L, include_singles=False)
            return qml.probs(wires=range(cfg_base.n))

        q_evens = []
        rng = onp.random.default_rng(123)
        for _ in range(5):
            W = np.array(0.7 * rng.standard_normal(num_params(cfg_base.n, pairs, quads, L, include_singles=False)),
                         requires_grad=False)
            q = onp.array(circuit_tmp(W), dtype=onp.float64)
            q = q / max(1e-15, float(q.sum()))
            q_even = float(q[support].sum())
            q_evens.append(q_even)
        q_even_means.append(float(onp.mean(q_evens)))
        q_even_stds.append(float(onp.std(q_evens)))
        print(f"  L={L}: q(S_even) mean={q_even_means[-1]:.6f} std={q_even_stds[-1]:.6f}  (odd≈1, even≈0.5)")

    plot_parity_sanity(L_vals, q_even_means, q_even_stds, outdir)

    # Build baseline loss objects (same data & features across depths)
    loss_objs_base = build_loss_objects(cfg_base, p_star, support, good_mask, use_oracle_moments=False)

    # (B) Baseline ensemble training for L=1/2/3
    print("\n[B] Baseline training L=1/2/3 (ensemble over init seeds) ...")
    baseline_results = {}
    for L in [1, 2, 3]:
        print(f"\n-> Baseline ensemble for L={L}")
        baseline_results[L] = ensemble_depth(cfg_base, L, p_star, support, good_mask, loss_objs_base, init_seeds)
    print_final_summary("BASELINE", baseline_results)

    baseline_title = f"Baseline (ZZ + ZZZZ only) | arch={cfg_base.arch}, n={cfg_base.n}, beta={cfg_base.beta}"
    plot_trainability_grid(
        baseline_results,
        pstar_top5,
        outdir,
        fname="02_trainability_baseline_L123.pdf",
        title=baseline_title
    )

    # EXTRA: baseline panels individually (no titles)
    plot_trainability_panels_individual(
        baseline_results,
        pstar_top5,
        outdir,
        fig_prefix_num="02",
        fig_prefix_name="trainability_baseline",
        suptitle=baseline_title
    )

    # (C) L=2 sweeps
    print("\n[C] Robustness sweeps for L=2 baseline ...")

    # LR sweep (plots KL)
    kl_means, kl_stds = [], []
    for lr in lr_list:
        cfg_lr = RunConfig(**{**cfg_base.__dict__, "lr": float(lr)})
        loss_objs = build_loss_objects(cfg_lr, p_star, support, good_mask, use_oracle_moments=False)
        print(f"\n-> LR sweep: lr={lr}")
        res = ensemble_depth(cfg_lr, 2, p_star, support, good_mask, loss_objs, init_seeds)
        kl_means.append(res["agg"]["kl"]["final_mean"])
        kl_stds.append(res["agg"]["kl"]["final_std"])

    plot_sweep_final_metric(
        lr_list, kl_means, kl_stds,
        xlabel="Learning rate",
        ylabel="KL (final)",
        title="L=2 baseline: LR sweep (final KL)",
        outpath=os.path.join(outdir, "03_lr_sweep_L2.pdf"),
        logx=True
    )

    # K sweep
    kl_means, kl_stds = [], []
    for K in K_list:
        cfg_K = RunConfig(**{**cfg_base.__dict__, "num_alpha": int(K)})
        loss_objs = build_loss_objects(cfg_K, p_star, support, good_mask, use_oracle_moments=False)
        print(f"\n-> K sweep: K={K}")
        res = ensemble_depth(cfg_K, 2, p_star, support, good_mask, loss_objs, init_seeds)
        kl_means.append(res["agg"]["kl"]["final_mean"])
        kl_stds.append(res["agg"]["kl"]["final_std"])

    plot_sweep_final_metric(
        K_list, kl_means, kl_stds,
        xlabel="K (parity features)",
        ylabel="KL (final)",
        title="L=2 baseline: K sweep (final KL)",
        outpath=os.path.join(outdir, "04_K_sweep_L2.pdf"),
        logx=True
    )

    # sigma sweep
    kl_means, kl_stds = [], []
    for sigma in sigma_list:
        cfg_s = RunConfig(**{**cfg_base.__dict__, "sigma": float(sigma)})
        loss_objs = build_loss_objects(cfg_s, p_star, support, good_mask, use_oracle_moments=False)
        print(f"\n-> sigma sweep: sigma={sigma}")
        res = ensemble_depth(cfg_s, 2, p_star, support, good_mask, loss_objs, init_seeds)
        kl_means.append(res["agg"]["kl"]["final_mean"])
        kl_stds.append(res["agg"]["kl"]["final_std"])

    plot_sweep_final_metric(
        sigma_list, kl_means, kl_stds,
        xlabel="sigma (kernel bandwidth)",
        ylabel="KL (final)",
        title="L=2 baseline: σ sweep (final KL)",
        outpath=os.path.join(outdir, "05_sigma_sweep_L2.pdf"),
        logx=True
    )

    # (D) Fix: Z-fields + parity feature
    print("\n[D] FIX: add Z-fields + force global parity feature ...")
    cfg_fix = RunConfig(**{**cfg_base.__dict__,
                           "include_singles": True,
                           "force_parity_feature": True,
                           "parity_weight": float(args.parity_weight)})

    loss_objs_fix = build_loss_objects(cfg_fix, p_star, support, good_mask, use_oracle_moments=False)

    fixed_results = {}
    for L in [1, 2, 3]:
        print(f"\n-> FIX ensemble for L={L}")
        fixed_results[L] = ensemble_depth(cfg_fix, L, p_star, support, good_mask, loss_objs_fix, init_seeds)

    print_final_summary("FIX (Z-fields + parity feature)", fixed_results)

    fixed_title = f"FIX (Z-fields + forced parity feature) | arch={cfg_base.arch}, n={cfg_base.n}, beta={cfg_base.beta}"
    plot_trainability_grid(
        fixed_results,
        pstar_top5,
        outdir,
        fname="06_trainability_fixed_L123.pdf",
        title=fixed_title
    )

    # EXTRA: FIX panels individually (no titles)
    plot_trainability_panels_individual(
        fixed_results,
        pstar_top5,
        outdir,
        fig_prefix_num="06",
        fig_prefix_name="trainability_fixed",
        suptitle=fixed_title
    )

    # Summary bar chart baseline vs fix
    plot_final_summary_bars(baseline_results, fixed_results, outdir)

    # EXTRA: summary panels individually (no titles)
    plot_final_summary_bars_individual(baseline_results, fixed_results, outdir)

    # Save summary JSON
    summary = {
        "config_baseline": cfg_base.__dict__,
        "config_fix": cfg_fix.__dict__,
        "pstar_top5": pstar_top5,
        "baseline_results": baseline_results,
        "fixed_results": fixed_results,
        "parity_sanity": {"L_vals": L_vals, "q_even_means": q_even_means, "q_even_stds": q_even_stds},
        "sweeps": {
            "lr_list": lr_list,
            "K_list": K_list,
            "sigma_list": sigma_list,
        }
    }
    path = os.path.join(outdir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {path}")
    print(f"Done. Results in ./{outdir}/")

if __name__ == "__main__":
    main()
