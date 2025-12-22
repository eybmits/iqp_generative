#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQP-QCBM: Temperature / Sharpness Sweep over β (Red & Black + Intersection)
==========================================================================

Goal:
  Vary the target sharpness parameter β and measure how well an IQP-QCBM
  matches the underlying distribution p*_β.

Primary metric (only):
  Final KL(p*_β || q_{θ,β}) after training.

Plot (single main figure):
  - x-axis: β (log-x), with explicit tick marks for sampled β values
  - y-axis (left): mean ± std of KL over k seeds (log-y)
  - y-axis (left): uniform baseline KL(p*_β || Unif) (same units, dashed black)
  - y-axis (right): target entropy H(p*_β) in bits (gray dotted), as "difficulty"
  - if an intersection exists between model KL and baseline KL:
      annotate β_x where KL_model == KL_uniform (approx. in log-domain)

Update in this version:
  - REDUCED BETA GRID: Fewer points at the high end to prevent visual overlap.
  - Adds journal-aware plotting style and sizes:
      --fig-target col  : RevTeX single-column-ready PDFs (fonts ~8–9pt)
      --fig-target full : figure*-ready PDFs (bigger fonts/lines)

Outputs (in --outdir):
  - beta_sweep_KL.pdf / .png
  - beta_sweep_results.json
  - beta_sweep_results.csv

Usage:
  python iqp_beta_sweep_kl.py --arch E --layers 1 --n 14 --steps 600 --train-m 1000 --k 5 \
      --outdir beta_sweep_E --fig-target col
"""

import os
import csv
import json
import math
import argparse
import itertools
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pennylane")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

import numpy as onp
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as np


# ------------------------------------------------------------------------------
# 1) Journal-aware Visual Style
# ------------------------------------------------------------------------------

COLORS = {
    "target": "#222222",   # almost black
    "model":  "#D62728",   # deep red
    "gray":   "#666666",
    "loss":   "#1F77B4",
}

# Practical widths for APS/RevTeX, in inches
COL_W  = 3.37  # single-column
FULL_W = 6.95  # two-column (figure*)

def fig_size(fig_target: str, h_col: float, h_full: float) -> Tuple[float, float]:
    if fig_target not in ("col", "full"):
        raise ValueError("fig_target must be 'col' or 'full'")
    return (COL_W, h_col) if fig_target == "col" else (FULL_W, h_full)

def set_style(fig_target: str = "col"):
    """
    - col  : RevTeX single-column-ready (8–9pt, thinner lines)
    - full : bigger "figure*" look (12–14pt, thicker lines)
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
            "lines.markersize": 6.0,

            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
# 2) Target distribution (Even parity + longest zero-run score)
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
    """
    p*_β(x) ∝ exp(beta*(1+z(x))) for even parity x; else 0.
    Returns: p_star (float64), support_mask (bool), entropy bits can be computed separately.
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
    return p_star.astype(onp.float64), support

def sample_indices(probs: onp.ndarray, m: int, seed: int) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    p = probs / probs.sum()
    return rng.choice(len(p), size=m, replace=True, p=p)

def empirical_dist(idxs: onp.ndarray, N: int) -> onp.ndarray:
    c = onp.bincount(idxs, minlength=N)
    return (c / max(1, c.sum())).astype(onp.float64)

def entropy_bits(p: onp.ndarray, eps: float = 1e-18) -> float:
    p = onp.clip(p, eps, 1.0)
    return float(-onp.sum(p * onp.log2(p)))

def kl_div(p: onp.ndarray, q: onp.ndarray, eps: float = 1e-12) -> float:
    q = onp.clip(q, eps, 1.0)
    p = onp.clip(p, 0.0, 1.0)
    return float(onp.sum(onp.where(p > 0, p * (onp.log(p + eps) - onp.log(q + eps)), 0.0)))


# ------------------------------------------------------------------------------
# 3) IQP architectures (A–E) + circuit
# ------------------------------------------------------------------------------

def get_iqp_topology(n: int, arch: str):
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

def iqp_circuit(W, wires, pairs, quads, layers=1):
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
# 4) Gaussian spectral MMD features
# ------------------------------------------------------------------------------

def p_sigma(sigma: float) -> float:
    if sigma <= 0.0:
        return 0.5
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma**2)))

def sample_alphas(n: int, sigma: float, K: int, seed: int) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas: onp.ndarray, n: int) -> onp.ndarray:
    N = 2 ** n
    xs = onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)
    mod_vals = (alphas @ xs.T) % 2
    return onp.where(mod_vals == 0, 1.0, -1.0).astype(onp.float64)


# ------------------------------------------------------------------------------
# 5) Training (single run)
# ------------------------------------------------------------------------------

@dataclass
class TrainCfg:
    n: int = 14
    arch: str = "D"
    layers: int = 1
    steps: int = 600
    lr: float = 0.05
    sigma: float = 1.0
    num_alpha: int = 512
    train_m: int = 1000

def train_one_beta_seed(beta: float, seed: int, cfg: TrainCfg) -> Dict:
    # Target
    p_star, _ = build_target_distribution(cfg.n, beta)

    # Training data
    beta_tag = int(round(beta * 1000))
    data_seed = 100_000 + 10_000 * beta_tag + seed
    idxs_train = sample_indices(p_star, cfg.train_m, seed=data_seed)
    p_emp = empirical_dist(idxs_train, 2 ** cfg.n)

    # MMD features
    alpha_seed = 200_000 + 10_000 * beta_tag + seed
    alphas = sample_alphas(cfg.n, cfg.sigma, cfg.num_alpha, seed=alpha_seed)
    P_mat = build_parity_matrix(alphas, cfg.n)
    z_data = onp.dot(P_mat, p_emp).astype(onp.float64)

    # Model
    dev = qml.device("default.qubit", wires=cfg.n)
    pairs, quads = get_iqp_topology(cfg.n, cfg.arch)
    n_params = (len(pairs) + len(quads)) * cfg.layers

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(cfg.n), pairs, quads, layers=cfg.layers)
        return qml.probs(wires=range(cfg.n))

    P_tensor = np.array(P_mat, requires_grad=False)
    z_tensor = np.array(z_data, requires_grad=False)

    init_seed = 300_000 + 10_000 * beta_tag + seed
    rng = onp.random.default_rng(init_seed)
    W = np.array(0.01 * rng.standard_normal(n_params), requires_grad=True)
    opt = qml.AdamOptimizer(cfg.lr)

    def loss_fn(w):
        q = circuit(w)
        z_model = P_tensor @ q
        return np.mean((z_tensor - z_model) ** 2)

    for _ in range(cfg.steps):
        W, _ = opt.step_and_cost(loss_fn, W)

    q_final = onp.array(circuit(W), dtype=onp.float64)
    q_final = onp.clip(q_final, 1e-15, 1.0)
    q_final = q_final / max(1e-15, float(q_final.sum()))

    kl = kl_div(p_star, q_final)
    Hbits = entropy_bits(p_star)

    return {
        "beta": float(beta),
        "seed": int(seed),
        "kl": float(kl),
        "entropy_bits": float(Hbits),
        "n_params": int(n_params),
    }


# ------------------------------------------------------------------------------
# 6) Beta grid + plotting
# ------------------------------------------------------------------------------

def default_beta_grid() -> List[float]:
    # Fine grid at low beta
    fine = [round(0.1 * i, 2) for i in range(1, 11)]  # 0.1, 0.2 ... 1.0
    
    # Coarse grid at high beta - THINNED OUT to prevent overlap
    # Removed 1.25, 1.75, 2.5
    coarse = [1.5, 2.0, 3.0]
    
    return fine + coarse

def parse_betas(arg: str) -> List[float]:
    if not arg:
        return default_beta_grid()
    xs = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok:
            xs.append(float(tok))
    return sorted(list(dict.fromkeys(xs)))

def _find_intersection_beta(betas, y_model, y_base) -> Optional[float]:
    """
    Find β where y_model(β) == y_base(β), if it exists.
    We use log-domain interpolation consistent with log-x/log-y plot:
      r(β)=log10(y_model/y_base); crossing when r changes sign.
    Returns beta_hat or None.
    """
    betas = onp.asarray(betas, dtype=float)
    y_model = onp.asarray(y_model, dtype=float)
    y_base = onp.asarray(y_base, dtype=float)
    eps = 1e-18

    r = onp.log10(onp.clip(y_model, eps, None)) - onp.log10(onp.clip(y_base, eps, None))
    logb = onp.log(onp.clip(betas, eps, None))

    for i in range(len(r)):
        if abs(float(r[i])) < 1e-6:
            return float(betas[i])

    for i in range(len(r) - 1):
        if (r[i] < 0 and r[i+1] > 0) or (r[i] > 0 and r[i+1] < 0):
            t = float(r[i] / (r[i] - r[i+1]))
            logb_hat = float(logb[i] + t * (logb[i+1] - logb[i]))
            return float(onp.exp(logb_hat))
    return None

def plot_beta_sweep(
    betas: List[float],
    kl_mean: onp.ndarray,
    kl_std: onp.ndarray,
    ent_bits: onp.ndarray,
    outdir: str,
    n: int,
    tick_mode: str,
    fig_target: str,
):
    betas = list(map(float, betas))
    kl_mean = onp.asarray(kl_mean, dtype=float)
    kl_std = onp.asarray(kl_std, dtype=float)
    ent_bits = onp.asarray(ent_bits, dtype=float)

    # Uniform baseline KL in natural log units:
    # KL(p || Unif) = ln(2^n) - H_nat(p) = n ln 2 - H_bits(p) ln 2
    baseline_kl = onp.array([(n * math.log(2.0)) - float(Hb) * math.log(2.0) for Hb in ent_bits], dtype=float)

    fig, ax = plt.subplots(figsize=fig_size(fig_target, 2.9, 3.9), constrained_layout=True)

    # Baseline curve (black dashed)
    ax.plot(
        betas, baseline_kl,
        color=COLORS["target"], linestyle="--",
        linewidth=(1.6 if fig_target == "col" else 2.0),
        label=r"Uniform baseline $\mathrm{KL}(p_\beta^*\Vert \mathrm{Unif})$",
        zorder=2
    )

    # Model KL (red with errorbars)
    ms = 4 if fig_target == "col" else 6
    lw = 1.9 if fig_target == "col" else 2.6
    cap = 3 if fig_target == "col" else 4
    ax.errorbar(
        betas, kl_mean, yerr=kl_std,
        color=COLORS["model"], marker="o", markersize=ms,
        linewidth=lw, capsize=cap,
        label=r"Model $\mathrm{KL}(p_\beta^*\Vert q_{\theta,\beta})$",
        zorder=5
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Target sharpness $\beta$")
    ax.set_ylabel(r"Final KL divergence (mean $\pm$ std)")

    ax.set_xticks(betas)
    if tick_mode == "all":
        labels = [f"{b:g}" for b in betas]
    else:
        # Adjusted 'keep' set to match the new sparse grid
        keep = set([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
        labels = [f"{b:g}" if (round(b, 2) in keep) else "" for b in betas]
    ax.set_xticklabels(labels)

    # Right axis: entropy in bits
    ax2 = ax.twinx()
    ax2.plot(
        betas, ent_bits,
        color=COLORS["gray"], linestyle=":", marker="x", markersize=ms,
        linewidth=(1.4 if fig_target == "col" else 2.0),
        label=r"Target entropy $H(p_\beta^*)$",
        zorder=3
    )
    ax2.set_ylabel(r"Target entropy $H(p_\beta^*)$ [bits]", color=COLORS["gray"])
    ax2.tick_params(axis="y", labelcolor=COLORS["gray"])

    # Intersection (if exists)
    beta_hat = _find_intersection_beta(betas, kl_mean, baseline_kl)
    if beta_hat is not None and (min(betas) <= beta_hat <= max(betas)):
        ax.axvline(beta_hat, color=COLORS["gray"], linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)
        y_top = float(onp.nanmax(onp.r_[baseline_kl, kl_mean]))
        ann_fs = 7 if fig_target == "col" else 10
        ax.text(
            beta_hat, y_top * (1.20 if fig_target == "col" else 1.15),
            rf"$\beta_{{\times}}\approx {beta_hat:.2f}$",
            color=COLORS["gray"], fontsize=ann_fs, ha="center", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc")
        )
        print(f"[Intersection] KL_model == KL_uniform at beta ≈ {beta_hat:.4f}")
    else:
        print("[Intersection] No crossing found between model KL and uniform baseline on the sampled β-grid.")

    # Legend (combine axes)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")

    fig.savefig(os.path.join(outdir, "beta_sweep_KL.pdf"))
    fig.savefig(os.path.join(outdir, "beta_sweep_KL.png"))
    plt.close(fig)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="beta_sweep_out")
    ap.add_argument("--arch", type=str, default="D", choices=["A","B","C","D","E"])
    ap.add_argument("--layers", type=int, default=1)

    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)

    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--num-alpha", dest="num_alpha", type=int, default=512)

    ap.add_argument("--train-m", dest="train_m", type=int, default=1000)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)

    ap.add_argument("--betas", type=str, default="", help="comma-separated betas (override default grid)")
    ap.add_argument("--tick-mode", type=str, default="sparse", choices=["sparse","all"], help="x-axis beta tick labels")

    # NEW: plot target (column vs full-width)
    ap.add_argument("--fig-target", type=str, default="col", choices=["col", "full"],
                    help="col: RevTeX single-column-ready PDFs, full: figure*-ready PDFs")
    args = ap.parse_args()

    set_style(args.fig_target)
    outdir = ensure_outdir(args.outdir)

    betas = parse_betas(args.betas)
    seeds = [args.seed0 + i for i in range(args.k)]

    cfg = TrainCfg(
        n=args.n,
        arch=args.arch,
        layers=args.layers,
        steps=args.steps,
        lr=args.lr,
        sigma=args.sigma,
        num_alpha=args.num_alpha,
        train_m=args.train_m,
    )

    rows = []
    print(f"[Config] arch={args.arch} layers={args.layers} n={args.n} steps={args.steps} train_m={args.train_m} k={args.k}")
    print(f"[Plot] fig_target={args.fig_target}")
    print(f"[Betas] {betas}")

    for b in betas:
        kl_list = []
        H_list = []
        for s in seeds:
            out = train_one_beta_seed(b, s, cfg)
            rows.append(out)
            kl_list.append(out["kl"])
            H_list.append(out["entropy_bits"])
        print(f"beta={b:>4.2f} | KL mean={onp.mean(kl_list):.4g} std={onp.std(kl_list):.3g} | H={onp.mean(H_list):.2f} bits")

    beta_to_kl: Dict[float, List[float]] = {b: [] for b in betas}
    beta_to_H: Dict[float, List[float]] = {b: [] for b in betas}
    for r in rows:
        beta_to_kl[r["beta"]].append(r["kl"])
        beta_to_H[r["beta"]].append(r["entropy_bits"])

    kl_mean = onp.array([onp.mean(beta_to_kl[b]) for b in betas], dtype=float)
    kl_std  = onp.array([onp.std(beta_to_kl[b])  for b in betas], dtype=float)
    ent     = onp.array([onp.mean(beta_to_H[b])  for b in betas], dtype=float)

    with open(os.path.join(outdir, "beta_sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "arch": args.arch,
                "layers": args.layers,
                "n": args.n,
                "steps": args.steps,
                "lr": args.lr,
                "sigma": args.sigma,
                "num_alpha": args.num_alpha,
                "train_m": args.train_m,
                "k": args.k,
                "seed0": args.seed0,
                "betas": betas,
                "tick_mode": args.tick_mode,
                "fig_target": args.fig_target,
            },
            "rows": rows,
            "aggregate": {
                "beta": betas,
                "kl_mean": kl_mean.tolist(),
                "kl_std": kl_std.tolist(),
                "entropy_bits": ent.tolist(),
            }
        }, f, indent=2)

    csv_path = os.path.join(outdir, "beta_sweep_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["beta","seed","kl","entropy_bits","n_params"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "beta": r["beta"],
                "seed": r["seed"],
                "kl": r["kl"],
                "entropy_bits": r["entropy_bits"],
                "n_params": r["n_params"],
            })

    plot_beta_sweep(
        betas, kl_mean, kl_std, ent,
        outdir=outdir, n=args.n, tick_mode=args.tick_mode, fig_target=args.fig_target
    )
    print(f"\nDone. Outputs in: {os.path.abspath(outdir)}")
    print(f"[Tip] LaTeX: if --fig-target col => include with width=\\columnwidth; if full => width=\\textwidth (figure*).")

if __name__ == "__main__":
    main()