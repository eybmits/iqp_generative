#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topological Ablation Study in IQP Generative Models (Holy Trinity)
=================================================================
"Red & Black Edition" - Strict adherence to the 'Hero' publication style.

Description:
  Compares different circuit topologies (A, B, C, D, E) on the Parity Problem.
  Generates publication-ready plots comparing convergence, pareto efficiency,
  and depth scaling.

Visual Style:
  - Hero Model (Arch D): Bold Red (#D62728).
  - Competitors (A,B,C,E): Grayscale with distinct linestyles/markers.
  - Fonts: Serif (DejaVu Serif / Times), specific sizes for columns.

Usage:
  python ablation_red_black.py --outdir results_ablation --arches A,B,C,D,E --layers 1,2,3 --k 5 --fig-target col

Outputs (<outdir>/):
  - curves/       (Dynamics per layer)
  - summary/      (Pareto fronts, Depth analysis)
  - runs/         (Raw JSON data)
"""

import os
import json
import math
import argparse
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress PennyLane warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pennylane")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

import numpy as onp
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as np


# ------------------------------------------------------------------------------
# 1) Journal-Aware Style Engine
# ------------------------------------------------------------------------------

# Colors: D is Red, others are Grayscale variants
COLORS = {
    "hero":     "#D62728",   # Deep Red (Arch D)
    "black":    "#222222",   # Almost Black (Arch C)
    "dark":     "#444444",   # Dark Gray (Arch A)
    "mid":      "#777777",   # Mid Gray (Arch B)
    "light":    "#999999",   # Light Gray (Arch E)
    "grid":     "#B0B0B0",
}

# Specific styles for each architecture to ensure D stands out
ARCH_STYLE = {
    "A": dict(color=COLORS["dark"],  marker="o", linestyle="--", label="A: Local Ring"),
    "B": dict(color=COLORS["mid"],   marker="s", linestyle="-.", label="B: High-Order"),
    "C": dict(color=COLORS["black"], marker="^", linestyle=":",  label="C: Ext. Ring"),
    "D": dict(color=COLORS["hero"],  marker="D", linestyle="-",  label="D: Hybrid (Hero)"),
    "E": dict(color=COLORS["light"], marker="X", linestyle="--", label="E: All-to-All"),
}

# Standard widths for columns (Nature/RevTeX)
COL_W  = 3.37  # single-column width in inches
FULL_W = 6.95  # two-column width in inches

def fig_size(fig_target: str, aspect_ratio: float = 0.75) -> Tuple[float, float]:
    """
    Calculates strict figure dimensions.
    aspect_ratio: Height / Width (default 0.75 is 4:3)
    """
    if fig_target not in ("col", "full"):
        raise ValueError("fig_target must be 'col' or 'full'")
    
    width = COL_W if fig_target == "col" else FULL_W
    height = width * aspect_ratio
    return (width, height)

def set_style(fig_target: str = "col"):
    """
    Configures Matplotlib for publication quality.
    """
    if fig_target == "col":
        base = 8
        plt.rcParams.update({
            "font.size": base,
            "axes.labelsize": base + 1,
            "axes.titlesize": base + 1,
            "legend.fontsize": base - 1,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.6,
        })
    else:
        # Full width (Poster/Slides/Big Figure)
        base = 12
        plt.rcParams.update({
            "font.size": base,
            "axes.labelsize": base + 2,
            "axes.titlesize": base + 2,
            "legend.fontsize": base,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "grid.linewidth": 1.0,
        })

    # Global Aesthetic
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.color": COLORS["grid"],
        "legend.frameon": False,     # Cleaner legend
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # Editable text in PDFs
        "ps.fonttype": 42,
    })

def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------
# 2) Physics & Logic (IQP / Parity)
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
    
    # Softmax stable
    m = onp.max(logits[support])
    unnorm = onp.zeros(N, dtype=onp.float64)
    unnorm[support] = onp.exp(logits[support] - m)
    p_star = unnorm / unnorm.sum()
    return p_star.astype(onp.float64), support, scores.astype(onp.float64)

def topk_mask(scores: onp.ndarray, support: onp.ndarray, frac: float = 0.05) -> onp.ndarray:
    valid = onp.where(support)[0]
    k = max(1, int(onp.floor(frac * valid.size)))
    valid_scores = scores[valid]
    top_indices = valid[onp.argsort(-valid_scores)[:k]]
    mask = onp.zeros_like(support, dtype=bool)
    mask[top_indices] = True
    return mask

def kl_div(p: onp.ndarray, q: onp.ndarray, eps: float = 1e-12) -> float:
    q = onp.clip(q, eps, 1.0)
    p = onp.clip(p, 0.0, 1.0)
    return float(onp.sum(onp.where(p > 0, p * (onp.log(p + eps) - onp.log(q + eps)), 0.0)))

def expected_unique_set(probs: onp.ndarray, mask: onp.ndarray, Q: int) -> float:
    pS = probs[mask].astype(onp.float64)
    if pS.size == 0: return 0.0
    return float(onp.sum(1.0 - onp.power(1.0 - pS, Q)))

def p_sigma(sigma: float) -> float:
    if sigma <= 0.0: return 0.5
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma**2)))

def sample_alphas(n: int, sigma: float, K: int, seed: int) -> onp.ndarray:
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas: onp.ndarray, n: int) -> onp.ndarray:
    N = 2 ** n
    xs = onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)
    # alphas: (K, n), xs.T: (n, N) -> (K, N)
    mod_vals = (alphas @ xs.T) % 2
    return onp.where(mod_vals == 0, 1.0, -1.0).astype(onp.float64)


# ------------------------------------------------------------------------------
# 3) Circuit Topologies
# ------------------------------------------------------------------------------

def get_iqp_topology(n: int, arch: str) -> Tuple[List[Tuple], List[Tuple]]:
    pairs, quads = [], []
    def clean(l): return sorted(list(set(tuple(sorted(x)) for x in l)))
    
    # A: Local 1D Ring (ZZ nearest neighbor)
    if arch in ["A", "B", "C", "D"]:
        pairs.extend([(i, (i + 1) % n) for i in range(n)])
    
    # C: Extended Ring (ZZ next-nearest)
    if arch in ["C", "D"]:
        pairs.extend([(i, (i + 2) % n) for i in range(n)])
    
    # B: High Order Ring (ZZZZ)
    if arch in ["B", "D"]:
        quads.extend([(i, (i + 1) % n, (i + 2) % n, (i + 3) % n) for i in range(n)])
    
    # E: All-to-All (ZZ only)
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

def param_count(n: int, arch: str, layers: int) -> int:
    p, q = get_iqp_topology(n, arch)
    return (len(p) + len(q)) * layers

def cnot_proxy(n: int, arch: str, layers: int) -> int:
    p, q = get_iqp_topology(n, arch)
    # Approx: ZZ -> 2 CNOTs, ZZZZ -> 6 CNOTs
    return layers * (2 * len(p) + 6 * len(q))


# ------------------------------------------------------------------------------
# 4) Training Loop
# ------------------------------------------------------------------------------

@dataclass
class TrainCfg:
    n: int = 12
    beta: float = 0.9
    steps: int = 600
    lr: float = 0.05
    sigma: float = 1.0
    num_alpha: int = 512
    train_m: int = 1000
    eval_every: int = 50
    diversity_Q: int = 5000

def train_one(arch, layers, seed, cfg, p_star, good_mask, P_mat, z_data) -> Dict:
    dev = qml.device("default.qubit", wires=cfg.n)
    pairs, quads = get_iqp_topology(cfg.n, arch)
    n_params = (len(pairs) + len(quads)) * layers

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(cfg.n), pairs, quads, layers=layers)
        return qml.probs(wires=range(cfg.n))

    P_tensor = np.array(P_mat, requires_grad=False)
    z_tensor = np.array(z_data, requires_grad=False)
    
    rng = onp.random.default_rng(seed)
    W = np.array(0.01 * rng.standard_normal(n_params), requires_grad=True)
    opt = qml.AdamOptimizer(cfg.lr)

    history = {"step": [], "loss": [], "kl": [], "top5": []}

    def loss_fn(w):
        q = circuit(w)
        return np.mean((z_tensor - P_tensor @ q) ** 2)

    for t in range(1, cfg.steps + 1):
        W, loss_val = opt.step_and_cost(loss_fn, W)
        
        if t % cfg.eval_every == 0 or t == 1 or t == cfg.steps:
            q_val = onp.clip(onp.array(circuit(W), dtype=onp.float64), 1e-15, 1.0)
            q_val /= q_val.sum()
            
            history["step"].append(t)
            history["loss"].append(float(loss_val))
            history["kl"].append(kl_div(p_star, q_val))
            history["top5"].append(float(q_val[good_mask].sum()))

    q_final = onp.clip(onp.array(circuit(W), dtype=onp.float64), 1e-15, 1.0)
    q_final /= q_final.sum()
    div = expected_unique_set(q_final, good_mask, cfg.diversity_Q)

    return {
        "arch": arch, "layers": int(layers), "seed": int(seed),
        "n_params": int(n_params), 
        "cnot_proxy": int(cnot_proxy(cfg.n, arch, layers)),
        "final": {
            "loss": float(history["loss"][-1]), 
            "kl": float(kl_div(p_star, q_final)),
            "top5": float(q_final[good_mask].sum()), 
            "div": float(div),
        },
        "history": history,
    }


# ------------------------------------------------------------------------------
# 5) Plotting: The Red & Black Edition
# ------------------------------------------------------------------------------

def plot_curves_by_layer(all_runs, layers_list, outdir, arches, fig_target):
    curves_dir = outdir / "curves"
    curves_dir.mkdir(exist_ok=True)
    
    metrics = [
        ("loss", r"MMD Loss $\mathcal{L}$", True),
        ("kl",   r"KL$(p^* \Vert q_\theta)$", False),
        ("top5", r"Top-5% Mass $q_\theta(G)$", False),
    ]

    for L in layers_list:
        for key, ylabel, logy in metrics:
            fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)
            
            # Plot reference line for Top5
            if key == "top5":
                ax.axhline(1.0, color=COLORS["black"], linestyle="--", linewidth=1.0, alpha=0.4)

            # Sort so D is last (on top)
            sorted_arches = sorted(arches, key=lambda x: 1 if x == "D" else 0)

            for arch in sorted_arches:
                runs = all_runs.get((arch, L), [])
                if not runs: continue
                
                # Compute Mean/Std
                steps = onp.array(runs[0]["history"]["step"])
                vals = onp.array([r["history"][key] for r in runs])
                mu = onp.nanmean(vals, axis=0)
                sd = onp.nanstd(vals, axis=0)

                st = ARCH_STYLE[arch]
                is_hero = (arch == "D")
                
                # Z-order: Hero on top
                z = 10 if is_hero else 5
                
                ax.plot(steps, mu, label=st["label"], color=st["color"], 
                        linestyle=st["linestyle"], marker=st["marker"],
                        markevery=max(1, len(steps)//6), zorder=z)
                
                fill_alpha = 0.15 if is_hero else 0.05
                ax.fill_between(steps, mu-sd, mu+sd, color=st["color"], alpha=fill_alpha, zorder=z-1, edgecolor=None)

            ax.set_xlabel("Training Steps")
            ax.set_ylabel(ylabel)
            if logy: ax.set_yscale("log")
            
            # Smart Legend placement
            ax.legend(loc="best")
            
            fname = f"{key}_L{L}.pdf"
            fig.savefig(curves_dir / fname)
            plt.close(fig)

def plot_pareto(summary, outdir, fig_target):
    summ_dir = outdir / "summary"
    summ_dir.mkdir(exist_ok=True)

    def _pareto_front(pts):
        # pts list of (x, y). Find min y for x.
        # Simple non-dominated sort
        sorted_pts = sorted(pts, key=lambda x: x[0])
        front = []
        curr_min_y = float('inf')
        for x, y in sorted_pts:
            if y < curr_min_y:
                front.append((x, y))
                curr_min_y = y
        return front

    # Plots: Params vs KL, CNOT vs KL
    scenarios = [
        ("n_params", "kl_mean", r"Parameter Count $N_\theta$"),
        ("cnot_proxy", "kl_mean", r"Est. CNOT Cost"),
    ]

    for xkey, ykey, xlabel in scenarios:
        fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)
        
        # Collect all points to calculate pareto front
        all_pts = []
        for row in summary:
            all_pts.append((row[xkey], row[ykey]))
        
        # Plot Pareto Front line
        front = _pareto_front(all_pts)
        fx, fy = zip(*front)
        ax.plot(fx, fy, color=COLORS["black"], linestyle="--", alpha=0.5, label="Pareto Frontier", zorder=1)

        # Plot Scatter Points
        # Group by Arch to use legend
        existing_labels = set()
        for row in summary:
            arch = row["arch"]
            st = ARCH_STYLE[arch]
            lbl = st["label"] if arch not in existing_labels else None
            existing_labels.add(arch)
            
            is_hero = (arch == "D")
            size = 60 if is_hero else 35
            edge = "black" if is_hero else "white"
            z = 10 if is_hero else 5

            ax.scatter(row[xkey], row[ykey], c=st["color"], marker=st["marker"], 
                       s=size, edgecolors=edge, linewidth=0.8, zorder=z, label=lbl)
            
            # Annotate Layer
            ax.text(row[xkey], row[ykey], f" L{row['layers']}", fontsize=6, 
                    color=st["color"], va="bottom", ha="left")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Final KL$(p^* \Vert q_\theta)$")
        ax.legend()
        
        fig.savefig(summ_dir / f"pareto_{xkey}.pdf")
        plt.close(fig)

def plot_depth_analysis(summary, outdir, fig_target, arches, layers_list):
    summ_dir = outdir / "summary"
    summ_dir.mkdir(exist_ok=True)
    
    # We want Mean +/- Std vs Depth for each Arch
    metrics = [("kl_mean", "kl_std", "Final KL"), ("div_mean", "div_std", "Diversity (Unique)")]

    width = 0.12
    # Offset x-axis for clarity
    x_offsets = {a: (i - len(arches)/2)*width for i, a in enumerate(arches)}

    for m_mean, m_std, m_lbl in metrics:
        fig, ax = plt.subplots(figsize=fig_size(fig_target), constrained_layout=True)
        
        for arch in arches:
            st = ARCH_STYLE[arch]
            xs, ys, yerrs = [], [], []
            
            for L in layers_list:
                # Find row
                rows = [r for r in summary if r["arch"] == arch and r["layers"] == L]
                if not rows: continue
                r = rows[0]
                
                xs.append(L + x_offsets[arch])
                ys.append(r[m_mean])
                yerrs.append(r[m_std])
            
            ax.errorbar(xs, ys, yerr=yerrs, fmt=st["marker"], color=st["color"], 
                        label=st["label"], capsize=3, linestyle=st["linestyle"])

        ax.set_xticks(layers_list)
        ax.set_xlabel("Layers $L$")
        ax.set_ylabel(m_lbl)
        ax.legend()
        
        fig.savefig(summ_dir / f"depth_{m_mean.split('_')[0]}.pdf")
        plt.close(fig)


# ------------------------------------------------------------------------------
# 6) Main Controller
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Holy Trinity Ablation Study (Red & Black)")
    parser.add_argument("--outdir", type=str, default="ablation_results")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--arches", type=str, default="A,B,C,D,E")
    parser.add_argument("--layers", type=str, default="1,2,3")
    parser.add_argument("--k", type=int, default=5, help="Number of seeds")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fig-target", type=str, default="col", choices=["col", "full"])
    parser.add_argument("--reuse", action="store_true", help="Skip training if json exists")
    
    # Fixed hyperparameters for simplicity
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.05)
    
    args = parser.parse_args()

    # 1. Setup
    outdir = ensure_outdir(args.outdir)
    runs_dir = ensure_outdir(outdir / "runs")
    set_style(args.fig_target)

    arches_list = [x.strip() for x in args.arches.split(",")]
    layers_list = [int(x) for x in args.layers.split(",")]
    seeds = range(args.k)

    # 2. Prepare Target
    print(f"[Info] Building Target n={args.n}...")
    p_star, support, scores = build_target_distribution(args.n, args.beta)
    good_mask = topk_mask(scores, support)
    
    # 3. Train / Load
    all_runs = {} # Key: (arch, layer) -> List[Dict]
    
    cfg = TrainCfg(n=args.n, beta=args.beta, steps=args.steps, lr=args.lr)

    # Pre-generate Data Distribution (Fixed per seed)
    data_cache = {}

    for seed in seeds:
        # Data generation (MMD target)
        if seed not in data_cache:
            rng_idx = onp.random.default_rng(1000 + seed)
            idxs = rng_idx.choice(len(p_star), size=cfg.train_m, p=p_star)
            p_emp = onp.bincount(idxs, minlength=2**cfg.n) / cfg.train_m
            
            alphas = sample_alphas(cfg.n, cfg.sigma, cfg.num_alpha, seed=2000 + seed)
            P_mat = build_parity_matrix(alphas, cfg.n)
            z_data = P_mat @ p_emp
            data_cache[seed] = (P_mat, z_data)
        
        P_mat, z_data = data_cache[seed]

        for arch, L in itertools.product(arches_list, layers_list):
            fname = f"run_A{arch}_L{L}_S{seed}.json"
            fpath = runs_dir / fname
            
            if args.reuse and fpath.exists():
                with open(fpath, "r") as f:
                    res = json.load(f)
            else:
                print(f"Training A={arch} L={L} Seed={seed}...")
                res = train_one(arch, L, seed, cfg, p_star, good_mask, P_mat, z_data)
                with open(fpath, "w") as f:
                    json.dump(res, f, indent=2)
            
            if (arch, L) not in all_runs: all_runs[(arch, L)] = []
            all_runs[(arch, L)].append(res)

    # 4. Summarize
    summary = []
    for (arch, L), runs in all_runs.items():
        # Average metrics
        kls = [r["final"]["kl"] for r in runs]
        divs = [r["final"]["div"] for r in runs]
        top5s = [r["final"]["top5"] for r in runs]
        
        summary.append({
            "arch": arch, "layers": L,
            "n_params": runs[0]["n_params"],
            "cnot_proxy": runs[0]["cnot_proxy"],
            "kl_mean": onp.mean(kls), "kl_std": onp.std(kls),
            "div_mean": onp.mean(divs), "div_std": onp.std(divs),
            "top5_mean": onp.mean(top5s), "top5_std": onp.std(top5s),
        })
    
    with open(outdir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 5. Plot
    print("[Info] Generating Plots...")
    plot_curves_by_layer(all_runs, layers_list, outdir, arches_list, args.fig_target)
    plot_pareto(summary, outdir, args.fig_target)
    plot_depth_analysis(summary, outdir, args.fig_target, arches_list, layers_list)
    
    print(f"[Done] Results saved to {outdir}")

if __name__ == "__main__":
    main()