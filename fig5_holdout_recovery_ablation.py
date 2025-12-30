#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQP-QCBM Holdout Ablation Study (Architectures A–E)
==================================================

Version 8 (Maximum Altitude):
- "Orbit High": Labels start very high above bars (base_pad=0.18).
- "Tower Stacking": Vertical stacking step increased to 0.18.
- Plot ceiling raised to 2.0 to ensure fit.

Outputs
-------
outdir/
  holdout_strings.txt
  runs/run_archA_seed42.json ...
  fig5_holdout_recovery_curves.pdf
  fig5_holdout_recovery_budgets.pdf  (FINAL)
  holdout_speed_summary.csv

Usage
-----
python holdout_ablation_final.py \
  --archs A B C D E \
  --layers 1 \
  --holdout-k 20 \
  --seeds 42 \
  --outdir ablation_holdout \
  --fig-target full
"""

import os
import json
import math
import argparse
import itertools
import warnings
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pennylane")
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")

import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

import pennylane as qml
from pennylane import numpy as np


# ------------------------------------------------------------------------------
# 0) Journal-aware sizes + style
# ------------------------------------------------------------------------------

COL_W  = 3.37  # single-column
FULL_W = 6.95  # two-column (figure*)

COLORS = {
    "black": "#111111",
    "gray":  "#666666",
    "light": "#DDDDDD",
    "hero":  "#D62728",   # deep red
}

ARCH_ORDER = ["A", "B", "C", "D", "E"]

ARCH_HATCH = {
    "A": "///",
    "B": "\\\\",
    "C": "xx",
    "D": None,
    "E": "..",
}

def fig_size(fig_target: str, h_col: float, h_full: float) -> Tuple[float, float]:
    return (COL_W, h_col) if fig_target == "col" else (FULL_W, h_full)

def set_style(fig_target: str = "full"):
    base = 8 if fig_target == "col" else 12
    lw = 1.6 if fig_target == "col" else 2.5
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": base,
        "axes.labelsize": base + 1 if fig_target == "col" else 16,
        "axes.titlesize": base + 1 if fig_target == "col" else 16,
        "legend.fontsize": base - 1 if fig_target == "col" else 12,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "white",
        "lines.linewidth": lw,
        "axes.linewidth": 0.8,
        "xtick.direction": "out" if fig_target=="col" else "in",
        "ytick.direction": "out" if fig_target=="col" else "in",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def _arch_linestyle(fig_target: str) -> Dict[str, Dict[str, Any]]:
    s = 0.78 if fig_target == "col" else 1.00
    return {
        "A": dict(color=COLORS["black"], ls="-",  lw=2.7 * s, alpha=0.98),
        "B": dict(color=COLORS["black"], ls="--", lw=2.7 * s, alpha=0.98),
        "C": dict(color=COLORS["black"], ls=":",  lw=2.9 * s, alpha=0.98),
        "D": dict(color=COLORS["hero"],  ls="-",  lw=4.0 * s, alpha=1.00),
        "E": dict(color=COLORS["black"], ls="-.", lw=2.9 * s, alpha=0.98),
    }

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
# 1) Robust "Staircase" Label Placement (MAX HEIGHT)
# ------------------------------------------------------------------------------

def _place_all_percent_labels_panelB(
    ax,
    group_items: Dict[int, List[Tuple[str, float, float]]]
):
    """
    Places labels extremely high above bars.
    """
    base_fs = float(plt.rcParams.get("font.size", 8))
    fs = max(7, int(base_fs - 1))
    
    # --- ADJUSTED PARAMETERS FOR "ORBIT HEIGHT" ---
    base_pad = 0.3       # Start extremely high (was 0.12)
    vertical_step = 0.15  # Huge stacking jump (was 0.15)
    collision_x = 0.8    # Wide check
    collision_y = 0.10    # Tall check

    for j, items in group_items.items():
        items = [(a, xc, h) for (a, xc, h) in items if onp.isfinite(h) and h > 0.01]
        items.sort(key=lambda t: t[1])
        if not items: continue

        placed_labels = [] # List of (x, y)

        for (arch, xc, h) in items:
            txt = f"{100*h:.0f}%"
            is_hero = (arch == "D")
            
            # Start position
            y_cand = h + base_pad
            
            # Collision Resolution
            collision_found = True
            while collision_found:
                collision_found = False
                for (px, py) in placed_labels:
                    dx = abs(xc - px)
                    dy = abs(y_cand - py)
                    
                    if dx < collision_x and dy < collision_y:
                        # Collision! Move up significantly
                        y_cand = py + vertical_step
                        collision_found = True
                        break 

            placed_labels.append((xc, y_cand))
            
            bbox_style = dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.90)
            
            ax.text(
                xc, y_cand, txt,
                ha="center", va="center",
                fontsize=fs,
                fontweight="bold" if is_hero else "normal",
                color=COLORS["hero"] if is_hero else COLORS["black"],
                bbox=bbox_style,
                zorder=60
            )


# ------------------------------------------------------------------------------
# 2) Physics Logic (Unchanged)
# ------------------------------------------------------------------------------

def int2bits(k: int, n: int) -> onp.ndarray:
    return onp.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=onp.int8)

def bits_str(bits: onp.ndarray) -> str:
    return "".join("1" if int(b) else "0" for b in bits)

def parity_even(bits: onp.ndarray) -> bool:
    return (int(onp.sum(bits)) % 2) == 0

def longest_zero_run_between_ones(bits: onp.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2: return 0
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
    return p_star, support, scores

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

def _min_hamming_to_set(bit_vec, sel_bits):
    if sel_bits.shape[0] == 0: return bit_vec.shape[0]
    return int(onp.min(onp.sum(sel_bits != bit_vec[None, :], axis=1)))

def select_holdout_smart(p_star, good_mask, bits_table, m_train, holdout_k, pool_size, seed):
    if holdout_k <= 0: return onp.zeros_like(good_mask, dtype=bool)
    good_idxs = onp.where(good_mask)[0]
    
    taus = [1.0/max(1, m_train), 0.5/max(1, m_train), 0.0]
    cand = None
    for tau in taus:
        cand = good_idxs[p_star[good_idxs] >= tau]
        if cand.size >= holdout_k: break
    
    cand = cand[onp.argsort(-p_star[cand])]
    pool = cand[:min(pool_size, cand.size)]

    selected = [int(pool[0])]
    selected_bits = onp.vstack([onp.zeros((0, bits_table.shape[1]), dtype=onp.int8), bits_table[selected[-1]]])

    while len(selected) < holdout_k and len(selected) < pool.size:
        best_idx = None
        best_d = -1
        for idx in pool:
            idx = int(idx)
            if idx in selected: continue
            d = _min_hamming_to_set(bits_table[idx], selected_bits)
            if d > best_d:
                best_d, best_idx = d, idx
            elif d == best_d and best_idx is not None:
                if p_star[idx] > p_star[best_idx]: best_idx = idx
        if best_idx is None: break
        selected.append(int(best_idx))
        selected_bits = onp.vstack([selected_bits, bits_table[best_idx]])

    holdout = onp.zeros_like(good_mask, dtype=bool)
    holdout[onp.array(selected, dtype=int)] = True
    return holdout

def save_holdout_list(holdout_mask, bits_table, p_star, scores, outdir):
    idxs = onp.where(holdout_mask)[0]
    sorted_idxs = idxs[onp.argsort(-p_star[idxs])] if idxs.size > 0 else idxs
    path = os.path.join(outdir, "holdout_strings.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Holdout Strings (k={len(idxs)})\n")
        for idx in sorted_idxs:
            b_str = bits_str(bits_table[int(idx)])
            f.write(f"{int(idx):<8d} {b_str:<16s} {scores[int(idx)]:<8.1f} {p_star[int(idx)]:.6e}\n")


# ------------------------------------------------------------------------------
# 3) Circuit & Math
# ------------------------------------------------------------------------------

def get_iqp_topology(n, arch):
    pairs, quads = [], []
    if arch in ["A", "B", "C", "D"]:
        pairs.extend([tuple(sorted((i, (i + 1) % n))) for i in range(n)])
    if arch in ["C", "D"]:
        pairs.extend([tuple(sorted((i, (i + 2) % n))) for i in range(n)])
    if arch in ["B", "D"]:
        quads.extend([tuple(sorted((i, (i + 1) % n, (i + 2) % n, (i + 3) % n))) for i in range(n)])
    if arch == "E":
        pairs = list(itertools.combinations(range(n), 2))
    return sorted(list(set(pairs))), sorted(list(set(quads)))

def iqp_circuit(W, wires, pairs, quads, layers=1):
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

def p_sigma(sigma):
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma ** 2))) if sigma > 0 else 0.5

def sample_alphas(n, sigma, K, seed):
    rng = onp.random.default_rng(seed)
    return rng.binomial(1, p_sigma(sigma), size=(K, n)).astype(onp.int8)

def build_parity_matrix(alphas, n):
    N = 2 ** n
    xs = onp.array([int2bits(i, n) for i in range(N)], dtype=onp.int8)
    return onp.where(((alphas @ xs.T) % 2) == 0, 1.0, -1.0).astype(onp.float64)

def expected_unique_from_probs(p_vec: onp.ndarray, Q_vals: onp.ndarray) -> onp.ndarray:
    if p_vec.size == 0: return onp.zeros_like(Q_vals, dtype=onp.float64)
    p = onp.clip(p_vec.astype(onp.float64), 0.0, 1.0)[:, None]
    Q = Q_vals[None, :].astype(onp.int64)
    return onp.sum(1.0 - onp.power(1.0 - p, Q), axis=0)

def recovery_fraction(p_vec: onp.ndarray, Q: int) -> float:
    H = int(p_vec.size)
    if H == 0: return 0.0
    return float(expected_unique_from_probs(p_vec, onp.array([int(Q)]))[0] / H)

def expected_recovery_curve(p_vec: onp.ndarray, Q_grid: onp.ndarray) -> onp.ndarray:
    H = int(p_vec.size)
    if H == 0: return onp.zeros_like(Q_grid, dtype=onp.float64)
    return expected_unique_from_probs(p_vec, Q_grid) / H

def min_Q_for_recovery(p_vec: onp.ndarray, target_frac: float, Q_cap: int = 2_000_000) -> float:
    H = int(p_vec.size)
    if H == 0: return float("nan")
    max_reachable = float(onp.count_nonzero(onp.array(p_vec) > 0.0) / H)
    if max_reachable + 1e-12 < target_frac: return float("inf")
    hi = 1
    while hi < Q_cap and recovery_fraction(p_vec, hi) < target_frac: hi *= 2
    if recovery_fraction(p_vec, hi) < target_frac: return float("inf")
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if recovery_fraction(p_vec, mid) >= target_frac: hi = mid
        else: lo = mid
    return float(hi)


# ------------------------------------------------------------------------------
# 4) Config + Prep
# ------------------------------------------------------------------------------

@dataclass
class Config:
    n: int = 14
    layers: int = 1
    beta: float = 0.9
    steps: int = 600
    lr: float = 0.05
    sigma: float = 1.0
    num_alpha: int = 512
    train_m: int = 1000
    eval_every: int = 50
    holdout_k: int = 20
    holdout_pool: int = 400
    good_frac: float = 0.05
    holdout_seed: int = 123
    data_seed: int = 7
    feature_seed: int = 222
    Q_list: Tuple[int, ...] = (1000, 5000, 10000)
    Qmax: int = 10000
    outdir: str = "ablation_holdout"
    fig_target: str = "full"

def prepare_shared(cfg: Config) -> Dict[str, Any]:
    outdir = ensure_outdir(cfg.outdir)
    ensure_outdir(os.path.join(outdir, "runs"))
    N = 2 ** cfg.n
    bits_table = make_bits_table(cfg.n)
    p_star, support, scores = build_target_distribution(cfg.n, cfg.beta)
    good_mask = topk_mask(scores, support, frac=cfg.good_frac)

    holdout_mask = select_holdout_smart(p_star, good_mask, bits_table, cfg.train_m, cfg.holdout_k, cfg.holdout_pool, cfg.holdout_seed)
    holdout_idxs = onp.where(holdout_mask)[0].astype(int)
    save_holdout_list(holdout_mask, bits_table, p_star, scores, outdir)

    p_train = p_star.copy()
    if holdout_idxs.size > 0:
        p_train[holdout_idxs] = 0.0
        p_train /= p_train.sum()

    idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.data_seed)
    emp_dist = empirical_dist(idxs_train, N)
    alphas = sample_alphas(cfg.n, cfg.sigma, cfg.num_alpha, seed=cfg.feature_seed)
    P_mat = build_parity_matrix(alphas, cfg.n)
    z_data = onp.dot(P_mat, emp_dist).astype(onp.float64)

    return dict(outdir=outdir, N=N, bits_table=bits_table, p_star=p_star,
                holdout_idxs=holdout_idxs, P_mat=P_mat, z_data=z_data, good_mask=good_mask)


# ------------------------------------------------------------------------------
# 5) Training
# ------------------------------------------------------------------------------

def train_one_architecture(cfg: Config, shared: Dict[str, Any], arch: str, seed: int) -> Dict[str, Any]:
    outdir = shared["outdir"]
    P_mat, z_data = shared["P_mat"], shared["z_data"]
    holdout_idxs = shared["holdout_idxs"]

    dev = qml.device("default.qubit", wires=cfg.n)
    pairs, quads = get_iqp_topology(cfg.n, arch)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit(W, range(cfg.n), pairs, quads, layers=cfg.layers)
        return qml.probs(wires=range(cfg.n))

    P_mat_tensor = np.array(P_mat, requires_grad=False)
    z_data_tensor = np.array(z_data, requires_grad=False)
    num_params = (len(pairs) + len(quads)) * cfg.layers
    rng = onp.random.default_rng(seed)
    W = np.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(cfg.lr)

    print(f"\n--- Training arch={arch} | seed={seed} ---")
    for t in range(1, cfg.steps + 1):
        def loss_fn(w):
            q = circuit(w)
            return np.mean((z_data_tensor - P_mat_tensor @ q) ** 2)
        W, l_val = opt.step_and_cost(loss_fn, W)
        if t % 100 == 0:
            print(f"Step {t}/{cfg.steps} | Loss {l_val:.2e}")

    q_final = onp.clip(onp.array(circuit(W), dtype=onp.float64), 0.0, 1.0)
    q_final /= max(1e-15, float(q_final.sum()))
    holdout_probs_final = q_final[holdout_idxs].astype(onp.float64) if holdout_idxs.size > 0 else onp.array([])

    run = {
        "arch": arch, "seed": int(seed), "params": int(num_params),
        "holdout_probs": holdout_probs_final.tolist()
    }
    
    with open(os.path.join(shared["outdir"], "runs", f"run_arch{arch}_seed{seed}.json"), "w") as f:
        json.dump(run, f)
    return run


# ------------------------------------------------------------------------------
# 6) Plot: Curves
# ------------------------------------------------------------------------------

def plot_holdout_recovery_curves(cfg: Config, agg: Dict[str, Any], outdir: str) -> None:
    Q_grid = agg["Q_grid"]
    curves = agg["curves"]
    target_curve = agg["target_curve"]
    target_speed = agg["target_speed"]
    speed = agg["speed"]

    st_map = _arch_linestyle(cfg.fig_target)
    # Changed size to match budgets plot (3.1, 4.2)
    fig, ax = plt.subplots(figsize=fig_size(cfg.fig_target, 3.1, 4.2), constrained_layout=True)

    # Target
    ax.plot(Q_grid, target_curve, color=COLORS["black"], lw=(1.8 if cfg.fig_target == "col" else 2.4),
            ls=":", label=r"Target $p^*$", zorder=6)

    # Archs
    for arch in ARCH_ORDER:
        if arch not in curves: continue
        st = st_map[arch]
        m, s = curves[arch]["mean"], curves[arch]["sem"]
        ax.plot(Q_grid, m, color=st["color"], lw=st["lw"], ls=st["ls"], alpha=st["alpha"],
                label=f"Arch {arch}", zorder=12 if arch == "D" else 8)
        if onp.any(s > 0):
            ax.fill_between(Q_grid, onp.clip(m - s, 0, 1), onp.clip(m + s, 0, 1),
                            color=COLORS["black"], alpha=0.06, linewidth=0, zorder=2)

    for Q in cfg.Q_list:
        ax.axvline(Q, color=COLORS["light"], lw=1.0, ls="--", alpha=0.55, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(Q_grid.min(), Q_grid.max())
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Sampling Budget $Q$ (log)")
    ax.set_ylabel(r"Holdout Recovery  $U_H(Q)/|H|$")
    # Title removed per request
    ax.legend(loc="lower right")

    # Speed text box constructed with VPacker to allow individual red line for Arch D
    box_fs = max(7, int(plt.rcParams.get("font.size", 8) - 1))
    
    pack_children = []
    # Header
    pack_children.append(TextArea(r"$Q_{80}$ (samples for 80% recovery)", textprops=dict(color=COLORS["black"], fontsize=box_fs)))
    
    # Target
    t_val = int(target_speed['Q80']) if onp.isfinite(target_speed['Q80']) else '∞'
    pack_children.append(TextArea(f"Target: {t_val}", textprops=dict(color=COLORS["black"], fontsize=box_fs)))
    
    # Archs
    for arch in ARCH_ORDER:
        if arch not in speed: continue
        m80 = speed[arch]
        txt = f"{int(m80)}" if onp.isfinite(m80) else "∞"
        prefix = "Arch D: " if arch == "D" else f"Arch {arch}: "
        col = COLORS["hero"] if arch == "D" else COLORS["black"]
        pack_children.append(TextArea(prefix + txt, textprops=dict(color=col, fontsize=box_fs)))
        
    # Stack vertically
    box_packer = VPacker(children=pack_children, align="left", pad=0, sep=4)
    
    # Anchor to axes
    anchored_box = AnchoredOffsetbox(loc='upper left', child=box_packer, pad=0.4, frameon=True,
                                     bbox_to_anchor=(0.03, 0.97), bbox_transform=ax.transAxes, borderpad=0.0)
    anchored_box.patch.set_boxstyle("round,pad=0.3")
    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_alpha(0.92)
    anchored_box.patch.set_edgecolor(COLORS["light"])
    
    ax.add_artist(anchored_box)

    path = os.path.join(outdir, "fig5_holdout_recovery_curves.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[Saved] {path}")


# ------------------------------------------------------------------------------
# 7) Plot: Budgets (FINAL LAYOUT)
# ------------------------------------------------------------------------------
def plot_holdout_recovery_budgets(cfg: Config, agg: Dict[str, Any], outdir: str) -> None:
    bars = agg["bars"]
    target_bars = agg["target_bars"]
    Qs = list(cfg.Q_list)
    active_archs = [a for a in ARCH_ORDER if any(a in bars.get(Q, {}) for Q in Qs)]
    
    fig, ax = plt.subplots(figsize=fig_size(cfg.fig_target, 3.1, 4.2))

    group_gap = 2.2 
    x = onp.arange(len(Qs)) * group_gap
    bar_w = 0.13
    bar_sep = 0.28
    group_items = {j: [] for j in range(len(Qs))}

    for i, arch in enumerate(active_archs):
        vals, errs = [], []
        for Q in Qs:
            v = bars.get(Q, {}).get(arch, {}).get("mean", 0)
            e = bars.get(Q, {}).get(arch, {}).get("sem", 0)
            vals.append(v)
            errs.append(e)

        offs = (i - (len(active_archs) - 1) / 2) * bar_sep
        face, edge, hatch, alpha, z = (COLORS["hero"], COLORS["black"], "", 0.92, 7) if arch == "D" \
                                 else ("white", COLORS["black"], ARCH_HATCH.get(arch, "///"), 0.99, 6)
        
        rects = ax.bar(x + offs, vals, width=bar_w, color=face, edgecolor=edge, linewidth=1.0,
                       hatch=hatch, alpha=alpha, yerr=errs if any(e > 0 for e in errs) else None,
                       capsize=3.0, zorder=z)

        for j, r in enumerate(rects):
            h = float(r.get_height())
            xc = float(r.get_x() + r.get_width() / 2)
            group_items[j].append((arch, xc, h))

    # Target diamonds
    ax.plot(x, [target_bars[Q] for Q in Qs], marker="D", ms=(6 if cfg.fig_target == "col" else 8),
            color=COLORS["black"], ls="none", label="_nolegend_", zorder=20)

    # Full recovery line
    ax.axhline(1.0, color=COLORS["gray"], lw=1.4, ls="--", alpha=0.55, zorder=2)
    ax.text(0.985, 0.92, "Full recovery", transform=ax.transAxes, ha="right", va="center",
            fontsize=max(7, int(plt.rcParams.get("font.size", 8))), color=COLORS["gray"], style="italic",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.90), zorder=25)

    # Percent labels (unchanged)
    _place_all_percent_labels_panelB(ax, group_items)

    ax.set_xticks(x)
    fs_xt = 10 if cfg.fig_target == "col" else 14
    ax.set_xticklabels([f"Q={int(Q):,}".replace(",", " ") for Q in Qs], fontsize=fs_xt)
    ax.set_xlim(x[0] - 1.10, x[-1] + 1.10)

    # ✅ Keep headroom so the plot stays "like before"
    ax.set_ylim(0.0, 2.00)

    # ✅ But "stop" the y-axis labeling at 1.0 (ticks only up to 1.0)
    yt = onp.linspace(0.0, 1.0, 6)  # 0.0,0.2,...,1.0
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:.1f}" for t in yt])

    ax.set_xlabel(r"Sampling Budget $Q$")
    ax.set_ylabel("Recovered fraction")

    # Legend inside Top-Left (like your good-looking original)
    legend_handles = [Line2D([0], [0], marker="D", linestyle="None", markersize=7,
                             markerfacecolor=COLORS["black"], markeredgecolor=COLORS["black"], label=r"Target $p^*$")]
    for arch in active_archs:
        legend_handles.append(Patch(facecolor=COLORS["hero"] if arch=="D" else "white", 
                                    edgecolor=COLORS["black"], hatch=ARCH_HATCH.get(arch, "///"), 
                                    label=f"Arch {arch}"))
    
    fs_legend = 7 if cfg.fig_target == "col" else 10
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98),
              ncol=2, columnspacing=1.2, frameon=True, borderaxespad=0.0,
              fontsize=fs_legend)

    path = os.path.join(outdir, "fig5_holdout_recovery_budgets.pdf")
    plt.savefig(path)
    plt.close()
    print(f"[Saved] {path}")

# ------------------------------------------------------------------------------
# 8) Main & Aggregation
# ------------------------------------------------------------------------------

def aggregate_runs(cfg, shared, runs_by_arch):
    holdout_idxs = shared["holdout_idxs"]
    pH_star = shared["p_star"][holdout_idxs] if holdout_idxs.size > 0 else onp.array([])
    
    Q_grid = onp.unique(onp.logspace(0, math.log10(cfg.Qmax), 260).astype(int))
    Q_grid = Q_grid[Q_grid >= 1]
    
    target_curve = expected_recovery_curve(pH_star, Q_grid)
    target_bars = {Q: recovery_fraction(pH_star, int(Q)) for Q in cfg.Q_list}
    
    bars, curves, speed = {}, {}, {}

    for arch in ARCH_ORDER:
        runs = runs_by_arch.get(arch, [])
        if not runs: continue
        
        bar_stack = {Q: [] for Q in cfg.Q_list}
        curve_stack = []
        q80_stack = []
        
        for r in runs:
            pH = onp.array(r["holdout_probs"], dtype=onp.float64)
            curve_stack.append(expected_recovery_curve(pH, Q_grid))
            for Q in cfg.Q_list:
                bar_stack[Q].append(recovery_fraction(pH, int(Q)))
            q80_stack.append(min_Q_for_recovery(pH, 0.80))
            
        c_arr = onp.array(curve_stack)
        curves[arch] = {
            "mean": onp.mean(c_arr, axis=0),
            "sem": onp.std(c_arr, axis=0, ddof=1)/math.sqrt(c_arr.shape[0]) if c_arr.shape[0]>1 else onp.zeros_like(Q_grid)
        }
        
        for Q in cfg.Q_list:
            arr = onp.array(bar_stack[Q])
            bars.setdefault(Q, {})[arch] = {
                "mean": float(arr.mean()),
                "sem": float(arr.std(ddof=1)/math.sqrt(arr.size)) if arr.size > 1 else 0.0
            }
        
        q80arr = onp.array([x for x in q80_stack if onp.isfinite(x)])
        speed[arch] = float(q80arr.mean()) if q80arr.size > 0 else float("inf")

    return {
        "H": holdout_idxs.size, "Q_grid": Q_grid,
        "curves": curves, "bars": bars, "speed": speed,
        "target_curve": target_curve, "target_bars": target_bars,
        "target_speed": {"Q80": min_Q_for_recovery(pH_star, 0.80)}
    }

def save_speed_summary_csv(cfg: Config, runs_by_arch: Dict, agg: Dict, outdir: str) -> None:
    csv_path = os.path.join(outdir, "holdout_speed_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arch", "params", "Rec@Q", "rec_mean", "rec_sem", "Q80_mean"])
        for arch in ARCH_ORDER:
            runs = runs_by_arch.get(arch, [])
            if not runs: continue
            params = runs[0]["params"]
            q80m = agg["speed"][arch]
            for Q in cfg.Q_list:
                bm = agg["bars"].get(Q, {}).get(arch, {}).get("mean", 0)
                bs = agg["bars"].get(Q, {}).get(arch, {}).get("sem", 0)
                w.writerow([arch, params, Q, bm, bs, q80m])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=14)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--holdout-k", type=int, default=20)
    p.add_argument("--Q-list", type=int, nargs="+", default=[1000, 5000, 10000])
    p.add_argument("--archs", type=str, nargs="+", default=["A", "B", "C", "D", "E"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--outdir", type=str, default="ablation_holdout")
    p.add_argument("--fig-target", type=str, default="full", choices=["col", "full"])
    
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--num-alpha", type=int, default=512)
    p.add_argument("--train-m", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--holdout-pool", type=int, default=400)
    p.add_argument("--good-frac", type=float, default=0.05)
    p.add_argument("--holdout-seed", type=int, default=123)
    p.add_argument("--data-seed", type=int, default=7)
    p.add_argument("--feature-seed", type=int, default=222)
    p.add_argument("--Qmax", type=int, default=10000)

    args = p.parse_args()
    set_style(args.fig_target)
    
    cfg = Config(
        n=args.n, layers=args.layers, steps=args.steps,
        holdout_k=args.holdout_k, Q_list=tuple(args.Q_list),
        outdir=args.outdir, fig_target=args.fig_target,
        beta=args.beta, lr=args.lr, sigma=args.sigma, num_alpha=args.num_alpha,
        train_m=args.train_m, eval_every=args.eval_every, holdout_pool=args.holdout_pool,
        good_frac=args.good_frac, holdout_seed=args.holdout_seed, data_seed=args.data_seed,
        feature_seed=args.feature_seed, Qmax=args.Qmax
    )

    print(f"[Run] Archs: {args.archs} | Seeds: {args.seeds}")
    shared = prepare_shared(cfg)
    
    runs_by_arch = {a: [] for a in args.archs}
    for seed in args.seeds:
        for arch in args.archs:
            run = train_one_architecture(cfg, shared, arch, seed)
            runs_by_arch[arch].append(run)

    agg = aggregate_runs(cfg, shared, runs_by_arch)
    
    print("Generating fig5_holdout_recovery_curves.pdf...")
    plot_holdout_recovery_curves(cfg, agg, shared["outdir"])
    
    print("Generating fig5_holdout_recovery_budgets.pdf (FINAL Spacing)...")
    plot_holdout_recovery_budgets(cfg, agg, shared["outdir"])
    
    save_speed_summary_csv(cfg, runs_by_arch, agg, shared["outdir"])
    print("Done.")

if __name__ == "__main__":
    main()