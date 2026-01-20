#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hero_finite_shot_recovery_noise_scaling_styled.py

Finite-shot recovery curves (with CI) + simple hardware-noise simulation + scaling.

This is the same experiment logic as your original
`hero_finite_shot_recovery_noise_scaling.py`, but with:
  - Experiment_1 / PRA figure styling (single-column: 3.375in x 2.6in)
  - NO vertical reference line in the scaling plot
  - a generalized CLI for *any number* of system sizes n
  - grayscale palette for all n except a highlighted n (default n=12 in red)

Outputs (in --outdir):
  scaling_finite_shot_CI.pdf
  noise_finite_shot_CI.pdf
  summary.json

Requires:
  pip install numpy matplotlib pennylane

Examples:
  # Default: run n in {10,12,14}, highlight n=12 in red, others grayscale
  python3 hero_finite_shot_recovery_noise_scaling_styled.py --outdir demo_out

  # Run a custom list of sizes
  python3 hero_finite_shot_recovery_noise_scaling_styled.py --n 8 --n 10 --n 12 --n 14 --outdir demo_out

  # Choose which n is highlighted (red)
  python3 hero_finite_shot_recovery_noise_scaling_styled.py --n 10 --n 12 --n 14 --highlight-n 14

  # Select which n is used for the noise plot (default: max(n))
  python3 hero_finite_shot_recovery_noise_scaling_styled.py --n 10 --n 12 --n 14 --noise-n 12
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# PennyLane for IQP-QCBM training (statevector)
import pennylane as qml
from pennylane import numpy as pnp


# -----------------------------
# Style (match Experiment_1)
# -----------------------------

COL_W = 3.375
COL_H = 2.6

COLOR_RED = "#D62728"
COLOR_BLACK = "#222222"
COLOR_GRAY = "#666666"

# Clean dash-dot similar to the Experiment_1 spectral completion style
DASH_DOT = (0, (7, 2, 1.2, 2))


def fig_size_col(h: float = COL_H) -> Tuple[float, float]:
    return (COL_W, float(h))


def set_style(base: int = 8) -> None:
    """Experiment_1-like matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": base,
            "axes.labelsize": base + 1,
            "axes.titlesize": base + 1,
            "legend.fontsize": base - 1,
            "legend.frameon": False,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            # IMPORTANT: Experiment_1 figures have no background grid
            "axes.grid": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------
# Target distribution (paper benchmark: even parity support + exponential tilt)
# -----------------------------


def int2bits(k: int, n: int) -> np.ndarray:
    # bits[0] is MSB (wire 0)
    return np.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int8)


def parity_even(bits: np.ndarray) -> bool:
    return (int(np.sum(bits)) % 2) == 0


def longest_zero_run_between_ones(bits: np.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0


def make_bits_table(n: int) -> np.ndarray:
    N = 2**n
    return np.array([int2bits(i, n) for i in range(N)], dtype=np.int8)


def build_target_distribution(n: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Even-parity support + exponential score tilt (Appendix-A benchmark)."""
    N = 2**n
    scores = np.full(N, -100.0, dtype=np.float64)
    support = np.zeros(N, dtype=bool)

    for k in range(N):
        b = int2bits(k, n)
        if parity_even(b):
            support[k] = True
            scores[k] = 1.0 + float(longest_zero_run_between_ones(b))

    logits = np.full(N, -np.inf, dtype=np.float64)
    logits[support] = beta * scores[support]
    m = float(np.max(logits[support]))
    unnorm = np.zeros(N, dtype=np.float64)
    unnorm[support] = np.exp(logits[support] - m)
    p_star = unnorm / max(1e-15, float(unnorm.sum()))

    return p_star.astype(np.float64), support, scores.astype(np.float64)


def topk_mask(scores: np.ndarray, support: np.ndarray, frac: float = 0.05) -> np.ndarray:
    valid = np.where(support)[0]
    k = max(1, int(np.floor(frac * valid.size)))
    local_order = np.argsort(-scores[valid])
    top_indices = valid[local_order[:k]]
    mask = np.zeros_like(support, dtype=bool)
    mask[top_indices] = True
    return mask


def sample_indices(probs: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = probs / max(1e-15, float(probs.sum()))
    return rng.choice(len(p), size=m, replace=True, p=p)


def empirical_dist(idxs: np.ndarray, N: int) -> np.ndarray:
    c = np.bincount(idxs, minlength=N)
    s = max(1, int(c.sum()))
    return (c / s).astype(np.float64)


# -----------------------------
# Holdout selection (same heuristic as your script)
# -----------------------------


def _min_hamming_to_set(bit_vec: np.ndarray, sel_bits: np.ndarray) -> int:
    if sel_bits.shape[0] == 0:
        return int(bit_vec.shape[0])
    d = np.sum(sel_bits != bit_vec[None, :], axis=1)
    return int(np.min(d))


def select_holdout_smart(
    p_star: np.ndarray,
    good_mask: np.ndarray,
    bits_table: np.ndarray,
    m_train: int,
    holdout_k: int,
    pool_size: int,
    seed: int,
) -> np.ndarray:
    if holdout_k <= 0:
        return np.zeros_like(good_mask, dtype=bool)

    rng = np.random.default_rng(seed)

    good_idxs = np.where(good_mask)[0]
    taus = [1.0 / max(1, m_train), 0.5 / max(1, m_train), 0.25 / max(1, m_train), 0.0]

    cand: Optional[np.ndarray] = None
    for tau in taus:
        cand = good_idxs[p_star[good_idxs] >= tau]
        if cand.size >= holdout_k:
            break
    if cand is None or cand.size == 0:
        raise RuntimeError("No candidates for holdout selection. Check G or m_train.")

    order = np.argsort(-p_star[cand])
    cand = cand[order]
    pool = cand[: min(pool_size, cand.size)].copy()
    rng.shuffle(pool)

    pool_sorted = pool[np.argsort(-p_star[pool])]
    selected = [int(pool_sorted[0])]
    selected_bits = bits_table[[selected[-1]]].copy()

    while len(selected) < holdout_k and len(selected) < pool.size:
        best_idx = None
        best_d = -1
        best_p = -1.0
        for idx in pool:
            idx_i = int(idx)
            if idx_i in selected:
                continue
            d = _min_hamming_to_set(bits_table[idx_i], selected_bits)
            p = float(p_star[idx_i])
            if (d > best_d) or (d == best_d and p > best_p):
                best_d = d
                best_p = p
                best_idx = idx_i
        if best_idx is None:
            break
        selected.append(int(best_idx))
        selected_bits = np.vstack([selected_bits, bits_table[best_idx]])

    holdout = np.zeros_like(good_mask, dtype=bool)
    holdout[np.array(selected, dtype=int)] = True
    return holdout


# -----------------------------
# Parity features (Walsh)
# -----------------------------


def p_sigma(sigma: float) -> float:
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma**2))) if sigma > 0 else 0.5


def sample_alphas(n: int, sigma: float, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = p_sigma(sigma)
    alphas = rng.binomial(1, p, size=(K, n)).astype(np.int8)

    # avoid all-zero alpha
    zero = np.where(np.sum(alphas, axis=1) == 0)[0]
    while zero.size > 0:
        alphas[zero] = rng.binomial(1, p, size=(zero.size, n)).astype(np.int8)
        zero = np.where(np.sum(alphas, axis=1) == 0)[0]

    return alphas


def build_parity_matrix(alphas: np.ndarray, bits_table: np.ndarray) -> np.ndarray:
    A = alphas.astype(np.int16)
    X = bits_table.astype(np.int16).T
    par = (A @ X) & 1
    return np.where(par == 0, 1.0, -1.0).astype(np.float64)


# -----------------------------
# IQP-QCBM (NN+NNN ZZ ring)
# -----------------------------


def get_iqp_pairs_nn_nnn(n: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        pairs.append(tuple(sorted((i, (i + 1) % n))))
        pairs.append(tuple(sorted((i, (i + 2) % n))))
    pairs = sorted(list(set(pairs)))
    return pairs


def iqp_circuit_zz_only(W, n: int, pairs: List[Tuple[int, int]], layers: int = 1):
    idx = 0
    for w in range(n):
        qml.Hadamard(wires=w)
    for _ in range(layers):
        for (i, j) in pairs:
            qml.IsingZZ(W[idx], wires=[i, j])
            idx += 1
        for w in range(n):
            qml.Hadamard(wires=w)


def train_iqp_qcbm(
    n: int,
    layers: int,
    steps: int,
    lr: float,
    P: np.ndarray,
    z_data: np.ndarray,
    seed_init: int,
) -> Tuple[np.ndarray, float]:
    """Moment-MSE training; returns exact probability vector."""

    dev = qml.device("default.qubit", wires=n)
    pairs = get_iqp_pairs_nn_nnn(n)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit_zz_only(W, n=n, pairs=pairs, layers=layers)
        return qml.probs(wires=range(n))

    P_t = pnp.array(P, requires_grad=False)
    z_t = pnp.array(z_data, requires_grad=False)

    num_params = len(pairs) * layers
    rng = np.random.default_rng(seed_init)
    W = pnp.array(0.01 * rng.standard_normal(num_params), requires_grad=True)

    opt = qml.AdamOptimizer(lr)

    def loss_fn(w):
        q = circuit(w)
        return pnp.mean((z_t - P_t @ q) ** 2)

    loss_val = float("nan")
    for _ in range(int(steps)):
        W, loss_val = opt.step_and_cost(loss_fn, W)

    q_final = np.array(circuit(W), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final /= max(1e-15, float(q_final.sum()))

    return q_final, float(loss_val)


# -----------------------------
# Recovery metrics
# -----------------------------


def expected_recovery_fraction(probs: np.ndarray, holdout_mask: np.ndarray, Q_vals: np.ndarray) -> np.ndarray:
    """Expectation-level recovery R(Q)."""
    Q_vals = np.array(Q_vals, dtype=int)
    H = int(np.sum(holdout_mask))
    if H == 0:
        return np.zeros_like(Q_vals, dtype=np.float64)
    pS = probs[holdout_mask].astype(np.float64)[:, None]
    return np.sum(1.0 - np.power(1.0 - pS, Q_vals[None, :]), axis=0) / H


def find_Q_threshold_expected(probs: np.ndarray, holdout_mask: np.ndarray, thr: float, Qmax: int = 200000) -> float:
    """Smallest Q with expected recovery >= thr, or inf if not reached by Qmax."""
    H = int(np.sum(holdout_mask))
    if H == 0:
        return float("nan")

    def frac(Q: int) -> float:
        return float(expected_recovery_fraction(probs, holdout_mask, np.array([Q]))[0])

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


def sample_based_recovery_curve(
    probs: np.ndarray,
    holdout_mask: np.ndarray,
    Q_vals: np.ndarray,
    n_trials: int,
    seed: int,
    thr: float = 0.8,
) -> Dict[str, np.ndarray]:
    """Monte-Carlo estimator of recovery curve with trial-to-trial variability."""

    Q_vals = np.array(Q_vals, dtype=int)
    Q_vals = np.unique(Q_vals)
    Q_vals.sort()
    Qmax = int(Q_vals[-1])

    H_idxs = np.where(holdout_mask)[0]
    H_size = int(H_idxs.size)
    if H_size == 0:
        raise ValueError("Holdout set is empty.")

    N = probs.size
    map_arr = np.full(N, -1, dtype=np.int32)
    map_arr[H_idxs] = np.arange(H_size, dtype=np.int32)

    rng = np.random.default_rng(seed)

    R_trials = np.zeros((n_trials, Q_vals.size), dtype=np.float64)
    Qthr_trials = np.full(n_trials, np.inf, dtype=np.float64)

    p = probs / max(1e-15, float(probs.sum()))

    for t in range(n_trials):
        samples = rng.choice(N, size=Qmax, replace=True, p=p)

        seen = np.zeros(H_size, dtype=bool)
        seen_count = 0

        j = 0
        next_Q = int(Q_vals[j])
        qthr_set = False

        for i, s in enumerate(samples, start=1):
            pos = int(map_arr[int(s)])
            if pos >= 0 and (not seen[pos]):
                seen[pos] = True
                seen_count += 1

            if (not qthr_set) and (seen_count / H_size >= thr):
                Qthr_trials[t] = float(i)
                qthr_set = True

            if i == next_Q:
                R_trials[t, j] = seen_count / H_size
                j += 1
                if j >= Q_vals.size:
                    break
                next_Q = int(Q_vals[j])

        if j < Q_vals.size:
            R_trials[t, j:] = seen_count / H_size

    mean = np.mean(R_trials, axis=0)
    lo = np.quantile(R_trials, 0.025, axis=0)
    hi = np.quantile(R_trials, 0.975, axis=0)

    finite = np.isfinite(Qthr_trials)
    if finite.any():
        qthr_med = np.median(Qthr_trials[finite])
        qthr_lo = np.quantile(Qthr_trials[finite], 0.025)
        qthr_hi = np.quantile(Qthr_trials[finite], 0.975)
    else:
        qthr_med, qthr_lo, qthr_hi = float("inf"), float("inf"), float("inf")

    return {
        "Q_vals": Q_vals,
        "R_mean": mean,
        "R_lo": lo,
        "R_hi": hi,
        "R_trials": R_trials,
        "Qthr_trials": Qthr_trials,
        "Qthr_median": np.array([qthr_med], dtype=np.float64),
        "Qthr_lo": np.array([qthr_lo], dtype=np.float64),
        "Qthr_hi": np.array([qthr_hi], dtype=np.float64),
    }


# -----------------------------
# Simple hardware-noise model on distributions
# -----------------------------


def apply_depolarizing_mixture(q: np.ndarray, p_dep: float) -> np.ndarray:
    p_dep = float(np.clip(p_dep, 0.0, 1.0))
    if p_dep <= 0:
        return q
    u = np.ones_like(q, dtype=np.float64) / q.size
    out = (1.0 - p_dep) * q + p_dep * u
    out = np.clip(out, 0.0, None)
    out /= max(1e-15, float(out.sum()))
    return out.astype(np.float64)


def apply_readout_bitflip(q: np.ndarray, n: int, p_ro: float) -> np.ndarray:
    """Independent classical bit-flip on each qubit applied to the measurement distribution."""
    p_ro = float(np.clip(p_ro, 0.0, 0.5))
    if p_ro <= 0:
        return q

    qn = q.astype(np.float64).copy()
    N = qn.size

    # qubit 0 is MSB -> integer bit position n-1
    for qubit in range(n):
        bitpos = (n - 1 - qubit)
        block = 1 << bitpos
        period = block << 1
        for base in range(0, N, period):
            i0 = base
            i1 = base + block
            a = qn[i0:i1].copy()
            b = qn[i1:i1 + block].copy()
            qn[i0:i1] = (1.0 - p_ro) * a + p_ro * b
            qn[i1:i1 + block] = p_ro * a + (1.0 - p_ro) * b

    qn = np.clip(qn, 0.0, None)
    qn /= max(1e-15, float(qn.sum()))
    return qn


def apply_simple_hardware_noise(q: np.ndarray, n: int, p_dep: float, p_ro: float) -> np.ndarray:
    q2 = apply_depolarizing_mixture(q, p_dep=p_dep)
    q3 = apply_readout_bitflip(q2, n=n, p_ro=p_ro)
    return q3


# -----------------------------
# Experiment bundle
# -----------------------------


@dataclass
class RunCfg:
    n: int
    beta: float
    train_m: int
    holdout_k: int
    holdout_pool: int
    sigma: float
    K: int
    seed: int
    iqp_steps: int
    iqp_lr: float
    iqp_layers: int


def train_one_setting(cfg: RunCfg) -> Dict[str, object]:
    bits_table = make_bits_table(cfg.n)
    p_star, support, scores = build_target_distribution(cfg.n, cfg.beta)
    good_mask = topk_mask(scores, support, frac=0.05)

    holdout_mask = select_holdout_smart(
        p_star=p_star,
        good_mask=good_mask,
        bits_table=bits_table,
        m_train=cfg.train_m,
        holdout_k=cfg.holdout_k,
        pool_size=cfg.holdout_pool,
        seed=cfg.seed + 111,
    )

    # Masked training distribution (holdout removed)
    p_train = p_star.copy()
    p_train[holdout_mask] = 0.0
    p_train /= max(1e-15, float(p_train.sum()))

    idxs_train = sample_indices(p_train, cfg.train_m, seed=cfg.seed + 7)
    emp = empirical_dist(idxs_train, p_star.size)

    # Features
    alphas = sample_alphas(cfg.n, cfg.sigma, cfg.K, seed=cfg.seed + 222)
    P = build_parity_matrix(alphas, bits_table)
    z = P @ emp

    # Train IQP
    q_iqp, loss = train_iqp_qcbm(
        n=cfg.n,
        layers=cfg.iqp_layers,
        steps=cfg.iqp_steps,
        lr=cfg.iqp_lr,
        P=P,
        z_data=z,
        seed_init=cfg.seed + 10000,
    )

    return {
        "bits_table": bits_table,
        "p_star": p_star,
        "holdout_mask": holdout_mask,
        "q_iqp": q_iqp,
        "train_loss": float(loss),
    }


# -----------------------------
# Plotting helpers
# -----------------------------


def _dedupe_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def colors_for_sizes(n_list: List[int], highlight_n: int = 12) -> Dict[int, str]:
    """Map each n to a color: highlight_n -> red, others -> grayscale."""
    n_sorted = sorted(n_list)
    others = [n for n in n_sorted if n != highlight_n]

    colors: Dict[int, str] = {}
    if highlight_n in n_sorted:
        colors[highlight_n] = COLOR_RED

    if len(others) == 0:
        return colors

    # Dark -> light (but avoid pure white for visibility)
    # Greys(1) is black, Greys(0) is white.
    if len(others) == 1:
        ts = [0.80]
    else:
        # spread t in [0.85, 0.30]
        ts = np.linspace(0.85, 0.30, num=len(others)).tolist()

    for n, t in zip(others, ts):
        colors[n] = plt.cm.Greys(float(t))

    return colors


def plot_scaling_finite_shot(
    outpath: str,
    curves: List[Dict[str, object]],
    thr: float,
    highlight_n: int,
) -> None:
    """Overlay multiple system sizes on x-axis normalized by expected Q_thr.

    To avoid overplotting when curves collapse under normalization, we use:
      - highlight_n: full styling (mean + 95% CI band + faint expectation curve)
      - all other n: thin mean curve only (no CI band), grayscale, high transparency

    Visual conventions:
      - NO vertical reference line
      - threshold horizontal dotted line
      - legend at center left
    """

    fig, ax = plt.subplots(figsize=fig_size_col(), constrained_layout=True)

    # Threshold only
    ax.axhline(thr, color="#444444", linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    # Plot non-highlight first, highlight last (so the red curve sits on top).
    curves_sorted = sorted(curves, key=lambda c: 1 if int(c["n"]) == int(highlight_n) else 0)

    # jitter step for Q80 markers (avoid overlap)
    m = max(1, len(curves_sorted))
    jitter = min(0.02, 0.12 / m)

    for idx, c in enumerate(curves_sorted):
        n = int(c["n"])
        is_hi = (n == int(highlight_n))

        color = c["color"]
        label = c["label"]

        Q = np.asarray(c["Q_vals"], dtype=np.float64)
        Q80_exp = float(c["Q80_exp"])
        Q80_norm = Q80_exp if np.isfinite(Q80_exp) and Q80_exp > 0 else float(Q[-1])
        x = Q / max(1.0, Q80_norm)

        # Highlight: show expectation curve + CI band + thick mean
        if is_hi:
            ax.plot(
                x,
                c["R_exp"],
                color=color,
                linewidth=1.2,
                alpha=0.35,
                zorder=2,
            )
            ax.fill_between(
                x,
                c["R_lo"],
                c["R_hi"],
                color=color,
                alpha=0.14,
                linewidth=0.0,
                zorder=2,
            )
            ax.plot(
                x,
                c["R_mean"],
                color=color,
                linewidth=2.1,
                label=label,
                zorder=4,
            )
        else:
            # Non-highlight: mean only (thin + transparent), no band.
            ax.plot(
                x,
                c["R_mean"],
                color=color,
                linewidth=1.0,
                alpha=0.28,
                label=label,
                zorder=2,
            )

        # Mark finite-shot Q80 (median + 95% CI) as a horizontal errorbar.
        q80_med = float(c["Q80_mc_med"])
        q80_lo = float(c["Q80_mc_lo"])
        q80_hi = float(c["Q80_mc_hi"])

        if np.isfinite(q80_med) and q80_med > 0 and np.isfinite(Q80_norm) and Q80_norm > 0:
            x_med = q80_med / max(1.0, Q80_norm)
            x_lo = (q80_lo / max(1.0, Q80_norm)) if np.isfinite(q80_lo) else x_med
            x_hi = (q80_hi / max(1.0, Q80_norm)) if np.isfinite(q80_hi) else x_med

            y_mark = max(0.02, thr - jitter * idx)

            # Make highlight marker slightly stronger.
            alpha = 0.95 if is_hi else 0.45
            ms = 4.0 if is_hi else 3.2
            elw = 1.0 if is_hi else 0.8

            ax.errorbar(
                x_med,
                y_mark,
                xerr=[[max(0.0, x_med - x_lo)], [max(0.0, x_hi - x_med)]],
                fmt="o",
                markersize=ms,
                mfc="white",
                mec=color,
                ecolor=color,
                elinewidth=elw,
                capsize=2,
                capthick=1.0,
                alpha=alpha,
                zorder=5 if is_hi else 3,
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"Normalized budget $Q / Q_{80}^{\mathrm{exp}}$")
    ax.set_ylabel(r"Recovery $\hat R(Q)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center left", handlelength=2.0)

    fig.savefig(outpath)
    plt.close(fig)


def plot_noise_finite_shot(
    outpath: str,
    Q_vals: np.ndarray,
    ideal: Dict[str, np.ndarray],
    noise_low: Dict[str, np.ndarray],
    noise_high: Dict[str, np.ndarray],
    R_exp_ideal: np.ndarray,
    R_exp_low: np.ndarray,
    R_exp_high: np.ndarray,
    thr: float,
    Q80_exp_ideal: float,
    Q80_exp_low: float,
    Q80_exp_high: float,
    ideal_color: str,
) -> None:
    """Finite-shot recovery under a simple (simulated) hardware-noise model."""

    fig, ax = plt.subplots(figsize=fig_size_col(), constrained_layout=True)

    # Threshold
    ax.axhline(thr, color="#444444", linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    # Colors (grayscale; optionally red if ideal is highlighted)
    c_ideal = COLOR_RED  # ideal always red
    c_low = "#555555"
    c_high = "#999999"

    # Expectation-level curves (faint)
    ax.plot(Q_vals, R_exp_ideal, color=c_ideal, alpha=0.30, linewidth=1.2, linestyle="-", zorder=2)
    ax.plot(Q_vals, R_exp_low, color=c_low, alpha=0.30, linewidth=1.2, linestyle="--", zorder=2)
    ax.plot(Q_vals, R_exp_high, color=c_high, alpha=0.30, linewidth=1.2, linestyle=DASH_DOT, zorder=2)

    def band(curve: Dict[str, np.ndarray], color: str, label: str, ls, band_alpha: float):
        ax.fill_between(Q_vals, curve["R_lo"], curve["R_hi"], color=color, alpha=band_alpha, linewidth=0, zorder=2)
        ax.plot(Q_vals, curve["R_mean"], color=color, linewidth=2.0, linestyle=ls, label=label, zorder=3)

    band(ideal, c_ideal, f"Ideal (Q80_exp={Q80_exp_ideal:.0f})", "-", 0.14)
    band(noise_low, c_low, f"Noise low (Q80_exp={Q80_exp_low:.0f})", "--", 0.12)
    band(noise_high, c_high, f"Noise high (Q80_exp={Q80_exp_high:.0f})", DASH_DOT, 0.10)

    # Q80 markers (median + CI)
    jitter = 0.02
    for idx, (curve, color) in enumerate([(ideal, c_ideal), (noise_low, c_low), (noise_high, c_high)]):
        med = float(curve["Qthr_median"][0])
        lo = float(curve["Qthr_lo"][0])
        hi = float(curve["Qthr_hi"][0])
        if np.isfinite(med) and med > 0:
            y_mark = max(0.02, thr - jitter * idx)
            ax.errorbar(
                med,
                y_mark,
                xerr=[[max(0.0, med - lo)], [max(0.0, hi - med)]],
                fmt="o",
                markersize=4.0,
                mfc="white",
                mec=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                capthick=1.0,
                alpha=0.95,
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"Samples $Q$")
    ax.set_ylabel(r"Recovery $\hat R(Q)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center left", handlelength=2.0)

    fig.savefig(outpath)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str, default="hero_finite_shot_PRA")

    # System sizes: repeatable --n
    parser.add_argument(
        "--n",
        type=int,
        action="append",
        dest="n_list",
        help="System size(s) to run. Repeat flag to add more. If omitted, defaults to [10,12,14].",
    )

    # Backward-compat (optional): if users still pass --n-small/--n-large
    parser.add_argument("--n-small", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-large", type=int, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--highlight-n", type=int, default=12, help="The n to highlight in red (default: 12).")
    parser.add_argument(
        "--noise-n",
        type=int,
        default=None,
        help="Which n to use for the noise plot (default: max(n)).",
    )

    # Paper-like target
    parser.add_argument("--beta", type=float, default=0.9)

    # Data / holdout
    parser.add_argument("--train-m", type=int, default=1000)
    parser.add_argument("--holdout-k", type=int, default=20)
    parser.add_argument("--holdout-pool", type=int, default=400)

    # Feature band
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=128)

    # IQP training budget
    parser.add_argument("--iqp-steps", type=int, default=20)
    parser.add_argument("--iqp-lr", type=float, default=0.05)
    parser.add_argument("--iqp-layers", type=int, default=1)

    # Finite-shot evaluation
    parser.add_argument("--n-trials", type=int, default=120)
    parser.add_argument("--Qmax", type=int, default=100000)
    parser.add_argument("--thr", type=float, default=0.8)

    # Noise settings
    parser.add_argument("--pdep-low", type=float, default=0.05)
    parser.add_argument("--pro-low", type=float, default=0.02)
    parser.add_argument("--pdep-high", type=float, default=0.30)
    parser.add_argument("--pro-high", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_style(8)
    outdir = ensure_outdir(args.outdir)

    # Determine n_list
    if args.n_list is not None and len(args.n_list) > 0:
        n_list = _dedupe_keep_order([int(x) for x in args.n_list])
    elif args.n_small is not None or args.n_large is not None:
        # legacy usage
        n_small = int(args.n_small) if args.n_small is not None else 10
        n_large = int(args.n_large) if args.n_large is not None else 14
        n_list = _dedupe_keep_order([n_small, n_large])
    else:
        n_list = [10, 12, 14]

    # Choose which n is used for noise plot
    noise_n = int(args.noise_n) if args.noise_n is not None else int(max(n_list))
    if noise_n not in n_list:
        n_list = _dedupe_keep_order(n_list + [noise_n])

    # Q grid (log-spaced + include 1 and Qmax)
    Q_vals = np.unique(
        np.concatenate(
            [
                np.array([1, 2, 5, 10, 20, 50, 100, 200, 500], dtype=int),
                np.unique(np.logspace(3, math.log10(args.Qmax), 14).astype(int)),
                np.array([args.Qmax], dtype=int),
            ]
        )
    )
    Q_vals = Q_vals[(Q_vals >= 1) & (Q_vals <= args.Qmax)]
    Q_vals.sort()

    # Colors
    color_map = colors_for_sizes(n_list, highlight_n=int(args.highlight_n))

    # Train each n + compute finite-shot curves
    artifacts: Dict[int, Dict[str, object]] = {}
    curves_for_plot: List[Dict[str, object]] = []

    for i, n in enumerate(n_list):
        cfg = RunCfg(
            n=int(n),
            beta=float(args.beta),
            train_m=int(args.train_m),
            holdout_k=int(args.holdout_k),
            holdout_pool=int(args.holdout_pool),
            sigma=float(args.sigma),
            K=int(args.K),
            seed=int(args.seed) + 13 * i,
            iqp_steps=int(args.iqp_steps),
            iqp_lr=float(args.iqp_lr),
            iqp_layers=int(args.iqp_layers),
        )

        print(f"[Train] n={cfg.n} K={cfg.K} sigma={cfg.sigma} steps={cfg.iqp_steps}")
        art = train_one_setting(cfg)
        artifacts[cfg.n] = art

        q = art["q_iqp"]  # type: ignore
        holdout = art["holdout_mask"]  # type: ignore

        R_exp = expected_recovery_fraction(q, holdout, Q_vals)
        Q80_exp = find_Q_threshold_expected(q, holdout, thr=float(args.thr), Qmax=max(200000, int(args.Qmax)))

        mc = sample_based_recovery_curve(
            probs=q,
            holdout_mask=holdout,
            Q_vals=Q_vals,
            n_trials=int(args.n_trials),
            seed=int(cfg.seed) + 900,
            thr=float(args.thr),
        )

        color = color_map.get(cfg.n, plt.cm.Greys(0.65))
        label = fr"IQP $n$={cfg.n}"

        curves_for_plot.append(
            {
                "n": cfg.n,
                "color": color,
                "label": label,
                "Q_vals": mc["Q_vals"],
                "R_mean": mc["R_mean"],
                "R_lo": mc["R_lo"],
                "R_hi": mc["R_hi"],
                "R_exp": R_exp,
                "Q80_exp": float(Q80_exp),
                "Q80_mc_med": float(mc["Qthr_median"][0]),
                "Q80_mc_lo": float(mc["Qthr_lo"][0]),
                "Q80_mc_hi": float(mc["Qthr_hi"][0]),
            }
        )

        print(
            f"[Scaling] n={cfg.n} loss={art['train_loss']:.3e} "
            f"Q80_exp={Q80_exp:.0f} Q80_MC_med={float(mc['Qthr_median'][0]):.0f} "
            f"(95% [{float(mc['Qthr_lo'][0]):.0f}, {float(mc['Qthr_hi'][0]):.0f}])"
        )

    # Scaling plot
    scaling_path = os.path.join(outdir, "scaling_finite_shot_CI.pdf")
    plot_scaling_finite_shot(scaling_path, curves_for_plot, thr=float(args.thr), highlight_n=int(args.highlight_n))
    print(f"[Saved] {scaling_path}")

    # Noise plot on selected n
    art_noise = artifacts[int(noise_n)]
    q_ideal = art_noise["q_iqp"]  # type: ignore
    holdout_noise = art_noise["holdout_mask"]  # type: ignore

    q_low = apply_simple_hardware_noise(q_ideal, n=int(noise_n), p_dep=float(args.pdep_low), p_ro=float(args.pro_low))
    q_high = apply_simple_hardware_noise(q_ideal, n=int(noise_n), p_dep=float(args.pdep_high), p_ro=float(args.pro_high))

    R_exp_ideal = expected_recovery_fraction(q_ideal, holdout_noise, Q_vals)
    R_exp_low = expected_recovery_fraction(q_low, holdout_noise, Q_vals)
    R_exp_high = expected_recovery_fraction(q_high, holdout_noise, Q_vals)

    Q80_exp_ideal = find_Q_threshold_expected(q_ideal, holdout_noise, thr=float(args.thr), Qmax=max(200000, int(args.Qmax)))
    Q80_exp_low = find_Q_threshold_expected(q_low, holdout_noise, thr=float(args.thr), Qmax=max(200000, int(args.Qmax)))
    Q80_exp_high = find_Q_threshold_expected(q_high, holdout_noise, thr=float(args.thr), Qmax=max(200000, int(args.Qmax)))

    mc_ideal = sample_based_recovery_curve(q_ideal, holdout_noise, Q_vals, int(args.n_trials), seed=int(args.seed) + 1111, thr=float(args.thr))
    mc_low = sample_based_recovery_curve(q_low, holdout_noise, Q_vals, int(args.n_trials), seed=int(args.seed) + 2222, thr=float(args.thr))
    mc_high = sample_based_recovery_curve(q_high, holdout_noise, Q_vals, int(args.n_trials), seed=int(args.seed) + 3333, thr=float(args.thr))

    ideal_color = COLOR_RED  # ideal always red

    print(f"[Noise] n={noise_n} ideal Q80_exp={Q80_exp_ideal:.0f} | low Q80_exp={Q80_exp_low:.0f} | high Q80_exp={Q80_exp_high:.0f}")
    print(
        f"[Noise] n={noise_n} ideal Q80_MC_med={float(mc_ideal['Qthr_median'][0]):.0f} "
        f"| low Q80_MC_med={float(mc_low['Qthr_median'][0]):.0f} "
        f"| high Q80_MC_med={float(mc_high['Qthr_median'][0]):.0f}"
    )

    noise_path = os.path.join(outdir, "noise_finite_shot_CI.pdf")
    plot_noise_finite_shot(
        noise_path,
        Q_vals=Q_vals,
        ideal=mc_ideal,
        noise_low=mc_low,
        noise_high=mc_high,
        R_exp_ideal=R_exp_ideal,
        R_exp_low=R_exp_low,
        R_exp_high=R_exp_high,
        thr=float(args.thr),
        Q80_exp_ideal=float(Q80_exp_ideal),
        Q80_exp_low=float(Q80_exp_low),
        Q80_exp_high=float(Q80_exp_high),
        ideal_color=ideal_color,
    )
    print(f"[Saved] {noise_path}")

    # Summary JSON
    summary = {
        "n_list": [int(x) for x in n_list],
        "highlight_n": int(args.highlight_n),
        "noise_n": int(noise_n),
        "cfg": {
            "beta": float(args.beta),
            "train_m": int(args.train_m),
            "holdout_k": int(args.holdout_k),
            "holdout_pool": int(args.holdout_pool),
            "sigma": float(args.sigma),
            "K": int(args.K),
            "iqp_steps": int(args.iqp_steps),
            "iqp_lr": float(args.iqp_lr),
            "iqp_layers": int(args.iqp_layers),
            "n_trials": int(args.n_trials),
            "Qmax": int(args.Qmax),
            "thr": float(args.thr),
            "noise": {
                "p_dep_low": float(args.pdep_low),
                "p_ro_low": float(args.pro_low),
                "p_dep_high": float(args.pdep_high),
                "p_ro_high": float(args.pro_high),
            },
            "seed": int(args.seed),
        },
        "Q_vals": [int(x) for x in Q_vals.tolist()],
        "scaling": [
            {
                "n": int(c["n"]),
                "Q80_exp": float(c["Q80_exp"]),
                "Q80_mc_median": float(c["Q80_mc_med"]),
                "Q80_mc_lo": float(c["Q80_mc_lo"]),
                "Q80_mc_hi": float(c["Q80_mc_hi"]),
            }
            for c in curves_for_plot
        ],
        "noise": {
            "n": int(noise_n),
            "Q80_exp": {
                "ideal": float(Q80_exp_ideal),
                "low": float(Q80_exp_low),
                "high": float(Q80_exp_high),
            },
            "Q80_mc_median": {
                "ideal": float(mc_ideal["Qthr_median"][0]),
                "low": float(mc_low["Qthr_median"][0]),
                "high": float(mc_high["Qthr_median"][0]),
            },
        },
    }

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
