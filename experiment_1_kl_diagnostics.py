#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 1: standalone KL diagnostics figure panels.

Protocol:
- panels (a) and (b): illustrative fixed-beta, fixed-seed sigma-K landscape
- panel (c): 10 matched seeds, seedwise-best parity over the full sigma-K grid vs IQP MSE
- the current panel aesthetics are the approved locked-in Experiment 1 standard
- raw runs are saved so figures can be rerendered without recomputation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

HAS_PENNYLANE = False
try:
    import pennylane as qml  # type: ignore
    from pennylane import numpy as qnp  # type: ignore

    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False
    qml = None  # type: ignore[assignment]
    qnp = None  # type: ignore[assignment]

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from final_plot_style import (
    IEEE_THREEUP_PANEL_H_IN,
    IEEE_THREEUP_PANEL_W_IN,
    apply_ieee_latex_style,
    save_exact_figure,
)
from model_labels import IQP_MSE_LABEL, IQP_PARITY_LABEL


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_1_kl_diagnostics.py"

FIG_W = IEEE_THREEUP_PANEL_W_IN
FIG_H = IEEE_THREEUP_PANEL_H_IN
HEATMAP_FIGSIZE = (FIG_W, FIG_H)
HEATMAP_SQUARE_CELLS_FIGSIZE = (FIG_W, FIG_H * 1.55)
BAR_FIGSIZE = (FIG_W, FIG_H)
LOLLIPOP_FIGSIZE = (FIG_W, FIG_H)

COLOR_TARGET = "#2d6a4f"
COLOR_IQP = "#d90429"
COLOR_IQP_MSE = "#5B9BE6"
COLOR_NEUTRAL = "#999999"
COLOR_TEXT = "#333333"
COLOR_SUBTEXT = "#888888"
COLOR_LIGHT_TEXT = "#444444"
COLOR_DARK = "#1a0505"
COLOR_GRID = "#FFFFFF"
COLOR_AXIS = "#cfcfcf"
CMAP_KL = colors.LinearSegmentedColormap.from_list(
    "kl_red",
    ["#ff8a8a", "#d90429", "#1a0505"],
    N=256,
)
CMAP_KL_CONSISTENT = colors.LinearSegmentedColormap.from_list(
    "kl_softred_babyblue",
    ["#ea8a7d", "#f6f1ee", "#86afe8"],
    N=256,
)
CMAP_RANK = colors.LinearSegmentedColormap.from_list(
    "rank_blue_lightred_black",
    [COLOR_TEXT, "#ea8a7d", "#86afe8"],
    N=256,
)

SIGMA_VALUES = [0.5, 1.0, 2.0, 3.0]
K_VALUES = [128, 256, 512]
CI95_T_DF9 = 2.2621571628540993
PANEL_C_INCLUDE_UNIFORM = False
PANEL_A_STYLE = {
    "aspect": "auto",
    "top_tick_label_mode": "plain_k_values_with_explicit_k_axis_label",
    "minor_grid_color": COLOR_GRID,
    "best_cell_border_color": COLOR_DARK,
    "colorbar_orientation": "horizontal",
}
PANEL_B_STYLE = {
    "mode": "dot_rank_guides",
    "guide_color": "#e7e2de",
    "guide_linewidth": 0.9,
    "guide_extent": "stop_at_point",
    "dot_size": 54,
    "x_major_tick_step": 0.1,
    "title": "none",
    "color_scale": "baby_blue_bad__light_red_mid__black_best_at_kl_zero",
    "value_label_pad": 0.014,
}
PANEL_C_STYLE = {
    "mode": "benchmark_lollipop_zoomed",
    "target_color": COLOR_TEXT,
    "best_parity_color": "#ea8a7d",
    "iqp_mse_color": "#86afe8",
    "x_min": -0.015,
    "x_floor": 0.59,
    "x_max_padding": 0.035,
    "major_xtick_step": 0.2,
    "minor_xtick_step": 0.1,
    "include_uniform": PANEL_C_INCLUDE_UNIFORM,
    "selection_rule": "fixed_global_best_over_panel_c_seeds_by_mean",
}


def _approved_style_metadata() -> dict:
    return {
        "style_standard": "experiment_1_locked_visual_standard_2026-03-23",
        "panel_a_style": dict(PANEL_A_STYLE),
        "panel_b_style": dict(PANEL_B_STYLE),
        "panel_c_style": dict(PANEL_C_STYLE),
    }


def apply_style() -> None:
    apply_ieee_latex_style(use_tex=True)
    plt.rcParams.update(
        {
            "legend.edgecolor": COLOR_SUBTEXT,
            "grid.color": "#E2DFDB",
            "grid.alpha": 0.65,
        }
    )


def _fmt_sigma(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:g}"


def _luminance(rgba: tuple[float, float, float, float]) -> float:
    r, g, b, _ = rgba
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _save_fig(fig: plt.Figure, path: Path) -> None:
    save_exact_figure(fig, path)
    fig.savefig(path.with_suffix(".png"), format="png")
    plt.close(fig)


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _ci95_halfwidth(vals: np.ndarray) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    tcrit = CI95_T_DF9 if arr.size == 10 else 1.96
    return tcrit * sem


def _panel_c_indices(data: dict) -> np.ndarray:
    all_seeds = np.asarray(data["all_seeds"], dtype=np.int64)
    panel_c_seeds = np.asarray(data["panel_c_seeds"], dtype=np.int64)
    all_seed_to_idx = {int(seed): idx for idx, seed in enumerate(all_seeds.tolist())}
    return np.asarray([all_seed_to_idx[int(seed)] for seed in panel_c_seeds.tolist()], dtype=np.int64)


def _panel_c_global_best_summary(data: dict) -> dict:
    panel_c_idx = _panel_c_indices(data)
    sigma_values = np.asarray(data["sigma_values"], dtype=np.float64)
    k_values = np.asarray(data["k_values"], dtype=np.int64)
    kl_grids = np.asarray(data["kl_grid_by_seed"], dtype=np.float64)[panel_c_idx]
    mean_grid = np.mean(kl_grids, axis=0)
    best_flat_idx = int(np.argmin(mean_grid))
    best_i, best_j = np.unravel_index(best_flat_idx, mean_grid.shape)
    best_vals = np.asarray(kl_grids[:, best_i, best_j], dtype=np.float64)
    return {
        "best_sigma": float(sigma_values[best_i]),
        "best_k": int(k_values[best_j]),
        "best_vals": best_vals,
        "best_mean": float(np.mean(best_vals)),
        "best_ci95": float(_ci95_halfwidth(best_vals)),
    }


def int2bits(k: int, n: int) -> np.ndarray:
    return np.array([(k >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int8)


def parity_even(bits: np.ndarray) -> bool:
    return (int(np.sum(bits)) % 2) == 0


def make_bits_table(n: int) -> np.ndarray:
    return np.array([int2bits(i, n) for i in range(2**n)], dtype=np.int8)


def longest_zero_run_between_ones(bits: np.ndarray) -> int:
    idx = [i for i, b in enumerate(bits) if b == 1]
    if len(idx) < 2:
        return 0
    gaps = [idx[i + 1] - idx[i] - 1 for i in range(len(idx) - 1)]
    return max(gaps) if gaps else 0


def build_target_distribution_paper(n: int, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = 2**n
    support = np.zeros(N, dtype=bool)
    scores = np.zeros(N, dtype=np.float64)
    for k in range(N):
        bits = int2bits(k, n)
        if parity_even(bits):
            support[k] = True
            scores[k] = 1.0 + float(longest_zero_run_between_ones(bits))
    logits = np.full(N, -np.inf, dtype=np.float64)
    logits[support] = float(beta) * scores[support]
    m = float(np.max(logits[support]))
    unnorm = np.zeros(N, dtype=np.float64)
    unnorm[support] = np.exp(logits[support] - m)
    p_star = unnorm / max(1e-15, float(np.sum(unnorm)))
    return p_star.astype(np.float64), support, scores.astype(np.float64)


def sample_indices(probs: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    p = np.asarray(probs, dtype=np.float64)
    p = p / max(1e-15, float(np.sum(p)))
    return rng.choice(p.size, size=int(m), replace=True, p=p)


def empirical_dist(idxs: np.ndarray, N: int) -> np.ndarray:
    counts = np.bincount(np.asarray(idxs, dtype=np.int64), minlength=int(N))
    return (counts / max(1, int(np.sum(counts)))).astype(np.float64)


def p_sigma(sigma: float) -> float:
    if sigma <= 0.0:
        return 0.5
    return 0.5 * (1.0 - math.exp(-1.0 / (2.0 * sigma**2)))


def sample_alphas(n: int, sigma: float, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    p = p_sigma(float(sigma))
    alphas = rng.binomial(1, p, size=(int(K), int(n))).astype(np.int8)
    zero = np.where(np.sum(alphas, axis=1) == 0)[0]
    while zero.size > 0:
        alphas[zero] = rng.binomial(1, p, size=(zero.size, int(n))).astype(np.int8)
        zero = np.where(np.sum(alphas, axis=1) == 0)[0]
    return alphas


def build_parity_matrix(alphas: np.ndarray, bits_table: np.ndarray) -> np.ndarray:
    A = alphas.astype(np.int16)
    X = bits_table.astype(np.int16).T
    par = (A @ X) & 1
    return np.where(par == 0, 1.0, -1.0).astype(np.float64)


def get_iqp_pairs_nn_nnn(n: int) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(int(n)):
        pairs.append(tuple(sorted((i, (i + 1) % int(n)))))
        pairs.append(tuple(sorted((i, (i + 2) % int(n)))))
    return sorted(list(set(pairs)))


def iqp_circuit_zz_only(W, wires, pairs: List[Tuple[int, int]], layers: int = 1) -> None:
    idx = 0
    for w in wires:
        qml.Hadamard(wires=w)
    for _ in range(int(layers)):
        for i, j in pairs:
            qml.IsingZZ(W[idx], wires=[wires[i], wires[j]])
            idx += 1
        for w in wires:
            qml.Hadamard(wires=w)


def train_iqp_qcbm(
    *,
    n: int,
    layers: int,
    steps: int,
    lr: float,
    P: np.ndarray,
    z_data: np.ndarray,
    seed_init: int,
) -> np.ndarray:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP training.")

    dev = qml.device("default.qubit", wires=int(n))
    pairs = get_iqp_pairs_nn_nnn(int(n))

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit_zz_only(W, range(int(n)), pairs, layers=int(layers))
        return qml.probs(wires=range(int(n)))

    P_t = qnp.array(np.asarray(P, dtype=np.float64), requires_grad=False)
    z_t = qnp.array(np.asarray(z_data, dtype=np.float64), requires_grad=False)
    num_params = len(pairs) * int(layers)
    rng = np.random.default_rng(int(seed_init))
    W = qnp.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(float(lr))

    def loss_fn(w):
        q = circuit(w)
        return qnp.mean((z_t - P_t @ q) ** 2)

    for _ in range(1, int(steps) + 1):
        W, _ = opt.step_and_cost(loss_fn, W)

    q_final = np.asarray(circuit(W), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(np.sum(q_final)))
    return q_final.astype(np.float64)


def train_iqp_qcbm_prob_mse(
    *,
    n: int,
    layers: int,
    steps: int,
    lr: float,
    emp_dist: np.ndarray,
    seed_init: int,
) -> np.ndarray:
    if not HAS_PENNYLANE:
        raise RuntimeError("Pennylane is required for IQP training.")

    dev = qml.device("default.qubit", wires=int(n))
    pairs = get_iqp_pairs_nn_nnn(int(n))

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(W):
        iqp_circuit_zz_only(W, range(int(n)), pairs, layers=int(layers))
        return qml.probs(wires=range(int(n)))

    emp_t = qnp.array(np.asarray(emp_dist, dtype=np.float64), requires_grad=False)
    num_params = len(pairs) * int(layers)
    rng = np.random.default_rng(int(seed_init))
    W = qnp.array(0.01 * rng.standard_normal(num_params), requires_grad=True)
    opt = qml.AdamOptimizer(float(lr))

    def loss_fn(w):
        q = circuit(w)
        return qnp.mean((q - emp_t) ** 2)

    for _ in range(1, int(steps) + 1):
        W, _ = opt.step_and_cost(loss_fn, W)

    q_final = np.asarray(circuit(W), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(np.sum(q_final)))
    return q_final.astype(np.float64)


def forward_kl(p_star: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    pv = np.clip(np.asarray(p_star, dtype=np.float64), float(eps), 1.0)
    qv = np.clip(np.asarray(q, dtype=np.float64), float(eps), 1.0)
    pv /= max(1e-15, float(np.sum(pv)))
    qv /= max(1e-15, float(np.sum(qv)))
    return float(np.sum(pv * np.log(pv / qv)))


def render_heatmap_panel(
    *,
    outdir: Path,
    sigma_values: list[float],
    k_values: list[int],
    panel_ab_grid: np.ndarray,
    panel_ab_best_sigma: float,
    panel_ab_best_k: int,
    square_cells: bool = False,
    outname: str = "experiment_1_panel_a_sigma_k_heatmap.pdf",
    figsize: tuple[float, float] | None = None,
    target_cell_width_height_ratio: float | None = None,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(
        figsize=figsize if figsize is not None else (HEATMAP_SQUARE_CELLS_FIGSIZE if square_cells else HEATMAP_FIGSIZE),
        constrained_layout=True,
    )

    norm = colors.Normalize(vmin=float(np.min(panel_ab_grid)), vmax=float(np.max(panel_ab_grid)))
    im = ax.imshow(panel_ab_grid, cmap=CMAP_KL, norm=norm, aspect=str(PANEL_A_STYLE["aspect"]))
    if square_cells or target_cell_width_height_ratio is not None:
        target_ratio = 1.0 if square_cells else float(target_cell_width_height_ratio)
        # Convert desired per-cell width/height into the enclosing axes box aspect.
        ax.set_box_aspect(len(sigma_values) / max(1e-9, target_ratio * len(k_values)))
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values], fontweight="normal")
    ax.set_yticks(np.arange(len(sigma_values)))
    ax.set_yticklabels([_fmt_sigma(s) for s in sigma_values], fontweight="normal")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_ticks_position("top")
    ax.set_xlabel(r"$K$", labelpad=8.5)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(r"$\sigma$", labelpad=6)
    ax.set_xticks(np.arange(-0.5, len(k_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sigma_values), 1), minor=True)
    ax.grid(which="minor", color=str(PANEL_A_STYLE["minor_grid_color"]), linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    best_i = sigma_values.index(float(panel_ab_best_sigma))
    best_j = k_values.index(int(panel_ab_best_k))
    for i in range(len(sigma_values)):
        for j in range(len(k_values)):
            kl_val = float(panel_ab_grid[i, j])
            rgba = CMAP_KL(norm(kl_val))
            text_color = COLOR_DARK if norm(kl_val) <= 0.25 else "#FFFFFF"
            is_best = i == best_i and j == best_j
            ax.text(
                j,
                i - 0.02,
                f"{kl_val:.3f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color=text_color,
                fontweight="bold" if is_best else "normal",
            )
            if is_best:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        linewidth=1.0,
                        edgecolor=str(PANEL_A_STYLE["best_cell_border_color"]),
                    )
                )
    cbar = fig.colorbar(im, ax=ax, orientation=str(PANEL_A_STYLE["colorbar_orientation"]), fraction=0.046, pad=0.08, shrink=0.85)
    cbar.set_label(r"IQP Parity $D_{\mathrm{KL}}(p^{*}\,\|\,q)$", labelpad=4, fontsize=9)
    cbar.set_ticks([float(np.min(panel_ab_grid)), float(np.median(panel_ab_grid)), float(np.max(panel_ab_grid))])
    cbar.set_ticklabels(
        [
            f"{float(np.min(panel_ab_grid)):.3f}",
            f"{float(np.median(panel_ab_grid)):.3f}",
            f"{float(np.max(panel_ab_grid)):.3f}",
        ]
    )
    cbar.ax.tick_params(axis="x", length=2.5, labelsize=7, colors=COLOR_LIGHT_TEXT)
    cbar.outline.set_linewidth(0.4)

    out = outdir / str(outname)
    _save_fig(fig, out)
    return out


def render_rank_panel(
    *,
    outdir: Path,
    sigma_values: list[float],
    k_values: list[int],
    panel_ab_grid: np.ndarray,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE, constrained_layout=True)

    entries = []
    for i, sigma in enumerate(sigma_values):
        for j, kval in enumerate(k_values):
            entries.append({"sigma": float(sigma), "K": int(kval), "kl": float(panel_ab_grid[i, j])})
    entries = sorted(entries, key=lambda row: float(row["kl"]))
    y = np.arange(len(entries))[::-1]

    kl_min = float(min(row["kl"] for row in entries))
    kl_max = float(max(row["kl"] for row in entries))
    norm = colors.Normalize(vmin=0.0, vmax=kl_max)
    rank_cmap = CMAP_RANK
    x_left = 0.05 * math.floor((kl_min - 0.01) / 0.05)
    x_right = 0.05 * math.ceil((kl_max + 0.02) / 0.05)
    major_start = 0.1 * math.ceil(x_left / 0.1)
    label_pad = float(PANEL_B_STYLE["value_label_pad"])

    ax.set_xlim(x_left - 0.02, x_right + 0.01)
    ax.set_ylim(-0.5, len(entries) - 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [rf"$\sigma\!=\!{_fmt_sigma(row['sigma'])},\; K\!=\!{int(row['K'])}$" for row in entries],
        fontsize=8.0,
    )
    ax.set_xticks(np.arange(major_start, x_right + 1e-9, float(PANEL_B_STYLE["x_major_tick_step"])))
    ax.set_xlabel(r"$D_{\mathrm{KL}}(p^{*}\,\|\,q)$ (lower better)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for yi, row in zip(y, entries):
        kl_val = float(row["kl"])
        dot_color = rank_cmap(norm(kl_val))
        ax.hlines(yi, x_left, kl_val, color=str(PANEL_B_STYLE["guide_color"]), linewidth=float(PANEL_B_STYLE["guide_linewidth"]), zorder=0)
        ax.scatter(
            kl_val,
            yi,
            s=float(PANEL_B_STYLE["dot_size"]),
            color=dot_color,
            edgecolors="none",
            zorder=3,
        )
        ax.text(
            kl_val + label_pad,
            yi,
            f"{kl_val:.3f}",
            va="center",
            ha="left",
            fontsize=7.0,
            color=COLOR_TEXT,
            fontweight="bold" if abs(kl_val - kl_min) < 1e-12 else "normal",
        )

    out = outdir / "experiment_1_panel_b_sigma_k_rank_ordering.pdf"
    _save_fig(fig, out)
    return out


def render_heatmap_panel_legacy_consistent(
    *,
    outdir: Path,
    sigma_values: list[float],
    k_values: list[int],
    panel_ab_grid: np.ndarray,
    panel_ab_best_sigma: float,
    panel_ab_best_k: int,
    outname: str = "fig2_kl_summary_panels_heatmap_only_legacy_consistent_palette.pdf",
) -> Path:
    apply_style()
    fig = plt.figure(figsize=HEATMAP_FIGSIZE, constrained_layout=False)
    # Pin the heatmap/cbar geometry so the visible plot height matches the
    # neighboring paper panels instead of shrinking under constrained_layout.
    # Match the Experiment 1 panel-B/panel-C axes geometry so the heatmap's
    # visible plot box has the same height and x-axis baseline in composites.
    ax = fig.add_axes([0.12, 0.2266, 0.66, 0.7572])
    cax = fig.add_axes([0.80, 0.2266, 0.032, 0.7572])

    vmin = float(np.min(panel_ab_grid))
    vmax = float(np.max(panel_ab_grid))
    tick_lo = math.floor(vmin / 0.05) * 0.05
    tick_hi = math.floor(vmax / 0.05) * 0.05
    # Align the norm lower edge with the displayed lowest tick so the colorbar
    # does not show an empty under-range sliver below the first valid color.
    norm = colors.Normalize(vmin=tick_lo, vmax=vmax)
    im = ax.imshow(panel_ab_grid, cmap=CMAP_KL_CONSISTENT, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values], fontweight="normal")
    ax.set_yticks(np.arange(len(sigma_values)))
    ax.set_yticklabels([_fmt_sigma(s) for s in sigma_values], fontweight="normal")
    ax.tick_params(axis="x", pad=3.6)
    ax.tick_params(axis="y", pad=2)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_xlabel(r"$K$", labelpad=8)
    ax.set_ylabel(r"$\sigma$", labelpad=4)
    ax.yaxis.set_label_coords(-0.07, 0.5)

    best_i = sigma_values.index(float(panel_ab_best_sigma))
    best_j = k_values.index(int(panel_ab_best_k))
    best_rgba = CMAP_KL_CONSISTENT(norm(float(panel_ab_grid[best_i, best_j])))
    best_border_color = "#FFFFFF" if _luminance(best_rgba) < 0.62 else COLOR_DARK
    ax.add_patch(
        Rectangle(
            (best_j - 0.5, best_i - 0.5),
            1.0,
            1.0,
            fill=False,
            linewidth=1.4,
            linestyle="--",
            edgecolor=best_border_color,
        )
    )

    for i in range(len(sigma_values)):
        for j in range(len(k_values)):
            kl_val = float(panel_ab_grid[i, j])
            rgba = CMAP_KL_CONSISTENT(norm(kl_val))
            text_color = COLOR_DARK if _luminance(rgba) > 0.64 else "#FFFFFF"
            ax.text(
                j,
                i,
                f"{kl_val:.3f}",
                ha="center",
                va="center",
                fontsize=10.1,
                color=text_color,
                fontweight="bold" if (i == best_i and j == best_j) else "normal",
            )

    cbar_ticks = np.arange(tick_lo, tick_hi + 1e-9, 0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_ticks(cbar_ticks.tolist())
    if cbar.solids is not None:
        # Avoid renderer seam artifacts that show up as white slivers in the colorbar.
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_linewidth(0.0)
    cbar.ax.tick_params(axis="y", length=3.0, labelsize=8, colors=COLOR_LIGHT_TEXT)
    cbar.outline.set_linewidth(0.8)

    out = outdir / str(outname)
    _save_fig(fig, out)
    return out


def render_benchmark_panel(
    *,
    outdir: Path,
    best_mean: float,
    best_ci: float,
    mse_mean: float,
    mse_ci: float,
    uniform_kl: float,
    include_uniform: bool,
    best_sublabel: str = "",
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=LOLLIPOP_FIGSIZE, constrained_layout=True)

    benchmark_entries = [
        ("Target p*", COLOR_TEXT, 0.0, 0.0, ""),
        (f"Best {IQP_PARITY_LABEL}", "#ea8a7d", float(best_mean), float(best_ci), str(best_sublabel)),
        (IQP_MSE_LABEL, "#86afe8", float(mse_mean), float(mse_ci), ""),
    ]
    if include_uniform:
        benchmark_entries.append(("Uniform", "#cfd2d7", float(uniform_kl), 0.0, ""))
    y_positions = np.arange(len(benchmark_entries))[::-1]

    right_edge = max(val + ci for _, _, val, ci, _ in benchmark_entries)
    x_min = float(PANEL_C_STYLE["x_min"])
    x_max = max(float(PANEL_C_STYLE["x_floor"]), right_edge + float(PANEL_C_STYLE["x_max_padding"]))
    label_pad = max(0.014, 0.026 * (x_max - x_min))
    major_tick_step = float(PANEL_C_STYLE["major_xtick_step"])
    minor_tick_step = float(PANEL_C_STYLE["minor_xtick_step"])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, len(benchmark_entries) - 0.4)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, major_tick_step))
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, minor_tick_step), minor=True)
    ax.set_xlabel(r"$D_{\mathrm{KL}}(p^{*}\,\|\,q)$ (lower better)")
    ax.grid(True, axis="x", alpha=0.14, linestyle="--", dashes=(2, 2))
    ax.grid(True, which="minor", axis="x", alpha=0.08, linestyle="--", dashes=(2, 2))
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(0.0, color=COLOR_AXIS, linewidth=1.1, zorder=0)

    ytrans = ax.get_yaxis_transform()

    for y_pos, (label, color, kl_val, ci_val, sublabel) in zip(y_positions, benchmark_entries):
        display_kl = kl_val
        bar_end = display_kl if ci_val <= 0.0 else min(x_max, display_kl + 0.010)
        ax.plot([0.0, bar_end], [y_pos, y_pos], color=color, linewidth=8.0, alpha=0.88, solid_capstyle="butt", zorder=1)
        if ci_val > 0.0:
            ax.errorbar(
                display_kl,
                y_pos,
                xerr=np.asarray([[ci_val], [ci_val]], dtype=np.float64),
                fmt="none",
                ecolor=color,
                elinewidth=2.2,
                capsize=4.2,
                capthick=2.2,
                zorder=4,
            )
        marker_edge = color if ci_val <= 0.0 else "white"
        marker_lw = 1.0 if ci_val <= 0.0 else 1.4
        ax.scatter(display_kl, y_pos, s=72, color=color, edgecolors=marker_edge, linewidths=marker_lw, zorder=5, clip_on=False)
        label_x = -0.060 if abs(kl_val) <= 1e-12 else -0.035
        value_pad = label_pad if (abs(kl_val) > 1e-12 or ci_val > 0.0) else max(label_pad, 0.032)
        ax.text(
            label_x,
            y_pos + 0.02,
            label,
            transform=ytrans,
            fontsize=10.4,
            fontweight="normal",
            va="center",
            ha="right",
            color=COLOR_TEXT,
            clip_on=False,
        )
        ax.text(
            display_kl + ci_val + value_pad,
            y_pos,
            f"KL {kl_val:.3f}" if ci_val <= 0.0 else f"KL {kl_val:.3f} ± {ci_val:.3f}",
            fontsize=7.6,
            fontweight="normal",
            va="center",
            ha="left",
            color=COLOR_TEXT,
            clip_on=False,
        )
        if sublabel:
            ax.text(
                -0.035,
                y_pos - 0.34,
                sublabel,
                transform=ytrans,
                fontsize=6.1,
                va="top",
                ha="right",
                fontstyle="italic",
                color=COLOR_SUBTEXT,
                clip_on=False,
            )

    out = outdir / "experiment_1_panel_c_fixed_beta_comparison.pdf"
    _save_fig(fig, out)
    return out


def compute_experiment_data(
    *,
    beta: float,
    panel_ab_seed: int,
    panel_c_seeds: list[int],
    n: int,
    train_m: int,
    layers: int,
    iqp_steps: int,
    iqp_lr: float,
) -> dict:
    all_seeds = []
    seen = set()
    for seed in [int(panel_ab_seed)] + [int(s) for s in panel_c_seeds]:
        if seed not in seen:
            seen.add(seed)
            all_seeds.append(seed)

    p_star, _support, _scores = build_target_distribution_paper(int(n), float(beta))
    bits_table = make_bits_table(int(n))
    kl_grid_by_seed = np.zeros((len(all_seeds), len(SIGMA_VALUES), len(K_VALUES)), dtype=np.float64)
    mse_kl_by_seed = np.zeros(len(all_seeds), dtype=np.float64)
    rows = []

    for seed_idx, seed in enumerate(all_seeds):
        idxs_train = sample_indices(p_star, int(train_m), seed=int(seed))
        emp = empirical_dist(idxs_train, 2 ** int(n))

        for i, sigma in enumerate(SIGMA_VALUES):
            for j, kval in enumerate(K_VALUES):
                alphas = sample_alphas(int(n), float(sigma), int(kval), seed=int(seed) + 222)
                P = build_parity_matrix(alphas, bits_table)
                z_data = P @ emp
                q_parity = train_iqp_qcbm(
                    n=int(n),
                    layers=int(layers),
                    steps=int(iqp_steps),
                    lr=float(iqp_lr),
                    P=P,
                    z_data=z_data,
                    seed_init=int(seed) + 10000 + 7 * int(kval),
                )
                kl_val = forward_kl(p_star, q_parity)
                kl_grid_by_seed[seed_idx, i, j] = float(kl_val)
                rows.append(
                    {
                        "family": "iqp_parity",
                        "seed": int(seed),
                        "sigma": float(sigma),
                        "K": int(kval),
                        "KL_pstar_to_q": float(kl_val),
                    }
                )

        q_mse = train_iqp_qcbm_prob_mse(
            n=int(n),
            layers=int(layers),
            steps=int(iqp_steps),
            lr=float(iqp_lr),
            emp_dist=emp,
            seed_init=int(seed) + 20000 + 7 * 512,
        )
        kl_mse = forward_kl(p_star, q_mse)
        mse_kl_by_seed[seed_idx] = float(kl_mse)
        rows.append(
            {
                "family": "iqp_mse",
                "seed": int(seed),
                "sigma": "",
                "K": "",
                "KL_pstar_to_q": float(kl_mse),
            }
        )

    all_seed_to_idx = {int(seed): idx for idx, seed in enumerate(all_seeds)}
    panel_ab_idx = int(all_seed_to_idx[int(panel_ab_seed)])
    panel_ab_grid = np.asarray(kl_grid_by_seed[panel_ab_idx], dtype=np.float64)
    panel_ab_best_flat_idx = int(np.argmin(panel_ab_grid))
    panel_ab_best_i, panel_ab_best_j = np.unravel_index(panel_ab_best_flat_idx, panel_ab_grid.shape)
    panel_ab_best_sigma = float(SIGMA_VALUES[panel_ab_best_i])
    panel_ab_best_k = int(K_VALUES[panel_ab_best_j])

    panel_c_indices = np.asarray([all_seed_to_idx[int(s)] for s in panel_c_seeds], dtype=np.int64)
    panel_c_kl_grids = np.asarray(kl_grid_by_seed[panel_c_indices], dtype=np.float64)
    panel_c_mse_vals = np.asarray(mse_kl_by_seed[panel_c_indices], dtype=np.float64)

    kl_flat = panel_c_kl_grids.reshape(len(panel_c_seeds), -1)
    panel_c_seedwise_best_flat_idx = np.argmin(kl_flat, axis=1)
    panel_c_seedwise_best_vals = kl_flat[np.arange(len(panel_c_seeds)), panel_c_seedwise_best_flat_idx]
    panel_c_seedwise_best_sigma = np.asarray(
        [SIGMA_VALUES[int(idx // len(K_VALUES))] for idx in panel_c_seedwise_best_flat_idx],
        dtype=np.float64,
    )
    panel_c_seedwise_best_k = np.asarray(
        [K_VALUES[int(idx % len(K_VALUES))] for idx in panel_c_seedwise_best_flat_idx],
        dtype=np.int64,
    )

    return {
        "beta": float(beta),
        "all_seeds": np.asarray(all_seeds, dtype=np.int64),
        "panel_ab_seed": np.asarray([int(panel_ab_seed)], dtype=np.int64),
        "panel_c_seeds": np.asarray(panel_c_seeds, dtype=np.int64),
        "sigma_values": np.asarray(SIGMA_VALUES, dtype=np.float64),
        "k_values": np.asarray(K_VALUES, dtype=np.int64),
        "kl_grid_by_seed": np.asarray(kl_grid_by_seed, dtype=np.float64),
        "mse_kl_by_seed": np.asarray(mse_kl_by_seed, dtype=np.float64),
        "panel_ab_grid": panel_ab_grid,
        "panel_ab_best_sigma": np.asarray([panel_ab_best_sigma], dtype=np.float64),
        "panel_ab_best_k": np.asarray([panel_ab_best_k], dtype=np.int64),
        "panel_c_seedwise_best_vals": np.asarray(panel_c_seedwise_best_vals, dtype=np.float64),
        "panel_c_seedwise_best_sigma": np.asarray(panel_c_seedwise_best_sigma, dtype=np.float64),
        "panel_c_seedwise_best_k": np.asarray(panel_c_seedwise_best_k, dtype=np.int64),
        "panel_c_mse_vals": np.asarray(panel_c_mse_vals, dtype=np.float64),
        "uniform_kl": np.asarray([forward_kl(p_star, np.ones_like(p_star, dtype=np.float64) / float(p_star.size))], dtype=np.float64),
        "points_rows": rows,
    }


def save_experiment_data(outdir: Path, data: dict, *, n: int, train_m: int, layers: int, iqp_steps: int, iqp_lr: float) -> None:
    np.savez_compressed(
        outdir / "experiment_1_data.npz",
        beta=np.asarray([data["beta"]], dtype=np.float64),
        all_seeds=np.asarray(data["all_seeds"], dtype=np.int64),
        panel_ab_seed=np.asarray(data["panel_ab_seed"], dtype=np.int64),
        panel_c_seeds=np.asarray(data["panel_c_seeds"], dtype=np.int64),
        sigma_values=np.asarray(data["sigma_values"], dtype=np.float64),
        k_values=np.asarray(data["k_values"], dtype=np.int64),
        kl_grid_by_seed=np.asarray(data["kl_grid_by_seed"], dtype=np.float64),
        mse_kl_by_seed=np.asarray(data["mse_kl_by_seed"], dtype=np.float64),
        panel_ab_grid=np.asarray(data["panel_ab_grid"], dtype=np.float64),
        panel_ab_best_sigma=np.asarray(data["panel_ab_best_sigma"], dtype=np.float64),
        panel_ab_best_k=np.asarray(data["panel_ab_best_k"], dtype=np.int64),
        panel_c_seedwise_best_vals=np.asarray(data["panel_c_seedwise_best_vals"], dtype=np.float64),
        panel_c_seedwise_best_sigma=np.asarray(data["panel_c_seedwise_best_sigma"], dtype=np.float64),
        panel_c_seedwise_best_k=np.asarray(data["panel_c_seedwise_best_k"], dtype=np.int64),
        panel_c_mse_vals=np.asarray(data["panel_c_mse_vals"], dtype=np.float64),
        uniform_kl=np.asarray(data["uniform_kl"], dtype=np.float64),
    )

    with (outdir / "experiment_1_points.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "seed", "sigma", "K", "KL_pstar_to_q"])
        writer.writeheader()
        writer.writerows(list(data["points_rows"]))

    panel_c_summary = _panel_c_global_best_summary(data)
    panel_c_best_mean = float(panel_c_summary["best_mean"])
    panel_c_best_ci95 = float(panel_c_summary["best_ci95"])
    panel_c_mse_mean = float(np.mean(np.asarray(data["panel_c_mse_vals"], dtype=np.float64)))
    panel_c_mse_ci95 = float(_ci95_halfwidth(np.asarray(data["panel_c_mse_vals"], dtype=np.float64)))
    style_metadata = _approved_style_metadata()

    with (outdir / "RUN_CONFIG.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "script": SCRIPT_REL,
                "outdir": str(outdir.relative_to(ROOT)),
                "protocol": "panel_ab_fixed_seed__panel_c_global_best_over_grid_by_mean",
                "beta": float(data["beta"]),
                "panel_ab_seed": int(np.asarray(data["panel_ab_seed"])[0]),
                "panel_c_seeds": np.asarray(data["panel_c_seeds"], dtype=np.int64).tolist(),
                "n": int(n),
                "train_m": int(train_m),
                "layers": int(layers),
                "iqp_steps": int(iqp_steps),
                "iqp_lr": float(iqp_lr),
                "sigma_values": np.asarray(data["sigma_values"], dtype=np.float64).tolist(),
                "k_values": np.asarray(data["k_values"], dtype=np.int64).tolist(),
                "panel_ab_best_sigma": float(np.asarray(data["panel_ab_best_sigma"])[0]),
                "panel_ab_best_k": int(np.asarray(data["panel_ab_best_k"])[0]),
                "panel_c_global_best_sigma": float(panel_c_summary["best_sigma"]),
                "panel_c_global_best_k": int(panel_c_summary["best_k"]),
                "panel_c_best_parity_mean": panel_c_best_mean,
                "panel_c_best_parity_ci95": panel_c_best_ci95,
                "panel_c_iqp_mse_mean": panel_c_mse_mean,
                "panel_c_iqp_mse_ci95": panel_c_mse_ci95,
                "uniform_kl": float(np.asarray(data["uniform_kl"])[0]),
                "panel_c_include_uniform": PANEL_C_INCLUDE_UNIFORM,
                **style_metadata,
            },
            f,
            indent=2,
        )

    with (outdir / "RERENDER_CONFIG.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "script": SCRIPT_REL,
                "data_npz": str((outdir / "experiment_1_data.npz").relative_to(ROOT)),
                "outdir": str(outdir.relative_to(ROOT)),
                "rerender_command": f"MPLCONFIGDIR=/tmp/mpl-cache python {SCRIPT_REL} --rerender-only 1 --data-npz {(outdir / 'experiment_1_data.npz').relative_to(ROOT)} --outdir {outdir.relative_to(ROOT)}",
            },
            f,
            indent=2,
        )


def load_experiment_data(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        return {
            "beta": float(z["beta"][0]),
            "all_seeds": np.asarray(z["all_seeds"], dtype=np.int64),
            "panel_ab_seed": np.asarray(z["panel_ab_seed"], dtype=np.int64),
            "panel_c_seeds": np.asarray(z["panel_c_seeds"], dtype=np.int64),
            "sigma_values": np.asarray(z["sigma_values"], dtype=np.float64),
            "k_values": np.asarray(z["k_values"], dtype=np.int64),
            "kl_grid_by_seed": np.asarray(z["kl_grid_by_seed"], dtype=np.float64),
            "mse_kl_by_seed": np.asarray(z["mse_kl_by_seed"], dtype=np.float64),
            "panel_ab_grid": np.asarray(z["panel_ab_grid"], dtype=np.float64),
            "panel_ab_best_sigma": np.asarray(z["panel_ab_best_sigma"], dtype=np.float64),
            "panel_ab_best_k": np.asarray(z["panel_ab_best_k"], dtype=np.int64),
            "panel_c_seedwise_best_vals": np.asarray(z["panel_c_seedwise_best_vals"], dtype=np.float64),
            "panel_c_seedwise_best_sigma": np.asarray(z["panel_c_seedwise_best_sigma"], dtype=np.float64),
            "panel_c_seedwise_best_k": np.asarray(z["panel_c_seedwise_best_k"], dtype=np.int64),
            "panel_c_mse_vals": np.asarray(z["panel_c_mse_vals"], dtype=np.float64),
            "uniform_kl": np.asarray(z["uniform_kl"], dtype=np.float64),
        }


def render_all_panels(*, outdir: Path, data: dict) -> None:
    panel_c_summary = _panel_c_global_best_summary(data)
    panel_c_best_mean = float(panel_c_summary["best_mean"])
    panel_c_best_ci95 = float(panel_c_summary["best_ci95"])
    panel_c_mse_mean = float(np.mean(np.asarray(data["panel_c_mse_vals"], dtype=np.float64)))
    panel_c_mse_ci95 = float(_ci95_halfwidth(np.asarray(data["panel_c_mse_vals"], dtype=np.float64)))

    render_heatmap_panel(
        outdir=outdir,
        sigma_values=[float(x) for x in np.asarray(data["sigma_values"], dtype=np.float64).tolist()],
        k_values=[int(x) for x in np.asarray(data["k_values"], dtype=np.int64).tolist()],
        panel_ab_grid=np.asarray(data["panel_ab_grid"], dtype=np.float64),
        panel_ab_best_sigma=float(np.asarray(data["panel_ab_best_sigma"])[0]),
        panel_ab_best_k=int(np.asarray(data["panel_ab_best_k"])[0]),
    )
    render_heatmap_panel_legacy_consistent(
        outdir=outdir,
        sigma_values=[float(x) for x in np.asarray(data["sigma_values"], dtype=np.float64).tolist()],
        k_values=[int(x) for x in np.asarray(data["k_values"], dtype=np.int64).tolist()],
        panel_ab_grid=np.asarray(data["panel_ab_grid"], dtype=np.float64),
        panel_ab_best_sigma=float(np.asarray(data["panel_ab_best_sigma"])[0]),
        panel_ab_best_k=int(np.asarray(data["panel_ab_best_k"])[0]),
    )
    render_rank_panel(
        outdir=outdir,
        sigma_values=[float(x) for x in np.asarray(data["sigma_values"], dtype=np.float64).tolist()],
        k_values=[int(x) for x in np.asarray(data["k_values"], dtype=np.int64).tolist()],
        panel_ab_grid=np.asarray(data["panel_ab_grid"], dtype=np.float64),
    )
    render_benchmark_panel(
        outdir=outdir,
        best_mean=panel_c_best_mean,
        best_ci=panel_c_best_ci95,
        mse_mean=panel_c_mse_mean,
        mse_ci=panel_c_mse_ci95,
        uniform_kl=float(np.asarray(data["uniform_kl"])[0]),
        include_uniform=PANEL_C_INCLUDE_UNIFORM,
        best_sublabel=rf"fixed global-best $(\sigma,K)=({panel_c_summary['best_sigma']:g},{int(panel_c_summary['best_k'])})$",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 1 standalone KL diagnostics panels.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "plots" / "experiment_1_kl_diagnostics"),
    )
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--panel-ab-seed", type=int, default=42)
    ap.add_argument("--panel-c-seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--train-m", type=int, default=200)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--iqp-steps", type=int, default=600)
    ap.add_argument("--iqp-lr", type=float, default=0.05)
    ap.add_argument("--data-npz", type=str, default="")
    ap.add_argument("--rerender-only", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_npz = Path(args.data_npz) if str(args.data_npz).strip() else (outdir / "experiment_1_data.npz")

    if int(args.rerender_only) == 1:
        if not data_npz.exists():
            raise FileNotFoundError(f"Missing data npz for rerender: {data_npz}")
        data = load_experiment_data(data_npz)
        render_all_panels(outdir=outdir, data=data)
    else:
        if not HAS_PENNYLANE:
            raise RuntimeError("Pennylane is required for this standalone script.")
        panel_c_seeds = _parse_int_list(args.panel_c_seeds)
        data = compute_experiment_data(
            beta=float(args.beta),
            panel_ab_seed=int(args.panel_ab_seed),
            panel_c_seeds=panel_c_seeds,
            n=int(args.n),
            train_m=int(args.train_m),
            layers=int(args.layers),
            iqp_steps=int(args.iqp_steps),
            iqp_lr=float(args.iqp_lr),
        )
        save_experiment_data(
            outdir,
            data,
            n=int(args.n),
            train_m=int(args.train_m),
            layers=int(args.layers),
            iqp_steps=int(args.iqp_steps),
            iqp_lr=float(args.iqp_lr),
        )
        render_all_panels(outdir=outdir, data=data)

    print(f"[saved] {outdir / 'experiment_1_panel_a_sigma_k_heatmap.pdf'}")
    print(f"[saved] {outdir / 'experiment_1_panel_b_sigma_k_rank_ordering.pdf'}")
    print(f"[saved] {outdir / 'experiment_1_panel_c_fixed_beta_comparison.pdf'}")


if __name__ == "__main__":
    main()
