#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master script: generate all 4 composite paper figures.

Produces:
  outputs/paper_figures/fig1.pdf  — Recovery & Mechanism (3 panels)
  outputs/paper_figures/fig2.pdf  — Robustness & Budget Law (3 panels)
  outputs/paper_figures/fig3.pdf  — Fit ≠ Discovery (2 panels)
  outputs/paper_figures/fig4.pdf  — Fair Classical Baseline (3 panels)

Usage:
  python experiments/make_paper_figures.py

Dependencies:
  pip install numpy matplotlib scipy pennylane torch
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTDIR = ROOT / "outputs" / "paper_figures"
SPEC_COLOR = "#555555"
SPEC_LS = (0, (7, 2, 1.2, 2))  # dash-gap-dot-gap

LEGEND_STYLE = dict(
    loc="lower right",
    fontsize=7.0,
    frameon=True,
    framealpha=0.90,
    facecolor="white",
    edgecolor="none",
    handlelength=1.6,
    labelspacing=0.25,
    borderpad=0.25,
    handletextpad=0.5,
    borderaxespad=0.2,
)

CMAP_REDBLACK = LinearSegmentedColormap.from_list("RedBlack", ["#FF0000", "#000000"])
CMAP_BLUEBLACK = LinearSegmentedColormap.from_list(
    "BlueBlack", ["#DCEBFA", "#1F77B4", "#000000"]
)


def _q_grid(Qmax: int = 10000) -> np.ndarray:
    Q = np.unique(
        np.concatenate([
            np.unique(np.logspace(0, 3.5, 120).astype(int)),
            np.linspace(1000, Qmax, 160).astype(int),
            np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
        ])
    )
    return Q[(Q >= 0) & (Q <= Qmax)]


def _ising_color(rank: int, m: int):
    if m <= 1:
        t, r = 0.95, 0.0
    else:
        r = rank / (m - 1)
        t = 0.95 - 0.90 * r
    color = plt.cm.Blues(t)
    alpha = max(0.35, 0.90 - 0.50 * r)
    return color, alpha


def _best_case_Q80_lb(qH_ratio: float, N: int, thr: float = 0.8) -> float:
    if not np.isfinite(qH_ratio) or qH_ratio <= 0:
        return float("nan")
    p = float(qH_ratio) / float(N)
    if not (0.0 < p < 1.0):
        return float("nan")
    return float(np.log(1.0 - thr) / np.log(1.0 - p))


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _base_config(outdir: str) -> hv.Config:
    return hv.Config(
        n=12,
        beta=0.9,
        train_m=1000,
        holdout_k=20,
        holdout_pool=400,
        seed=42,
        good_frac=0.05,
        sigmas=[0.5, 1.0, 2.0, 3.0],
        Ks=[128, 256, 512],
        Qmax=10000,
        Q80_thr=0.8,
        Q80_search_max=200000,
        use_iqp=True,
        use_classical=True,
        iqp_steps=600,
        iqp_lr=0.05,
        iqp_eval_every=50,
        iqp_layers=1,
        adversarial=False,
        outdir=outdir,
    )


def _setup_target(cfg: hv.Config):
    bits_table = hv.make_bits_table(cfg.n)
    p_star, support, scores = hv.build_target_distribution_paper(cfg.n, cfg.beta)
    good_mask = hv.topk_mask_by_scores(scores, support, frac=cfg.good_frac)
    holdout_mask = hv.select_holdout_smart(
        p_star=p_star,
        good_mask=good_mask,
        bits_table=bits_table,
        m_train=cfg.train_m,
        holdout_k=cfg.holdout_k,
        pool_size=cfg.holdout_pool,
        seed=cfg.seed + 111,
    )
    return bits_table, p_star, support, scores, good_mask, holdout_mask


# ===========================================================================
# Figure 1 — Recovery & Mechanism  (3 panels)
# ===========================================================================

def _plot_recovery_panel(
    ax,
    cfg: hv.Config,
    p_star: np.ndarray,
    holdout_mask: np.ndarray,
    bits_table: np.ndarray,
    results: List[Dict],
    sigma_show: float,
    K_show: int,
):
    """Plot recovery R(Q) panel with IQP, spectral, all Ising, uniform."""
    Q = _q_grid(cfg.Qmax)

    # Best IQP at the given (sigma, K)
    cfg_best = replace(cfg, use_iqp=True, use_classical=False, sigmas=[sigma_show], Ks=[K_show])
    art = hv.rerun_single_setting(
        cfg=cfg_best, p_star=p_star, holdout_mask=holdout_mask,
        bits_table=bits_table, sigma=sigma_show, K=K_show, return_hist=False,
    )
    q_iqp = art["q_iqp"]
    q_spec = art["q_spec"]

    u = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
    y_unif = hv.expected_unique_fraction(u, holdout_mask, Q)
    y_iqp = hv.expected_unique_fraction(q_iqp, holdout_mask, Q)
    y_spec = hv.expected_unique_fraction(q_spec, holdout_mask, Q)

    # All Ising controls (ranked)
    ising_rows = []
    for r in results:
        ising_rows.append((float(r["sigma"]), int(r["K"]),
                           float(r["Q80_class"]), float(r["qH_ratio_class"])))

    def _key(tup):
        _, _, Q80c, qHr = tup
        finite = np.isfinite(Q80c)
        return (0 if finite else 1, Q80c if finite else 1e99, -qHr)

    ising_rows.sort(key=_key)
    m = len(ising_rows)

    cfg_cl = replace(cfg, use_iqp=False, use_classical=True)
    for idx, (sig, k, _, _) in enumerate(ising_rows):
        art_cl = hv.rerun_single_setting(
            cfg=replace(cfg_cl, sigmas=[sig], Ks=[k]),
            p_star=p_star, holdout_mask=holdout_mask,
            bits_table=bits_table, sigma=sig, K=k, return_hist=False,
        )
        q_cl = art_cl["q_class"]
        y_cl = hv.expected_unique_fraction(q_cl, holdout_mask, Q)
        color, alpha = _ising_color(idx, m)
        ax.plot(Q, y_cl, color=color, alpha=alpha, linewidth=1.2, zorder=2)

    ax.plot(Q, y_unif, color=hv.COLORS["gray"], linestyle="--", linewidth=1.5, alpha=0.9, zorder=1)
    ax.plot(Q, y_spec, color=SPEC_COLOR, linestyle=SPEC_LS, linewidth=1.9, zorder=4)
    ax.plot(Q, y_star, color=hv.COLORS["target"], linewidth=1.9, zorder=5)
    ax.plot(Q, y_iqp, color=hv.COLORS["model"], linewidth=2.4, zorder=6)

    ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlim(0, cfg.Qmax)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")

    legend_handles = [
        Line2D([0], [0], color=hv.COLORS["target"], lw=1.9, label=r"Target $p^*$"),
        Line2D([0], [0], color=hv.COLORS["model"], lw=2.4,
               label=fr"IQP-QCBM ($\sigma$={sigma_show:g}, $K$={K_show})"),
        Line2D([0], [0], color=SPEC_COLOR, lw=1.9, ls=SPEC_LS, label=r"Spectral $\tilde{q}$"),
        Line2D([0], [0], color=plt.cm.Blues(0.90), lw=1.6, label="Ising controls"),
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform"),
    ]
    ax.legend(handles=legend_handles, **LEGEND_STYLE)


def _visibility_computation(
    n: int, beta: float, sigma: float, K: int,
    holdout_k: int, train_m: int, seed: int,
    good_frac: float,
):
    """Replicate exp03 visibility logic: return masks, completions, p_star."""
    from experiments.exp03_visibility_minvis import (
        build_target_distribution, topk_mask, sample_alphas,
        build_parity_matrix, greedy_min_abs_sum,
        linear_band_reconstruction, completion_by_axioms,
        sample_indices, empirical_dist, expected_unique_fraction as euf_vis,
    )
    N = 2 ** n
    bits_table = np.array([[(k >> i) & 1 for i in range(n - 1, -1, -1)]
                           for k in range(N)], dtype=np.int8)
    p_star, support, scores = build_target_distribution(n, beta)
    good_mask = topk_mask(scores, support, frac=good_frac)

    s_int = scores.astype(int)
    score_level = 7
    cand = np.where(good_mask & (s_int == score_level))[0]
    if cand.size < 2 * holdout_k:
        levels = sorted(set(s_int[good_mask].tolist()))
        viable = [lv for lv in levels if np.where(good_mask & (s_int == lv))[0].size >= 2 * holdout_k]
        score_level = min(viable, key=lambda lv: abs(lv - 7))
        cand = np.where(good_mask & (s_int == score_level))[0]

    # Try sigma grid for best cancellation
    sigma_grid = [0.5, 1.0, 2.0, 3.0]
    best = None
    for gi, sig in enumerate(sigma_grid):
        alphas = sample_alphas(n, sig, K, seed=seed + 999 + 17 * gi)
        P = build_parity_matrix(alphas, bits_table)
        z_ref = P @ p_star
        r_ref = P.T @ z_ref
        cand_vals = r_ref[cand]

        H_inv, _ = greedy_min_abs_sum(
            cand_idxs=cand, cand_vals=cand_vals,
            k=holdout_k, improve_steps=2000,
            seed=seed + 12345 + 31 * gi,
        )
        inv_set = set(int(i) for i in H_inv.tolist())
        remaining = np.array([int(i) for i in cand.tolist() if int(i) not in inv_set], dtype=int)
        if remaining.size >= holdout_k:
            rem_vals = r_ref[remaining]
            order_vis = np.argsort(-rem_vals)
            H_vis = remaining[order_vis[:holdout_k]]
        else:
            order_vis = np.argsort(-cand_vals)
            H_vis = cand[order_vis[:holdout_k]]

        Vis_inv = abs(float(np.sum(r_ref[H_inv]) / N))
        Vis_vis = abs(float(np.sum(r_ref[H_vis]) / N))
        if best is None or Vis_inv < best["obj"]:
            best = {"sigma": sig, "P": P, "H_vis": H_vis, "H_inv": H_inv, "obj": Vis_inv}

    sigma_used = best["sigma"]
    P = best["P"]
    H_vis = best["H_vis"]
    H_inv = best["H_inv"]

    def _completion(holdout_idxs, seed_offset):
        mask = np.zeros(N, dtype=bool)
        mask[holdout_idxs] = True
        p_train = p_star.copy()
        p_train[mask] = 0.0
        p_train /= float(np.sum(p_train))
        idxs_train = sample_indices(p_train, train_m, seed=seed + 7 + seed_offset)
        emp = empirical_dist(idxs_train, N)
        z_train = P @ emp
        q_lin = linear_band_reconstruction(P, z_train, n)
        q_tilde = completion_by_axioms(q_lin)
        return q_tilde, mask

    q_vis, mask_vis = _completion(H_vis, seed_offset=0)
    q_inv, mask_inv = _completion(H_inv, seed_offset=123)

    return p_star, mask_vis, mask_inv, q_vis, q_inv


def figure_1(outdir: Path) -> None:
    """Recovery & Mechanism: (a) σ=2,K=256; (b) σ=1,K=512; (c) Visibility."""
    print("[Fig 1] Computing recovery panels + visibility...")
    cfg = _base_config(str(outdir))
    bits_table, p_star, support, scores, good_mask, holdout_mask = _setup_target(cfg)
    Q = _q_grid(cfg.Qmax)

    # Run full sweep once
    results = hv.run_sweep(cfg, p_star, holdout_mask, good_mask, bits_table)

    fig, axes = plt.subplots(1, 3, figsize=hv.fig_size("full", 2.6))
    plt.subplots_adjust(wspace=0.38)

    # Panel (a): σ=2, K=256
    hv._panel_label(axes[0], "(a)")
    _plot_recovery_panel(axes[0], cfg, p_star, holdout_mask, bits_table, results,
                         sigma_show=2.0, K_show=256)

    # Panel (b): σ=1, K=512
    hv._panel_label(axes[1], "(b)")
    _plot_recovery_panel(axes[1], cfg, p_star, holdout_mask, bits_table, results,
                         sigma_show=1.0, K_show=512)

    # Panel (c): Visibility
    hv._panel_label(axes[2], "(c)")
    p_vis, mask_vis, mask_inv, q_vis, q_inv = _visibility_computation(
        n=12, beta=0.9, sigma=1.0, K=512, holdout_k=20,
        train_m=1000, seed=42, good_frac=0.05,
    )

    y_star = hv.expected_unique_fraction(p_vis, mask_vis, Q)
    y_vis = hv.expected_unique_fraction(q_vis, mask_vis, Q)
    y_inv = hv.expected_unique_fraction(q_inv, mask_inv, Q)
    u = np.ones_like(p_vis, dtype=np.float64) / p_vis.size
    y_u = hv.expected_unique_fraction(u, mask_vis, Q)

    ax = axes[2]
    ax.plot(Q, y_star, color=hv.COLORS["target"], linewidth=1.9, zorder=5)
    ax.plot(Q, y_vis, color="#1f77b4", linewidth=2.2, zorder=6)
    ax.plot(Q, y_inv, color="#1f77b4", linestyle="--", linewidth=2.0, zorder=4)
    ax.plot(Q, y_u, color=hv.COLORS["gray"], linestyle=":", linewidth=1.6, zorder=3)
    ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)
    ax.set_xlim(0, cfg.Qmax)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")

    vis_handles = [
        Line2D([0], [0], color=hv.COLORS["target"], lw=1.9, label=r"Target $p^*$"),
        Line2D([0], [0], color="#1f77b4", lw=2.2, label=r"Visible $\mathcal{H}_{\mathrm{vis}}$"),
        Line2D([0], [0], color="#1f77b4", lw=2.0, ls="--",
               label=r"Invisible $\mathcal{H}_{\mathrm{inv}}$"),
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.6, ls=":", label="Uniform"),
    ]
    ax.legend(handles=vis_handles, **LEGEND_STYLE)

    fig.savefig(str(outdir / "fig1.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {outdir / 'fig1.pdf'}")
    return results  # reuse for figure 2


# ===========================================================================
# Figure 2 — Robustness & Budget Law  (3 panels)
# ===========================================================================

def _heatmap_on_ax(ax, mat, row_labels, col_labels, cmap, cbar_label, fmt="{:.0f}", log10=False):
    """Render heatmap onto an existing axis."""
    data = np.array(mat, dtype=np.float64)
    if log10:
        plot_data = hv._safe_log10_for_heatmap(data)
    else:
        plot_data = data

    im = ax.imshow(plot_data, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("K")
    ax.set_ylabel(r"$\sigma$")

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.18)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(cbar_label)

    max_pd = float(np.max(plot_data[np.isfinite(plot_data)])) if np.isfinite(plot_data).any() else 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = r"$\infty$" if np.isinf(val) else fmt.format(val)
            text_col = "white"
            if isinstance(cmap, str) and cmap in ("Reds", "Blues"):
                if plot_data[i, j] < max_pd * 0.5:
                    text_col = "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=text_col, fontsize=7, fontweight="bold")


def figure_2(outdir: Path, cfg: hv.Config, results: List[Dict]) -> None:
    """Robustness & Budget Law: (a) IQP Q80 heatmap; (b) Ising Q80 heatmap; (c) budget-law scatter."""
    print("[Fig 2] Computing heatmaps + budget-law scatter...")

    Q80_iqp = hv.build_result_matrix(results, cfg.sigmas, cfg.Ks, key="Q80_iqp")
    Q80_cl = hv.build_result_matrix(results, cfg.sigmas, cfg.Ks, key="Q80_class")

    row_labels = [str(s) for s in cfg.sigmas]
    col_labels = [str(k) for k in cfg.Ks]

    fig, axes = plt.subplots(1, 3, figsize=hv.fig_size("full", 2.6))
    plt.subplots_adjust(wspace=0.45)

    # Panel (a): IQP Q80 heatmap
    hv._panel_label(axes[0], "(a)")
    _heatmap_on_ax(axes[0], Q80_iqp, row_labels, col_labels,
                   cmap=CMAP_REDBLACK, cbar_label=r"$\log_{10} Q_{80}$",
                   fmt="{:.0f}", log10=True)

    # Panel (b): Ising Q80 heatmap
    hv._panel_label(axes[1], "(b)")
    _heatmap_on_ax(axes[1], Q80_cl, row_labels, col_labels,
                   cmap=CMAP_BLUEBLACK, cbar_label=r"$\log_{10} Q_{80}$",
                   fmt="{:.0f}", log10=True)

    # Panel (c): Budget-law scatter
    hv._panel_label(axes[2], "(c)")
    ax = axes[2]
    N = 2 ** cfg.n
    thr = cfg.Q80_thr
    cap = float(cfg.Q80_search_max)

    iqp_xy, cl_xy = [], []
    iqp_cap, cl_cap = [], []

    for r in results:
        # IQP
        qHr = float(r["qH_ratio_iqp"])
        Q80 = float(r["Q80_iqp"])
        x = _best_case_Q80_lb(qHr, N, thr)
        if np.isfinite(x) and x > 0:
            if np.isfinite(Q80) and Q80 > 0:
                iqp_xy.append((x, Q80))
            else:
                iqp_cap.append((x, cap))
        # Classical
        qHr_c = float(r["qH_ratio_class"])
        Q80_c = float(r["Q80_class"])
        x_c = _best_case_Q80_lb(qHr_c, N, thr)
        if np.isfinite(x_c) and x_c > 0:
            if np.isfinite(Q80_c) and Q80_c > 0:
                cl_xy.append((x_c, Q80_c))
            else:
                cl_cap.append((x_c, cap))

    IQP_COLOR = hv.COLORS["model"]
    ISING_COLOR = "#1F77B4"

    if iqp_xy:
        ax.scatter([x for x, y in iqp_xy], [y for x, y in iqp_xy],
                   s=26, c=IQP_COLOR, marker="o", edgecolors="white",
                   linewidths=0.4, alpha=0.90, zorder=3)
    if cl_xy:
        ax.scatter([x for x, y in cl_xy], [y for x, y in cl_xy],
                   s=26, c=ISING_COLOR, marker="o", edgecolors="white",
                   linewidths=0.4, alpha=0.90, zorder=3)

    all_vals = [v for pts in (iqp_xy, iqp_cap, cl_xy, cl_cap)
                for x, y in pts for v in (x, y) if np.isfinite(v) and v > 0]
    if all_vals:
        lo = max(min(all_vals) * 0.8, 1e-6)
        hi = max(all_vals) * 1.25
    else:
        lo, hi = 1, 1e5

    xx = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(xx, xx, color=hv.COLORS["gray"], ls=":", lw=1.2, alpha=0.9, zorder=1)
    ax.plot(xx, 2 * xx, color=hv.COLORS["gray"], ls="--", lw=1.2, alpha=0.9, zorder=1)
    ax.plot(xx, 5 * xx, color=hv.COLORS["gray"], ls="-.", lw=1.2, alpha=0.9, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Best-case $Q_{80}^{\mathrm{lb}}$")
    ax.set_ylabel(r"Measured $Q_{80}$")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.45)

    scatter_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=IQP_COLOR,
               markeredgecolor="white", markeredgewidth=0.4, markersize=5.5, label="IQP-QCBM"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ISING_COLOR,
               markeredgecolor="white", markeredgewidth=0.4, markersize=5.5, label="Ising control"),
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.2, ls=":", label=r"$y=x$"),
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.2, ls="--", label=r"$y=2x$"),
        Line2D([0], [0], color=hv.COLORS["gray"], lw=1.2, ls="-.", label=r"$y=5x$"),
    ]
    ax.legend(handles=scatter_handles, **LEGEND_STYLE)

    fig.savefig(str(outdir / "fig2.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {outdir / 'fig2.pdf'}")


# ===========================================================================
# Figure 3 — Fit ≠ Discovery  (2 panels)
# ===========================================================================

def figure_3(outdir: Path) -> None:
    """Fit ≠ Discovery: (a) parity vs prob recovery; (b) Q80 ratio across K."""
    print("[Fig 3] Computing parity vs prob-MSE comparison...")

    cfg = _base_config(str(outdir))
    bits_table, p_star, support, scores, good_mask, holdout_mask = _setup_target(cfg)
    Q = _q_grid(cfg.Qmax)
    N = p_star.size
    H_size = int(np.sum(holdout_mask))
    q_unif = np.ones(N, dtype=np.float64) / N
    qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0

    # Best setting: σ=2, K=256
    sigma, K_best = 2.0, 256
    Ks = [128, 256, 512]

    # Train parity-MSE at best setting
    cfg_parity = replace(cfg, sigmas=[sigma], Ks=[K_best], use_classical=False, iqp_loss="parity_mse")
    art_parity = hv.rerun_single_setting(
        cfg_parity, p_star, holdout_mask, bits_table,
        sigma=sigma, K=K_best, return_hist=False, iqp_loss="parity_mse",
    )
    q_parity = art_parity["q_iqp"]
    q_spec = art_parity["q_spec"]

    # Train prob-MSE at best setting
    cfg_prob = replace(cfg, sigmas=[sigma], Ks=[K_best], use_classical=False, iqp_loss="prob_mse")
    art_prob = hv.rerun_single_setting(
        cfg_prob, p_star, holdout_mask, bits_table,
        sigma=sigma, K=K_best, return_hist=False, iqp_loss="prob_mse",
    )
    q_prob = art_prob["q_iqp"]

    fig, axes = plt.subplots(1, 2, figsize=hv.fig_size("full", 2.6))
    plt.subplots_adjust(wspace=0.35)

    # Panel (a): Recovery comparison
    hv._panel_label(axes[0], "(a)")
    ax = axes[0]
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
    y_parity = hv.expected_unique_fraction(q_parity, holdout_mask, Q)
    y_prob = hv.expected_unique_fraction(q_prob, holdout_mask, Q)
    y_spec = hv.expected_unique_fraction(q_spec, holdout_mask, Q)
    y_u = hv.expected_unique_fraction(q_unif, holdout_mask, Q)

    ax.plot(Q, y_star, color=hv.COLORS["target"], lw=1.9, label=r"Target $p^*$", zorder=6)
    ax.plot(Q, y_parity, color=hv.COLORS["model"], lw=2.2, label="IQP (parity MSE)", zorder=7)
    ax.plot(Q, y_prob, color=hv.COLORS["blue"], lw=2.0, label="IQP (prob MSE)", zorder=5)
    ax.plot(Q, y_spec, color=SPEC_COLOR, ls="-.", lw=1.9, label=r"Spectral $\tilde{q}$", zorder=4)
    ax.plot(Q, y_u, color=hv.COLORS["gray"], ls="--", lw=1.5, alpha=0.9, label="Uniform", zorder=3)
    ax.axhline(1.0, color=hv.COLORS["gray"], ls=":", alpha=0.7)
    ax.set_xlim(0, cfg.Qmax)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.legend(**LEGEND_STYLE)

    # Panel (b): Q80 ratio across K
    hv._panel_label(axes[1], "(b)")
    ax = axes[1]
    q80_parity_vals, q80_prob_vals = [], []
    for K in Ks:
        cfg_p = replace(cfg, sigmas=[sigma], Ks=[K], use_classical=False, iqp_loss="parity_mse")
        art_p = hv.rerun_single_setting(
            cfg_p, p_star, holdout_mask, bits_table,
            sigma=sigma, K=K, return_hist=False, iqp_loss="parity_mse",
        )
        met_p = hv.compute_metrics_for_q(art_p["q_iqp"], holdout_mask, qH_unif, H_size,
                                          cfg.Q80_thr, cfg.Q80_search_max)

        cfg_q = replace(cfg, sigmas=[sigma], Ks=[K], use_classical=False, iqp_loss="prob_mse")
        art_q = hv.rerun_single_setting(
            cfg_q, p_star, holdout_mask, bits_table,
            sigma=sigma, K=K, return_hist=False, iqp_loss="prob_mse",
        )
        met_q = hv.compute_metrics_for_q(art_q["q_iqp"], holdout_mask, qH_unif, H_size,
                                          cfg.Q80_thr, cfg.Q80_search_max)
        q80_parity_vals.append(float(met_p["Q80"]))
        q80_prob_vals.append(float(met_q["Q80"]))

    ratios = [p / b if np.isfinite(p) and np.isfinite(b) and b > 0 else float("nan")
              for p, b in zip(q80_parity_vals, q80_prob_vals)]

    x_pos = np.arange(len(Ks))
    width = 0.35
    ax.bar(x_pos - width / 2, q80_parity_vals, width, color=hv.COLORS["model"],
           alpha=0.85, label="Parity MSE")
    ax.bar(x_pos + width / 2, q80_prob_vals, width, color=hv.COLORS["blue"],
           alpha=0.85, label="Prob MSE")

    for i, r in enumerate(ratios):
        if np.isfinite(r):
            y_max = max(q80_parity_vals[i], q80_prob_vals[i])
            ax.text(x_pos[i], y_max * 1.05, f"{r:.2f}x",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel(r"$Q_{80}$ (lower is better)")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right", fontsize=7)

    fig.savefig(str(outdir / "fig3.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {outdir / 'fig3.pdf'}")


# ===========================================================================
# Figure 4 — Fair Classical Baseline  (3 panels)
# ===========================================================================

def _build_holdout_mask_for_mode(
    mode: str, p_star, support, good_mask, bits_table,
    m_train_for_holdout, holdout_k, holdout_pool, seed,
):
    if mode == "high_value":
        candidate_mask = good_mask
    elif mode == "global":
        candidate_mask = support.astype(bool)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return hv.select_holdout_smart(
        p_star=p_star, good_mask=candidate_mask, bits_table=bits_table,
        m_train=m_train_for_holdout, holdout_k=holdout_k,
        pool_size=holdout_pool, seed=seed + 111,
    )


def _all_pairs_dense(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _train_classical_boltzmann(
    n, layers, steps, lr, seed_init, P, z_data, loss_mode, emp_dist,
    topology="nn_nnn", include_fields=True,
):
    """Reimplementation from exp10 for self-containment."""
    qml = hv.qml
    anp = hv.np

    if topology == "nn_nnn":
        pairs = hv.get_iqp_pairs_nn_nnn(n)
    elif topology == "dense":
        pairs = _all_pairs_dense(n)
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    bits = hv.make_bits_table(n)
    spins = 1.0 - 2.0 * bits.astype(np.float64)
    N = spins.shape[0]

    pair_feats = np.zeros((len(pairs), N), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        pair_feats[k] = spins[:, i] * spins[:, j]

    feat_blocks = [pair_feats]
    if include_fields:
        feat_blocks.append(spins.T.copy())
    F = np.concatenate(feat_blocks, axis=0)
    num_features = F.shape[0]

    F_t = anp.array(F, requires_grad=False)
    P_t = anp.array(P, requires_grad=False)
    z_t = anp.array(z_data, requires_grad=False)
    emp_t = anp.array(emp_dist, requires_grad=False)
    emp_t = emp_t / anp.sum(emp_t)

    rng = np.random.default_rng(seed_init)
    theta = anp.array(0.01 * rng.standard_normal(num_features), requires_grad=True)
    opt = qml.AdamOptimizer(lr)

    def _softmax(logits):
        m = anp.max(logits)
        ex = anp.exp(logits - m)
        return ex / anp.sum(ex)

    loss_name = str(loss_mode).lower()

    def _loss(theta_flat):
        q = _softmax(anp.dot(theta_flat, F_t))
        if loss_name == "parity_mse":
            return anp.mean((z_t - P_t @ q) ** 2)
        if loss_name == "prob_mse":
            return anp.mean((q - emp_t) ** 2)
        q_clip = anp.clip(q, 1e-12, 1.0)
        return -anp.sum(emp_t * anp.log(q_clip))

    for _ in range(steps):
        theta, _ = opt.step_and_cost(_loss, theta)

    q_final = np.array(_softmax(anp.dot(theta, F_t)), dtype=np.float64)
    q_final = np.clip(q_final, 0.0, 1.0)
    q_final = q_final / max(1e-15, float(q_final.sum()))
    return q_final


def _train_transformer(bits_table, idxs_train, n, seed, epochs=300):
    """Minimal autoregressive transformer baseline."""
    import torch
    import torch.nn as tnn
    import torch.nn.functional as tF

    torch.manual_seed(seed)
    device = torch.device("cpu")

    class _AR(tnn.Module):
        def __init__(self):
            super().__init__()
            d = 64
            self.tok = tnn.Embedding(3, d)
            self.pos = tnn.Parameter(torch.zeros(1, n, d))
            enc = tnn.TransformerEncoderLayer(d, 4, 128, 0.0, "gelu", batch_first=True)
            self.enc = tnn.TransformerEncoder(enc, 2)
            self.out = tnn.Linear(d, 1)

        def forward(self, inp):
            x = self.tok(inp) + self.pos[:, :inp.shape[1], :]
            t = inp.shape[1]
            causal = torch.triu(torch.ones(t, t, device=inp.device, dtype=torch.bool), 1)
            return self.out(self.enc(x, mask=causal)).squeeze(-1)

    x_train = torch.from_numpy(bits_table[idxs_train].astype(np.int64)).to(device)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train),
        batch_size=256, shuffle=True, drop_last=False,
        generator=torch.Generator("cpu").manual_seed(seed + 11),
    )

    model = _AR().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for (xb,) in dl:
            bos = torch.full((xb.shape[0], 1), 2, device=device, dtype=torch.long)
            inp = torch.cat([bos, xb[:, :-1]], dim=1)
            loss = tF.binary_cross_entropy_with_logits(model(inp), xb.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    x_all = torch.from_numpy(bits_table.astype(np.int64)).to(device)
    with torch.no_grad():
        bos = torch.full((x_all.shape[0], 1), 2, device=device, dtype=torch.long)
        inp = torch.cat([bos, x_all[:, :-1]], dim=1)
        probs = torch.sigmoid(model(inp))
        x_f = x_all.float()
        logp = x_f * torch.log(probs.clamp(1e-12)) + (1 - x_f) * torch.log((1 - probs).clamp(1e-12))
        logp = logp.sum(1)
        logp = logp - torch.logsumexp(logp, 0)
        q = torch.exp(logp).cpu().numpy().astype(np.float64)
    q = np.clip(q, 0.0, 1.0)
    return q / max(1e-15, float(q.sum()))


def _train_maxent(P, z_data, seed, steps=2500, lr=5e-2):
    """MaxEnt parity model."""
    import torch
    import torch.nn as tnn
    torch.manual_seed(seed)
    device = torch.device("cpu")

    P_t = torch.from_numpy(P.astype(np.float32)).to(device)
    z_t = torch.from_numpy(z_data.astype(np.float32)).to(device)
    theta = tnn.Parameter(torch.zeros(P_t.shape[0], device=device))
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        logits = torch.matmul(theta, P_t)
        loss = torch.logsumexp(logits, 0) - torch.dot(theta, z_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = torch.matmul(theta, P_t)
        logits = logits - torch.logsumexp(logits, 0)
        q = torch.exp(logits).cpu().numpy().astype(np.float64)
    q = np.clip(q, 0.0, 1.0)
    return q / max(1e-15, float(q.sum()))


def _plot_baseline_recovery_panel(
    ax, p_star, holdout_mask, model_rows, Qmax=10000,
):
    """Plot recovery for a list of model rows onto an existing ax."""
    Q = _q_grid(Qmax)
    y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
    q_unif = np.ones_like(p_star, dtype=np.float64) / p_star.size
    y_unif = hv.expected_unique_fraction(q_unif, holdout_mask, Q)

    ax.plot(Q, y_star, color=hv.COLORS["target"], lw=2.0, label=r"Target $p^*$", zorder=10)
    for idx, row in enumerate(model_rows):
        y = hv.expected_unique_fraction(row["q"], holdout_mask, Q)
        ax.plot(Q, y, color=row["color"], ls=row.get("ls", "-"),
                lw=row.get("lw", 1.9), alpha=row.get("alpha", 1.0),
                label=row["label"], zorder=9 - idx)
    ax.plot(Q, y_unif, color=hv.COLORS["gray"], lw=1.5, ls="--", alpha=0.9, label="Uniform", zorder=1)
    ax.axhline(1.0, color=hv.COLORS["gray"], ls=":", alpha=0.7)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(**LEGEND_STYLE)


def _compute_baseline_models(
    mode: str, p_star, support, good_mask, bits_table, scores,
    seed, train_m, sigma, K, layers, iqp_steps, iqp_lr, iqp_eval_every,
    holdout_k, holdout_pool, holdout_m_train,
    q80_thr, q80_search_max,
):
    """Compute all baseline models for a given mode, return (holdout_mask, model_rows, metrics)."""
    holdout_mask = _build_holdout_mask_for_mode(
        mode=mode, p_star=p_star, support=support, good_mask=good_mask,
        bits_table=bits_table, m_train_for_holdout=holdout_m_train,
        holdout_k=holdout_k, holdout_pool=holdout_pool, seed=seed,
    )
    H_size = int(np.sum(holdout_mask))
    N = p_star.size

    cfg = hv.Config(
        n=int(np.log2(N)), beta=0.9, train_m=train_m,
        holdout_k=holdout_k, holdout_pool=holdout_pool, seed=seed,
        good_frac=0.05, sigmas=[sigma], Ks=[K], Qmax=10000,
        Q80_thr=q80_thr, Q80_search_max=q80_search_max,
        use_iqp=True, use_classical=True,
        iqp_steps=iqp_steps, iqp_lr=iqp_lr, iqp_eval_every=iqp_eval_every,
        iqp_layers=layers, iqp_loss="parity_mse", adversarial=False,
        outdir=str(OUTDIR),
    )

    art = hv.rerun_single_setting(
        cfg, p_star, holdout_mask, bits_table,
        sigma=sigma, K=K, return_hist=False, iqp_loss="parity_mse",
    )
    q_iqp = art["q_iqp"]
    P = art["P"]
    z_data = art["z"]

    # IQP prob-MSE
    cfg_prob = replace(cfg, use_classical=False, iqp_loss="prob_mse")
    art_prob = hv.rerun_single_setting(
        cfg_prob, p_star, holdout_mask, bits_table,
        sigma=sigma, K=K, return_hist=False, iqp_loss="prob_mse",
    )
    q_iqp_prob = art_prob["q_iqp"]

    # Empirical dist
    p_train = p_star.copy()
    if H_size > 0:
        p_train[holdout_mask] = 0.0
        p_train /= p_train.sum()
    idxs_train = hv.sample_indices(p_train, train_m, seed=seed + 7)
    emp = hv.empirical_dist(idxs_train, N)

    # Strong baselines
    n = int(np.log2(N))
    q_nnn_fields = _train_classical_boltzmann(
        n, layers, iqp_steps, iqp_lr, seed + 30001,
        P, z_data, "parity_mse", emp, "nn_nnn", True,
    )
    q_dense_xent = _train_classical_boltzmann(
        n, layers, iqp_steps, iqp_lr, seed + 30004,
        P, z_data, "xent", emp, "dense", True,
    )
    q_maxent = _train_maxent(P, z_data, seed + 36001)
    q_transformer = _train_transformer(bits_table, idxs_train, n, seed + 35501)

    model_rows = [
        {"key": "iqp_parity", "label": "IQP (parity)", "q": q_iqp,
         "color": hv.COLORS["model"], "ls": "-", "lw": 2.2},
        {"key": "iqp_prob", "label": "IQP (prob-MSE)", "q": q_iqp_prob,
         "color": hv.COLORS["model_prob_mse"], "ls": "--", "lw": 2.0},
        {"key": "ising_fields", "label": "Ising+fields (NN+NNN)", "q": q_nnn_fields,
         "color": "#005A9C", "ls": "-", "lw": 1.9},
        {"key": "dense_xent", "label": "Dense Ising (xent)", "q": q_dense_xent,
         "color": "#8C564B", "ls": (0, (5, 2)), "lw": 1.9},
        {"key": "maxent", "label": "MaxEnt parity", "q": q_maxent,
         "color": "#9467BD", "ls": "--", "lw": 2.1},
        {"key": "transformer", "label": "AR Transformer", "q": q_transformer,
         "color": "#1AA7A1", "ls": "--", "lw": 2.0},
    ]

    # Compute Q80 for all models
    q_unif = np.ones(N, dtype=np.float64) / N
    qH_unif = float(q_unif[holdout_mask].sum()) if H_size > 0 else 1.0
    metrics = {}
    for row in model_rows:
        met = hv.compute_metrics_for_q(
            row["q"], holdout_mask, qH_unif, H_size, q80_thr, q80_search_max)
        metrics[row["key"]] = met

    return holdout_mask, model_rows, metrics


def figure_4(outdir: Path) -> None:
    """Fair Classical Baseline: (a) HV holdout; (b) Global holdout; (c) Q80 comparison."""
    print("[Fig 4] Computing strong baselines for both holdout modes...")

    n = 12
    bits_table = hv.make_bits_table(n)
    p_star, support, scores = hv.build_target_distribution_paper(n, 0.9)
    good_mask = hv.topk_mask_by_scores(scores, support, frac=0.05)

    # Best settings from exp09/exp10
    hv_seed, hv_m, hv_sigma, hv_K = 44, 5000, 2.0, 256
    gl_seed, gl_m, gl_sigma, gl_K = 46, 5000, 2.0, 512

    common = dict(
        p_star=p_star, support=support, good_mask=good_mask,
        bits_table=bits_table, scores=scores, layers=1,
        iqp_steps=400, iqp_lr=0.05, iqp_eval_every=50,
        holdout_k=20, holdout_pool=400, holdout_m_train=5000,
        q80_thr=0.8, q80_search_max=200000,
    )

    hv_mask, hv_rows, hv_met = _compute_baseline_models(
        mode="high_value", seed=hv_seed, train_m=hv_m, sigma=hv_sigma, K=hv_K, **common)
    gl_mask, gl_rows, gl_met = _compute_baseline_models(
        mode="global", seed=gl_seed, train_m=gl_m, sigma=gl_sigma, K=gl_K, **common)

    fig, axes = plt.subplots(1, 3, figsize=hv.fig_size("full", 2.6))
    plt.subplots_adjust(wspace=0.38)

    # Panel (a): High-value holdout
    hv._panel_label(axes[0], "(a)")
    _plot_baseline_recovery_panel(axes[0], p_star, hv_mask, hv_rows)

    # Panel (b): Global holdout
    hv._panel_label(axes[1], "(b)")
    _plot_baseline_recovery_panel(axes[1], p_star, gl_mask, gl_rows)

    # Panel (c): Q80 comparison bars
    hv._panel_label(axes[2], "(c)")
    ax = axes[2]

    model_keys = [r["key"] for r in hv_rows]
    model_labels = [r["label"] for r in hv_rows]
    model_colors = [r["color"] for r in hv_rows]

    q80_hv = [float(hv_met[k]["Q80"]) for k in model_keys]
    q80_gl = [float(gl_met[k]["Q80"]) for k in model_keys]

    x_pos = np.arange(len(model_keys))
    width = 0.35

    bars_hv = ax.bar(x_pos - width / 2, q80_hv, width, color=[c for c in model_colors],
                     alpha=0.85, edgecolor="black", linewidth=0.5)
    bars_gl = ax.bar(x_pos + width / 2, q80_gl, width, color=[c for c in model_colors],
                     alpha=0.45, edgecolor="black", linewidth=0.5, hatch="//")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=6)
    ax.set_ylabel(r"$Q_{80}$ (lower is better)")
    ax.set_yscale("log")

    # Custom legend for HV vs Global
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#888888", alpha=0.85, edgecolor="black", label="High-value"),
        Patch(facecolor="#888888", alpha=0.45, edgecolor="black", hatch="//", label="Global"),
    ], frameon=False, loc="upper right", fontsize=7)

    fig.savefig(str(outdir / "fig4.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {outdir / 'fig4.pdf'}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    hv.set_style(base=8)
    outdir = OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = _base_config(str(outdir))

    # Figure 1 also returns sweep results for reuse in Figure 2
    results = figure_1(outdir)

    # Figure 2 reuses sweep results
    figure_2(outdir, cfg, results)

    # Figure 3 (independent)
    figure_3(outdir)

    # Figure 4 (independent, heavy computation)
    figure_4(outdir)

    print(f"\nAll figures saved to {outdir}/")


if __name__ == "__main__":
    main()
