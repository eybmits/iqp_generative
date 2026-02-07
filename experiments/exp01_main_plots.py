#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Main plots

Recovery overlays (for BOTH targets):
  - fig1_recovery_paper_overlay.pdf
  - fig1_recovery_paper_nonparity_overlay.pdf

Each overlay plot shows:
  - Target p* (black)
  - Best IQP-QCBM (red)  [best = minimal finite Q80_iqp]
  - Spectral completion q~ (gray dash-dot)  [at the SAME (sigma,K) as best IQP]
  - Many Ising controls (blue -> white) across the (sigma,K) grid
  - Uniform (gray dashed)

Heatmaps (paper target):
  - fig2a_paper_qH_ratio_iqp.pdf      [RED]
  - fig2a_paper_qH_ratio_class.pdf    [BLUE]
  - fig2c_paper_Q80_iqp.pdf           [RED]
  - fig2c_paper_Q80_class.pdf         [BLUE]

Heatmaps (paper_nonparity target):
  - fig2a_paper_nonparity_qH_ratio_iqp.pdf   [RED]
  - fig2a_paper_nonparity_qH_ratio_class.pdf [BLUE]
  - fig2c_paper_nonparity_Q80_iqp.pdf        [RED]
  - fig2c_paper_nonparity_Q80_class.pdf      [BLUE]

ONLY CHANGE vs previous:
  - Overlay legend: lightly white-backed AND placed at lower right.

Run:
  python3 experiments/exp01_main_plots.py
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _q_grid(Qmax: int = 10000) -> np.ndarray:
    Q = np.unique(
        np.concatenate(
            [
                np.unique(np.logspace(0, 3.5, 120).astype(int)),
                np.linspace(1000, Qmax, 160).astype(int),
                np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100], dtype=int),
            ]
        )
    )
    return Q[(Q >= 0) & (Q <= Qmax)]


def _ising_color(rank: int, m: int):
    """Dark blue for best (rank=0), fading to near-white for worst (rank=m-1)."""
    if m <= 1:
        t = 0.95
        r = 0.0
    else:
        r = rank / (m - 1)
        t = 0.95 - 0.90 * r
    color = plt.cm.Blues(t)
    alpha = 0.90 - 0.50 * r
    alpha = max(0.35, float(alpha))
    return color, alpha


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    if not getattr(hv, "HAS_PENNYLANE", False):
        raise RuntimeError(
            "Pennylane is required for this experiment (IQP model + Ising control).\n"
            "Install with: pip install pennylane"
        )

    # Paper-grade style
    hv.set_style(base=8)
    outdir = hv.ensure_outdir(os.path.join(ROOT, "outputs", "exp01_main_plots"))

    # Colormaps for heatmaps:
    cmap_redblack = LinearSegmentedColormap.from_list("RedBlack", ["#FF0000", "#000000"])
    cmap_blueblack = LinearSegmentedColormap.from_list("BlueBlack", ["#DCEBFA", "#1F77B4", "#000000"])

    # Spectral completion style (clear dash-dot)
    SPEC_COLOR = "#555555"
    SPEC_LS = (0, (7, 2, 1.2, 2))  # dash, gap, dot, gap

    # Legend style: lightly white-backed, LOWER RIGHT
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

    # Base config (paper-default values)
    base_cfg = hv.Config(
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

    bits_table = hv.make_bits_table(base_cfg.n)
    Q = _q_grid(Qmax=base_cfg.Qmax)

    # -------------------------------------------------------------------------
    # Heatmap bundle (both models) for one target -> returns results for overlay
    # -------------------------------------------------------------------------
    def heatmaps_bundle(prefix: str, cfg, p_star, holdout_mask, good_mask) -> List[Dict]:
        cfg_heat = replace(cfg, use_iqp=True, use_classical=True, adversarial=False)
        results = hv.run_sweep(cfg_heat, p_star, holdout_mask, good_mask, bits_table)

        qH_ratio_iqp = hv.build_result_matrix(results, cfg_heat.sigmas, cfg_heat.Ks, key="qH_ratio_iqp")
        Q80_iqp = hv.build_result_matrix(results, cfg_heat.sigmas, cfg_heat.Ks, key="Q80_iqp")
        qH_ratio_cl = hv.build_result_matrix(results, cfg_heat.sigmas, cfg_heat.Ks, key="qH_ratio_class")
        Q80_cl = hv.build_result_matrix(results, cfg_heat.sigmas, cfg_heat.Ks, key="Q80_class")

        row_labels = [str(s) for s in cfg_heat.sigmas]
        col_labels = [str(k) for k in cfg_heat.Ks]

        hv.plot_heatmap(
            qH_ratio_iqp,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$q_\theta(H)/q_{\mathrm{unif}}(H)$",
            outpath=os.path.join(outdir, f"fig2a_{prefix}_qH_ratio_iqp.pdf"),
            log10=False,
            fmt="{:.1f}",
            cmap="Reds",
            mode="col",
        )
        hv.plot_heatmap(
            qH_ratio_cl,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$q_{\mathrm{cl}}(H)/q_{\mathrm{unif}}(H)$",
            outpath=os.path.join(outdir, f"fig2a_{prefix}_qH_ratio_class.pdf"),
            log10=False,
            fmt="{:.1f}",
            cmap="Blues",
            mode="col",
        )

        hv.plot_heatmap(
            Q80_iqp,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$\log_{10} Q_{80}$",
            outpath=os.path.join(outdir, f"fig2c_{prefix}_Q80_iqp.pdf"),
            log10=True,
            fmt="{:.0f}",
            cmap=cmap_redblack,
            mode="col",
        )
        hv.plot_heatmap(
            Q80_cl,
            row_labels=row_labels,
            col_labels=col_labels,
            cbar_label=r"$\log_{10} Q_{80}$",
            outpath=os.path.join(outdir, f"fig2c_{prefix}_Q80_class.pdf"),
            log10=True,
            fmt="{:.0f}",
            cmap=cmap_blueblack,
            mode="col",
        )

        return results

    # -------------------------------------------------------------------------
    # Overlay plot: best IQP (red) vs spectral completion (gray) vs ALL Ising (blue->white)
    # -------------------------------------------------------------------------
    def plot_best_iqp_vs_all_ising_with_spec(
        outpath: str,
        cfg,
        p_star: np.ndarray,
        holdout_mask: np.ndarray,
        results: List[Dict],
    ) -> None:
        best = hv.pick_best_setting(results, prefer="iqp")
        best_sigma = float(best["sigma"])
        best_K = int(best["K"])

        cfg_best = replace(cfg, use_iqp=True, use_classical=False, adversarial=False)
        best_art = hv.rerun_single_setting(
            cfg=cfg_best,
            p_star=p_star,
            holdout_mask=holdout_mask,
            bits_table=bits_table,
            sigma=best_sigma,
            K=best_K,
            return_hist=False,
        )
        q_iqp_best = best_art["q_iqp"]  # type: ignore
        q_spec_best = best_art["q_spec"]  # type: ignore

        u = np.ones_like(p_star, dtype=np.float64) / p_star.size
        y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
        y_unif = hv.expected_unique_fraction(u, holdout_mask, Q)
        y_iqp = hv.expected_unique_fraction(q_iqp_best, holdout_mask, Q)
        y_spec = hv.expected_unique_fraction(q_spec_best, holdout_mask, Q)

        ising_rows: List[Tuple[float, int, float, float]] = []
        for r in results:
            sigma = float(r["sigma"])
            K = int(r["K"])
            Q80c = float(r["Q80_class"])
            qHr = float(r["qH_ratio_class"])
            ising_rows.append((sigma, K, Q80c, qHr))

        def _key(tup):
            _, _, Q80c, qHr = tup
            finite = np.isfinite(Q80c)
            return (0 if finite else 1, Q80c if finite else 1e99, -qHr)

        ising_rows.sort(key=_key)
        m = len(ising_rows)

        fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)

        cfg_class_only = replace(cfg, use_iqp=False, use_classical=True, adversarial=False)
        for idx, (sigma, K, _, _) in enumerate(ising_rows):
            art = hv.rerun_single_setting(
                cfg=cfg_class_only,
                p_star=p_star,
                holdout_mask=holdout_mask,
                bits_table=bits_table,
                sigma=float(sigma),
                K=int(K),
                return_hist=False,
            )
            q_cl = art["q_class"]  # type: ignore
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
                   label=fr"Best IQP-QCBM, $\sigma$={best_sigma:g}, $K$={best_K:d}"),
            Line2D([0], [0], color=SPEC_COLOR, lw=1.9, ls=SPEC_LS, label=r"Spectral completion $\~q$"),
            Line2D([0], [0], color=plt.cm.Blues(0.90), lw=1.6, label="Ising controls"),
            Line2D([0], [0], color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform"),
        ]
        ax.legend(handles=legend_handles, **LEGEND_STYLE)

        fig.savefig(outpath)
        plt.close(fig)
        print(f"[Saved] {outpath}")

    # -------------------------------------------------------------------------
    # Overlay plot: only best IQP, best Ising, uniform, and target
    # -------------------------------------------------------------------------
    def plot_best_iqp_vs_best_ising_only(
        outpath: str,
        cfg,
        p_star: np.ndarray,
        holdout_mask: np.ndarray,
        results: List[Dict],
    ) -> None:
        best_iqp = hv.pick_best_setting(results, prefer="iqp")
        best_cl = hv.pick_best_setting(results, prefer="classical")
        best_iqp_sigma = float(best_iqp["sigma"])
        best_iqp_K = int(best_iqp["K"])
        best_cl_sigma = float(best_cl["sigma"])
        best_cl_K = int(best_cl["K"])

        cfg_iqp_only = replace(cfg, use_iqp=True, use_classical=False, adversarial=False)
        cfg_cl_only = replace(cfg, use_iqp=False, use_classical=True, adversarial=False)

        iqp_art = hv.rerun_single_setting(
            cfg=cfg_iqp_only,
            p_star=p_star,
            holdout_mask=holdout_mask,
            bits_table=bits_table,
            sigma=best_iqp_sigma,
            K=best_iqp_K,
            return_hist=False,
        )
        cl_art = hv.rerun_single_setting(
            cfg=cfg_cl_only,
            p_star=p_star,
            holdout_mask=holdout_mask,
            bits_table=bits_table,
            sigma=best_cl_sigma,
            K=best_cl_K,
            return_hist=False,
        )

        q_iqp_best = iqp_art["q_iqp"]  # type: ignore
        q_cl_best = cl_art["q_class"]  # type: ignore

        u = np.ones_like(p_star, dtype=np.float64) / p_star.size
        y_star = hv.expected_unique_fraction(p_star, holdout_mask, Q)
        y_unif = hv.expected_unique_fraction(u, holdout_mask, Q)
        y_iqp = hv.expected_unique_fraction(q_iqp_best, holdout_mask, Q)
        y_cl = hv.expected_unique_fraction(q_cl_best, holdout_mask, Q)

        fig, ax = plt.subplots(figsize=hv.fig_size("col", 2.6), constrained_layout=True)

        ax.plot(Q, y_unif, color=hv.COLORS["gray"], linestyle="--", linewidth=1.5, alpha=0.9, zorder=1)
        ax.plot(Q, y_cl, color=hv.COLORS["blue"], linestyle=":", linewidth=2.0, zorder=3)
        ax.plot(Q, y_star, color=hv.COLORS["target"], linewidth=1.9, zorder=4)
        ax.plot(Q, y_iqp, color=hv.COLORS["model"], linewidth=2.4, zorder=5)

        ax.axhline(1.0, color=hv.COLORS["gray"], linestyle=":", alpha=0.7)

        ax.set_xlim(0, cfg.Qmax)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$Q$ samples from model")
        ax.set_ylabel(r"Recovery $R(Q)$")

        legend_handles = [
            Line2D([0], [0], color=hv.COLORS["target"], lw=1.9, label=r"Target $p^*$"),
            Line2D([0], [0], color=hv.COLORS["model"], lw=2.4,
                   label=fr"Best IQP-QCBM, $\sigma$={best_iqp_sigma:g}, $K$={best_iqp_K:d}"),
            Line2D([0], [0], color=hv.COLORS["blue"], lw=2.0, ls=":",
                   label=fr"Best Ising, $\sigma$={best_cl_sigma:g}, $K$={best_cl_K:d}"),
            Line2D([0], [0], color=hv.COLORS["gray"], lw=1.5, ls="--", label="Uniform"),
        ]
        ax.legend(handles=legend_handles, **LEGEND_STYLE)

        fig.savefig(outpath)
        plt.close(fig)
        print(f"[Saved] {outpath}")

    # =========================================================================
    # PAPER target
    # =========================================================================
    cfg_paper = replace(base_cfg, target_family="paper_even")

    p_star_paper, support_paper, scores_paper = hv.build_target_distribution_paper(cfg_paper.n, cfg_paper.beta)
    good_paper = hv.topk_mask_by_scores(scores_paper, support_paper, frac=cfg_paper.good_frac)
    holdout_paper = hv.select_holdout_smart(
        p_star=p_star_paper,
        good_mask=good_paper,
        bits_table=bits_table,
        m_train=cfg_paper.train_m,
        holdout_k=cfg_paper.holdout_k,
        pool_size=cfg_paper.holdout_pool,
        seed=cfg_paper.seed + 111,
    )

    results_paper = heatmaps_bundle("paper", cfg_paper, p_star_paper, holdout_paper, good_paper)
    plot_best_iqp_vs_best_ising_only(
        outpath=os.path.join(outdir, "fig1_recovery_paper_overlay.pdf"),
        cfg=cfg_paper,
        p_star=p_star_paper,
        holdout_mask=holdout_paper,
        results=results_paper,
    )
    plot_best_iqp_vs_all_ising_with_spec(
        outpath=os.path.join(outdir, "fig1_recovery_paper_overlay_full.pdf"),
        cfg=cfg_paper,
        p_star=p_star_paper,
        holdout_mask=holdout_paper,
        results=results_paper,
    )

    # =========================================================================
    # PAPER-NONPARITY target
    # =========================================================================
    cfg_nonparity = replace(base_cfg, target_family="paper_nonparity")

    p_star_nonparity, support_nonparity, scores_nonparity = hv.build_target_distribution_paper_nonparity(
        cfg_nonparity.n,
        cfg_nonparity.beta,
    )
    good_nonparity = hv.topk_mask_by_scores(scores_nonparity, support_nonparity, frac=cfg_nonparity.good_frac)
    holdout_nonparity = hv.select_holdout_smart(
        p_star=p_star_nonparity,
        good_mask=good_nonparity,
        bits_table=bits_table,
        m_train=cfg_nonparity.train_m,
        holdout_k=cfg_nonparity.holdout_k,
        pool_size=cfg_nonparity.holdout_pool,
        seed=cfg_nonparity.seed + 111,
    )

    results_nonparity = heatmaps_bundle(
        "paper_nonparity",
        cfg_nonparity,
        p_star_nonparity,
        holdout_nonparity,
        good_nonparity,
    )
    plot_best_iqp_vs_all_ising_with_spec(
        outpath=os.path.join(outdir, "fig1_recovery_paper_nonparity_overlay.pdf"),
        cfg=cfg_nonparity,
        p_star=p_star_nonparity,
        holdout_mask=holdout_nonparity,
        results=results_nonparity,
    )


if __name__ == "__main__":
    main()
