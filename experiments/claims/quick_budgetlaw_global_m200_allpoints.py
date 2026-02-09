#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budget-law scatter variants for global holdout (m=200):
1) IQP vs baselines (all points, no y=2x or y=5x)
2) IQP sigma/K sweep overlay vs baselines (many red points)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


SUMMARY_CSV = (
    ROOT
    / "outputs"
    / "paper_even_final"
    / "34_claim_beta_sweep_bestparams"
    / "global_m200_sigma1_k512"
    / "summary"
    / "beta_sweep_metrics_long.csv"
)

ABLATION_CSV = (
    ROOT
    / "outputs"
    / "paper_even_final"
    / "33_claim_iqp_sigma_k_ablation_b07to12_steps200"
    / "ablation_metrics_long.csv"
)

OUTDIR = (
    ROOT
    / "outputs"
    / "paper_even_final"
    / "35_claim_budgetlaw_global_m200_custom"
)

def _finite_pairs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    return x[good], y[good]


def _style_axes(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    xs = np.geomspace(lo, hi, 400)
    ax.plot(xs, xs, linestyle="--", linewidth=2.6, color="#2f2f2f", alpha=0.98, zorder=20)
    ax.set_xlabel("Predicted Q80")
    ax.set_ylabel(r"Measured $Q_{80}$")


def plot_iqp_vs_baselines_allpoints(df: pd.DataFrame, outpath: Path) -> None:
    iqp_par = df["model_key"] == "iqp_parity_mse"
    iqp_prob = df["model_key"] == "iqp_prob_mse"
    classical = ~(iqp_par | iqp_prob)

    x_all, y_all = _finite_pairs(df["Q80_pred"].to_numpy(), df["Q80"].to_numpy())
    if x_all.size == 0:
        raise RuntimeError("No finite points available for plotting.")
    xmin = max(1.0, float(np.nanmin(x_all)) * 0.75)
    xmax = float(np.nanmax(x_all)) * 1.25
    ymin = max(1.0, float(np.nanmin(y_all)) * 0.75)
    ymax = float(np.nanmax(y_all)) * 1.25

    fig, ax = plt.subplots(figsize=hv.fig_size("col", 3.2), constrained_layout=True)

    def _scatter(mask: np.ndarray, color: str, marker: str, size: float, alpha: float = 0.88) -> None:
        x, y = _finite_pairs(df.loc[mask, "Q80_pred"].to_numpy(), df.loc[mask, "Q80"].to_numpy())
        if x.size > 0:
            ax.scatter(x, y, c=color, s=size, marker=marker, alpha=alpha, linewidths=0.0)

    _scatter(classical.to_numpy(), "#1f77b4", "o", 44)
    _scatter(iqp_prob.to_numpy(), "#ff7f7f", "D", 48, alpha=0.82)
    _scatter(iqp_par.to_numpy(), "#d62728", "o", 54, alpha=0.92)

    _style_axes(ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="none", markersize=7, label="IQP-QCBM (parity)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#ff7f7f", markeredgecolor="none", markersize=6.5, label="IQP (prob-MSE)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=6.5, label="Classical baselines"),
        Line2D([0], [0], color="#2f2f2f", lw=2.2, ls="--", label=r"$y=x$"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=7.0)
    fig.savefig(outpath)
    plt.close(fig)


def plot_iqp_sweep_overlay(df_summary: pd.DataFrame, df_ablation: pd.DataFrame, outpath: Path) -> None:
    d_base = df_summary[~df_summary["model_key"].isin(["iqp_parity_mse", "iqp_prob_mse"])].copy()
    d_iqp_ref = df_summary[df_summary["model_key"] == "iqp_parity_mse"].copy()
    d_iqp_sw = df_ablation[
        (df_ablation["holdout_mode"] == "global")
        & (df_ablation["train_m"] == 200)
    ].copy()
    if d_base.empty or d_iqp_sw.empty:
        raise RuntimeError("No points found for sigma/K sweep.")

    x_collect = np.concatenate(
        [
            d_base["Q80_pred"].to_numpy(dtype=np.float64),
            d_iqp_ref["Q80_pred"].to_numpy(dtype=np.float64),
            d_iqp_sw["Q80_pred"].to_numpy(dtype=np.float64),
        ]
    )
    y_collect = np.concatenate(
        [
            d_base["Q80"].to_numpy(dtype=np.float64),
            d_iqp_ref["Q80"].to_numpy(dtype=np.float64),
            d_iqp_sw["Q80"].to_numpy(dtype=np.float64),
        ]
    )
    x_all, y_all = _finite_pairs(x_collect, y_collect)
    if x_all.size == 0:
        raise RuntimeError("No finite points available for sigma/K sweep plot.")
    xmin = max(1.0, float(np.nanmin(x_all)) * 0.75)
    xmax = float(np.nanmax(x_all)) * 1.25
    ymin = max(1.0, float(np.nanmin(y_all)) * 0.75)
    ymax = float(np.nanmax(y_all)) * 1.25

    fig, ax = plt.subplots(figsize=hv.fig_size("col", 3.2), constrained_layout=True)

    # Classical baselines (blue)
    xb, yb = _finite_pairs(d_base["Q80_pred"].to_numpy(), d_base["Q80"].to_numpy())
    if xb.size > 0:
        ax.scatter(xb, yb, c="#1f77b4", s=46, marker="o", alpha=0.88, linewidths=0.0)

    # IQP sigma/K sweep (many red points)
    xs, ys = _finite_pairs(d_iqp_sw["Q80_pred"].to_numpy(), d_iqp_sw["Q80"].to_numpy())
    if xs.size > 0:
        ax.scatter(xs, ys, c="#d62728", s=32, marker="o", alpha=0.35, linewidths=0.0)

    # Highlight chosen IQP reference setting (global m=200, sigma=1 K=512 across beta sweep)
    xr, yr = _finite_pairs(d_iqp_ref["Q80_pred"].to_numpy(), d_iqp_ref["Q80"].to_numpy())
    if xr.size > 0:
        ax.scatter(xr, yr, c="#d62728", s=78, marker="o", alpha=0.95, edgecolors="white", linewidths=0.6)

    _style_axes(ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="none", alpha=0.35, markersize=6.0, label=r"IQP sweep ($\sigma,K,\beta$)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="white", markeredgewidth=0.7, markersize=7.2, label=r"IQP ref ($\sigma=1,K=512$, beta sweep)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markeredgecolor="none", markersize=6.5, label="Classical baselines"),
        Line2D([0], [0], color="#2f2f2f", lw=2.2, ls="--", label=r"$y=x$"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=7.0)
    fig.savefig(outpath)
    plt.close(fig)


def main() -> None:
    hv.set_style(base=8)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df_summary = pd.read_csv(SUMMARY_CSV)
    df_ablation = pd.read_csv(ABLATION_CSV)

    out1 = OUTDIR / "budgetlaw_scatter_global_m200_iqp_vs_baselines_allpoints_no2x5x.pdf"
    out2 = OUTDIR / "budgetlaw_scatter_global_m200_iqp_sigmak_sweep_overlay_no2x5x.pdf"
    plot_iqp_vs_baselines_allpoints(df_summary, out1)
    plot_iqp_sweep_overlay(df_summary, df_ablation, out2)

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")


if __name__ == "__main__":
    main()
