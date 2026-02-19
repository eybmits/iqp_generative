#!/usr/bin/env python3
"""Faceted budgetlaw scatter with discrete beta bins.

Creates a multi-panel plot (one panel per model) from the claim-35 points CSV.
Design goals:
- less overlap via faceting and mild jitter,
- discrete beta bins instead of continuous color scale,
- larger markers with thicker edges,
- publication-ready export (>=300 dpi).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


MODEL_ORDER: List[Tuple[str, str, str]] = [
    ("iqp_parity_mse", "IQP parity", "o"),
    ("iqp_prob_mse", "IQP MSE", "p"),
    ("classical_nnn_fields_parity", "Ising+fields", "s"),
    ("classical_dense_fields_xent", "Dense Ising xent", "D"),
    ("classical_transformer_mle", "AR Transformer", "^"),
    ("classical_maxent_parity", "MaxEnt parity", "v"),
]

BIN_LABELS = [
    "β in [0.6, 0.8)",
    "β in [0.8, 1.0)",
    "β in [1.0, 1.2)",
    "β in [1.2, 1.4]",
]

BIN_COLORS: Dict[str, str] = {
    "β in [0.6, 0.8)": "#d84a3a",
    "β in [0.8, 1.0)": "#ea9965",
    "β in [1.0, 1.2)": "#a9c7df",
    "β in [1.2, 1.4]": "#5f83be",
}


def _beta_bin(beta: float) -> str:
    b = float(beta)
    if b < 0.8:
        return BIN_LABELS[0]
    if b < 1.0:
        return BIN_LABELS[1]
    if b < 1.2:
        return BIN_LABELS[2]
    return BIN_LABELS[3]


def _line_angle_deg(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle (degrees) of a data-space segment in screen coordinates."""
    (x1d, y1d) = ax.transData.transform((x1, y1))
    (x2d, y2d) = ax.transData.transform((x2, y2))
    return math.degrees(math.atan2(y2d - y1d, x2d - x1d))


def main() -> None:
    ap = argparse.ArgumentParser(description="Faceted budgetlaw scatter with discrete beta bins.")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "outputs/paper_even_final/35_claim_budgetlaw_global_m200_custom/"
            "budgetlaw_dual_holdout_m200_beta06to14_all_experiments_points.csv"
        ),
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=Path(
            "outputs/paper_even_final/35_claim_budgetlaw_global_m200_custom/"
            "budgetlaw_scatter_dual_holdout_m200_beta06to14_faceted_discrete.pdf"
        ),
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path(
            "outputs/paper_even_final/35_claim_budgetlaw_global_m200_custom/"
            "budgetlaw_scatter_dual_holdout_m200_beta06to14_faceted_discrete.png"
        ),
    )
    ap.add_argument("--dpi", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--jitter-log-sigma",
        type=float,
        default=0.018,
        help="Gaussian jitter sigma in log10-space (0 disables jitter).",
    )
    ap.add_argument("--alpha", type=float, default=0.82)
    ap.add_argument("--marker-size", type=float, default=120.0)
    ap.add_argument("--edge-width", type=float, default=1.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    model_keys = [m[0] for m in MODEL_ORDER]
    df = df[df["model_key"].isin(model_keys)].copy()
    df = df[
        np.isfinite(df["Q80_pred_plot"].to_numpy(float))
        & np.isfinite(df["Q80_plot"].to_numpy(float))
        & (df["Q80_pred_plot"].to_numpy(float) > 0)
        & (df["Q80_plot"].to_numpy(float) > 0)
    ].copy()

    if df.empty:
        raise RuntimeError("No finite points found after filtering.")

    df["beta_bin"] = df["beta"].map(_beta_bin)

    # Deterministic mild jitter in log-space to separate dense overlaps.
    rng = np.random.default_rng(int(args.seed))
    x = df["Q80_pred_plot"].to_numpy(float)
    y = df["Q80_plot"].to_numpy(float)
    if float(args.jitter_log_sigma) > 0:
        x = x * np.power(10.0, rng.normal(0.0, float(args.jitter_log_sigma), size=x.size))
        y = y * np.power(10.0, rng.normal(0.0, float(args.jitter_log_sigma), size=y.size))
    df["x_plot"] = x
    df["y_plot"] = y

    x_min = float(np.nanmin(df["x_plot"]))
    x_max = float(np.nanmax(df["x_plot"]))
    y_min = float(np.nanmin(df["y_plot"]))
    y_max = float(np.nanmax(df["y_plot"]))
    lo = min(x_min, y_min) * 0.75
    hi = max(x_max, y_max) * 1.12
    shade_x = np.logspace(np.log10(lo), np.log10(hi), 320)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,  # >= 7pt for print
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.9), dpi=args.dpi, sharex=True, sharey=True)
    panel_axes = axes.ravel()

    if len(MODEL_ORDER) != len(panel_axes):
        raise RuntimeError(
            f"Expected exactly {len(panel_axes)} models for 2x3 faceting, got {len(MODEL_ORDER)}."
        )

    for ax, (model_key, title, marker) in zip(panel_axes, MODEL_ORDER):
        sub = df[df["model_key"] == model_key].copy()
        if sub.empty:
            ax.set_title(f"{title} (no points)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            continue

        for bl in BIN_LABELS:
            ssub = sub[sub["beta_bin"] == bl]
            if ssub.empty:
                continue
            ax.scatter(
                ssub["x_plot"].to_numpy(float),
                ssub["y_plot"].to_numpy(float),
                s=float(args.marker_size),
                marker=marker,
                c=BIN_COLORS[bl],
                edgecolors="#2e2e2e",
                linewidths=float(args.edge_width),
                alpha=float(args.alpha),
                zorder=4,
            )

        ax.plot([lo, hi], [lo, hi], "--", color="#9a9a9a", linewidth=1.15, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title, pad=6)
        ax.set_facecolor("#ececec")  # keep lower triangle clearly gray
        ax.set_axisbelow(True)
        ax.grid(which="major", linestyle="-", color="#c3c3c3", alpha=0.42, linewidth=0.65)
        ax.grid(which="minor", linestyle="-", color="#d3d3d3", alpha=0.30, linewidth=0.45)
        # Keep upper-left triangle white, but below grid so grid lines stay visible there.
        ax.fill_between(shade_x, shade_x, hi, color="white", alpha=1.0, linewidth=0, zorder=0.2)

        # Bring back the semantic annotations in each panel.
        ax.text(
            0.36,
            0.10,
            r"unreachable: $Q_{80} < Q_{80}^{lb}$",
            transform=ax.transAxes,
            color="#6a6a6a",
            fontsize=10.2,
            ha="center",
            va="center",
            zorder=1,
        )

    for ax in axes[1, :]:
        ax.set_xlabel(r"Predicted $Q_{80}$")
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel(r"$Q_{80}$")

    color_legend = [
        Patch(facecolor=BIN_COLORS[bl], edgecolor="#2e2e2e", linewidth=0.9, label=bl)
        for bl in BIN_LABELS
    ]

    panel_axes[0].legend(
        handles=color_legend,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        frameon=True,
        facecolor="white",
        edgecolor="#b8b8b8",
        framealpha=1.0,
        title="Discrete β bins",
        borderpad=0.48,
        labelspacing=0.36,
        fontsize=8.6,
        title_fontsize=9.2,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.965, bottom=0.095, wspace=0.08, hspace=0.12)

    # Add y=x annotation after final layout so angle matches the drawn diagonal.
    fig.canvas.draw()
    for ax in panel_axes:
        x_lab = np.sqrt(lo * hi) * 1.05
        y_lab = x_lab * 0.80  # farther below y=x
        angle = _line_angle_deg(ax, lo, lo, hi, hi)
        ax.text(
            x_lab,
            y_lab,
            r"$y=x$ (uniform holdout-mass bound)",
            transform=ax.transData,
            rotation=angle,
            rotation_mode="anchor",
            color="#666666",
            fontsize=10.8,
            ha="center",
            va="center",
            zorder=1,
        )

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, dpi=args.dpi)
    fig.savefig(args.out_png, dpi=args.dpi)
    plt.close(fig)

    print(f"[saved] {args.out_pdf}")
    print(f"[saved] {args.out_png}")


if __name__ == "__main__":
    main()
