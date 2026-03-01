#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final plot script: IQP sigma-K ablation recovery panel (data-driven)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Final locked style
FIG_W = 243.12 / 72.0
FIG_H = 185.52 / 72.0
PNG_DPI = 300
LEGEND_FONTSIZE = 7.2


def apply_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": LEGEND_FONTSIZE,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "gray",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.03,
        }
    )


def make_figure():
    return plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)


def _pure_red_shades(n: int) -> List[Tuple[float, float, float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(0.92, 0.62, 0.64, 1.0)]
    c0 = np.array([0.97, 0.85, 0.86])
    c1 = np.array([0.84, 0.30, 0.34])
    out: List[Tuple[float, float, float, float]] = []
    for i in range(n):
        t = i / float(n - 1)
        c = (1.0 - t) * c0 + t * c1
        out.append((float(c[0]), float(c[1]), float(c[2]), 1.0))
    return out


def run() -> None:
    ap = argparse.ArgumentParser(description="Final IQP sigma-K ablation recovery plot (data-driven).")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig2_iqp_sigmak_ablation_recovery"),
    )
    ap.add_argument(
        "--data-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig2_iqp_sigmak_ablation_recovery" / "fig2_data_default.npz"),
    )
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_npz = Path(args.data_npz)
    if not data_npz.exists():
        raise FileNotFoundError(f"Missing data file: {data_npz}")

    with np.load(data_npz, allow_pickle=False) as z:
        Q = np.asarray(z["Q"], dtype=np.int64)
        y_target = np.asarray(z["y_target"], dtype=np.float64)
        y_unif = np.asarray(z["y_unif"], dtype=np.float64)
        y_ref = np.asarray(z["y_ref"], dtype=np.float64)
        y_best = np.asarray(z["y_best"], dtype=np.float64)
        y_other = np.asarray(z["y_other"], dtype=np.float64)
        best_sigma = float(z["best_sigma"])
        best_K = int(z["best_K"])
        reference_loss = str(z["reference_loss"]) if "reference_loss" in z else "prob_mse"

    apply_final_style()
    fig, ax = make_figure()

    ax.plot(Q, y_target, color="#111111", linewidth=2.2, zorder=30)
    ax.plot(Q, y_unif, color="#6E6E6E", linewidth=1.6, linestyle="--", zorder=5)

    reds = _pure_red_shades(max(1, y_other.shape[0]))
    for i in range(y_other.shape[0]):
        ax.plot(Q, y_other[i], color=reds[i], linewidth=1.3, alpha=0.98, zorder=10)

    ax.plot(Q, y_best, color="#C40000", linewidth=2.8, zorder=40)
    ax.plot(Q, y_ref, color="#1F77B4", linewidth=1.9, zorder=20)

    ax.set_xlim(0, 10000)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.grid(True, alpha=0.16, linestyle="--")

    ref_label = "IQP MSE" if reference_loss.lower() == "prob_mse" else f"IQP {reference_loss.upper()}"
    handles = [
        Line2D([0], [0], color="#111111", lw=2.2, label=r"Target $p^*$"),
        Line2D([0], [0], color="#C40000", lw=2.8, label=fr"IQP Parity ($\sigma={best_sigma:g}, K={best_K}$)"),
        Line2D([0], [0], color=_pure_red_shades(1)[0], lw=1.5, label=fr"IQP Parity (n={int(y_other.shape[0])})"),
        Line2D([0], [0], color="#1F77B4", lw=1.9, label=ref_label),
        Line2D([0], [0], color="#6E6E6E", lw=1.6, ls="--", label="Uniform"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.975, 0.055),
        borderaxespad=0.18,
        frameon=True,
        fontsize=6.6,
        handlelength=1.6,
        labelspacing=0.25,
        borderpad=0.25,
        facecolor="white",
        edgecolor="#bfbfbf",
    )

    out_pdf = outdir / "fig2_iqp_sigmak_ablation_recovery.pdf"
    out_png = outdir / "fig2_iqp_sigmak_ablation_recovery.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    run()
