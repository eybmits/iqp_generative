#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final plot script: visible vs invisible holdout recovery (data-driven)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


def run() -> None:
    ap = argparse.ArgumentParser(description="Final visible vs invisible holdout recovery plot (data-driven).")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig5_visibility_visible_invisible_recovery"),
    )
    ap.add_argument(
        "--data-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig5_visibility_visible_invisible_recovery" / "fig5_data_default.npz"),
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
        y_star = np.asarray(z["y_star"], dtype=np.float64)
        y_vis = np.asarray(z["y_vis"], dtype=np.float64)
        y_inv = np.asarray(z["y_inv"], dtype=np.float64)
        y_unif = np.asarray(z["y_unif"], dtype=np.float64)

    apply_final_style()
    fig, ax = make_figure()
    ax.plot(Q, y_star, color="#111111", linestyle="-", linewidth=2.2, label=r"Target $p^*$", zorder=7)
    ax.plot(Q, y_vis, color="#1f77b4", linestyle=(0, (8, 3)), linewidth=2.0, label=r"Visible $\mathcal{H}_{\mathrm{vis}}$", zorder=6)
    ax.plot(Q, y_inv, color="#0b3d91", linestyle=(0, (2, 2)), linewidth=2.0, label=r"Invisible $\mathcal{H}_{\mathrm{inv}}$", zorder=5)
    ax.plot(Q, y_unif, color="#8A8A8A", linestyle="--", linewidth=1.6, label="Uniform", zorder=4)
    ax.set_xlim(0, int(np.max(Q)))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"$Q$ samples from model")
    ax.set_ylabel(r"Recovery $R(Q)$")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.93),
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        facecolor="white",
        edgecolor="#bfbfbf",
    )

    out_pdf = outdir / "fig5_visibility_visible_invisible_recovery.pdf"
    out_png = outdir / "fig5_visibility_visible_invisible_recovery.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    run()
