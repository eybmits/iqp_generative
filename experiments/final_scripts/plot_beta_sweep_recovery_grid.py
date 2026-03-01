#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final plot script: beta-sweep recovery grid (data-driven)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# Final constants
PNG_DPI = 300
COLOR_TARGET = "#222222"
COLOR_GRAY = "#666666"
LW_TARGET = 1.95
LW_UNIFORM = 1.35
LW_MODEL_SCALE = 1.0
LW_Q80 = 1.2
MS_Q80 = 6.0


def _tv_alpha_and_marker(tv_vals: np.ndarray) -> Dict[int, Dict[str, float]]:
    tv_min = float(np.min(tv_vals))
    tv_max = float(np.max(tv_vals))
    span = max(1e-12, tv_max - tv_min)
    out: Dict[int, Dict[str, float]] = {}
    for i, tv in enumerate(tv_vals.tolist()):
        quality = 1.0 - ((float(tv) - tv_min) / span)
        alpha = 0.35 + 0.65 * quality
        msize = 2.2 + 3.0 * quality
        out[i] = {"alpha": float(alpha), "msize": float(msize)}
    return out


def _first_q_crossing(Q: np.ndarray, y: np.ndarray, thr: float) -> float:
    idx = np.where(y >= thr)[0]
    if idx.size == 0:
        return float("inf")
    i = int(idx[0])
    if i == 0:
        return float(Q[0])
    x0, x1 = float(Q[i - 1]), float(Q[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 <= y0 + 1e-12:
        return x1
    t = (float(thr) - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return x0 + t * (x1 - x0)


def _legend_handles_compact(model_labels: List[str], style_color: List[str], style_ls: List[object], style_lw: List[float]) -> List[Line2D]:
    handles: List[Line2D] = [
        Line2D([0], [0], color=COLOR_TARGET, lw=LW_TARGET, ls="-", label=r"Target $p^*$"),
    ]
    for lab, col, ls, lw in zip(model_labels, style_color, style_ls, style_lw):
        handles.append(
            Line2D(
                [0],
                [0],
                color=str(col),
                lw=max(1.0, float(lw) * LW_MODEL_SCALE),
                ls=ls,
                label=str(lab),
            )
        )
    handles.append(Line2D([0], [0], color=COLOR_GRAY, lw=LW_UNIFORM, ls="--", label="Uniform"))
    return handles


def run() -> None:
    ap = argparse.ArgumentParser(description="Final beta-sweep recovery grid plot (data-driven).")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid"),
    )
    ap.add_argument(
        "--data-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig6_beta_sweep_recovery_grid" / "fig6_data_default.npz"),
    )
    ap.add_argument("--q80-thr", type=float, default=0.8)
    ap.add_argument("--grid-cols", type=int, default=4)
    ap.add_argument("--qmax", type=int, default=10000)
    ap.add_argument("--log-x", type=int, default=0, choices=[0, 1])
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_npz = Path(args.data_npz)
    if not data_npz.exists():
        raise FileNotFoundError(f"Missing data file: {data_npz}")

    # base style aligned with final fig6 look
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.12,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    with np.load(data_npz, allow_pickle=True) as z:
        betas = np.asarray(z["betas"], dtype=np.float64)
        Q = np.asarray(z["Q"], dtype=np.int64)
        y_target = np.asarray(z["y_target"], dtype=np.float64)
        y_unif = np.asarray(z["y_unif"], dtype=np.float64)
        y_models = np.asarray(z["y_models"], dtype=np.float64)
        tv_vals = np.asarray(z["tv_vals"], dtype=np.float64)
        model_order = [str(x) for x in z["model_order"].tolist()]
        model_labels = [str(x) for x in z["model_labels"].tolist()]
        style_color = [str(x) for x in z["style_color"].tolist()]
        style_ls = z["style_ls"].tolist()
        style_lw = [float(x) for x in z["style_lw"].tolist()]
        style_z = [float(x) for x in z["style_z"].tolist()]

    ncols = max(1, int(args.grid_cols))
    nrows = int(np.ceil(len(betas) / ncols))
    panel_w = 3.0
    panel_h = 2.18
    fig_w = max(6.0, panel_w * ncols)
    fig_h = max(4.4, panel_h * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    x_max = int(args.qmax)
    x_min = 1 if bool(int(args.log_x)) else 0

    for i, beta in enumerate(betas):
        ax = axes[i]
        yt = y_target[i]
        yu = y_unif[i]
        ym = y_models[i]
        tv = tv_vals[i]
        enc = _tv_alpha_and_marker(tv)

        ax.plot(Q, yt, color=COLOR_TARGET, linewidth=LW_TARGET, zorder=20)
        ax.plot(Q, yu, color=COLOR_GRAY, linewidth=LW_UNIFORM, linestyle="--", alpha=0.85, zorder=1)

        for j, _key in enumerate(model_order):
            y = ym[j]
            alpha = enc[j]["alpha"]
            msize = enc[j]["msize"]
            ax.plot(
                Q,
                y,
                color=style_color[j],
                linestyle=style_ls[j],
                linewidth=max(1.0, float(style_lw[j]) * LW_MODEL_SCALE),
                alpha=alpha,
                zorder=style_z[j],
            )
            y_end = float(np.interp(float(np.max(Q)), Q.astype(np.float64), y.astype(np.float64)))
            ax.plot([float(np.max(Q))], [y_end], marker="o", markersize=msize, color=style_color[j], alpha=alpha, zorder=style_z[j] + 0.2)

        candidate_curves: Dict[str, np.ndarray] = {model_order[j]: ym[j] for j in range(len(model_order))}
        candidate_curves["uniform_random"] = yu
        candidate_colors: Dict[str, str] = {model_order[j]: style_color[j] for j in range(len(model_order))}
        candidate_colors["uniform_random"] = COLOR_GRAY

        winner_key = None
        winner_q80 = float("inf")
        for key, y_curve in candidate_curves.items():
            q80 = _first_q_crossing(Q.astype(np.float64), y_curve.astype(np.float64), float(args.q80_thr))
            if np.isfinite(q80) and q80 < winner_q80:
                winner_q80 = float(q80)
                winner_key = key

        if winner_key is not None and np.isfinite(winner_q80):
            q80_mark = float(np.clip(winner_q80, x_min, x_max))
            wcolor = candidate_colors[winner_key]
            y_q80 = float(np.interp(q80_mark, Q.astype(np.float64), candidate_curves[winner_key].astype(np.float64)))
            ax.axvspan(q80_mark, x_max, color="#FFFFFF", alpha=0.42, zorder=25)
            ax.axvline(q80_mark, color=wcolor, linestyle="--", linewidth=LW_Q80, alpha=0.95, zorder=28)
            ax.plot([q80_mark], [y_q80], marker="o", markersize=MS_Q80, markerfacecolor=wcolor, markeredgecolor="white", markeredgewidth=0.8, zorder=30)
            q80_text_x = float(min(x_max, q80_mark + 260.0))
            ax.text(
                q80_text_x,
                0.07,
                "Q80",
                color=wcolor,
                fontsize=7,
                rotation=0,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6),
                zorder=31,
            )

        if bool(int(args.log_x)):
            ax.set_xscale("log")
            ax.set_xlim(1, x_max)
        else:
            ax.set_xlim(0, x_max)
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color=COLOR_GRAY, linestyle=":", alpha=0.6, linewidth=0.9)
        ax.set_title(fr"$\beta={beta:g}$")
        ax.set_xlabel("Q samples from model", fontsize=12)

        if i % ncols == 0:
            ax.set_ylabel(r"Recovery $R(Q)$")
        else:
            ax.tick_params(labelleft=False)

    for j in range(len(betas), len(axes)):
        axes[j].axis("off")

    if len(axes) > 0:
        legend = axes[0].legend(
            handles=_legend_handles_compact(model_labels, style_color, style_ls, style_lw),
            loc="lower right",
            bbox_to_anchor=(0.985, 0.03),
            fontsize=7.2,
            frameon=True,
            framealpha=1.0,
            facecolor="#FFFFFF",
            edgecolor="#D8D8D8",
            handlelength=2.7,
            labelspacing=0.24,
            borderpad=0.26,
            handletextpad=0.55,
            borderaxespad=0.0,
        )
        legend.set_zorder(60)

    out_pdf = outdir / "fig6_beta_sweep_recovery_grid.pdf"
    out_png = outdir / "fig6_beta_sweep_recovery_grid.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    run()
