#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final Fig.1 script: target sharpness under a beta sweep.

This script is standalone and produces only the Fig.1 plot (PDF + PNG):
  - faint gray background beta-family curves
  - highlighted betas: 0.6, 0.8, 1.0, 1.2, 1.4
  - x-axis score level uses s = (paper score) - 1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

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


def _parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _beta_grid(b0: float, b1: float, step: float) -> List[float]:
    k = int(round((b1 - b0) / step))
    return [round(b0 + i * step, 10) for i in range(k + 1)]


def _int_to_bits(value: int, n: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(n - 1, -1, -1)], dtype=np.int8)


def _parity_even(bits: np.ndarray) -> bool:
    return (int(np.sum(bits)) % 2) == 0


def _longest_zero_run_between_ones(bits: np.ndarray) -> int:
    ones = [i for i, b in enumerate(bits.tolist()) if b == 1]
    if len(ones) < 2:
        return 0
    gaps = [ones[i + 1] - ones[i] - 1 for i in range(len(ones) - 1)]
    return max(gaps) if gaps else 0


def _build_target_distribution_paper(n: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper-style target p*: even parity support, exp-tilt by score."""
    N = 2 ** n
    scores = np.full(N, -100.0, dtype=np.float64)
    support = np.zeros(N, dtype=bool)

    for x in range(N):
        bits = _int_to_bits(x, n)
        if _parity_even(bits):
            support[x] = True
            scores[x] = 1.0 + float(_longest_zero_run_between_ones(bits))

    logits = np.full(N, -np.inf, dtype=np.float64)
    logits[support] = float(beta) * scores[support]
    max_logit = float(np.max(logits[support]))
    unnorm = np.zeros(N, dtype=np.float64)
    unnorm[support] = np.exp(logits[support] - max_logit)
    p_star = unnorm / max(1e-15, float(np.sum(unnorm)))
    return p_star, support, scores


def _score_mass_curve(n: int, beta: float, no_plus_one: bool = True) -> Dict[str, np.ndarray]:
    p_star, support, scores = _build_target_distribution_paper(n=n, beta=beta)
    s = scores[support].astype(np.int64)
    if no_plus_one:
        s = s - 1
    levels = np.arange(int(np.min(s)), int(np.max(s)) + 1, dtype=np.int64)
    idx_support = np.where(support)[0]
    masses = np.zeros(levels.shape[0], dtype=np.float64)
    for i, lv in enumerate(levels):
        m = s == int(lv)
        if np.any(m):
            masses[i] = float(np.sum(p_star[idx_support[m]]))
    return {"levels": levels, "mass": masses}


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.fabs(float(a) - float(b)) <= tol


def main() -> None:
    ap = argparse.ArgumentParser(description="Final Fig.1 plot: target sharpness under beta sweep.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig1_target_sharpness"),
    )
    ap.add_argument("--highlight-betas", type=str, default="0.6,0.8,1.0,1.2,1.4")
    ap.add_argument("--bg-beta-min", type=float, default=0.1)
    ap.add_argument("--bg-beta-max", type=float, default=2.0)
    ap.add_argument("--bg-beta-step", type=float, default=0.1)
    ap.add_argument("--bg-alpha", type=float, default=0.18)
    ap.add_argument("--bg-color", type=str, default="#8F8F8F")
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    apply_final_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    highlight = _parse_floats(args.highlight_betas)
    bg_betas = _beta_grid(args.bg_beta_min, args.bg_beta_max, args.bg_beta_step)
    all_betas = sorted({float(b) for b in bg_betas} | {float(b) for b in highlight})
    curves = {float(b): _score_mass_curve(n=int(args.n), beta=float(b), no_plus_one=True) for b in all_betas}
    levels_ref = curves[all_betas[0]]["levels"]

    fig, ax = make_figure()

    for b in bg_betas:
        if any(_close(float(b), hb) for hb in highlight):
            continue
        c = curves[float(b)]
        ax.plot(c["levels"], c["mass"], color=str(args.bg_color), linewidth=0.9, alpha=float(args.bg_alpha), zorder=1)

    hl_colors = ["#111111", "#5E0B0B", "#8C1010", "#C01010", "#EE2222"]
    while len(hl_colors) < len(highlight):
        hl_colors.append("#C01010")

    legend_handles = []
    legend_labels = []
    for i, b in enumerate(highlight):
        c = curves[float(b)]
        line, = ax.plot(
            c["levels"],
            c["mass"],
            color=hl_colors[i],
            linewidth=1.8,
            marker="o",
            markersize=4.8,
            zorder=10 + i,
        )
        legend_handles.append(line)
        legend_labels.append(fr"$\beta={b:g}$")

    ax.set_xlim(int(np.min(levels_ref)) - 0.15, int(np.max(levels_ref)) + 0.15)
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Score level s")
    ax.set_ylabel(r"Target mass $p^*(S=s)$")
    ax.grid(True, alpha=0.13, linestyle="--")
    ax.legend(legend_handles, legend_labels, loc="upper left", frameon=False, fontsize=LEGEND_FONTSIZE)

    pdf = outdir / "fig1_target_sharpness_beta_sweep.pdf"
    png = outdir / "fig1_target_sharpness_beta_sweep.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[saved] {pdf}")
    print(f"[saved] {png}")


if __name__ == "__main__":
    main()
