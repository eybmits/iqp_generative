#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Target score-mass profile with highlighted betas and gray background beta family.

Default behavior:
  - highlight: beta = [0.6, 0.8, 1.0, 1.2, 1.4]
  - background: beta in [0.1, 2.0] step 0.1 (light gray curves)
  - score on x-axis uses no-plus-one convention: s = (paper score) - 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iqp_generative import core as hv  # noqa: E402


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _beta_grid(b0: float, b1: float, step: float) -> List[float]:
    k = int(round((b1 - b0) / step))
    return [round(b0 + i * step, 10) for i in range(k + 1)]


def _score_mass_curve(n: int, beta: float, no_plus_one: bool = True) -> Dict[str, np.ndarray]:
    p_star, support, scores = hv.build_target_distribution_paper(n, beta)
    s = scores[support].astype(np.int64)
    if no_plus_one:
        s = s - 1
    s_min, s_max = int(np.min(s)), int(np.max(s))
    levels = np.arange(s_min, s_max + 1, dtype=np.int64)
    idx = np.where(support)[0]
    masses = np.zeros_like(levels, dtype=np.float64)
    for i, lv in enumerate(levels):
        mask = s == lv
        if np.any(mask):
            masses[i] = float(np.sum(p_star[idx[mask]]))
    return {"levels": levels, "mass": masses}


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def run() -> None:
    ap = argparse.ArgumentParser(description="Target score profile with gray beta background.")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "paper_even_final" / "35_target_problem_5betas_nature"),
    )
    ap.add_argument("--highlight-betas", type=str, default="0.6,0.8,1.0,1.2,1.4")
    ap.add_argument("--bg-beta-min", type=float, default=0.1)
    ap.add_argument("--bg-beta-max", type=float, default=2.0)
    ap.add_argument("--bg-beta-step", type=float, default=0.1)
    ap.add_argument("--bg-alpha", type=float, default=0.18)
    ap.add_argument("--bg-color", type=str, default="#8F8F8F")
    ap.add_argument("--no-plus-one", action="store_true", default=True)
    args = ap.parse_args()

    hv.set_style(base=8)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    highlight = _parse_floats(args.highlight_betas)
    bg_betas = _beta_grid(args.bg_beta_min, args.bg_beta_max, args.bg_beta_step)

    # Curves for all required betas.
    all_needed = sorted({float(b) for b in bg_betas} | {float(b) for b in highlight})
    curves = {float(b): _score_mass_curve(args.n, float(b), no_plus_one=bool(args.no_plus_one)) for b in all_needed}
    levels_ref = curves[all_needed[0]]["levels"]

    fig, ax = plt.subplots(figsize=(5.1, 3.7))

    # Background gray family.
    for b in bg_betas:
        if any(_close(b, hb) for hb in highlight):
            continue
        c = curves[float(b)]
        ax.plot(
            c["levels"],
            c["mass"],
            color=args.bg_color,
            linewidth=1.0,
            alpha=float(args.bg_alpha),
            zorder=1,
        )

    # Highlight curves (same style as existing figure).
    hl_colors = ["#111111", "#5E0B0B", "#8C1010", "#C01010", "#EE2222"]
    while len(hl_colors) < len(highlight):
        hl_colors.append("#C01010")
    legend_handles = []
    legend_labels = []
    for i, b in enumerate(highlight):
        c = curves[float(b)]
        ln, = ax.plot(
            c["levels"],
            c["mass"],
            color=hl_colors[i],
            linewidth=1.8,
            marker="o",
            markersize=5.2,
            zorder=10 + i,
        )
        legend_handles.append(ln)
        legend_labels.append(fr"$\beta={b:g}$")

    ax.set_xlim(int(np.min(levels_ref)) - 0.15, int(np.max(levels_ref)) + 0.15)
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Score level s")
    ax.set_ylabel(r"Target probability mass $p^*(S=s)$")
    ax.grid(True, alpha=0.13, linestyle="--")
    ax.legend(legend_handles, legend_labels, loc="upper left", frameon=False, fontsize=8)

    pdf = outdir / "nature_plotA_target_score_5betas_hd_no_plus1_with_bg_beta0p1_to_2p0.pdf"
    png = outdir / "nature_plotA_target_score_5betas_hd_no_plus1_with_bg_beta0p1_to_2p0.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)

    # Save table for reproducibility.
    csv_path = outdir / "target_score_mass_all_betas_0p1_to_2p0.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "score_level", "mass"])
        for b in all_needed:
            c = curves[b]
            for s, m in zip(c["levels"].tolist(), c["mass"].tolist()):
                w.writerow([float(b), int(s), float(m)])

    print(f"[saved] {pdf}")
    print(f"[saved] {png}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    run()

