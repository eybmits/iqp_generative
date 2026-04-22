#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example wide target-mass sweep plot for visual comparison with the KL panels.

This plot is derived directly from the target distribution p* across the beta
sweep and mirrors the wider aspect ratio used for the approved Experiment 2/3
pointcloud figures, while keeping a simple line-plot style.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiment_1_kl_diagnostics import build_target_distribution_paper
from final_plot_style import TEXT_DARK, apply_final_style


ROOT = Path(__file__).resolve().parent
SCRIPT_REL = "experiment_10_target_mass_beta_sweep.py"
OUTPUT_STEM = "experiment_10_target_mass_beta_sweep"
DEFAULT_OUTDIR = ROOT / "plots" / OUTPUT_STEM

FIG_W = 320.0 / 72.0
FIG_H = 185.52 / 72.0
FIG_4X3 = (4.0, 3.0)
PLOT_LEFT = 0.17
PLOT_RIGHT = 0.985
PLOT_BOTTOM = 0.22
PLOT_TOP = 0.93
PLOT_4X3_LEFT = 0.125
PLOT_4X3_RIGHT = 0.992
PLOT_4X3_BOTTOM = 0.165
PLOT_4X3_TOP = 0.988

ALL_BETAS = np.asarray([x / 10.0 for x in range(1, 21)], dtype=np.float64)
HIGHLIGHT_BETAS = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4], dtype=np.float64)

GRAY_SWEEP_COLOR = "#CFCFCF"
HIGHLIGHT_COLORS = ["#151515", "#5D1C18", "#8B2A22", "#BF3A2E", "#E34234"]


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _bucket_masses(dist: np.ndarray, scores: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score_vals = np.asarray(sorted(int(s) for s in np.unique(scores[support])), dtype=np.int64)
    masses = np.zeros(score_vals.size, dtype=np.float64)
    for idx, s in enumerate(score_vals.tolist()):
        masses[idx] = float(np.sum(np.asarray(dist, dtype=np.float64)[(scores == float(s)) & support]))
    return score_vals, masses


def _parse_float_list(text: str) -> np.ndarray:
    return np.asarray([float(x.strip()) for x in str(text).split(",") if x.strip()], dtype=np.float64)


def _render_plot(
    *,
    out_pdf: Path,
    out_png: Path,
    figsize: tuple[float, float],
    all_betas: np.ndarray,
    highlight_betas: np.ndarray,
    score_display: np.ndarray,
    masses_by_beta: Dict[float, np.ndarray],
    subplot_adjust: tuple[float, float, float, float] = (PLOT_LEFT, PLOT_RIGHT, PLOT_BOTTOM, PLOT_TOP),
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_final_style()
    fig, ax = plt.subplots(figsize=figsize)
    left, right, bottom, top = subplot_adjust
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

    for beta in all_betas.tolist():
        y = masses_by_beta[float(beta)]
        ax.plot(
            score_display,
            y,
            color=GRAY_SWEEP_COLOR,
            lw=0.9,
            alpha=0.33,
            zorder=1,
        )

    for beta, color in zip(highlight_betas.tolist(), HIGHLIGHT_COLORS):
        y = masses_by_beta[float(beta)]
        ax.plot(
            score_display,
            y,
            color=color,
            lw=2.1,
            marker="o",
            markersize=5.0,
            label=rf"$\beta = {beta:g}$",
            zorder=3,
        )

    ax.set_xlabel(r"Score level $s$", labelpad=2.0)
    ax.set_ylabel(r"Target mass $p^*(\ell = s)$", labelpad=4.0)
    ax.set_xlim(float(score_display.min()) - 0.15, float(score_display.max()) + 0.15)
    ax.set_xticks(score_display.tolist())
    ymax = max(float(np.max(vals)) for vals in masses_by_beta.values())
    ax.set_ylim(0.0, ymax * 1.12)
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)

    handles = [
        Line2D([0], [0], color=GRAY_SWEEP_COLOR, lw=1.2, alpha=0.9, label=r"gray: $\beta \in [0.1, 2]$, $\Delta\beta = 0.1$")
    ]
    for beta, color in zip(highlight_betas.tolist(), HIGHLIGHT_COLORS):
        handles.append(
            Line2D([0], [0], color=color, lw=2.1, marker="o", markersize=5.0, label=rf"$\beta = {beta:g}$")
        )
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.6,
        handletextpad=0.5,
        fontsize=7.2,
    )
    legend.set_zorder(100)

    ax.spines["bottom"].set_color(TEXT_DARK)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["bottom"].set_zorder(10)

    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def run() -> None:
    ap = argparse.ArgumentParser(description="Wide target-mass beta-sweep example plot.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--betas", type=str, default=",".join(f"{x:.1f}" for x in ALL_BETAS.tolist()))
    ap.add_argument("--highlight-betas", type=str, default=",".join(f"{x:.1f}" for x in HIGHLIGHT_BETAS.tolist()))
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    all_betas = _parse_float_list(str(args.betas))
    highlight_betas = _parse_float_list(str(args.highlight_betas))

    masses_by_beta: Dict[float, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    raw_score_vals = None
    for beta in all_betas.tolist():
        p_star, support, scores = build_target_distribution_paper(int(args.n), float(beta))
        score_vals, masses = _bucket_masses(p_star, scores, support)
        if raw_score_vals is None:
            raw_score_vals = np.asarray(score_vals, dtype=np.int64)
        masses_by_beta[float(beta)] = np.asarray(masses, dtype=np.float64)
        for idx, score_raw in enumerate(score_vals.tolist()):
            rows.append(
                {
                    "n": int(args.n),
                    "beta": float(beta),
                    "score_level_raw": int(score_raw),
                    "score_level_display": int(score_raw - score_vals.min()),
                    "target_mass": float(masses[idx]),
                }
            )

    if raw_score_vals is None:
        raise RuntimeError("No score levels were generated.")
    score_display = np.asarray(raw_score_vals - int(raw_score_vals.min()), dtype=np.int64)

    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"
    out_fig1_4x3_pdf = outdir / "fig1_target_sharpness_beta_sweep_4x3.pdf"
    out_fig1_4x3_png = outdir / "fig1_target_sharpness_beta_sweep_4x3.png"
    data_csv = outdir / f"{OUTPUT_STEM}.csv"
    run_json = outdir / "RUN_CONFIG.json"

    _write_csv(data_csv, rows)
    _render_plot(
        out_pdf=out_pdf,
        out_png=out_png,
        figsize=(FIG_W, FIG_H),
        all_betas=all_betas,
        highlight_betas=highlight_betas,
        score_display=score_display,
        masses_by_beta=masses_by_beta,
        subplot_adjust=(PLOT_LEFT, PLOT_RIGHT, PLOT_BOTTOM, PLOT_TOP),
    )
    _render_plot(
        out_pdf=out_fig1_4x3_pdf,
        out_png=out_fig1_4x3_png,
        figsize=FIG_4X3,
        all_betas=all_betas,
        highlight_betas=highlight_betas,
        score_display=score_display,
        masses_by_beta=masses_by_beta,
        subplot_adjust=(PLOT_4X3_LEFT, PLOT_4X3_RIGHT, PLOT_4X3_BOTTOM, PLOT_4X3_TOP),
    )
    _write_json(
        run_json,
        {
            "script": SCRIPT_REL,
            "outdir": str(outdir.relative_to(ROOT) if outdir.is_relative_to(ROOT) else outdir),
            "n": int(args.n),
            "betas": [float(x) for x in all_betas.tolist()],
            "highlight_betas": [float(x) for x in highlight_betas.tolist()],
            "pdf": str(out_pdf.relative_to(ROOT) if out_pdf.is_relative_to(ROOT) else out_pdf),
            "png": str(out_png.relative_to(ROOT) if out_png.is_relative_to(ROOT) else out_png),
            "publication_pdf": str(out_fig1_4x3_pdf.relative_to(ROOT) if out_fig1_4x3_pdf.is_relative_to(ROOT) else out_fig1_4x3_pdf),
            "publication_png": str(out_fig1_4x3_png.relative_to(ROOT) if out_fig1_4x3_png.is_relative_to(ROOT) else out_fig1_4x3_png),
            "csv": str(data_csv.relative_to(ROOT) if data_csv.is_relative_to(ROOT) else data_csv),
        },
    )
    print(f"[experiment10] wrote {out_pdf}", flush=True)
    print(f"[experiment10] wrote {out_fig1_4x3_pdf}", flush=True)


if __name__ == "__main__":
    run()
