#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the three KL-summary panels as one LaTeX-sized textwidth figure.

This is a rerender-only composition of the three existing panels used in the
manuscript draft:

1. Experiment 8 score-level marginal fit,
2. Experiment 1 legacy sigma-K heat map,
3. Experiment 1 benchmark-only fixed-beta comparison.

The output PDF has a deterministic natural width of 516 pt, matching a common
IEEEtran full text width for direct ``\\includegraphics[width=1.0\\textwidth]``
inclusion without further font-size drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiment_1_kl_diagnostics import (  # noqa: E402
    CMAP_KL_CONSISTENT,
    COLOR_AXIS,
    COLOR_DARK,
    COLOR_IQP_MSE,
    COLOR_LIGHT_TEXT,
    COLOR_SUBTEXT,
    COLOR_TEXT,
    _luminance,
)
from final_plot_style import (  # noqa: E402
    PARITY_COLOR,
    TARGET_COLOR,
    TEXT_DARK,
    apply_ieee_latex_style,
    save_exact_figure,
)
from model_labels import IQP_MSE_LABEL, IQP_PARITY_LABEL  # noqa: E402


ROOT = Path(__file__).resolve().parent
OUTPUT_STEM = "fig_kl_triptych_aligned_textwidth"
DEFAULT_OUTDIR = ROOT / "plots" / "aligned_kl_triptych"

SCORE_CSV = (
    ROOT
    / "plots"
    / "experiment_8_fixed_beta_bucket_fit_iqp_vs_mse"
    / "experiment_8_fixed_beta_bucket_fit_iqp_vs_mse_bucket_masses.csv"
)
EXP1_NPZ = ROOT / "plots" / "experiment_1_kl_diagnostics" / "experiment_1_data.npz"

TEXTWIDTH_PT = 516.0
TEX_PT_PER_IN = 72.27
FIG_W_IN = TEXTWIDTH_PT / TEX_PT_PER_IN
FIG_H_IN = 150.0 / TEX_PT_PER_IN

PANEL_C_BEST_MEAN = 0.402
PANEL_C_BEST_CI = 0.021
PANEL_C_MSE_MEAN = 0.492
PANEL_C_MSE_CI = 0.058
PANEL_BOX_ASPECT = 0.7204825053354299


def _load_score_rows(path: Path) -> dict[str, np.ndarray | float]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    score_vals = np.asarray([int(row["score_level"]) for row in rows], dtype=np.int64)
    return {
        "score_vals": score_vals,
        "score_display": score_vals - int(np.min(score_vals)),
        "target": np.asarray([row["target_mass"] for row in rows], dtype=np.float64),
        "parity": np.asarray([row["iqp_parity_mass"] for row in rows], dtype=np.float64),
        "mse": np.asarray([row["iqp_mse_mass"] for row in rows], dtype=np.float64),
        "parity_kl": float(rows[0]["parity_kl"]),
        "mse_kl": float(rows[0]["mse_kl"]),
    }


def _fmt_sigma(val: float) -> str:
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    return f"{val:g}"


def _panel_label(fig: plt.Figure, x: float, text: str) -> None:
    fig.text(
        x,
        0.080,
        text,
        ha="center",
        va="center",
        fontsize=8.8,
        color=TEXT_DARK,
    )


def _render_score_panel(ax: plt.Axes, score_data: dict[str, np.ndarray | float]) -> None:
    score_display = np.asarray(score_data["score_display"], dtype=np.int64)
    target = np.asarray(score_data["target"], dtype=np.float64)
    parity = np.asarray(score_data["parity"], dtype=np.float64)
    mse = np.asarray(score_data["mse"], dtype=np.float64)
    parity_kl = float(score_data["parity_kl"])
    mse_kl = float(score_data["mse_kl"])

    group_step = 1.60
    x = np.arange(score_display.size, dtype=np.float64) * group_step
    width = 0.40
    parity_offset = width
    mse_offset = 2.0 * width
    ax.bar(
        x,
        target,
        width=width,
        color=TARGET_COLOR,
        alpha=0.85,
        edgecolor="none",
        linewidth=0.0,
        label=r"Target $p^*$",
        zorder=3,
    )
    ax.bar(
        x + parity_offset,
        parity,
        width=width,
        color=PARITY_COLOR,
        alpha=0.88,
        edgecolor="none",
        linewidth=0.0,
        label=rf"{IQP_PARITY_LABEL} ($KL = {parity_kl:.3f}$)",
        zorder=3,
    )
    ax.bar(
        x + mse_offset,
        mse,
        width=width,
        color=COLOR_IQP_MSE,
        alpha=0.90,
        edgecolor="none",
        linewidth=0.0,
        label=rf"{IQP_MSE_LABEL} ($KL = {mse_kl:.3f}$)",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(s)) for s in score_display.tolist()])
    ax.set_xlim(x[0] - 0.45, x[-1] + mse_offset + width / 2.0 + 0.45)
    ax.set_xlabel(r"Score level $s$", labelpad=1.5)
    ax.set_ylabel(r"$p(\ell = s)$", labelpad=2.5)
    ax.yaxis.set_label_coords(-0.235, 0.5)
    ax.grid(True, axis="y", ls="--", lw=0.45, alpha=0.24, zorder=0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_linewidth(0.8)
    ax.spines["right"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color(TEXT_DARK)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.axhline(0.0, color=TEXT_DARK, lw=0.9, zorder=9, clip_on=False)
    ax.set_ylim(0.0, max(float(np.max(target)), float(np.max(parity)), float(np.max(mse))) * 1.25)
    ax.tick_params(axis="both", which="major", pad=1.2)
    ax.set_box_aspect(PANEL_BOX_ASPECT)

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        borderpad=0.14,
        labelspacing=0.08,
        handlelength=1.02,
        handletextpad=0.30,
        borderaxespad=0.35,
        fontsize=7.7,
    )
    legend.set_zorder(100)


def _render_heatmap_panel(ax: plt.Axes, cax: plt.Axes, exp1_data: dict[str, np.ndarray]) -> None:
    grid = np.asarray(exp1_data["panel_ab_grid"], dtype=np.float64)
    sigma_values = [float(x) for x in np.asarray(exp1_data["sigma_values"], dtype=np.float64).tolist()]
    k_values = [int(x) for x in np.asarray(exp1_data["k_values"], dtype=np.int64).tolist()]
    best_sigma = float(np.asarray(exp1_data["panel_ab_best_sigma"], dtype=np.float64)[0])
    best_k = int(np.asarray(exp1_data["panel_ab_best_k"], dtype=np.int64)[0])

    vmin = float(np.min(grid))
    vmax = float(np.max(grid))
    tick_lo = math.floor(vmin / 0.05) * 0.05
    tick_hi = math.floor(vmax / 0.05) * 0.05
    norm = colors.Normalize(vmin=tick_lo, vmax=vmax)
    im = ax.imshow(grid, cmap=CMAP_KL_CONSISTENT, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_yticks(np.arange(len(sigma_values)))
    ax.set_yticklabels([_fmt_sigma(s) for s in sigma_values])
    ax.tick_params(axis="x", pad=1.4)
    ax.tick_params(axis="y", pad=1.2)
    ax.set_xlabel(r"$K$", labelpad=2.0)
    ax.set_ylabel(r"$\sigma$", labelpad=0.2)
    ax.yaxis.set_label_coords(-0.070, 0.5)
    ax.set_box_aspect(PANEL_BOX_ASPECT)
    ax.grid(False)

    best_i = sigma_values.index(best_sigma)
    best_j = k_values.index(best_k)

    for i in range(len(sigma_values)):
        for j in range(len(k_values)):
            val = float(grid[i, j])
            rgba = CMAP_KL_CONSISTENT(norm(val))
            text_color = COLOR_DARK if _luminance(rgba) > 0.64 else "#FFFFFF"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color=text_color,
                fontweight="bold" if (i == best_i and j == best_j) else "normal",
            )

    cbar = ax.figure.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_ticks(np.arange(tick_lo, tick_hi + 1e-9, 0.05).tolist())
    if cbar.solids is not None:
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_linewidth(0.0)
    cbar.ax.tick_params(axis="y", length=2.4, labelsize=7.8, colors=COLOR_LIGHT_TEXT, pad=1.2)
    cbar.outline.set_linewidth(0.55)


def _render_benchmark_panel(ax: plt.Axes) -> None:
    entries = [
        ("Target $p^*$", COLOR_TEXT, 0.0, 0.0, ""),
        (f"Best {IQP_PARITY_LABEL}", "#ea8a7d", PANEL_C_BEST_MEAN, PANEL_C_BEST_CI, ""),
        (IQP_MSE_LABEL, "#86afe8", PANEL_C_MSE_MEAN, PANEL_C_MSE_CI, ""),
    ]
    y_positions = np.arange(len(entries))[::-1]
    x_min = -0.015
    x_max = 0.92

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, 0.2))
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, 0.1), minor=True)
    ax.set_xlabel(r"$D_{\mathrm{KL}}(p^* \parallel q)$ (lower better)", labelpad=2.0)
    ax.set_box_aspect(PANEL_BOX_ASPECT)
    ax.grid(True, axis="x", alpha=0.14, linestyle="--", dashes=(2, 2), lw=0.55)
    ax.grid(True, which="minor", axis="x", alpha=0.08, linestyle="--", dashes=(2, 2), lw=0.55)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(0.0, color=COLOR_AXIS, linewidth=0.9, zorder=0)
    ax.tick_params(axis="x", which="major", pad=1.5)

    for y_pos, (label, color, kl_val, ci_val, sublabel) in zip(y_positions, entries):
        bar_end = kl_val if ci_val <= 0.0 else min(x_max, kl_val + 0.010)
        ax.plot(
            [0.0, bar_end],
            [y_pos, y_pos],
            color=color,
            linewidth=6.2,
            alpha=0.88,
            solid_capstyle="butt",
            zorder=1,
        )
        if ci_val > 0.0:
            ax.errorbar(
                kl_val,
                y_pos,
                xerr=np.asarray([[ci_val], [ci_val]], dtype=np.float64),
                fmt="none",
                ecolor=color,
                elinewidth=1.7,
                capsize=3.2,
                capthick=1.7,
                zorder=4,
            )
        marker_edge = "white"
        marker_lw = 1.1
        ax.scatter(
            kl_val,
            y_pos,
            s=50,
            color=color,
            edgecolors=marker_edge,
            linewidths=marker_lw,
            zorder=5,
            clip_on=False,
        )
        value_text = f"KL {kl_val:.3f}" if ci_val <= 0.0 else rf"KL {kl_val:.3f} $\pm$ {ci_val:.3f}"
        value_x = kl_val + ci_val + (0.042 if ci_val <= 0.0 else 0.024)
        value_ha = "left"
        if value_x > x_max - 0.070:
            value_x = x_max - 0.010
            value_ha = "right"
        ax.text(
            value_x,
            y_pos,
            value_text,
            fontsize=7.2,
            va="center",
            ha=value_ha,
            color=COLOR_TEXT,
            clip_on=False,
        )
        if sublabel:
            ax.text(
                -0.035,
                y_pos - 0.34,
                sublabel,
                transform=ax.get_yaxis_transform(),
                fontsize=6.0,
                va="top",
                ha="right",
                fontstyle="italic",
                color=COLOR_SUBTEXT,
                clip_on=False,
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="o",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=5.0,
            label=label,
        )
        for label, color, _kl_val, _ci_val, _sublabel in entries
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        facecolor="white",
        edgecolor="#888888",
        framealpha=0.95,
        borderpad=0.14,
        labelspacing=0.08,
        handlelength=0.65,
        handletextpad=0.48,
        borderaxespad=0.28,
        fontsize=6.8,
    )
    legend.set_zorder(100)


def render(outdir: Path, *, use_tex: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    score_data = _load_score_rows(SCORE_CSV)
    with np.load(EXP1_NPZ, allow_pickle=True) as z:
        exp1_data = {key: np.asarray(z[key]) for key in z.files}

    apply_ieee_latex_style(use_tex=use_tex)
    plt.rcParams.update(
        {
            "font.size": 8.4,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.15,
        }
    )

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
    ax_score = fig.add_axes([0.095, 0.320, 0.252, 0.590])
    ax_heat = fig.add_axes([0.385, 0.320, 0.238, 0.590])
    cax_heat = fig.add_axes([0.628, 0.320, 0.013, 0.590])
    ax_bench = fig.add_axes([0.720, 0.320, 0.238, 0.590])

    _render_score_panel(ax_score, score_data)
    _render_heatmap_panel(ax_heat, cax_heat, exp1_data)
    _render_benchmark_panel(ax_bench)

    _panel_label(fig, 0.221, r"(a) Score-level marginal fit")
    _panel_label(fig, 0.504, r"(b) $\sigma$-$K$ KL heat map")
    _panel_label(fig, 0.839, r"(c) Same-architecture fixed-$\beta$ comparison")

    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"
    save_exact_figure(fig, out_pdf)
    fig.savefig(out_png, format="png", dpi=450, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

    meta = {
        "script": "make_aligned_kl_triptych.py",
        "output_pdf": str(out_pdf.relative_to(ROOT) if out_pdf.is_relative_to(ROOT) else out_pdf),
        "output_png": str(out_png.relative_to(ROOT) if out_png.is_relative_to(ROOT) else out_png),
        "natural_width_pt": TEXTWIDTH_PT,
        "natural_height_pt": FIG_H_IN * TEX_PT_PER_IN,
        "intended_latex_include": r"\includegraphics[width=1.0\textwidth]{...}",
        "source_score_csv": str(SCORE_CSV.relative_to(ROOT)),
        "source_exp1_npz": str(EXP1_NPZ.relative_to(ROOT)),
        "panel_c_values": {
            "best_iqp_parity_mean": PANEL_C_BEST_MEAN,
            "best_iqp_parity_ci": PANEL_C_BEST_CI,
            "iqp_mse_mean": PANEL_C_MSE_MEAN,
            "iqp_mse_ci": PANEL_C_MSE_CI,
            "source_panel": "plots/experiment_1_kl_diagnostics/fig2_kl_summary_panels_benchmark_only.pdf",
        },
        "use_tex": bool(use_tex),
    }
    (outdir / f"{OUTPUT_STEM}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Render aligned textwidth KL triptych.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--use-tex", type=int, default=1)
    args = ap.parse_args()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    out_pdf = render(outdir, use_tex=bool(int(args.use_tex)))
    print(f"[saved] {out_pdf}")


if __name__ == "__main__":
    main()
