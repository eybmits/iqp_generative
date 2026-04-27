#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legend-only rerender of the existing cross-class pointcloud diagnostics.

This intentionally mirrors the existing pointcloud plots:

* plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary_pointcloud.pdf
* plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q1000_pointcloud.pdf

Only the legends are changed: marker-only handles, shortened Ising label, and
compact right-aligned spacing. Geometry, axes, data, jitter, lines, points, and
error bars follow the existing pointcloud renderers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from final_plot_style import (  # noqa: E402
    IEEE_TWOUP_PANEL_H_IN,
    IEEE_TWOUP_PANEL_W_IN,
    apply_ieee_latex_style,
    save_exact_figure,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_STEM = "fig_cross_class_diagnostics_aligned_textwidth"
DEFAULT_OUTDIR = ROOT / "plots" / "aligned_cross_class_diagnostics"
DOCUMENTS_COPY = Path("/Users/superposition/Documents/fig_cross_class_diagnostics_aligned_textwidth.pdf")

KL_METRICS_CSV = ROOT / "plots" / "experiment_2_beta_kl_summary" / "experiment_2_beta_kl_summary_metrics_per_seed.csv"
COVERAGE_METRICS_CSV = (
    ROOT / "plots" / "experiment_3_beta_quality_coverage" / "experiment_3_beta_quality_coverage_metrics_per_seed.csv"
)

TEXTWIDTH_PT = 516.0
TEX_PT_PER_IN = 72.27
FIG_W_IN = TEXTWIDTH_PT / TEX_PT_PER_IN
FIG_H_IN = 150.0 / TEX_PT_PER_IN

POINTCLOUD_LEFT = 0.19
POINTCLOUD_RIGHT = 0.985
POINTCLOUD_BOTTOM = 0.24
POINTCLOUD_TOP = 0.95
POINTCLOUD_XPAD_IN_STEPS = 0.35

MODEL_ORDER = [
    "iqp_parity_mse",
    "classical_nnn_fields_parity",
    "classical_dense_fields_xent",
    "classical_transformer_mle",
    "classical_maxent_parity",
]

MODEL_STYLE = {
    "iqp_parity_mse": {"label": "IQP-parity", "color": "#D62728", "ls": "-", "lw": 2.35},
    "classical_nnn_fields_parity": {"label": "Ising+fields", "color": "#1f77b4", "ls": "-", "lw": 1.85},
    "classical_dense_fields_xent": {
        "label": "Dense Ising+fields",
        "color": "#8c564b",
        "ls": (0, (5, 2)),
        "lw": 1.85,
    },
    "classical_transformer_mle": {"label": "AR Transformer", "color": "#17becf", "ls": "--", "lw": 1.90},
    "classical_maxent_parity": {"label": "MaxEnt-parity", "color": "#9467bd", "ls": "-.", "lw": 1.90},
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _group_metric(rows: list[dict[str, str]], metric: str) -> dict[str, dict[float, list[float]]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model_key = str(row["model_key"])
        if model_key not in MODEL_ORDER:
            continue
        grouped[model_key][float(row["beta"])].append(float(row[metric]))
    return grouped


def _major_beta_ticks(all_betas: list[float]) -> list[float]:
    if not all_betas:
        return []
    candidates = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
    lo = min(all_betas) - 1e-9
    hi = max(all_betas) + 1e-9
    return [tick for tick in candidates if lo <= tick <= hi]


def _legend_handles() -> list[Line2D]:
    handles: list[Line2D] = []
    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        color = str(style["color"])
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markersize=5.0,
                markerfacecolor=color,
                markeredgecolor=color,
                label=str(style["label"]),
            )
        )
    return handles


def _apply_compact_legend(ax: plt.Axes) -> None:
    legend = ax.legend(
        handles=_legend_handles(),
        loc="upper right",
        alignment="right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#BFBFBF",
        fontsize=8.2,
        borderpad=0.15,
        labelspacing=0.10,
        handlelength=0.42,
        handletextpad=0.34,
        borderaxespad=0.22,
    )
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(100)


def _apply_pointcloud_axes(ax: plt.Axes, all_betas: list[float]) -> None:
    beta_step = min(abs(all_betas[idx + 1] - all_betas[idx]) for idx in range(len(all_betas) - 1))
    x_pad = POINTCLOUD_XPAD_IN_STEPS * beta_step
    ax.set_xlim(float(min(all_betas)) - x_pad, float(max(all_betas)) + x_pad)
    ax.set_xticks(_major_beta_ticks(all_betas))
    ax.set_xticklabels([f"{tick:.1f}" for tick in _major_beta_ticks(all_betas)])
    ax.grid(True, ls="--", lw=0.5, alpha=0.25)


def _draw_pointcloud(
    ax: plt.Axes,
    grouped: dict[str, dict[float, list[float]]],
    *,
    metric_kind: str,
) -> tuple[float, float]:
    all_betas = sorted({beta for values_by_beta in grouped.values() for beta in values_by_beta})
    rng = np.random.default_rng(0)
    ymin = float("inf")
    ymax = float("-inf")

    for model_key in MODEL_ORDER:
        style = MODEL_STYLE[model_key]
        color = str(style["color"])
        mean_x: list[float] = []
        mean_y: list[float] = []
        std_y: list[float] = []
        for beta in all_betas:
            vals = np.asarray(grouped.get(model_key, {}).get(float(beta), []), dtype=np.float64)
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.028, 0.028, size=vals.size)
            ax.scatter(
                np.full(vals.size, float(beta)) + jitter,
                vals,
                s=16,
                color=color,
                alpha=0.18,
                edgecolors="none",
                zorder=2,
            )
            mean_x.append(float(beta))
            mean_y.append(float(np.mean(vals)))
            std_y.append(float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0)
            ymin = min(ymin, float(np.min(vals)))
            ymax = max(ymax, float(np.max(vals)))

        if not mean_x:
            continue

        x_arr = np.asarray(mean_x, dtype=np.float64)
        y_arr = np.asarray(mean_y, dtype=np.float64)
        sd_arr = np.asarray(std_y, dtype=np.float64)
        if metric_kind == "kl":
            elinewidth, capsize, capthick, lw_scale, marker_size = 1.0, 2.2, 1.0, 0.74, 24
        else:
            elinewidth, capsize, capthick, lw_scale, marker_size = 1.15, 2.6, 1.15, 0.85, 28
        ax.errorbar(
            x_arr,
            y_arr,
            yerr=sd_arr,
            fmt="none",
            ecolor=color,
            elinewidth=elinewidth,
            capsize=capsize,
            capthick=capthick,
            alpha=0.55,
            zorder=3,
        )
        ax.plot(
            x_arr,
            y_arr,
            color=color,
            ls=style["ls"],
            lw=float(style["lw"]) * lw_scale,
            alpha=0.9,
            zorder=4,
        )
        ax.scatter(
            x_arr,
            y_arr,
            s=marker_size,
            color=color,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.6,
            zorder=5,
        )

    _apply_pointcloud_axes(ax, all_betas)
    return ymin, ymax


def _draw_kl_panel(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    grouped = _group_metric(rows, "KL_pstar_to_q")
    _draw_pointcloud(ax, grouped, metric_kind="kl")
    ax.set_xlabel(r"$\beta$", labelpad=2.0)
    ax.set_ylabel(r"$D_{\mathrm{KL}}(p^* \parallel q)$", labelpad=4.0)
    ax.set_ylim(0.0, 4.0)
    _apply_compact_legend(ax)


def _draw_coverage_panel(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    grouped = _group_metric(rows, "quality_coverage_Q1000")
    ymin, ymax = _draw_pointcloud(ax, grouped, metric_kind="coverage")
    ax.set_xlabel(r"$\beta$", labelpad=2.0)
    ax.set_ylabel(r"$C_q(Q=1{,}000)$", labelpad=4.0)
    pad = 0.08 * max(ymax - ymin, 1e-6)
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    _apply_compact_legend(ax)


def _save_single_panels(outdir: Path, kl_rows: list[dict[str, str]], cov_rows: list[dict[str, str]]) -> tuple[Path, Path]:
    kl_pdf = outdir / "experiment_2_beta_kl_summary_pointcloud_legend_only.pdf"
    cov_pdf = outdir / "experiment_3_beta_quality_coverage_q1000_pointcloud_legend_only.pdf"

    apply_ieee_latex_style(use_tex=True)
    fig, ax = plt.subplots(figsize=(IEEE_TWOUP_PANEL_W_IN, IEEE_TWOUP_PANEL_H_IN))
    fig.subplots_adjust(left=POINTCLOUD_LEFT, right=POINTCLOUD_RIGHT, bottom=POINTCLOUD_BOTTOM, top=POINTCLOUD_TOP)
    _draw_kl_panel(ax, kl_rows)
    save_exact_figure(fig, kl_pdf)
    plt.close(fig)

    apply_ieee_latex_style(use_tex=True)
    fig, ax = plt.subplots(figsize=(IEEE_TWOUP_PANEL_W_IN, IEEE_TWOUP_PANEL_H_IN))
    fig.subplots_adjust(left=POINTCLOUD_LEFT, right=POINTCLOUD_RIGHT, bottom=POINTCLOUD_BOTTOM, top=POINTCLOUD_TOP)
    _draw_coverage_panel(ax, cov_rows)
    save_exact_figure(fig, cov_pdf)
    plt.close(fig)
    return kl_pdf, cov_pdf


def render(outdir: Path, *, copy_to_documents: bool) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    kl_rows = _read_rows(KL_METRICS_CSV)
    cov_rows = _read_rows(COVERAGE_METRICS_CSV)

    single_kl_pdf, single_cov_pdf = _save_single_panels(outdir, kl_rows, cov_rows)

    apply_ieee_latex_style(use_tex=True)
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), constrained_layout=False)
    ax_kl = fig.add_axes([0.092, 0.305, 0.382, 0.635])
    ax_cov = fig.add_axes([0.604, 0.305, 0.382, 0.635])
    _draw_kl_panel(ax_kl, kl_rows)
    _draw_coverage_panel(ax_cov, cov_rows)
    fig.text(0.283, 0.070, r"(a) Forward KL across the full $\beta$ sweep", ha="center", fontsize=7.7)
    fig.text(0.795, 0.070, r"(b) High-value-state coverage at $Q=1000$", ha="center", fontsize=7.7)

    out_pdf = outdir / f"{OUTPUT_STEM}.pdf"
    out_png = outdir / f"{OUTPUT_STEM}.png"
    save_exact_figure(fig, out_pdf)
    fig.savefig(out_png, format="png", dpi=500, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)

    if copy_to_documents:
        shutil.copyfile(out_pdf, DOCUMENTS_COPY)

    meta = {
        "script": Path(__file__).name,
        "output_pdf": _rel(out_pdf),
        "output_png": _rel(out_png),
        "documents_copy": str(DOCUMENTS_COPY) if copy_to_documents else None,
        "single_kl_pdf": _rel(single_kl_pdf),
        "single_coverage_pdf": _rel(single_cov_pdf),
        "source_kl_pointcloud_pdf": "plots/experiment_2_beta_kl_summary/experiment_2_beta_kl_summary_pointcloud.pdf",
        "source_coverage_pointcloud_pdf": "plots/experiment_3_beta_quality_coverage/experiment_3_beta_quality_coverage_q1000_pointcloud.pdf",
        "legend_only_changes": "Marker-only legend handles, shortened Ising+fields label, compact right-aligned legend.",
    }
    (outdir / f"{OUTPUT_STEM}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Legend-only exact-style cross-class diagnostics.")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--copy-to-documents", type=int, default=1)
    args = ap.parse_args()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    out_pdf = render(outdir, copy_to_documents=bool(int(args.copy_to_documents)))
    print(f"[saved] {out_pdf}")


if __name__ == "__main__":
    main()
