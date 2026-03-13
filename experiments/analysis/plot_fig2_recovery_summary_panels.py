#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a Fig2 companion summary: sigma-K recovery summary panels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_REL = "experiments/analysis/plot_fig2_recovery_summary_panels.py"
OUTDIR_REL_DEFAULT = "outputs/analysis/fig2_recovery_summary_panels"
STEM = "fig2_recovery_summary_panels"

FIG_W = 243.12 / 72.0
FIG_H = 185.52 / 72.0
PNG_DPI = 300

TARGET_COLOR = "#2F2A2B"
PARITY_COLOR = "#E46C5B"
MSE_COLOR = "#5B9BE6"
UNIFORM_COLOR = "#C6C9CF"
HEATMAP_LOW = "#F04B4C"
HEATMAP_MID = "#8E111B"
HEATMAP_HIGH = "#0D0D0F"
ACCENT_DARK = "#171717"
TEXT_DARK = "#222222"
TEXT_MID = "#8A8A8A"
AXIS_ZERO = "#C7C7C7"
HEATMAP_CBAR_LABEL = r"IQP Parity $Q_{80}$ (samples, lower is better)"


def apply_final_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 7.2,
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


def _first_q_crossing(q: np.ndarray, y: np.ndarray, thr: float) -> float:
    mask = y >= thr
    if not np.any(mask):
        return float("inf")
    j = int(np.flatnonzero(mask)[0])
    if j == 0:
        return float(q[0])
    q0, q1 = float(q[j - 1]), float(q[j])
    y0, y1 = float(y[j - 1]), float(y[j])
    if y1 <= y0:
        return q1
    t = (thr - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return q0 + t * (q1 - q0)


def _curve_metrics(q: np.ndarray, y: np.ndarray, q_thr: float) -> Dict[str, float]:
    q80 = _first_q_crossing(q=q, y=y, thr=q_thr)
    trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc_norm = float(trapz_fn(y, q) / (q[-1] - q[0]))
    return {
        "Q80": float(q80),
        "AUC_norm": auc_norm,
        "R10000": float(y[-1]),
    }


def _luminance(rgba: Tuple[float, float, float, float]) -> float:
    r, g, b, _ = rgba
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _fmt_int(value: float) -> str:
    return f"{int(round(float(value))):,}"


def _render_heatmap_panel(
    ax,
    *,
    sigma_values: List[float],
    k_values: List[int],
    q80_grid: np.ndarray,
    best_mask: np.ndarray,
    cmap,
    norm,
) -> None:
    im = ax.imshow(q80_grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_xticklabels([f"{k}" for k in k_values], fontweight="bold")
    ax.set_yticks(np.arange(len(sigma_values)))
    ax.set_yticklabels([f"{sigma:g}" for sigma in sigma_values], fontweight="bold")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.xaxis.set_ticks_position("top")
    ax.set_ylabel(r"$\sigma$", labelpad=6)
    ax.set_xticks(np.arange(-0.5, len(k_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sigma_values), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(sigma_values)):
        for j in range(len(k_values)):
            q80_val = float(q80_grid[i, j])
            rgba = cmap(norm(q80_val))
            text_color = "#F7F7F7" if _luminance(rgba) < 0.48 else "#1A1A1A"
            ax.text(
                j,
                i - 0.02,
                _fmt_int(q80_val),
                ha="center",
                va="center",
                fontsize=10.9,
                color=text_color,
                fontweight="bold" if best_mask[i, j] else "normal",
            )
            if best_mask[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        linewidth=3.0,
                        edgecolor=ACCENT_DARK,
                    )
                )
                ax.add_patch(
                    Polygon(
                        [
                            (j + 0.32, i - 0.50),
                            (j + 0.50, i - 0.50),
                            (j + 0.50, i - 0.32),
                        ],
                        closed=True,
                        facecolor=ACCENT_DARK,
                        edgecolor=ACCENT_DARK,
                        linewidth=0.0,
                    )
                )
    return im


def _render_benchmark_panel(
    ax,
    *,
    target_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    mse_metrics: Dict[str, float],
    uniform_metrics: Dict[str, float],
) -> None:
    benchmark_entries = [
        ("Target p*", TARGET_COLOR, target_metrics),
        ("Best IQP Parity", PARITY_COLOR, best_metrics),
        ("IQP MSE", MSE_COLOR, mse_metrics),
        ("Uniform", UNIFORM_COLOR, uniform_metrics),
    ]
    y_positions = np.arange(len(benchmark_entries))[::-1]
    max_q80 = max(float(m["Q80"]) for _, _, m in benchmark_entries)
    ax.set_xlim(0.0, 8400.0)
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.set_ylim(-0.6, len(benchmark_entries) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel(r"$Q_{80}$ (samples, lower is better)")
    ax.grid(True, axis="x", alpha=0.14, linestyle="--", dashes=(2, 2))
    ax.grid(False, axis="y")
    ax.axvline(0.0, color=AXIS_ZERO, linewidth=1.1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ytrans = ax.get_yaxis_transform()
    metric_label_dx = max_q80 * 0.045
    for y_pos, (label, color, metrics) in zip(y_positions, benchmark_entries):
        q80_val = float(metrics["Q80"])
        auc_val = float(metrics["AUC_norm"])
        r_end = float(metrics["R10000"])
        ax.plot([0.0, q80_val], [y_pos, y_pos], color=color, linewidth=12.0, solid_capstyle="butt", alpha=0.74)
        ax.scatter(q80_val, y_pos, s=58, color=color, edgecolors="white", linewidths=1.5, zorder=4)
        ax.text(
            -0.006,
            y_pos + 0.05,
            label,
            transform=ytrans,
            fontsize=10.4,
            va="center",
            ha="right",
            color=TEXT_DARK,
            fontweight="bold",
            clip_on=False,
        )
        if label == "Best IQP Parity":
            ax.text(
                -0.006,
                y_pos - 0.28,
                r"$\sigma = 1, K = 512$",
                transform=ytrans,
                fontsize=8.2,
                va="center",
                ha="right",
                color=TEXT_MID,
                clip_on=False,
            )
        ax.text(
            q80_val + metric_label_dx,
            y_pos + 0.07,
            f"AUC {auc_val:.3f}",
            fontsize=9.2,
            fontweight="bold",
            va="center",
            ha="left",
            color=TEXT_DARK,
        )
        ax.text(
            q80_val + metric_label_dx,
            y_pos - 0.24,
            rf"$R(10^4)$ {r_end:.3f}",
            fontsize=8.1,
            va="center",
            ha="left",
            color=TEXT_MID,
        )


def _style_heatmap_colorbar(cbar, cax, *, q80_min: float, q80_max: float) -> None:
    q80_mid = 0.5 * (float(q80_min) + float(q80_max))
    cbar.set_label(HEATMAP_CBAR_LABEL, labelpad=6, fontsize=10.5)
    cbar.set_ticks([float(q80_min), q80_mid, float(q80_max)])
    cbar.set_ticklabels([_fmt_int(q80_min), _fmt_int(q80_mid), _fmt_int(q80_max)])
    cbar.outline.set_visible(False)
    cax.tick_params(axis="x", length=3, colors="#555555")


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_config(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _write_readme(
    path: Path,
    data_npz_rel: str,
    q_thr: float,
    best_sigma: float,
    best_k: int,
    best_metrics: Dict[str, float],
    mse_metrics: Dict[str, float],
    target_metrics: Dict[str, float],
    uniform_metrics: Dict[str, float],
) -> None:
    delta_best = mse_metrics["Q80"] - best_metrics["Q80"]
    lines = [
        "# Fig2 Recovery Summary Panels",
        "",
        "This directory contains a companion summary view for the frozen Fig2 sigma-K ablation.",
        "",
        "Inputs:",
        "",
        f"- frozen snapshot: `{data_npz_rel}`",
        "- no retraining; all metrics are derived from the stored recovery curves",
        "- heatmap palette: fixed red-black house style",
        "",
        "Primary summary metric:",
        "",
        f"- `Q80`: first interpolated `Q` where `R(Q) >= {q_thr:.1f}`",
        "- `AUC_norm`: normalized area under the recovery curve on `[0, 10000]`",
        "- `R10000`: terminal recovery at `Q=10000`",
        "",
        "Main visual:",
        "",
        "- left panel: heatmap over the 12 parity settings in the frozen sigma-K grid",
        "- heatmap cell text: absolute `Q80(IQP Parity)` in samples",
        "- heatmap color: same absolute `Q80(IQP Parity)` quantity",
        "- lower `Q80` settings are rendered in brighter red; higher `Q80` settings darken toward black",
        "- cell text and color now encode the same parity-only `Q80` metric",
        "",
        "Right benchmark panel:",
        "",
        "- `Target p*`, `Best IQP Parity`, `IQP MSE`, `Uniform`",
        "- x-axis is `Q80` in samples; lower is better",
        "- inline labels report `AUC` and `R10000`",
        "",
        "Headline result:",
        "",
        f"- best parity setting: `sigma={best_sigma:g}, K={best_k}`",
        f"- best parity `Q80 = {best_metrics['Q80']:.2f}`",
        f"- IQP MSE `Q80 = {mse_metrics['Q80']:.2f}`",
        f"- best parity advantage vs IQP MSE: `{delta_best:.2f}` fewer samples to reach 80% recovery",
        f"- target `Q80 = {target_metrics['Q80']:.2f}`",
        f"- uniform `Q80 = {uniform_metrics['Q80']:.2f}`",
        "",
        "Kept files:",
        "",
        f"- `{STEM}.pdf`",
        f"- `{STEM}.png`",
        f"- `{STEM}_heatmap_only.pdf`",
        f"- `{STEM}_heatmap_only.png`",
        f"- `{STEM}_benchmark_only.pdf`",
        f"- `{STEM}_benchmark_only.png`",
        f"- `{STEM}_metrics.csv`",
        "- `RUN_CONFIG.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    ap = argparse.ArgumentParser(description="Render the Fig2 companion recovery-summary panels.")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / OUTDIR_REL_DEFAULT),
    )
    ap.add_argument(
        "--data-npz",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig2_iqp_sigmak_ablation_recovery" / "fig2_data_default.npz"),
    )
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    q_thr = 0.8
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_npz = Path(args.data_npz)
    if not data_npz.exists():
        raise FileNotFoundError(f"Missing data file: {data_npz}")
    try:
        data_npz_rel = str(data_npz.relative_to(ROOT))
    except ValueError:
        data_npz_rel = str(data_npz)
    try:
        outdir_rel = str(outdir.relative_to(ROOT))
    except ValueError:
        outdir_rel = str(outdir)

    with np.load(data_npz, allow_pickle=False) as z:
        q = np.asarray(z["Q"], dtype=np.float64)
        y_target = np.asarray(z["y_target"], dtype=np.float64)
        y_unif = np.asarray(z["y_unif"], dtype=np.float64)
        y_ref = np.asarray(z["y_ref"], dtype=np.float64)
        y_best = np.asarray(z["y_best"], dtype=np.float64)
        y_other = np.asarray(z["y_other"], dtype=np.float64)
        best_sigma = float(z["best_sigma"])
        best_k = int(z["best_K"])
        other_sigma = np.asarray(z["other_sigma"], dtype=np.float64)
        other_k = np.asarray(z["other_K"], dtype=np.int64)
        reference_loss = str(z["reference_loss"]) if "reference_loss" in z else "prob_mse"

    target_metrics = _curve_metrics(q=q, y=y_target, q_thr=q_thr)
    uniform_metrics = _curve_metrics(q=q, y=y_unif, q_thr=q_thr)
    mse_metrics = _curve_metrics(q=q, y=y_ref, q_thr=q_thr)
    best_metrics = _curve_metrics(q=q, y=y_best, q_thr=q_thr)

    parity_rows: List[Dict[str, object]] = [
        {
            "curve_group": "iqp_parity",
            "curve_label": "IQP Parity",
            "sigma": best_sigma,
            "K": best_k,
            "is_best_setting": 1,
            **best_metrics,
        }
    ]
    for sigma, kval, y in zip(other_sigma, other_k, y_other):
        parity_rows.append(
            {
                "curve_group": "iqp_parity",
                "curve_label": "IQP Parity",
                "sigma": float(sigma),
                "K": int(kval),
                "is_best_setting": 0,
                **_curve_metrics(q=q, y=np.asarray(y, dtype=np.float64), q_thr=q_thr),
            }
        )

    q80_mse = float(mse_metrics["Q80"])
    auc_mse = float(mse_metrics["AUC_norm"])
    rows: List[Dict[str, object]] = []
    for entry in parity_rows:
        rows.append(
            {
                **entry,
                "Q80_delta_vs_mse": float(q80_mse - float(entry["Q80"])),
                "AUC_delta_vs_mse": float(float(entry["AUC_norm"]) - auc_mse),
                "reference_loss": reference_loss,
            }
        )

    rows.extend(
        [
            {
                "curve_group": "benchmark",
                "curve_label": "Target p*",
                "sigma": np.nan,
                "K": np.nan,
                "is_best_setting": 0,
                **target_metrics,
                "Q80_delta_vs_mse": float(q80_mse - target_metrics["Q80"]),
                "AUC_delta_vs_mse": float(target_metrics["AUC_norm"] - auc_mse),
                "reference_loss": reference_loss,
            },
            {
                "curve_group": "benchmark",
                "curve_label": "IQP MSE",
                "sigma": np.nan,
                "K": np.nan,
                "is_best_setting": 0,
                **mse_metrics,
                "Q80_delta_vs_mse": 0.0,
                "AUC_delta_vs_mse": 0.0,
                "reference_loss": reference_loss,
            },
            {
                "curve_group": "benchmark",
                "curve_label": "Uniform",
                "sigma": np.nan,
                "K": np.nan,
                "is_best_setting": 0,
                **uniform_metrics,
                "Q80_delta_vs_mse": float(q80_mse - uniform_metrics["Q80"]),
                "AUC_delta_vs_mse": float(uniform_metrics["AUC_norm"] - auc_mse),
                "reference_loss": reference_loss,
            },
        ]
    )

    sigma_values = sorted(float(r["sigma"]) for r in parity_rows)
    sigma_values = sorted(set(sigma_values))
    k_values = sorted(int(r["K"]) for r in parity_rows)
    k_values = sorted(set(k_values))

    grid_shape = (len(sigma_values), len(k_values))
    q80_grid = np.full(grid_shape, np.nan, dtype=np.float64)
    best_mask = np.zeros(grid_shape, dtype=bool)
    for entry in rows:
        if str(entry["curve_group"]) != "iqp_parity":
            continue
        i = sigma_values.index(float(entry["sigma"]))
        j = k_values.index(int(entry["K"]))
        q80_grid[i, j] = float(entry["Q80"])
        best_mask[i, j] = bool(int(entry["is_best_setting"]))

    q80_min = float(np.nanmin(q80_grid))
    q80_max = float(np.nanmax(q80_grid))
    norm = colors.Normalize(vmin=q80_min, vmax=q80_max)
    cmap = colors.LinearSegmentedColormap.from_list(
        "parity_q80_red_black",
        [HEATMAP_LOW, HEATMAP_MID, HEATMAP_HIGH],
    )

    apply_final_style()
    fig = plt.figure(figsize=(FIG_W * 2.52, FIG_H * 1.46), facecolor="white")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.15],
        height_ratios=[15.0, 1.2],
        left=0.09,
        right=0.985,
        top=0.92,
        bottom=0.18,
        wspace=0.34,
        hspace=0.24,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bench = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 0])
    ax_pad = fig.add_subplot(gs[1, 1])
    ax_pad.axis("off")

    im = _render_heatmap_panel(
        ax_heat,
        sigma_values=sigma_values,
        k_values=k_values,
        q80_grid=q80_grid,
        best_mask=best_mask,
        cmap=cmap,
        norm=norm,
    )
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    _style_heatmap_colorbar(cbar, cax, q80_min=q80_min, q80_max=q80_max)
    _render_benchmark_panel(
        ax_bench,
        target_metrics=target_metrics,
        best_metrics=best_metrics,
        mse_metrics=mse_metrics,
        uniform_metrics=uniform_metrics,
    )

    metrics_csv = outdir / f"{STEM}_metrics.csv"
    run_config_json = outdir / "RUN_CONFIG.json"
    readme_md = outdir / "README.md"
    out_pdf = outdir / f"{STEM}.pdf"
    out_png = outdir / f"{STEM}.png"
    out_heat_pdf = outdir / f"{STEM}_heatmap_only.pdf"
    out_heat_png = outdir / f"{STEM}_heatmap_only.png"
    out_bench_pdf = outdir / f"{STEM}_benchmark_only.pdf"
    out_bench_png = outdir / f"{STEM}_benchmark_only.png"

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    fig_heat = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white", constrained_layout=True)
    gs_heat = GridSpec(
        2,
        1,
        figure=fig_heat,
        height_ratios=[15.0, 1.2],
        hspace=0.10,
    )
    ax_heat_only = fig_heat.add_subplot(gs_heat[0, 0])
    cax_heat_only = fig_heat.add_subplot(gs_heat[1, 0])
    im_heat_only = _render_heatmap_panel(
        ax_heat_only,
        sigma_values=sigma_values,
        k_values=k_values,
        q80_grid=q80_grid,
        best_mask=best_mask,
        cmap=cmap,
        norm=norm,
    )
    cbar_heat_only = fig_heat.colorbar(im_heat_only, cax=cax_heat_only, orientation="horizontal")
    _style_heatmap_colorbar(cbar_heat_only, cax_heat_only, q80_min=q80_min, q80_max=q80_max)
    fig_heat.savefig(out_heat_pdf)
    fig_heat.savefig(out_heat_png, dpi=int(args.dpi))
    plt.close(fig_heat)

    fig_bench, ax_bench_only = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="white", constrained_layout=True)
    _render_benchmark_panel(
        ax_bench_only,
        target_metrics=target_metrics,
        best_metrics=best_metrics,
        mse_metrics=mse_metrics,
        uniform_metrics=uniform_metrics,
    )
    fig_bench.savefig(out_bench_pdf)
    fig_bench.savefig(out_bench_png, dpi=int(args.dpi))
    plt.close(fig_bench)

    _write_csv(metrics_csv, rows)
    _write_run_config(
        run_config_json,
        {
            "selected_analysis_run": True,
            "script": SCRIPT_REL,
            "outdir": outdir_rel,
            "data_npz": data_npz_rel,
                "selected_output": f"{outdir_rel}/{STEM}.pdf",
                "q_threshold": q_thr,
                "reference_loss": reference_loss,
                "best_sigma": best_sigma,
                "best_K": best_k,
                "rerun_command": f"MPLCONFIGDIR=/tmp/mpl-cache python {SCRIPT_REL} --outdir {outdir_rel}",
                "output_files": [
                    f"{outdir_rel}/{STEM}.pdf",
                f"{outdir_rel}/{STEM}.png",
                f"{outdir_rel}/{STEM}_heatmap_only.pdf",
                f"{outdir_rel}/{STEM}_heatmap_only.png",
                f"{outdir_rel}/{STEM}_benchmark_only.pdf",
                f"{outdir_rel}/{STEM}_benchmark_only.png",
                f"{outdir_rel}/{STEM}_metrics.csv",
                f"{outdir_rel}/RUN_CONFIG.json",
                f"{outdir_rel}/README.md",
            ],
        },
    )
    _write_readme(
        path=readme_md,
        data_npz_rel=data_npz_rel,
        q_thr=q_thr,
        best_sigma=best_sigma,
        best_k=best_k,
        best_metrics=best_metrics,
        mse_metrics=mse_metrics,
        target_metrics=target_metrics,
        uniform_metrics=uniform_metrics,
    )

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_heat_pdf}")
    print(f"[saved] {out_heat_png}")
    print(f"[saved] {out_bench_pdf}")
    print(f"[saved] {out_bench_png}")
    print(f"[saved] {metrics_csv}")
    print(f"[saved] {run_config_json}")
    print(f"[saved] {readme_md}")


if __name__ == "__main__":
    run()
