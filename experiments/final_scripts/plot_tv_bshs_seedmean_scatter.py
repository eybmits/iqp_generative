#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final plot script: TV_score vs BSHS seed-mean scatter (single panel only).

The committed default CSV is a historical 12-seed frozen snapshot; benchmark-standard
20-seed reruns are documented under experiments/analysis/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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

MODEL_MARKERS: Dict[str, str] = {
    "iqp_parity_mse": "o",
    "classical_nnn_fields_parity": "s",
    "classical_dense_fields_xent": "D",
    "classical_transformer_mle": "^",
    "classical_maxent_parity": "X",
}

MODEL_LABELS: Dict[str, str] = {
    "iqp_parity_mse": "IQP (parity)",
    "classical_nnn_fields_parity": "Ising NN+NNN",
    "classical_dense_fields_xent": "Dense Ising",
    "classical_transformer_mle": "AR Transf.",
    "classical_maxent_parity": "MaxEnt",
}

MODEL_EDGE_COLORS: Dict[str, str] = {
    "iqp_parity_mse": "#C40000",
    "classical_nnn_fields_parity": "#1f77b4",
    "classical_dense_fields_xent": "#8c564b",
    "classical_transformer_mle": "#17becf",
    "classical_maxent_parity": "#9467bd",
}

MODEL_SHORT_LABELS: Dict[str, str] = {
    "iqp_parity_mse": "IQP",
    "classical_nnn_fields_parity": "NN+NNN",
    "classical_dense_fields_xent": "Dense",
    "classical_transformer_mle": "AR",
    "classical_maxent_parity": "MaxEnt",
}

MODEL_ORDER: List[str] = [
    "iqp_parity_mse",
    "classical_nnn_fields_parity",
    "classical_dense_fields_xent",
    "classical_transformer_mle",
    "classical_maxent_parity",
]


def _pareto_mask_minx_maxy(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    n = int(xv.size)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        xi = float(xv[i])
        yi = float(yv[i])
        dominated = False
        for j in range(n):
            if i == j:
                continue
            xj = float(xv[j])
            yj = float(yv[j])
            if (xj <= xi + eps) and (yj >= yi - eps) and ((xj < xi - eps) or (yj > yi + eps)):
                dominated = True
                break
        keep[i] = not dominated
    return keep


def _build_bxp_stats(vals: np.ndarray, y_min: float, y_max: float, min_box_h_frac: float = 0.03) -> Dict[str, object]:
    v = np.asarray(vals, dtype=float)
    q1 = float(np.percentile(v, 25))
    med = float(np.percentile(v, 50))
    q3 = float(np.percentile(v, 75))
    lo = float(np.min(v))
    hi = float(np.max(v))
    span = max(1e-9, float(y_max - y_min))
    min_box_h = max(1e-4, min_box_h_frac * span)
    if (q3 - q1) < min_box_h:
        half = 0.5 * min_box_h
        q1 = max(y_min, med - half)
        q3 = min(y_max, med + half)
        lo = min(lo, q1)
        hi = max(hi, q3)
    return {"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi, "fliers": []}


def _render_model_boxplot(
    df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    out_stem: str,
    outdir: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    ax.set_facecolor("white")

    model_keys = [k for k in MODEL_ORDER if k in set(df["model_key"].astype(str).tolist())]
    box_data: List[np.ndarray] = []
    box_labels: List[str] = []
    kept_keys: List[str] = []
    for key in model_keys:
        vals = df.loc[df["model_key"] == key, metric_col].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        box_data.append(vals)
        box_labels.append(MODEL_SHORT_LABELS.get(key, key))
        kept_keys.append(key)
    if not box_data:
        plt.close(fig)
        raise RuntimeError(f"No data for metric {metric_col}.")

    all_vals = np.concatenate(box_data)
    ymin_data = float(np.min(all_vals))
    ymax_data = float(np.max(all_vals))
    if metric_col == "BSHS":
        y_min = 0.0
        y_max = 1.0
    else:
        pad = 0.10 * max(1e-6, ymax_data - ymin_data)
        y_min = max(0.0, ymin_data - pad)
        y_max = ymax_data + pad
        if y_max <= y_min:
            y_max = y_min + 0.05

    bxp_stats: List[Dict[str, object]] = [_build_bxp_stats(v, y_min=y_min, y_max=y_max) for v in box_data]
    bp = ax.bxp(
        bxp_stats,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops={"linewidth": 1.2},
    )

    for i, patch in enumerate(bp["boxes"]):
        edge = MODEL_EDGE_COLORS.get(kept_keys[i], "#444444")
        patch.set_facecolor("white")
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.2)
    for i, med in enumerate(bp["medians"]):
        med.set_color(MODEL_EDGE_COLORS.get(kept_keys[i], "#333333"))
    whisker_cols: List[str] = []
    cap_cols: List[str] = []
    for k in kept_keys:
        whisker_cols.extend([MODEL_EDGE_COLORS.get(k, "#555555")] * 2)
        cap_cols.extend([MODEL_EDGE_COLORS.get(k, "#555555")] * 2)
    for ln, col in zip(bp["whiskers"], whisker_cols):
        ln.set_color(col)
        ln.set_linewidth(1.0)
    for ln, col in zip(bp["caps"], cap_cols):
        ln.set_color(col)
        ln.set_linewidth(1.0)

    ax.set_xticks(np.arange(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel(y_label)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    out_pdf = outdir / f"{out_stem}.pdf"
    out_png = outdir / f"{out_stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


def _style_bxp(
    bp: Dict[str, object],
    kept_keys: List[str],
    *,
    filled: bool,
    fill_alpha: float = 0.16,
    hatched: bool = False,
) -> None:
    for i, patch in enumerate(bp["boxes"]):
        edge = MODEL_EDGE_COLORS.get(kept_keys[i], "#444444")
        if filled:
            c = np.array(plt.matplotlib.colors.to_rgb(edge))
            face = (float(c[0]), float(c[1]), float(c[2]), float(fill_alpha))
            patch.set_facecolor(face)
        else:
            patch.set_facecolor("white")
        if hatched:
            patch.set_hatch("////")
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.35 if filled else 1.2)
    for i, med in enumerate(bp["medians"]):
        med.set_color(MODEL_EDGE_COLORS.get(kept_keys[i], "#333333"))
        med.set_linewidth(1.3)
    whisker_cols: List[str] = []
    cap_cols: List[str] = []
    for k in kept_keys:
        whisker_cols.extend([MODEL_EDGE_COLORS.get(k, "#555555")] * 2)
        cap_cols.extend([MODEL_EDGE_COLORS.get(k, "#555555")] * 2)
    for ln, col in zip(bp["whiskers"], whisker_cols):
        ln.set_color(col)
        ln.set_linewidth(1.0)
    for ln, col in zip(bp["caps"], cap_cols):
        ln.set_color(col)
        ln.set_linewidth(1.0)


def _render_dual_axis_boxplot(
    df: pd.DataFrame,
    out_stem: str,
    outdir: Path,
    dpi: int,
    right_metric_col: str = "TV_score",
    right_metric_axis_label: str = r"TV$_{score}$",
    right_metric_legend_label: str = "TVscore",
) -> None:
    fig, ax_l = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    ax_r = ax_l.twinx()
    ax_l.set_facecolor("white")

    model_keys = [k for k in MODEL_ORDER if k in set(df["model_key"].astype(str).tolist())]
    kept_keys: List[str] = []
    labels: List[str] = []
    bshs_data: List[np.ndarray] = []
    right_metric_data: List[np.ndarray] = []

    for key in model_keys:
        v_b = df.loc[df["model_key"] == key, "BSHS"].astype(float).to_numpy()
        v_t = df.loc[df["model_key"] == key, right_metric_col].astype(float).to_numpy()
        v_b = v_b[np.isfinite(v_b)]
        v_t = v_t[np.isfinite(v_t)]
        if v_b.size == 0 or v_t.size == 0:
            continue
        kept_keys.append(key)
        labels.append(MODEL_SHORT_LABELS.get(key, key))
        bshs_data.append(v_b)
        right_metric_data.append(v_t)

    if not kept_keys:
        plt.close(fig)
        raise RuntimeError("No finite data available for dual-axis boxplot.")

    x = np.arange(1, len(kept_keys) + 1, dtype=float)
    pos_b = x - 0.20
    pos_t = x + 0.20
    w = 0.30

    b_all = np.concatenate(bshs_data)
    b_min = float(np.min(b_all))
    b_max = float(np.max(b_all))
    b_pad = 0.10 * max(1e-6, b_max - b_min)
    b_lo = max(0.0, b_min - b_pad)
    b_hi = min(1.0, b_max + b_pad)
    if b_hi <= b_lo:
        b_hi = min(1.0, b_lo + 0.05)
    bshs_stats = [_build_bxp_stats(v, y_min=b_lo, y_max=b_hi, min_box_h_frac=0.03) for v in bshs_data]

    tv_all = np.concatenate(right_metric_data)
    tv_min = float(np.min(tv_all))
    tv_max = float(np.max(tv_all))
    tv_pad = 0.10 * max(1e-6, tv_max - tv_min)
    tv_lo = max(0.0, tv_min - tv_pad)
    tv_hi = tv_max + tv_pad
    if tv_hi <= tv_lo:
        tv_hi = tv_lo + 0.05
    tv_stats = [_build_bxp_stats(v, y_min=tv_lo, y_max=tv_hi, min_box_h_frac=0.05) for v in right_metric_data]

    bp_b = ax_l.bxp(
        bshs_stats,
        positions=pos_b,
        widths=w,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.2},
    )
    bp_t = ax_r.bxp(
        tv_stats,
        positions=pos_t,
        widths=w,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.2},
    )
    _style_bxp(bp_b, kept_keys, filled=False, hatched=False)
    _style_bxp(bp_t, kept_keys, filled=True, fill_alpha=0.20, hatched=True)

    ax_l.set_xlim(0.45, len(kept_keys) + 0.55)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_l.set_xlabel("Model")
    ax_l.set_ylabel(r"Support $BSHS(Q)$")
    ax_l.set_ylim(b_lo, b_hi)
    ax_l.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    ax_r.set_ylabel(right_metric_axis_label)
    ax_r.set_ylim(tv_lo, tv_hi)
    ax_r.grid(False)

    proxy_support = plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#444444", linewidth=1.2, label="Support")
    proxy_tv = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor=(0.4, 0.4, 0.4, 0.20),
        edgecolor="#444444",
        linewidth=1.2,
        hatch="////",
        label=right_metric_legend_label,
    )
    leg = ax_l.legend(
        handles=[proxy_support, proxy_tv],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        borderaxespad=0.0,
        frameon=True,
        fontsize=7.0,
        facecolor="white",
        edgecolor="#bfbfbf",
    )
    try:
        leg._legend_box.align = "center"
    except Exception:
        pass

    out_pdf = outdir / f"{out_stem}.pdf"
    out_png = outdir / f"{out_stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(dpi))
    plt.close(fig)
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


def run() -> None:
    ap = argparse.ArgumentParser(description="Final seed-mean TV_score vs BSHS scatter plot.")
    ap.add_argument(
        "--points-csv",
        type=str,
        default=str(
            ROOT
            / "outputs"
            / "final_plots"
            / "fig3_tv_bshs_seedmean_scatter"
            / "tv_bshs_points_multiseed_beta_q1000_no_iqp_mse_beta0p9_newseeds12.csv"
        ),
        help="Frozen historical 12-seed snapshot input. For benchmark-standard 20-seed reruns use the analysis drivers.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "outputs" / "final_plots" / "fig3_tv_bshs_seedmean_scatter"),
    )
    ap.add_argument("--q-eval", type=int, default=1000)
    ap.add_argument("--beta-fixed", type=float, default=0.90, help="If set, plot only this beta value.")
    ap.add_argument("--beta-tol", type=float, default=1e-9)
    ap.add_argument("--draw-pareto-front", type=int, default=1, choices=[0, 1])
    ap.add_argument("--draw-model-trends", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--support-boxplot-right",
        type=int,
        default=1,
        choices=[0, 1],
        help="For fixed beta plots, add a simple right-side boxplot of BSHS over seeds per model.",
    )
    ap.add_argument(
        "--beta-fixed-two-boxplots",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1 (and beta-fixed is set), create two separate boxplots: one for TV_score and one for BSHS.",
    )
    ap.add_argument(
        "--beta-fixed-dual-axis-boxplot",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1 (and beta-fixed is set), create one combined boxplot with left BSHS axis and right TV_score axis.",
    )
    ap.add_argument("--right-metric-col", type=str, default="TV_score")
    ap.add_argument("--right-metric-axis-label", type=str, default=r"TV$_{score}$")
    ap.add_argument("--right-metric-legend-label", type=str, default="TVscore")
    ap.add_argument("--hide-title", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dpi", type=int, default=PNG_DPI)
    args = ap.parse_args()

    points_csv = Path(args.points_csv)
    if not points_csv.exists():
        raise FileNotFoundError(f"points csv not found: {points_csv}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    apply_final_style()
    # Per user request: keep this panel on the compact 2-column style block.
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
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
        }
    )

    df = pd.read_csv(points_csv)
    required_cols = [str(args.right_metric_col), "BSHS", "beta", "seed"]
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid points in CSV.")

    if args.beta_fixed is not None:
        beta_target = float(args.beta_fixed)
        tol = float(args.beta_tol)
        df = df[np.isclose(df["beta"].astype(float), beta_target, atol=tol, rtol=0.0)].copy()
        if df.empty:
            raise RuntimeError(f"No rows found for beta={beta_target:g}.")

    if bool(int(args.beta_fixed_two_boxplots)):
        if args.beta_fixed is None:
            raise RuntimeError("--beta-fixed-two-boxplots requires --beta-fixed.")
        beta_tag = f"{float(args.beta_fixed):.2f}".replace(".", "p")
        _render_model_boxplot(
            df=df,
            metric_col="BSHS",
            y_label=r"Support $BSHS(Q)$",
            out_stem=f"fig3_tv_bshs_seedmean_scatter_beta_{beta_tag}_support_boxplot",
            outdir=outdir,
            dpi=int(args.dpi),
        )
        _render_model_boxplot(
            df=df,
            metric_col=str(args.right_metric_col),
            y_label=str(args.right_metric_axis_label),
            out_stem=f"fig3_tv_bshs_seedmean_scatter_beta_{beta_tag}_tvscore_boxplot",
            outdir=outdir,
            dpi=int(args.dpi),
        )
        return

    if bool(int(args.beta_fixed_dual_axis_boxplot)):
        if args.beta_fixed is None:
            raise RuntimeError("--beta-fixed-dual-axis-boxplot requires --beta-fixed.")
        beta_tag = f"{float(args.beta_fixed):.2f}".replace(".", "p")
        _render_dual_axis_boxplot(
            df=df,
            out_stem=f"fig3_tv_bshs_seedmean_scatter_beta_{beta_tag}_dual_axis_boxplot",
            outdir=outdir,
            dpi=int(args.dpi),
            right_metric_col=str(args.right_metric_col),
            right_metric_axis_label=str(args.right_metric_axis_label),
            right_metric_legend_label=str(args.right_metric_legend_label),
        )
        return

    # Seed-mean aggregation: one point per (model, beta).
    df_mean = (
        df.groupby(["model_key", "model_label", "beta"], as_index=False)
        .agg(
            TV_score_mean=("TV_score", "mean"),
            BSHS_mean=("BSHS", "mean"),
            seeds_n=("seed", "nunique"),
        )
        .sort_values(["model_key", "beta"], ascending=[True, True])
        .reset_index(drop=True)
    )

    use_support_box = bool(int(args.support_boxplot_right)) and (args.beta_fixed is not None)
    if use_support_box:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(FIG_W, FIG_H),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [4.1, 1.5], "wspace": 0.08},
        )
        ax = axes[0]
        ax_box = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
        ax_box = None
    ax.set_facecolor("white")

    betas = df_mean["beta"].astype(float).to_numpy()
    bmin = float(np.min(betas))
    bmax = float(np.max(betas))
    if bmax <= bmin:
        bmax = bmin + 1e-6
    norm = Normalize(vmin=bmin, vmax=bmax)
    cmap = plt.get_cmap("Reds")

    for model_key in MODEL_ORDER:
        g = df_mean[df_mean["model_key"] == model_key].copy()
        if g.empty:
            continue
        g = g.sort_values("beta")
        x = g["TV_score_mean"].astype(float).to_numpy()
        y = g["BSHS_mean"].astype(float).to_numpy()
        c = g["beta"].astype(float).to_numpy()
        edge = MODEL_EDGE_COLORS.get(model_key, "#444444")
        marker = MODEL_MARKERS.get(model_key, "o")

        if bool(int(args.draw_model_trends)):
            ax.plot(x, y, color=edge, linewidth=1.0, alpha=0.45, zorder=2)

        ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap,
            norm=norm,
            s=85,
            marker=marker,
            edgecolors=edge,
            linewidths=1.2,
            alpha=0.95,
            zorder=4,
        )

    if bool(int(args.draw_pareto_front)):
        x_all = df_mean["TV_score_mean"].astype(float).to_numpy()
        y_all = df_mean["BSHS_mean"].astype(float).to_numpy()
        mask = _pareto_mask_minx_maxy(x_all, y_all)
        df_pf = df_mean.loc[mask].copy().sort_values("TV_score_mean")
        if len(df_pf) >= 1:
            ax.plot(
                df_pf["TV_score_mean"].astype(float).to_numpy(),
                df_pf["BSHS_mean"].astype(float).to_numpy(),
                color="#202020",
                linewidth=1.4,
                linestyle="-",
                alpha=0.9,
                zorder=6,
            )

    ax.set_xlabel(r"TV$_{score}$ (lower better)")
    ax.set_ylabel(r"BSHS$(Q)$ (higher better)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    if not bool(int(args.hide_title)):
        ax.set_title(f"Seed-mean model comparison (Q={int(args.q_eval)})")

    if not use_support_box:
        legend_handles: List[Line2D] = []
        present_keys = set(df_mean["model_key"].astype(str).tolist())
        for key in MODEL_ORDER:
            if key not in present_keys:
                continue
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=MODEL_MARKERS.get(key, "o"),
                    linestyle="None",
                    markerfacecolor="white",
                    markeredgecolor=MODEL_EDGE_COLORS.get(key, "#444444"),
                    markeredgewidth=1.5,
                    markersize=6.3,
                    label=MODEL_LABELS.get(key, key),
                )
            )
        if bool(int(args.draw_pareto_front)):
            legend_handles.append(Line2D([0], [0], color="#202020", lw=1.4, linestyle="-", label="Pareto front"))
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            fontsize=9,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.6,
            handletextpad=0.45,
            borderpad=0.25,
            labelspacing=0.25,
            frameon=True,
            facecolor="white",
            edgecolor="#bfbfbf",
        )

    if use_support_box and ax_box is not None:
        model_keys = [k for k in MODEL_ORDER if k in set(df["model_key"].astype(str).tolist())]
        box_data: List[np.ndarray] = []
        box_labels: List[str] = []
        for key in model_keys:
            vals = df.loc[df["model_key"] == key, "BSHS"].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            box_data.append(vals)
            box_labels.append(MODEL_SHORT_LABELS.get(key, key))

        if len(box_data) > 0:
            # Ensure each category has a visibly sized box even for near-zero variance.
            min_box_h = 0.018
            bxp_stats = []
            for vals in box_data:
                v = np.asarray(vals, dtype=float)
                q1 = float(np.percentile(v, 25))
                med = float(np.percentile(v, 50))
                q3 = float(np.percentile(v, 75))
                lo = float(np.min(v))
                hi = float(np.max(v))
                if (q3 - q1) < min_box_h:
                    half = 0.5 * min_box_h
                    q1 = max(0.0, med - half)
                    q3 = min(1.0, med + half)
                    lo = min(lo, q1)
                    hi = max(hi, q3)
                bxp_stats.append({"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi, "fliers": []})

            bp = ax_box.bxp(
                bxp_stats,
                patch_artist=True,
                widths=0.58,
                showfliers=False,
                medianprops={"linewidth": 1.2},
            )
            kept_keys: List[str] = []
            for key in model_keys:
                vals = df.loc[df["model_key"] == key, "BSHS"].astype(float).to_numpy()
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    kept_keys.append(key)
            for i, patch in enumerate(bp["boxes"]):
                edge = MODEL_EDGE_COLORS.get(kept_keys[i], "#444444")
                patch.set_facecolor("white")
                patch.set_edgecolor(edge)
                patch.set_linewidth(1.2)
            for i, med in enumerate(bp["medians"]):
                med.set_color(MODEL_EDGE_COLORS.get(kept_keys[i], "#333333"))
            whisker_cols: List[str] = []
            cap_cols: List[str] = []
            for k in kept_keys:
                whisker_cols.extend([MODEL_EDGE_COLORS.get(k, "#555555")] * 2)
                cap_cols.extend([MODEL_EDGE_COLORS.get(k, "#555555")] * 2)
            for ln, col in zip(bp["whiskers"], whisker_cols):
                ln.set_color(col)
                ln.set_linewidth(1.0)
            for ln, col in zip(bp["caps"], cap_cols):
                ln.set_color(col)
                ln.set_linewidth(1.0)

            ax_box.set_xticks(np.arange(1, len(box_labels) + 1))
            ax_box.set_xticklabels(box_labels, rotation=90, ha="center", fontsize=6)
        ax_box.set_ylim(0.0, 1.02)
        ax_box.set_ylabel("")
        ax_box.set_xlabel("Support", fontsize=8)
        ax_box.tick_params(axis="y", labelsize=8)
        ax_box.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax_box.set_facecolor("white")

    if args.beta_fixed is None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$\beta$")

    if args.beta_fixed is None:
        stem = "fig3_tv_bshs_seedmean_scatter"
    else:
        stem = f"fig3_tv_bshs_seedmean_scatter_beta_{float(args.beta_fixed):.2f}".replace(".", "p")
    out_pdf = outdir / f"{stem}.pdf"
    out_png = outdir / f"{stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    run()
